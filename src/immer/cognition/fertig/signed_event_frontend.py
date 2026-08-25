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
from .discourse_ssa import (
    DiscourseNumber,
    DiscourseSSAError,
    TypedDiscourseSSA,
)
from .signed_expression import (
    AbsoluteExpr,
    CeilingExpr,
    ClosedShareExpr,
    Definition,
    ExpressionCompileError,
    ExpressionCompileResult,
    ExpressionProgram,
    ExpressionTarget,
    GroundProductExpr,
    LiteralExpr,
    MeanExpr,
    NumericEvidence,
    ProductExpr,
    QuotientExpr,
    RefExpr,
    SignedTerm,
    SumExpr,
    UnitConversionExpr,
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


_UNICODE_FRACTIONS = {
    "¼": Fraction(1, 4),
    "½": Fraction(1, 2),
    "¾": Fraction(3, 4),
    "⅐": Fraction(1, 7),
    "⅑": Fraction(1, 9),
    "⅒": Fraction(1, 10),
    "⅓": Fraction(1, 3),
    "⅔": Fraction(2, 3),
    "⅕": Fraction(1, 5),
    "⅖": Fraction(2, 5),
    "⅗": Fraction(3, 5),
    "⅘": Fraction(4, 5),
    "⅙": Fraction(1, 6),
    "⅚": Fraction(5, 6),
    "⅛": Fraction(1, 8),
    "⅜": Fraction(3, 8),
    "⅝": Fraction(5, 8),
    "⅞": Fraction(7, 8),
}

_TOKEN = re.compile(
    r"\$\s*(?:\d[\d,]*(?:\.\d+)?|\.\d+)|"
    r"(?:\d+\s*/\s*\d+|\d[\d,]*(?:\.\d+)?|\.\d+|"
    r"[\u00bc-\u00be\u2150-\u215e])|"
    r"[A-Za-z]+(?:['’\-][A-Za-z]+)*|[.!?;,:]"
)
_DIGIT = re.compile(
    r"^\$?\s*(?:\d+\s*/\s*\d+|\d[\d,]*(?:\.\d+)?|\.\d+|"
    r"[\u00bc-\u00be\u2150-\u215e])$"
)
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
MINUTE = Unit("minute", (("time", 1),), Fraction(60))
HOUR = Unit("hour", (("time", 1),), Fraction(3600))
FOOT = Unit.base("length", symbol="ft")
FOOT_PER_COUNT = FOOT / COUNT
MONTH = Unit("month", (("time", 1),), Fraction(1))
COUNT_PER_MONTH = COUNT / MONTH
PERCENT = Unit("%", (), Fraction(1, 100))
MILLIMETER = Unit.base("length", symbol="mm")
SECOND = Unit("second", (("time", 1),), Fraction(1))
POINT = Unit.base("score", symbol="point")
MASS = Unit.base("mass", symbol="lb")
VOLUME = Unit.base("volume", symbol="gal")
INCH = Unit("inch", (("length", 1),), Fraction(1, 12))

_LOCAL_CARDINALS = {
    "zero": Fraction(0),
    "one": Fraction(1),
    "two": Fraction(2),
    "three": Fraction(3),
    "four": Fraction(4),
    "five": Fraction(5),
    "six": Fraction(6),
    "seven": Fraction(7),
    "eight": Fraction(8),
    "nine": Fraction(9),
    "ten": Fraction(10),
    "eleven": Fraction(11),
    "twelve": Fraction(12),
    "twenty": Fraction(20),
    "thirty": Fraction(30),
}

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


def _ssa_family(planner):
    """Translate typed discourse failures into the frontend's closed statuses."""

    def guarded(source: str, clause_set: tuple[Clause, ...]):
        try:
            return planner(source, clause_set)
        except DiscourseSSAError as exc:
            status = (
                FrontendStatus.AMBIGUOUS
                if exc.ambiguous
                else FrontendStatus.INVALID
            )
            raise _Reject(status, exc.reason) from exc

    guarded.__name__ = planner.__name__
    return guarded


def _fraction(text: str) -> Fraction:
    raw = text.replace("$", "").replace(",", "").strip()
    if raw in _UNICODE_FRACTIONS:
        return _UNICODE_FRACTIONS[raw]
    if "/" in raw:
        numerator, denominator = raw.split("/", 1)
        return Fraction(int(numerator.strip()), int(denominator.strip()))
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
    if word == "feet":
        return "foot"
    if word == "jewell":
        return "jewel"
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


def _item_id(words: tuple[str, ...] | list[str]) -> str:
    parts = [word for word in words if word not in {"a", "an", "and", "one", "the"}]
    if not parts:
        raise _Reject(FrontendStatus.AMBIGUOUS, "item scope is empty")
    parts[-1] = _singular(parts[-1])
    return "_".join(parts)


def _cardinal_literal(
    builder: _Builder, token: Token, unit: Unit = SCALAR
) -> LiteralExpr:
    value = _LOCAL_CARDINALS.get(token.norm)
    if value is None:
        raise _Reject(
            FrontendStatus.UNSUPPORTED, "word cardinal is outside the grammar"
        )
    return builder.lexical_literal(token, value, unit)


def _word_cardinals(clause: Clause, *, after: int = 0) -> list[Token]:
    return [token for token in clause.tokens[after:] if token.norm in _LOCAL_CARDINALS]


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


def _require_single_target_marker(question: Clause) -> None:
    markers = [
        token
        for token in question.tokens
        if token.norm in {"calculate", "find", "how", "what"}
    ]
    _require(
        len(markers) == 1,
        "question contains multiple or missing target markers",
        FrontendStatus.AMBIGUOUS,
    )


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


def _discounted_purchase_ledger(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not (
        "spend" in question.norms
        and "items" in question.norms
        and "buy" in question.norms
    ):
        return None
    store_clause = _single_clause(
        clause_set,
        lambda clause: (
            "sells" in clause.norms
            and not _money_tokens(clause)
            and not _counts(clause)
        ),
        reason="purchase store scope is not unique",
    )
    sells_index = store_clause.norms.index("sells")
    _require(sells_index > 0, "purchase store noun is missing")
    store = _singular(store_clause.norms[sells_index - 1])
    prices_clause = _single_clause(
        clause_set,
        lambda clause: (
            "sold" in clause.norms
            and "each" in clause.norms
            and len(_money_tokens(clause)) > 1
        ),
        reason="purchase price schedule is not unique",
    )
    purchase = _single_clause(
        clause_set,
        lambda clause: (
            "wants" in clause.norms
            and "buy" in clause.norms
            and bool(_word_cardinals(clause))
        ),
        reason="purchase quantities are not unique",
    )
    discount_clause = _single_clause(
        clause_set,
        lambda clause: "discount" in clause.norms and bool(_counts(clause)),
        reason="purchase discount is not unique",
    )
    does = question.norms.index("does") if "does" in question.norms else -1
    wants_index = purchase.norms.index("wants") if "wants" in purchase.norms else -1
    buyer = question.norms[does + 1] if 0 <= does < len(question.tokens) - 1 else ""
    purchase_subject = purchase.norms[wants_index - 1] if wants_index > 0 else ""
    _require(
        buyer
        and purchase.norms[0] == buyer
        and purchase_subject in {buyer, "he", "she", "they"}
        and not any(
            token.text[:1].isupper() for token in purchase.tokens[1:wants_index]
        ),
        "purchase actor differs from the queried actor",
        FrontendStatus.AMBIGUOUS,
    )

    price_map: dict[str, Token] = {}
    prices = _money_tokens(prices_clause)
    for price in prices:
        index = prices_clause.tokens.index(price)
        at_offsets = [i for i in range(index) if prices_clause.norms[i] == "at"]
        _require(bool(at_offsets), "unit price lacks an at-item binding")
        at = at_offsets[-1]
        boundary = max(
            (i for i in range(at) if prices_clause.norms[i] in {",", "and"}),
            default=-1,
        )
        words = list(prices_clause.norms[boundary + 1 : at])
        if "is" in words:
            words = words[: words.index("is")]
        item = _item_id(words)
        _require(
            item not in price_map,
            "duplicate purchase price item",
            FrontendStatus.AMBIGUOUS,
        )
        price_map[item] = price

    quantity_map: dict[str, Token] = {}
    buy_index = purchase.norms.index("buy")
    for cardinal in _word_cardinals(purchase, after=buy_index + 1):
        index = purchase.tokens.index(cardinal)
        end = next(
            (
                i
                for i in range(index + 1, len(purchase.tokens))
                if purchase.norms[i] in {",", "and"}
            ),
            len(purchase.tokens),
        )
        item = _item_id(list(purchase.norms[index + 1 : end]))
        _require(
            item not in quantity_map,
            "duplicate purchase quantity item",
            FrontendStatus.AMBIGUOUS,
        )
        quantity_map[item] = cardinal
    _require(
        quantity_map.keys() == price_map.keys(),
        "purchase quantities and unit prices bind different items",
        FrontendStatus.AMBIGUOUS,
    )
    discount = _one_token(_counts(discount_clause), "discount percentage is incomplete")
    _require(
        discount.number is not None
        and 0 < discount.number < 100
        and source[discount.span.end : discount.span.end + 1] == "%"
        and discount_clause.norms[:2] == ("the", store)
        and _contains(discount_clause.norms, "on", "all", "the", "purchased", "items"),
        "discount basis is not explicit",
    )

    builder = _Builder(source, clause_set)
    subtotal_expr = _sum(
        *[
            _signed(
                1,
                _product(
                    _cardinal_literal(builder, quantity_map[item], COUNT),
                    builder.literal(price, PRICE_PER_COUNT),
                ),
                "purchase_line",
            )
            for item, price in price_map.items()
        ]
    )
    subtotal_symbol = SymbolKey(
        "buyer", "subtotal", "items", "purchase", "before_discount"
    )
    subtotal = Definition(subtotal_symbol, subtotal_expr, prices_clause.span)
    discount_expr = builder.literal(discount, PERCENT)
    expression = _sum(
        _signed(1, RefExpr(subtotal_symbol, question.span), "subtotal"),
        _signed(
            -1,
            _product(discount_expr, RefExpr(subtotal_symbol, question.span)),
            "discount",
        ),
    )
    return builder.finish(
        expression,
        "discounted_purchase_ledger",
        definitions=(subtotal,),
    )


def _batch_sale_profit(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not (
        "profit" in question.norms
        and "sell" in question.norms
        and "each" in question.norms
    ):
        return None
    input_clause = _single_clause(
        clause_set,
        lambda clause: "source" in clause.norms and "makes" in clause.norms,
        reason="input yield clause is not unique",
    )
    batch_clause = _single_clause(
        clause_set,
        lambda clause: "spend" in clause.norms and "per" in clause.norms,
        reason="batch cost clause is not unique",
    )
    source_index = input_clause.norms.index("source")
    spend_index = batch_clause.norms.index("spend")
    sell_index = question.norms.index("sell")
    _require(
        source_index == spend_index == 1
        and sell_index > 0
        and input_clause.norms[0]
        == batch_clause.norms[0]
        == question.norms[sell_index - 1],
        "profit clauses do not share one actor",
        FrontendStatus.AMBIGUOUS,
    )
    input_price = _one_token(_money_tokens(input_clause), "input price is incomplete")
    yield_count = _one_token(_counts(input_clause), "input yield is incomplete")
    batch_price = _one_token(_money_tokens(batch_clause), "batch price is incomplete")
    batch_count = _one_token(_counts(batch_clause), "batch size is incomplete")
    sale_price = _one_token(_money_tokens(question), "sale price is incomplete")
    sale_count = _one_token(_counts(question), "sale quantity is incomplete")
    batch_count_index = batch_clause.tokens.index(batch_count)
    per_offsets = [
        index for index, word in enumerate(batch_clause.norms) if word == "per"
    ]
    _require(
        len(per_offsets) == 1
        and per_offsets[0] + 1 == batch_count_index
        and batch_count_index + 2 == len(batch_clause.tokens),
        "batch cost has an additional or incomplete rate basis",
        FrontendStatus.AMBIGUOUS,
    )
    time_words = {
        "day",
        "days",
        "daily",
        "hour",
        "hours",
        "week",
        "weeks",
        "month",
        "months",
        "year",
        "years",
    }
    _require(
        not any(
            time_words.intersection(clause.norms)
            for clause in (input_clause, batch_clause, question)
        ),
        "batch profit has an unsupported time basis",
        FrontendStatus.AMBIGUOUS,
    )
    sale_item = _noun_after(question, sale_count)
    _require(
        sale_item
        == _noun_after(input_clause, yield_count)
        == _noun_after(batch_clause, batch_count),
        "yield, batch, and sale items differ",
        FrontendStatus.AMBIGUOUS,
    )
    price_index = input_clause.tokens.index(input_price)
    _require(
        price_index + 2 < len(input_clause.tokens)
        and input_clause.norms[price_index + 1] in {"a", "per"}
        and input_clause.norms[price_index + 2]
        in input_clause.norms[: input_clause.tokens.index(yield_count)],
        "input price unit is not bound to the yield source",
        FrontendStatus.AMBIGUOUS,
    )

    builder = _Builder(source, clause_set)
    sold_symbol = SymbolKey("seller", "quantity", sale_item, "sale", "sold")
    sold = Definition(
        sold_symbol,
        builder.literal(sale_count, COUNT),
        question.span,
    )
    sale_ref = RefExpr(sold_symbol, question.span)
    revenue = _product(
        sale_ref,
        builder.literal(sale_price, PRICE_PER_COUNT),
    )
    input_units = QuotientExpr(
        RefExpr(sold_symbol, input_clause.span),
        builder.literal(yield_count, SCALAR),
        _wide_span(sale_ref, yield_count, source),
    )
    batch_units = QuotientExpr(
        RefExpr(sold_symbol, batch_clause.span),
        builder.literal(batch_count, COUNT),
        _wide_span(sale_ref, batch_count, source),
    )
    expression = _sum(
        _signed(1, revenue, "sales"),
        _signed(
            -1,
            _product(input_units, builder.literal(input_price, PRICE_PER_COUNT)),
            "input_cost",
        ),
        _signed(
            -1,
            _product(batch_units, builder.literal(batch_price, MONEY)),
            "batch_cost",
        ),
    )
    return builder.finish(expression, "batch_sale_profit", definitions=(sold,))


def _bundle_relative_price_dag(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not (
        "price" in question.norms and _contains(question.norms, "for", "such", "a")
    ):
        return None
    bundle_index = next(
        index + 3
        for index in range(len(question.norms) - 3)
        if question.norms[index : index + 3] == ("for", "such", "a")
    )
    _require(bundle_index < len(question.tokens), "bundle target is missing")
    bundle_item = _singular(question.norms[bundle_index])
    components = _single_clause(
        clause_set,
        lambda clause: (
            bundle_item in {_singular(word) for word in clause.norms}
            and "consists" in clause.norms
            and len(_counts(clause)) > 1
        ),
        reason="bundle components are not unique",
    )
    base_clause = _single_clause(
        clause_set,
        lambda clause: (
            _contains(clause.norms, "twice", "as", "much")
            and bool(_money_tokens(clause))
        ),
        reason="base and first relative price are not unique",
    )
    chained_clause = _single_clause(
        clause_set,
        lambda clause: (
            _contains(clause.norms, "three", "times", "as", "much", "as")
            and "per" in clause.norms
        ),
        reason="second relative price is not unique",
    )
    count_map: dict[str, Token] = {}
    for count in _counts(components):
        item = _noun_after(components, count)
        _require(
            item not in count_map,
            "duplicate bundle component",
            FrontendStatus.AMBIGUOUS,
        )
        count_map[item] = count
    ones = [index for index, word in enumerate(base_clause.norms) if word == "one"]
    _require(len(ones) == 2, "relative-price items are incomplete")
    base_item = _singular(base_clause.norms[ones[0] + 1])
    dependent_item = _singular(base_clause.norms[ones[1] + 1])
    third_item = _singular(chained_clause.norms[0])
    as_offsets = [
        index for index, word in enumerate(chained_clause.norms) if word == "as"
    ]
    source_item = None
    if as_offsets:
        tail = chained_clause.norms[as_offsets[-1] + 1 :]
        source_words = [
            word for word in tail if word not in {"a", "an", "per", "piece"}
        ]
        if source_words:
            source_item = _singular(source_words[0])
    _require(
        source_item == dependent_item
        and {base_item, dependent_item, third_item} == count_map.keys(),
        "bundle and relative-price item scopes differ",
        FrontendStatus.AMBIGUOUS,
    )
    base_price = _one_token(_money_tokens(base_clause), "base item price is incomplete")
    twice = _one_token(
        [token for token in base_clause.tokens if token.norm == "twice"],
        "first price multiplier is incomplete",
    )
    three = _unique_word_token(clause_set, "three", clause=chained_clause)

    builder = _Builder(source, clause_set)
    symbols = {
        item: SymbolKey("bundle", "unit_price", item, bundle_item, "current")
        for item in count_map
    }
    definitions = (
        Definition(
            symbols[base_item],
            builder.literal(base_price, PRICE_PER_COUNT),
            base_clause.span,
        ),
        Definition(
            symbols[dependent_item],
            _product(
                builder.literal(twice, SCALAR),
                RefExpr(symbols[base_item], base_clause.span),
            ),
            base_clause.span,
        ),
        Definition(
            symbols[third_item],
            _product(
                _cardinal_literal(builder, three),
                RefExpr(symbols[dependent_item], chained_clause.span),
            ),
            chained_clause.span,
        ),
    )
    expression = _sum(
        *[
            _signed(
                1,
                _product(
                    builder.literal(count, COUNT),
                    RefExpr(symbols[item], components.span),
                ),
                "bundle_component",
            )
            for item, count in count_map.items()
        ]
    )
    return builder.finish(
        expression,
        "bundle_relative_price_dag",
        definitions=definitions,
    )


def _group_seat_purchase(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not (
        "group" in question.norms
        and "spent" in question.norms
        and bool(_money_tokens(question))
    ):
        return None
    group_clause = _single_clause(
        clause_set,
        lambda clause: (
            "friends" in clause.norms
            and "meet" in clause.norms
            and len(_counts(clause)) == 2
        ),
        reason="group composition is not unique",
    )
    capacity = _single_clause(
        clause_set,
        lambda clause: (
            "each" in clause.norms
            and "seat" in clause.norms
            and "person" in clause.norms
            and "drinks" in clause.norms
            and "snacks" in clause.norms
        ),
        reason="seat capacity is not unique",
    )
    reservation = _single_clause(
        clause_set,
        lambda clause: (
            "they" in clause.norms
            and "each" in clause.norms
            and "save" in clause.norms
            and "seat" in clause.norms
            and any(word in clause.norms for word in {"buy", "buys"})
            and "fill" in clause.norms
            and {"drinks", "snacks"}.issubset(clause.norms)
        ),
        reason="group-to-seat purchase scope is not unique",
    )
    singleton = group_clause.tokens[0]
    _require(
        singleton.text[:1].isupper()
        and any(
            _contains(group_clause.norms, "of", pronoun, "friends")
            for pronoun in {"his", "her", "their"}
        )
        and _contains(group_clause.norms, "more", "friends", "there"),
        "group singleton and friend scopes are not explicit",
        FrontendStatus.AMBIGUOUS,
    )
    _require(
        reservation.norms[0] == "they"
        and not any(token.text[:1].isupper() for token in reservation.tokens[1:]),
        "seat reservation is not bound to the declared group",
        FrontendStatus.AMBIGUOUS,
    )
    buy_index = next(
        index for index, word in enumerate(reservation.norms) if word in {"buy", "buys"}
    )
    then_offsets = [
        index for index in range(buy_index) if reservation.norms[index] == "then"
    ]
    _require(
        bool(then_offsets)
        and all(
            word in {"they"}
            for word in reservation.norms[then_offsets[-1] + 1 : buy_index]
        ),
        "purchase verb has a foreign local subject",
        FrontendStatus.AMBIGUOUS,
    )
    friend_counts = _counts(group_clause)
    one = _unique_word_token(clause_set, "one", clause=capacity)
    two = _unique_word_token(clause_set, "two", clause=capacity)
    three = _unique_word_token(clause_set, "three", clause=capacity)
    _require(
        _noun_after(capacity, one) == "person"
        and _noun_after(capacity, two) == "drink"
        and _noun_after(capacity, three) == "snack"
        and {"drinks", "snacks"}.issubset(question.norms),
        "seat capacity and purchased item scopes differ",
        FrontendStatus.AMBIGUOUS,
    )
    unit_price = _one_token(_money_tokens(question), "group unit price is incomplete")
    cost_index = question.norms.index("cost") if "cost" in question.norms else -1
    drinks_index = question.norms.index("drinks") if "drinks" in question.norms else -1
    _require(
        drinks_index >= 0
        and question.norms[:drinks_index] == ("if",)
        and cost_index >= 0
        and _contains(question.norms[:cost_index], "drinks", "and", "snacks")
        and "each" in question.norms[question.tokens.index(unit_price) :],
        "group price lacks each-item scope",
    )

    builder = _Builder(source, clause_set)
    group_symbol = SymbolKey("group", "size", "person", "cinema", "present")
    seats_symbol = SymbolKey("group", "seat_count", "seat", "cinema", "reserved")
    group = Definition(
        group_symbol,
        _sum(
            _signed(
                1,
                builder.lexical_literal(singleton, Fraction(1), COUNT),
                "named_person",
            ),
            *[
                _signed(1, builder.literal(count, COUNT), "friends")
                for count in friend_counts
            ],
        ),
        group_clause.span,
    )
    seats = Definition(
        seats_symbol,
        QuotientExpr(
            RefExpr(group_symbol, capacity.span),
            _cardinal_literal(builder, one),
            capacity.span,
        ),
        capacity.span,
    )
    purchased_items = _sum(
        _signed(
            1,
            _product(
                RefExpr(seats_symbol, capacity.span),
                _cardinal_literal(builder, two),
            ),
            "drinks",
        ),
        _signed(
            1,
            _product(
                RefExpr(seats_symbol, capacity.span),
                _cardinal_literal(builder, three),
            ),
            "snacks",
        ),
    )
    expression = _product(
        purchased_items,
        builder.literal(unit_price, PRICE_PER_COUNT),
    )
    return builder.finish(
        expression,
        "group_seat_purchase",
        definitions=(group, seats),
    )


def _funding_balance_residual(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not (
        "extra" in question.norms
        and "purse" in question.norms
        and bool(_money_tokens(question))
    ):
        return None
    intro = _single_clause(
        clause_set,
        lambda clause: "saving" in clause.norms and "new" in clause.norms,
        reason="funding owner and target are not unique",
    )
    new_index = intro.norms.index("new")
    _require(new_index + 1 < len(intro.tokens), "funding target item is missing")
    target_item = _singular(intro.norms[new_index + 1])
    price_clause = _single_clause(
        clause_set,
        lambda clause: (
            "costs" in clause.norms
            and target_item in {_singular(word) for word in clause.norms}
            and bool(_money_tokens(clause))
        ),
        reason="target purchase price is not unique",
    )
    reduction_clause = _single_clause(
        clause_set,
        lambda clause: (
            "traded" in clause.norms
            and "reduced" in clause.norms
            and bool(_money_tokens(clause))
        ),
        reason="trade-in reduction is not unique",
    )
    paid_clause = _single_clause(
        clause_set,
        lambda clause: (
            "paid" in clause.norms
            and _contains(clause.norms, "this", "week")
            and bool(_money_tokens(clause))
        ),
        reason="current income is not unique",
    )
    gift_clause = _single_clause(
        clause_set,
        lambda clause: (
            "mom" in clause.norms
            and "give" in clause.norms
            and bool(_money_tokens(clause))
        ),
        reason="current gift is not unique",
    )
    owner = intro.norms[0]
    does_index = question.norms.index("does") if "does" in question.norms else -1
    query_pronoun = (
        question.norms[does_index + 1]
        if 0 <= does_index < len(question.tokens) - 1
        else ""
    )
    possessive = {"he": "his", "she": "her", "they": "their"}.get(query_pronoun)
    object_pronoun = {"he": "him", "she": "her", "they": "them"}.get(query_pronoun)
    give_index = gift_clause.norms.index("give")
    traded_index = reduction_clause.norms.index("traded")
    trade_tail = reduction_clause.norms[traded_index + 1 :]
    trade_item = None
    if len(trade_tail) >= 4 and trade_tail[0] == "in" and trade_tail[2] == "old":
        trade_item = _singular(trade_tail[3])
    paid_index = paid_clause.norms.index("paid")
    paid_has = [
        index for index in range(paid_index) if paid_clause.norms[index] == "has"
    ]
    paid_and = [
        index
        for index in range(paid_has[-1] if paid_has else 0)
        if paid_clause.norms[index] == "and"
    ]
    local_pay_subject = (
        paid_clause.norms[paid_and[-1] + 1 : paid_has[-1]]
        if paid_has and paid_and
        else ()
    )
    _require(
        intro.tokens[0].text[:1].isupper()
        and owner in question.norms
        and possessive is not None
        and object_pronoun is not None
        and {"now", "only", "needs"}.issubset(question.norms)
        and paid_clause.norms[0] == query_pronoun
        and all(word == query_pronoun for word in local_pay_subject)
        and gift_clause.norms[0] in {f"{owner}'s", possessive}
        and give_index + 1 < len(gift_clause.tokens)
        and gift_clause.norms[give_index + 1] == object_pronoun
        and _contains(
            price_clause.norms, "the", target_item, query_pronoun, "wants", "costs"
        )
        and len(trade_tail) >= 4
        and trade_tail[1] == possessive
        and trade_item == target_item
        and _contains(
            reduction_clause.norms,
            "price",
            "of",
            "the",
            "new",
            "one",
            "would",
            "be",
            "reduced",
            "by",
        )
        and "purse" in paid_clause.norms
        and target_item in {_singular(word) for word in question.norms},
        "funding contributions do not share owner, item, and current scope",
        FrontendStatus.AMBIGUOUS,
    )
    amounts = (
        _one_token(_money_tokens(price_clause), "purchase price is incomplete"),
        _one_token(_money_tokens(reduction_clause), "trade reduction is incomplete"),
        _one_token(_money_tokens(paid_clause), "income is incomplete"),
        _one_token(_money_tokens(gift_clause), "gift is incomplete"),
        _one_token(_money_tokens(question), "remaining need is incomplete"),
    )
    builder = _Builder(source, clause_set)
    expression = _ledger(
        builder,
        (
            _Contribution(amounts[0], MONEY, "purchase_price"),
            _Contribution(amounts[1], MONEY, "trade_reduction", sign=-1),
            _Contribution(amounts[2], MONEY, "current_income", sign=-1),
            _Contribution(amounts[3], MONEY, "current_gift", sign=-1),
            _Contribution(amounts[4], MONEY, "remaining_need", sign=-1),
        ),
    )
    return builder.finish(expression, "funding_balance_residual")


def _mean_participant_totals(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not (
        _contains(question.norms, "average", "total", "distance")
        and "spat" in question.norms
    ):
        return None
    intro = _single_clause(
        clause_set,
        lambda clause: "contest" in clause.norms and "and" in clause.norms,
        reason="contest participants are not unique",
    )
    names = {token.norm for token in intro.tokens if token.text[:1].isupper()}
    rows = [
        clause
        for clause in clause_set
        if "has" in clause.norms
        and "spits" in clause.norms
        and len(_counts(clause)) == 2
    ]
    _require(
        len(names) == len(rows) == 2,
        "mean requires exactly two explicitly named contestant totals",
        FrontendStatus.AMBIGUOUS,
    )
    explicit_seed_rows = sum("seeds" in clause.norms for clause in rows)
    _require(
        "seed" in intro.norms and explicit_seed_rows >= 1,
        "elliptical contestant item has no parallel seed binding",
        FrontendStatus.AMBIGUOUS,
    )
    parsed: list[tuple[Token, Token]] = []
    owners: set[str] = set()
    actor_pronouns: set[str] = set()
    for clause in rows:
        owner = clause.norms[0].removesuffix("'s")
        counts = _counts(clause)
        seed_count, distance = counts
        distance_index = clause.tokens.index(distance)
        spits_index = clause.norms.index("spits")
        item_after_count = _noun_after(clause, seed_count)
        pronouns = [
            word
            for word in clause.norms[clause.tokens.index(seed_count) + 1 : spits_index]
            if word in {"he", "she", "they"}
        ]
        _require(
            owner in names
            and owner not in owners
            and (
                item_after_count == "seed"
                or (
                    item_after_count in {"he", "she", "they"}
                    and explicit_seed_rows == 1
                )
            )
            and distance_index + 1 < len(clause.tokens)
            and _singular(clause.norms[distance_index + 1]) == "foot"
            and _contains(clause.norms[:distance_index], "each", "one")
            and len(pronouns) == 1
            and not any(token.text[:1].isupper() for token in clause.tokens[1:]),
            "contestant count, item, or distance unit is not uniquely bound",
            FrontendStatus.AMBIGUOUS,
        )
        owners.add(owner)
        actor_pronouns.add(pronouns[0])
        parsed.append((seed_count, distance))
    _require(
        len(actor_pronouns) == 1,
        "contestant clauses use inconsistent actor pronouns",
        FrontendStatus.AMBIGUOUS,
    )

    builder = _Builder(source, clause_set)
    totals = tuple(
        _product(
            builder.literal(seed_count, COUNT),
            builder.literal(distance, FOOT_PER_COUNT),
        )
        for seed_count, distance in parsed
    )
    expression = MeanExpr(totals, Span(0, len(source), source))
    return builder.finish(expression, "mean_participant_totals")


def _balanced_percent_category_difference(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not (
        _contains(question.norms, "how", "many", "more") and "total" in question.norms
    ):
        return None
    relation = _single_clause(
        clause_set,
        lambda clause: (
            "fewer" in clause.norms
            and "more" in clause.norms
            and len(_counts(clause)) == 2
        ),
        reason="balanced percentage relations are not unique",
    )
    intro = _single_clause(
        clause_set,
        lambda clause: "containing" in clause.norms and "and" in clause.norms,
        reason="category enumeration is not unique",
    )
    percents = _counts(relation)
    _require(
        all(source[token.span.end : token.span.end + 1] == "%" for token in percents)
        and percents[0].number == percents[1].number,
        "percentage magnitudes are not an equal balanced pair",
    )
    bindings: dict[str, tuple[str, Token]] = {}
    for token in percents:
        index = relation.tokens.index(token)
        direction = (
            relation.norms[index + 1] if index + 1 < len(relation.tokens) else ""
        )
        than = next(
            (
                i
                for i in range(index + 1, len(relation.tokens))
                if relation.norms[i] == "than"
            ),
            -1,
        )
        _require(
            direction in {"fewer", "more"}
            and than > index + 2
            and than + 1 < len(relation.tokens),
            "percentage direction or baseline is missing",
        )
        bindings[direction] = (_singular(relation.norms[index + 2]), token)
        baseline = _singular(relation.norms[than + 1])
        bindings[f"{direction}_baseline"] = (baseline, token)
    _require(
        {"fewer", "more", "fewer_baseline", "more_baseline"}.issubset(bindings)
        and bindings["fewer_baseline"][0] == bindings["more_baseline"][0],
        "percentage relations do not share one baseline",
        FrontendStatus.AMBIGUOUS,
    )
    query_more = question.norms[question.norms.index("more") + 1]
    query_than = question.norms.index("than") if "than" in question.norms else -1
    _require(
        query_more == bindings["more"][0]
        and query_than >= 0
        and question.norms[query_than + 1] == bindings["fewer"][0],
        "question direction differs from the percentage relations",
        FrontendStatus.AMBIGUOUS,
    )
    categories = (
        bindings["fewer"][0],
        bindings["more_baseline"][0],
        bindings["more"][0],
    )
    intro_positions = [intro.norms.index(category) for category in categories]
    containing_index = intro.norms.index("containing")
    jelly_index = intro.norms.index("jelly") if "jelly" in intro.norms else -1
    enumerated = [
        word
        for word in intro.norms[containing_index + 1 : jelly_index]
        if word not in {",", "and"}
    ]
    _require(
        len(set(categories)) == 3
        and intro_positions == sorted(intro_positions)
        and tuple(enumerated) == categories
        and "jar" in relation.norms
        and question.norms[:3] == ("if", "the", "jar"),
        "category enumeration and relation scopes differ",
        FrontendStatus.AMBIGUOUS,
    )
    category_span = Span(
        intro.tokens[intro_positions[0]].span.start,
        intro.tokens[intro_positions[-1]].span.end,
        source,
    )
    category_token = Token(
        source[category_span.start : category_span.end],
        "enumerated_categories",
        category_span,
    )
    total = _one_token(_counts(question), "category total is incomplete")
    total_index = question.tokens.index(total)
    _require(
        question.norms[:total_index]
        == ("if", "the", "jar", "contains", "a", "total", "of")
        and total_index + 2 < len(question.tokens)
        and question.norms[total_index + 1 : total_index + 3] == ("jelly", "beans"),
        "category total is not scoped to the single declared jar",
        FrontendStatus.AMBIGUOUS,
    )
    builder = _Builder(source, clause_set)
    percent_mark_span = Span(percents[0].span.end, percents[0].span.end + 1, source)
    percent_origin = Token("%", "percent_origin", percent_mark_span)
    percent_sum = _sum(
        _signed(
            1,
            builder.lexical_literal(percent_origin, Fraction(0), SCALAR),
            "scalar_origin",
        ),
        *[
            _signed(
                1,
                builder.literal(token, PERCENT),
                "percent_delta",
            )
            for token in percents
        ],
    )
    numerator = _product(builder.literal(total, COUNT), percent_sum)
    expression = QuotientExpr(
        numerator,
        builder.lexical_literal(category_token, Fraction(3), SCALAR),
        Span(0, len(source), source),
    )
    return builder.finish(expression, "balanced_percent_category_difference")


def _affine_price_chain_total(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not _contains(question.norms, "total", "price", "for", "all", "three"):
        return None
    three_index = question.norms.index("three")
    _require(three_index + 1 < len(question.tokens), "catalog item type is missing")
    item_type = _singular(question.norms[three_index + 1])
    ratio_clause = _single_clause(
        clause_set,
        lambda clause: (
            _contains(clause.norms, "times", "as", "much", "as")
            and len(_counts(clause)) == 1
        ),
        reason="relative price ratio is not unique",
    )
    offset_clause = _single_clause(
        clause_set,
        lambda clause: "less" in clause.norms and bool(_money_tokens(clause)),
        reason="relative price offset is not unique",
    )
    base_clause = _single_clause(
        clause_set,
        lambda clause: "if" in clause.norms and bool(_money_tokens(clause)),
        reason="base price assignment is not unique",
    )
    ratio = _one_token(_counts(ratio_clause), "price ratio is incomplete")
    articles = [
        index + 2
        for index in range(len(ratio_clause.norms) - 2)
        if ratio_clause.norms[index : index + 3] == ("price", "of", "a")
    ]
    _require(
        len(articles) == 2
        and articles[0] + 2 < len(ratio_clause.tokens)
        and articles[1] + 2 < len(ratio_clause.tokens)
        and _singular(ratio_clause.norms[articles[0] + 2]) == item_type
        and _singular(ratio_clause.norms[articles[1] + 2]) == item_type,
        "ratio price items are not explicitly typed",
        FrontendStatus.AMBIGUOUS,
    )
    target_item = _singular(ratio_clause.norms[articles[0] + 1])
    base_item = _singular(ratio_clause.norms[articles[1] + 1])
    silver_offsets = [
        index for index, word in enumerate(offset_clause.norms) if word in {"a", "an"}
    ]
    _require(
        bool(silver_offsets)
        and silver_offsets[0] + 2 < len(offset_clause.tokens)
        and _singular(offset_clause.norms[silver_offsets[0] + 2]) == item_type,
        "offset target item is missing or untyped",
    )
    offset_item = _singular(offset_clause.norms[silver_offsets[0] + 1])
    less_index = offset_clause.norms.index("less")
    than_index = (
        offset_clause.norms.index("than") if "than" in offset_clause.norms else -1
    )
    _require(
        than_index == less_index + 1
        and than_index + 4 <= len(offset_clause.tokens)
        and _singular(offset_clause.norms[-1]) == target_item,
        "price offset source differs from the ratio target",
        FrontendStatus.AMBIGUOUS,
    )
    base_price = _one_token(_money_tokens(base_clause), "base price is incomplete")
    base_price_index = base_clause.tokens.index(base_price)
    base_prefix = (
        (
            base_clause.norms[0],
            base_clause.norms[1],
            _singular(base_clause.norms[2]),
            _singular(base_clause.norms[3]),
            base_clause.norms[4],
        )
        if base_price_index == 5
        else ()
    )
    _require(
        base_prefix == ("if", "a", base_item, item_type, "is")
        and len({base_item, target_item, offset_item}) == 3,
        "price chain categories are not unique",
        FrontendStatus.AMBIGUOUS,
    )
    offset = _one_token(_money_tokens(offset_clause), "price offset is incomplete")

    builder = _Builder(source, clause_set)
    symbols = {
        item: SymbolKey("catalog", "price", item, item_type, "current")
        for item in (base_item, target_item, offset_item)
    }
    definitions = (
        Definition(
            symbols[base_item],
            builder.literal(base_price, MONEY),
            base_clause.span,
        ),
        Definition(
            symbols[target_item],
            _product(
                builder.literal(ratio, SCALAR),
                RefExpr(symbols[base_item], ratio_clause.span),
            ),
            ratio_clause.span,
        ),
        Definition(
            symbols[offset_item],
            _sum(
                _signed(1, RefExpr(symbols[target_item], offset_clause.span), "base"),
                _signed(-1, builder.literal(offset, MONEY), "less"),
            ),
            offset_clause.span,
        ),
    )
    expression = _sum(
        *[
            _signed(1, RefExpr(symbols[item], question.span), "catalog_item")
            for item in (base_item, target_item, offset_item)
        ]
    )
    return builder.finish(
        expression,
        "affine_price_chain_total",
        definitions=definitions,
    )


def _ordinal_ratio_partition(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not (_contains(question.norms, "how", "many") and "third" in question.norms):
        return None
    intro = _single_clause(
        clause_set,
        lambda clause: (
            "three" in clause.norms and "homes" in clause.norms and not _counts(clause)
        ),
        reason="three-part partition scope is not unique",
    )
    relation = _single_clause(
        clause_set,
        lambda clause: (
            {"first", "second", "third", "double"}.issubset(clause.norms)
            and bool(_counts(clause))
        ),
        reason="ordinal partition relations are not unique",
    )
    numbers = _counts(relation)
    ratios = [
        token for token in numbers if token.number is not None and token.number < 1
    ]
    totals = [token for token in numbers if token not in ratios]
    ratio = _one_token(ratios, "partition ratio is incomplete")
    total = _one_token(totals, "partition total is incomplete")
    double = _unique_word_token(clause_set, "double", clause=relation)
    ratio_index = relation.tokens.index(ratio)
    total_index = relation.tokens.index(total)
    double_index = relation.norms.index("double")
    total_item_end = next(
        (
            index
            for index in range(total_index + 1, len(relation.tokens))
            if relation.norms[index] in {",", "with"}
        ),
        len(relation.tokens),
    )
    how_index = question.norms.index("many")
    query_item_end = next(
        (
            index
            for index in range(how_index + 1, len(question.tokens))
            if question.norms[index] in {"will", "does", "do", "did"}
        ),
        len(question.tokens),
    )
    _require(
        _contains(relation.norms[:ratio_index], "first", "house", "needing")
        and _contains(relation.norms[ratio_index + 1 :], "of", "the", "second")
        and _contains(relation.norms[:double_index], "third", "needing")
        and _contains(relation.norms[double_index + 1 :], "the", "first")
        and _item_id(relation.norms[total_index + 1 : total_item_end])
        == _item_id(question.norms[how_index + 1 : query_item_end])
        and _contains(question.norms, "third", "house", "need"),
        "ordinal ratio directions or target differ",
        FrontendStatus.AMBIGUOUS,
    )
    _require(
        _contains(intro.norms, "three", "homes")
        and _contains(relation.norms, "three", "homes")
        and ratio.number is not None
        and ratio.number > 0,
        "partition is not a closed positive three-part total",
    )
    denominator_value = Fraction(1) + ratio.number + 2 * ratio.number
    relation_span = Span(
        relation.tokens[relation.norms.index("first")].span.start,
        relation.tokens[double_index + 2].span.end,
        source,
    )
    denominator_token = Token(
        source[relation_span.start : relation_span.end],
        "derived_partition_coefficient",
        relation_span,
    )

    builder = _Builder(source, clause_set)
    numerator = _product(
        builder.literal(total, COUNT),
        builder.literal(ratio, SCALAR),
        builder.lexical_literal(double, Fraction(2), SCALAR),
    )
    expression = QuotientExpr(
        numerator,
        builder.lexical_literal(denominator_token, denominator_value, SCALAR),
        Span(0, len(source), source),
    )
    return builder.finish(expression, "ordinal_ratio_partition")


def _temporal_affine_score_chain(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not (
        _contains(question.norms, "how", "many", "total", "points")
        and _contains(question.norms, "both", "games")
    ):
        return None
    intro = _single_clause(
        clause_set,
        lambda clause: (
            "friends" in clause.norms
            and "basketball" in clause.norms
            and "teams" in clause.norms
            and not _counts(clause)
        ),
        reason="score participant scope is not unique",
    )
    assignment = _single_clause(
        clause_set,
        lambda clause: (
            "first" in clause.norms
            and "game" in clause.norms
            and "scored" in clause.norms
            and "fewer" not in clause.norms
            and len(_counts(clause)) == 1
        ),
        reason="first-game score assignment is not unique",
    )
    first_relation = _single_clause(
        clause_set,
        lambda clause: (
            "fewer" in clause.norms
            and "same" in clause.norms
            and "game" in clause.norms
            and len(_counts(clause)) == 1
        ),
        reason="same-game score relation is not unique",
    )
    second_relation = _single_clause(
        clause_set,
        lambda clause: (
            "fewer" in clause.norms
            and "second" in clause.norms
            and "first" in clause.norms
            and "score" in clause.norms
            and len(_counts(clause)) == 1
        ),
        reason="cross-game score relation is not unique",
    )
    target_owner = question.norms[question.norms.index("did") + 1]
    scored_index = assignment.norms.index("scored")
    assignment_owners = [
        token.norm
        for token in assignment.tokens[:scored_index]
        if token.text[:1].isupper() and token.norm not in {"in"}
    ]
    _require(len(assignment_owners) == 1, "first-game scorer is not unique")
    first_owner = assignment_owners[0]
    related_owner = first_relation.norms[0]
    intro_owners = {token.norm for token in intro.tokens if token.text[:1].isupper()}
    _require(
        intro_owners == {first_owner, related_owner}
        and target_owner == first_owner
        and related_owner != first_owner
        and _contains(
            first_relation.norms, "than", first_owner, "in", "the", "same", "game"
        )
        and second_relation.norms[0] == first_owner
        and _contains(second_relation.norms, "second", "game")
        and _contains(
            second_relation.norms,
            "than",
            f"{related_owner}'s",
            "score",
            "in",
            "the",
            "first",
            "game",
        ),
        "score owners or temporal states differ",
        FrontendStatus.AMBIGUOUS,
    )
    initial = _one_token(_counts(assignment), "first score is incomplete")
    first_offset = _one_token(_counts(first_relation), "same-game offset is incomplete")
    second_offset = _one_token(
        _counts(second_relation), "cross-game offset is incomplete"
    )
    builder = _Builder(source, clause_set)
    s1 = SymbolKey(first_owner, "score", "point", "game", "first")
    other1 = SymbolKey(related_owner, "score", "point", "game", "first")
    s2 = SymbolKey(first_owner, "score", "point", "game", "second")
    definitions = (
        Definition(s1, builder.literal(initial, COUNT), assignment.span),
        Definition(
            other1,
            _sum(
                _signed(1, RefExpr(s1, first_relation.span), "source_score"),
                _signed(-1, builder.literal(first_offset, COUNT), "fewer"),
            ),
            first_relation.span,
        ),
        Definition(
            s2,
            _sum(
                _signed(1, RefExpr(other1, second_relation.span), "prior_score"),
                _signed(-1, builder.literal(second_offset, COUNT), "fewer"),
            ),
            second_relation.span,
        ),
    )
    expression = _sum(
        _signed(1, RefExpr(s1, question.span), "first_game"),
        _signed(1, RefExpr(s2, question.span), "second_game"),
    )
    return builder.finish(
        expression,
        "temporal_affine_score_chain",
        definitions=definitions,
    )


def _entity_affine_chain_total(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not (
        _contains(question.norms, "how", "many")
        and "together" in question.norms
        and "all" in question.norms
    ):
        return None
    intro = _single_clause(
        clause_set,
        lambda clause: (
            "make" in clause.norms
            and "clay" in clause.norms
            and "dishes" in clause.norms
            and not _counts(clause)
        ),
        reason="maker enumeration is not unique",
    )
    relation = _single_clause(
        clause_set,
        lambda clause: (
            "twice" in clause.norms
            and "more" in clause.norms
            and "than" in clause.norms
        ),
        reason="maker affine relations are not unique",
    )
    names = [token.norm for token in intro.tokens if token.text[:1].isupper()]
    _require(
        len(names) == 3 and len(set(names)) == 3, "maker enumeration needs three names"
    )
    while_index = relation.norms.index("while") if "while" in relation.norms else -1
    than_index = relation.norms.index("than") if "than" in relation.norms else -1
    as_offsets = [index for index, word in enumerate(relation.norms) if word == "as"]
    _require(
        while_index > 0
        and than_index > while_index
        and as_offsets
        and as_offsets[-1] + 1 < while_index,
        "maker relation roles are incomplete",
    )
    scaled_owner = relation.norms[0]
    middle_owner = relation.norms[as_offsets[-1] + 1]
    offset_owner = relation.norms[while_index + 1]
    base_owner = relation.norms[than_index + 1]
    _require(
        middle_owner == offset_owner
        and {scaled_owner, middle_owner, base_owner} == set(names)
        and _contains(relation.norms, "clay", "dishes")
        and _contains(
            question.norms, "clay", "dishes", "they", "all", "make", "together"
        ),
        "maker entities or item scopes differ",
        FrontendStatus.AMBIGUOUS,
    )
    scale = _one_token(
        [token for token in relation.tokens if token.norm == "twice"],
        "maker scale is incomplete",
    )
    offset = _one_token(
        [token for token in _counts(relation) if token is not scale],
        "maker offset is incomplete",
    )
    base_values = _counts(question)
    base_value = _one_token(base_values, "maker base value is incomplete")
    if_index = question.norms.index("if") if "if" in question.norms else -1
    _require(
        if_index >= 0
        and if_index + 1 < len(question.tokens)
        and question.norms[if_index + 1] == base_owner
        and "made" in question.norms[: question.tokens.index(base_value)],
        "maker base assignment belongs to another entity",
        FrontendStatus.AMBIGUOUS,
    )

    builder = _Builder(source, clause_set)
    symbols = {
        name: SymbolKey(name, "quantity", "clay_dish", "project", "final")
        for name in names
    }
    definitions = (
        Definition(
            symbols[base_owner],
            builder.literal(base_value, COUNT),
            question.span,
        ),
        Definition(
            symbols[middle_owner],
            _sum(
                _signed(1, RefExpr(symbols[base_owner], relation.span), "base"),
                _signed(1, builder.literal(offset, COUNT), "more"),
            ),
            relation.span,
        ),
        Definition(
            symbols[scaled_owner],
            _product(
                builder.literal(scale, SCALAR),
                RefExpr(symbols[middle_owner], relation.span),
            ),
            relation.span,
        ),
    )
    expression = _sum(
        *[
            _signed(1, RefExpr(symbols[name], question.span), "maker_total")
            for name in names
        ]
    )
    return builder.finish(
        expression,
        "entity_affine_chain_total",
        definitions=definitions,
    )


def _cross_entity_property_dag(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not (
        "together" in question.norms
        and "total" in question.norms
        and {"socks", "dishes"}.issubset(question.norms)
    ):
        return None
    cross = _single_clause(
        clause_set,
        lambda clause: (
            "socks" in clause.norms
            and "dishes" in clause.norms
            and "half" in clause.norms
            and "twice" in clause.norms
        ),
        reason="cross-entity property relations are not unique",
    )
    within = _single_clause(
        clause_set,
        lambda clause: (
            "collected" in clause.norms
            and "twice" in clause.norms
            and "socks" in clause.norms
            and "dishes" in clause.norms
            and not clause.question
        ),
        reason="within-entity property relation is not unique",
    )
    first_owner = cross.norms[0]
    as_offsets = [index for index, word in enumerate(cross.norms) if word == "as"]
    _require(len(as_offsets) == 4, "cross-entity comparison roles are incomplete")
    second_owner = cross.norms[as_offsets[1] + 1]
    _require(
        second_owner == cross.norms[as_offsets[3] + 1] == within.norms[0]
        and first_owner != second_owner
        and _contains(
            cross.norms,
            "twice",
            "as",
            "many",
            "socks",
            "as",
            second_owner,
        )
        and _contains(
            cross.norms,
            "half",
            "times",
            "as",
            "many",
            "dishes",
            "as",
            second_owner,
        )
        and _contains(
            within.norms,
            "twice",
            "as",
            "many",
            "dishes",
            "as",
            "socks",
        ),
        "cross-entity owner, item, or awkward-half scope differs",
        FrontendStatus.AMBIGUOUS,
    )
    cross_twice = _one_token(
        [token for token in cross.tokens if token.norm == "twice"],
        "cross-owner sock scale is incomplete",
    )
    half = _one_token(
        [token for token in cross.tokens if token.norm == "half"],
        "cross-owner dish scale is incomplete",
    )
    within_twice = _one_token(
        [token for token in within.tokens if token.norm == "twice"],
        "within-owner scale is incomplete",
    )
    base_value = _one_token(_counts(question), "base dish count is incomplete")
    base_index = question.tokens.index(base_value)
    _require(
        second_owner in question.norms[:base_index]
        and base_index + 1 < len(question.tokens)
        and _singular(question.norms[base_index + 1]) == "dish"
        and "they" in question.norms,
        "base assignment or two-owner total scope differs",
        FrontendStatus.AMBIGUOUS,
    )

    builder = _Builder(source, clause_set)

    def key(owner: str, item: str) -> SymbolKey:
        return SymbolKey(owner, "quantity", item, "collection", "final")

    jack_dishes = key(second_owner, "dish")
    jack_socks = key(second_owner, "sock")
    peter_socks = key(first_owner, "sock")
    peter_dishes = key(first_owner, "dish")
    definitions = (
        Definition(
            jack_dishes,
            builder.literal(base_value, COUNT),
            question.span,
        ),
        Definition(
            jack_socks,
            QuotientExpr(
                RefExpr(jack_dishes, within.span),
                builder.literal(within_twice, SCALAR),
                within.span,
            ),
            within.span,
        ),
        Definition(
            peter_socks,
            _product(
                builder.literal(cross_twice, SCALAR),
                RefExpr(jack_socks, cross.span),
            ),
            cross.span,
        ),
        Definition(
            peter_dishes,
            _product(
                builder.literal(half, SCALAR),
                RefExpr(jack_dishes, cross.span),
            ),
            cross.span,
        ),
    )
    expression = _sum(
        *[
            _signed(1, RefExpr(symbol, question.span), "owner_item_total")
            for symbol in (jack_dishes, jack_socks, peter_dishes, peter_socks)
        ]
    )
    return builder.finish(
        expression,
        "cross_entity_property_dag",
        definitions=definitions,
    )


def _inverse_rate_time_difference(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not (
        _contains(question.norms, "how", "many", "more", "badges")
        and "compared" in question.norms
        and "year" in question.norms
    ):
        return None
    base_clause = _single_clause(
        clause_set,
        lambda clause: (
            "earns" in clause.norms
            and "badge" in clause.norms
            and "every" in clause.norms
            and "month" in clause.norms
            and len(_counts(clause)) == 1
        ),
        reason="base monthly badge rate is not unique",
    )
    inverse_clause = _single_clause(
        clause_set,
        lambda clause: (
            "twice" in clause.norms
            and "long" in clause.norms
            and "badge" in clause.norms
            and "than" in clause.norms
        ),
        reason="inverse duration relation is not unique",
    )
    scale_clause = _single_clause(
        clause_set,
        lambda clause: (
            "three" in clause.norms
            and "times" in clause.norms
            and "badges" in clause.norms
            and _contains(clause.norms, "same", "time", "frame")
        ),
        reason="same-frame rate scale is not unique",
    )
    base_owner = base_clause.norms[0]
    takes_index = inverse_clause.norms.index("takes")
    _require(
        takes_index + 1 < len(inverse_clause.tokens), "inverse-rate owner is missing"
    )
    inverse_owner = inverse_clause.norms[takes_index + 1]
    scaled_owner = scale_clause.norms[0]
    _require(
        len({base_owner, inverse_owner, scaled_owner}) == 3
        and inverse_clause.norms[-1] == base_owner
        and base_owner in scale_clause.norms
        and _contains(question.norms, "more", "badges", "does", scaled_owner)
        and _contains(question.norms, "compared", "to", inverse_owner),
        "rate owners or comparison direction differ",
        FrontendStatus.AMBIGUOUS,
    )
    base_rate = _one_token(_counts(base_clause), "base badge rate is incomplete")
    inverse_scale = _one_token(
        [token for token in inverse_clause.tokens if token.norm == "twice"],
        "inverse duration scale is incomplete",
    )
    forward_scale = _unique_word_token(clause_set, "three", clause=scale_clause)
    duration = _one_token(_counts(question), "year duration is incomplete")
    year = _unique_word_token(clause_set, "year", clause=question)
    base_index = base_clause.tokens.index(base_rate)
    every_index = base_clause.norms.index("every")
    _require(
        duration.number is not None
        and duration.number > 0
        and base_index < every_index
        and _singular(base_clause.norms[every_index - 1]) == "badge"
        and _noun_after(question, duration) == "year",
        "badge item or year duration unit differs",
        FrontendStatus.AMBIGUOUS,
    )

    builder = _Builder(source, clause_set)
    symbols = {
        owner: SymbolKey(owner, "rate", "badge", "earning", "monthly")
        for owner in (base_owner, inverse_owner, scaled_owner)
    }
    definitions = (
        Definition(
            symbols[base_owner],
            builder.literal(base_rate, COUNT_PER_MONTH),
            base_clause.span,
        ),
        Definition(
            symbols[inverse_owner],
            QuotientExpr(
                RefExpr(symbols[base_owner], inverse_clause.span),
                builder.literal(inverse_scale, SCALAR),
                inverse_clause.span,
            ),
            inverse_clause.span,
        ),
        Definition(
            symbols[scaled_owner],
            _product(
                _cardinal_literal(builder, forward_scale),
                RefExpr(symbols[base_owner], scale_clause.span),
            ),
            scale_clause.span,
        ),
    )
    rate_delta = _sum(
        _signed(1, RefExpr(symbols[scaled_owner], question.span), "faster_rate"),
        _signed(-1, RefExpr(symbols[inverse_owner], question.span), "slower_rate"),
    )
    expression = _product(
        rate_delta,
        builder.literal(duration, SCALAR),
        builder.lexical_literal(year, Fraction(12), MONTH),
    )
    return builder.finish(
        expression,
        "inverse_rate_time_difference",
        definitions=definitions,
    )


def _chained_inventory_residual(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not (
        _contains(question.norms, "how", "many")
        and "total" in question.norms
        and "cars" in question.norms
    ):
        return None
    assignment = _single_clause(
        clause_set,
        lambda clause: (
            "rink" in clause.norms
            and "has" in clause.norms
            and len(_counts(clause)) == 1
            and not clause.question
        ),
        reason="inventory base category is not unique",
    )
    offset_clause = _single_clause(
        clause_set,
        lambda clause: (
            "fewer" in clause.norms
            and "than" in clause.norms
            and "cars" in clause.norms
        ),
        reason="inventory offset category is not unique",
    )
    scale_clause = _single_clause(
        clause_set,
        lambda clause: (
            "times" in clause.norms
            and "number" in clause.norms
            and "cars" in clause.norms
        ),
        reason="inventory scaled category is not unique",
    )
    target_clause = _single_clause(
        clause_set,
        lambda clause: (
            "also" in clause.norms
            and "has" in clause.norms
            and "cars" in clause.norms
            and not _counts(clause)
        ),
        reason="inventory residual category is not unique",
    )
    base_count = _one_token(_counts(assignment), "base car count is incomplete")
    base_index = assignment.tokens.index(base_count)
    _require(base_index + 2 < len(assignment.tokens), "base car category is incomplete")
    base_category = _singular(assignment.norms[base_index + 1])
    fewer_index = offset_clause.norms.index("fewer")
    than_index = offset_clause.norms.index("than")
    offset_category = _singular(offset_clause.norms[fewer_index + 1])
    offset_source = _singular(offset_clause.norms[-2])
    of_offsets = [
        index for index, word in enumerate(scale_clause.norms) if word == "of"
    ]
    as_index = scale_clause.norms.index("as") if "as" in scale_clause.norms else -1
    _require(
        of_offsets and as_index > of_offsets[-1], "scaled car roles are incomplete"
    )
    scaled_category = _singular(scale_clause.norms[of_offsets[-1] + 1])
    scaled_source = _singular(scale_clause.norms[-2])
    target_category = _singular(target_clause.norms[-2])
    query_category = question.norms[question.norms.index("many") + 1]
    _require(
        offset_source == base_category
        and scaled_source == offset_category
        and query_category == target_category
        and len({base_category, offset_category, scaled_category, target_category}) == 4
        and all(clause.norms[0] == "they" for clause in (offset_clause, scale_clause))
        and target_clause.norms[:2] == ("the", "rink")
        and question.norms[:3] == ("if", "the", "rink"),
        "inventory actor, category chain, or target differs",
        FrontendStatus.AMBIGUOUS,
    )
    offset = _one_token(_counts(offset_clause), "inventory offset is incomplete")
    scale = _one_token(_counts(scale_clause), "inventory scale is incomplete")
    total = _one_token(_counts(question), "inventory total is incomplete")
    _require(
        than_index > fewer_index and total.number is not None and total.number > 0,
        "inventory direction or total is invalid",
    )

    builder = _Builder(source, clause_set)
    symbols = {
        category: SymbolKey("rink", "quantity", category, "car", "current")
        for category in (base_category, offset_category, scaled_category)
    }
    definitions = (
        Definition(
            symbols[base_category],
            builder.literal(base_count, COUNT),
            assignment.span,
        ),
        Definition(
            symbols[offset_category],
            _sum(
                _signed(1, RefExpr(symbols[base_category], offset_clause.span), "base"),
                _signed(-1, builder.literal(offset, COUNT), "fewer"),
            ),
            offset_clause.span,
        ),
        Definition(
            symbols[scaled_category],
            _product(
                builder.literal(scale, SCALAR),
                RefExpr(symbols[offset_category], scale_clause.span),
            ),
            scale_clause.span,
        ),
    )
    expression = _sum(
        _signed(1, builder.literal(total, COUNT), "inventory_total"),
        *[
            _signed(-1, RefExpr(symbols[category], question.span), "known_category")
            for category in (base_category, offset_category, scaled_category)
        ],
    )
    return builder.finish(
        expression,
        "chained_inventory_residual",
        definitions=definitions,
    )


def _part_scaled_period_total(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not (
        _contains(question.norms, "total", "amount")
        and _contains(question.norms, "two", "months")
        and {"first", "second"}.issubset(question.norms)
    ):
        return None
    relation = _single_clause(
        clause_set,
        lambda clause: (
            "sales" in clause.norms
            and "half" in clause.norms
            and len(_money_tokens(clause)) == 1
        ),
        reason="first-period part relation is not unique",
    )
    owner_clause = _single_clause(
        clause_set,
        lambda clause: (
            "shop" in clause.norms and "hats" in clause.norms and not _counts(clause)
        ),
        reason="period sales owner and item scope is not unique",
    )
    owner = owner_clause.norms[0]
    _require(
        owner_clause.tokens[0].text[:1].isupper()
        and relation.norms[0] in {"her", f"{owner}'s"}
        and _contains(relation.norms, "red", "hats")
        and _contains(relation.norms, "green", "hats")
        and _contains(
            relation.norms,
            "half",
            "the",
            "total",
            "amount",
            "she",
            "earned",
            "from",
            "selling",
            "green",
            "hats",
        )
        and _contains(question.norms, "second", "month")
        and _contains(question.norms, "first", "month"),
        "period sales actor, item, or state differs",
        FrontendStatus.AMBIGUOUS,
    )
    red = _one_token(_money_tokens(relation), "known category sales are incomplete")
    half = _one_token(
        [token for token in relation.tokens if token.norm == "half"],
        "part-to-whole ratio is incomplete",
    )
    second_ratio = _one_token(
        [
            token
            for token in _counts(question)
            if token.number is not None and token.number < 1
        ],
        "second-period ratio is incomplete",
    )
    _require(
        _contains(question.norms, "second", "month", "her", "sales", "were")
        and _contains(question.norms, "total", "sales", "of", "the", "first", "month"),
        "second-period ratio basis is not explicit",
        FrontendStatus.AMBIGUOUS,
    )

    builder = _Builder(source, clause_set)
    red_key = SymbolKey(owner, "sales", "red_hat", "month", "first")
    green_key = SymbolKey(owner, "sales", "green_hat", "month", "first")
    first_key = SymbolKey(owner, "sales", "all_hats", "month", "first")
    second_key = SymbolKey(owner, "sales", "all_hats", "month", "second")
    definitions = (
        Definition(red_key, builder.literal(red, MONEY), relation.span),
        Definition(
            green_key,
            QuotientExpr(
                RefExpr(red_key, relation.span),
                builder.literal(half, SCALAR),
                relation.span,
            ),
            relation.span,
        ),
        Definition(
            first_key,
            _sum(
                _signed(1, RefExpr(red_key, relation.span), "red_sales"),
                _signed(1, RefExpr(green_key, relation.span), "green_sales"),
            ),
            relation.span,
        ),
        Definition(
            second_key,
            _product(
                builder.literal(second_ratio, SCALAR),
                RefExpr(first_key, question.span),
            ),
            question.span,
        ),
    )
    expression = _sum(
        _signed(1, RefExpr(first_key, question.span), "first_month"),
        _signed(1, RefExpr(second_key, question.span), "second_month"),
    )
    return builder.finish(
        expression,
        "part_scaled_period_total",
        definitions=definitions,
    )


def _fractional_remnant_total(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not (
        _contains(question.norms, "total", "length")
        and _contains(question.norms, "not", "used")
    ):
        return None
    owner_clause = _single_clause(
        clause_set,
        lambda clause: (
            "three" in clause.norms
            and "glue" in clause.norms
            and "sticks" in clause.norms
            and "used" in clause.norms
        ),
        reason="remnant collection is not unique",
    )
    remnants = _single_clause(
        clause_set,
        lambda clause: "left" in clause.norms and len(_counts(clause)) == 3,
        reason="fractional remnants are not unique",
    )
    fractions = _counts(remnants)
    _require(
        all(token.number is not None and 0 < token.number <= 1 for token in fractions)
        and all(
            remnants.tokens.index(token) + 1 < len(remnants.tokens)
            and remnants.norms[remnants.tokens.index(token) + 1] == "left"
            for token in fractions
        )
        and {"one", "second", "third"}.issubset(remnants.norms),
        "remnant states are not a closed three-stick partition",
        FrontendStatus.AMBIGUOUS,
    )
    original = _one_token(_counts(question), "original stick length is incomplete")
    original_index = question.tokens.index(original)
    _require(
        owner_clause.tokens[0].text[:1].isupper()
        and _contains(question.norms, "glue", "stick")
        and _contains(question.norms, "glue", "sticks", "that", "are", "not", "used")
        and original_index + 1 < len(question.tokens)
        and _singular(question.norms[original_index + 1]) == "millimeter"
        and "originally" in question.norms,
        "remnant owner, item, state, or length unit differs",
        FrontendStatus.AMBIGUOUS,
    )
    builder = _Builder(source, clause_set)
    remaining_fraction = _sum(
        *[
            _signed(1, builder.literal(token, SCALAR), "fraction_left")
            for token in fractions
        ]
    )
    expression = _product(
        remaining_fraction,
        builder.literal(original, MILLIMETER),
    )
    return builder.finish(expression, "fractional_remnant_total")


def _reverse_affine_state_duration(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not (
        _contains(question.norms, "usual", "concerts", "run")
        and "minutes" in question.norms
    ):
        return None
    owner_clause = _single_clause(
        clause_set,
        lambda clause: (
            "final" in clause.norms
            and "concert" in clause.norms
            and "tour" in clause.norms
            and not _counts(clause)
        ),
        reason="concert owner and final state are not unique",
    )
    scale_clause = _single_clause(
        clause_set,
        lambda clause: (
            "twice" in clause.norms
            and "long" in clause.norms
            and "usual" in clause.norms
        ),
        reason="final-to-usual duration scale is not unique",
    )
    encore_clause = _single_clause(
        clause_set,
        lambda clause: "encore" in clause.norms and len(_counts(clause)) == 1,
        reason="final-state encore is not unique",
    )
    owner = owner_clause.norms[0]
    makes_index = scale_clause.norms.index("makes")
    performs_index = encore_clause.norms.index("performs")
    scale_subjects = [
        word
        for word in scale_clause.norms[:makes_index]
        if word in {owner, "he", "she", "they"}
    ]
    encore_subjects = [
        word
        for word in encore_clause.norms[:performs_index]
        if word in {owner, "he", "she", "they"}
    ]
    _require(
        owner_clause.tokens[0].text[:1].isupper()
        and len(scale_subjects) == len(encore_subjects) == 1
        and scale_subjects[0] == encore_subjects[0]
        and _contains(scale_clause.norms, "final", "concert")
        and _contains(scale_clause.norms, "usual", "concerts")
        and _contains(encore_clause.norms, "end", "of", "the", "concert"),
        "concert actor, item, or state differs",
        FrontendStatus.AMBIGUOUS,
    )
    scale = _one_token(
        [token for token in scale_clause.tokens if token.norm == "twice"],
        "final duration scale is incomplete",
    )
    encore = _one_token(_counts(encore_clause), "encore duration is incomplete")
    final = _one_token(
        [token for token in _counts(question) if token is not encore],
        "final runtime is incomplete",
    )
    final_index = question.tokens.index(final)
    _require(
        final_index + 1 < len(question.tokens)
        and _singular(question.norms[final_index + 1]) == "minute"
        and _contains(
            question.norms[:final_index], "runtime", "of", "this", "final", "concert"
        ),
        "final runtime value or unit is not explicitly bound",
        FrontendStatus.AMBIGUOUS,
    )
    builder = _Builder(source, clause_set)
    adjusted = _sum(
        _signed(1, builder.literal(final, MINUTE), "final_runtime"),
        _signed(-1, builder.literal(encore, MINUTE), "encore"),
    )
    expression = QuotientExpr(
        adjusted,
        builder.literal(scale, SCALAR),
        Span(0, len(source), source),
    )
    return builder.finish(expression, "reverse_affine_state_duration")


def _exact_trip_capacity_minimum(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not (
        "least" in question.norms
        and "berries" in question.norms
        and _contains(question.norms, "per", "trip")
        and _contains(question.norms, "same", "number", "of", "berries")
    ):
        return None
    trip_clause = _single_clause(
        clause_set,
        lambda clause: (
            "trip" in clause.norms
            and "hours" in clause.norms
            and "berries" in clause.norms
            and len(_counts(clause)) == 1
        ),
        reason="trip duration is not unique",
    )
    trip_duration = _one_token(_counts(trip_clause), "trip duration is incomplete")
    question_numbers = _counts(question)
    _require(len(question_numbers) == 2, "capacity target and horizon are incomplete")
    berries = next(
        (
            token
            for token in question_numbers
            if _noun_after(question, token) == "berry"
        ),
        None,
    )
    horizon = next(
        (token for token in question_numbers if _noun_after(question, token) == "hour"),
        None,
    )
    _require(
        berries is not None
        and horizon is not None
        and trip_duration.number is not None
        and horizon.number is not None
        and berries.number is not None
        and horizon.number > 0
        and trip_duration.number > 0
        and horizon.number % trip_duration.number == 0,
        "least-capacity case requires a positive integral trip count",
        FrontendStatus.INVALID,
    )
    trip_index = trip_clause.tokens.index(trip_duration)
    berry_index = question.tokens.index(berries)
    _require(
        trip_index + 1 < len(trip_clause.tokens)
        and _singular(trip_clause.norms[trip_index + 1]) == "hour"
        and "sloth" in trip_clause.norms
        and berry_index + 1 < len(question.tokens)
        and _contains(
            trip_clause.norms,
            "pick",
            "up",
            question.norms[berry_index + 1],
        )
        and _contains(
            question.norms,
            "collect",
            berries.norm,
            question.norms[berry_index + 1],
        )
        and _contains(
            question.norms,
            "least",
            "number",
            "of",
            question.norms[berry_index + 1],
            "he",
            "can",
            "pick",
            "up",
        )
        and _contains(question.norms, "if", "he", "wants", "to", "collect"),
        "trip actor, item, or time unit differs",
        FrontendStatus.AMBIGUOUS,
    )
    assert berries is not None and horizon is not None
    builder = _Builder(source, clause_set)
    numerator = _product(
        builder.literal(berries, COUNT),
        builder.literal(trip_duration, HOUR),
    )
    expression = CeilingExpr(
        QuotientExpr(
            numerator,
            builder.literal(horizon, HOUR),
            Span(0, len(source), source),
        ),
        Span(0, len(source), source),
    )
    return builder.finish(expression, "exact_trip_capacity_minimum")


def _weighted_bundle_residual_count(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not (
        _contains(question.norms, "how", "many")
        and {"red", "yellow", "balloons"}.issubset(question.norms)
    ):
        return None
    rates = _single_clause(
        clause_set,
        lambda clause: (
            "each" in clause.norms
            and "red" in clause.norms
            and "yellow" in clause.norms
            and len(_word_cardinals(clause)) == 2
        ),
        reason="per-color bundle rates are not unique",
    )
    owner_clause = _single_clause(
        clause_set,
        lambda clause: (
            "balloons" in clause.norms
            and "bologna" in clause.norms
            and not _counts(clause)
            and not _word_cardinals(clause)
        ),
        reason="bundle actor is not unique",
    )
    owner = owner_clause.norms[0]
    _require(
        owner_clause.tokens[0].text[:1].isupper()
        and rates.norms[0] in {owner, "he", "she", "they"}
        and owner in question.norms
        and _contains(
            rates.norms,
            "two",
            "pieces",
            "of",
            "bologna",
            "at",
            "each",
            "red",
            "balloon",
        )
        and _contains(
            rates.norms,
            "three",
            "pieces",
            "of",
            "bologna",
            "at",
            "each",
            "yellow",
            "balloon",
        )
        and _contains(question.norms, "pieces", "of", "bologna")
        and _contains(
            question.norms, "bundle", "of", "red", "and", "yellow", "balloons"
        )
        and _contains(question.norms, "balloons", "were", "red")
        and _contains(
            question.norms, "balloons", "in", "the", "bundle", "were", "yellow"
        ),
        "bundle actor, item, color, or target scope differs",
        FrontendStatus.AMBIGUOUS,
    )
    two = _unique_word_token(clause_set, "two", clause=rates)
    three = _unique_word_token(clause_set, "three", clause=rates)
    twenty = _unique_word_token(clause_set, "twenty", clause=question)
    total = _one_token(_counts(question), "bundle contribution total is incomplete")
    builder = _Builder(source, clause_set)
    known = _product(
        _cardinal_literal(builder, twenty, COUNT),
        _cardinal_literal(builder, two),
    )
    remainder = _sum(
        _signed(1, builder.literal(total, COUNT), "piece_total"),
        _signed(-1, known, "known_red_contribution"),
    )
    expression = QuotientExpr(
        remainder,
        _cardinal_literal(builder, three),
        Span(0, len(source), source),
    )
    return builder.finish(expression, "weighted_bundle_residual_count")


def _equal_allowance_purchase_balance(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not (
        "mother" in question.norms
        and _contains(question.norms, "each", "one", "of", "them")
    ):
        return None
    owner_clause = _single_clause(
        clause_set,
        lambda clause: (
            "allowance" in clause.norms
            and "same" in clause.norms
            and "mother" in clause.norms
        ),
        reason="equal allowance owners are not unique",
    )
    combine_clause = _single_clause(
        clause_set,
        lambda clause: (
            "two" in clause.norms
            and "girls" in clause.norms
            and "combine" in clause.norms
        ),
        reason="allowance pooling relation is not unique",
    )
    cake_clause = _single_clause(
        clause_set,
        lambda clause: "cake" in clause.norms and len(_money_tokens(clause)) == 1,
        reason="cake purchase is not unique",
    )
    balloon_clause = _single_clause(
        clause_set,
        lambda clause: (
            "dozen" in clause.norms
            and "balloons" in clause.norms
            and len(_money_tokens(clause)) == 1
        ),
        reason="balloon batch purchase is not unique",
    )
    ice_clause = _single_clause(
        clause_set,
        lambda clause: (
            "remaining" in clause.norms
            and "ice" in clause.norms
            and "each" in clause.norms
        ),
        reason="remaining-money purchase is not unique",
    )
    names = {token.norm for token in owner_clause.tokens if token.text[:1].isupper()}
    question_names = {word.removesuffix("'s") for word in question.norms}
    _require(
        len(names) == 2
        and _contains(combine_clause.norms, "combine", "their", "allowance")
        and names.issubset(question_names)
        and _contains(balloon_clause.norms, "for", "2", "balloons")
        and _contains(
            ice_clause.norms, "remaining", "money", "was", "used", "to", "buy"
        )
        and _contains(ice_clause.norms, "tubs", "of", "ice", "cream"),
        "allowance owners, pool, batch, or exhaustive spend scope differs",
        FrontendStatus.AMBIGUOUS,
    )
    cake = _one_token(_money_tokens(cake_clause), "cake cost is incomplete")
    dozen_count = _one_token(_counts(balloon_clause)[:1], "dozen count is incomplete")
    balloon_price = _one_token(
        _money_tokens(balloon_clause), "balloon batch price is incomplete"
    )
    balloon_batch = _one_token(
        [token for token in _counts(balloon_clause) if token is not dozen_count],
        "balloon price batch is incomplete",
    )
    dozen = _unique_word_token(clause_set, "dozen", clause=balloon_clause)
    ice_count = _one_token(_counts(ice_clause), "ice-cream count is incomplete")
    ice_price = _one_token(_money_tokens(ice_clause), "ice-cream price is incomplete")
    recipients = _unique_word_token(clause_set, "two", clause=combine_clause)

    builder = _Builder(source, clause_set)
    balloon_units = QuotientExpr(
        _product(
            builder.literal(dozen_count, SCALAR),
            builder.lexical_literal(dozen, Fraction(12), COUNT),
        ),
        builder.literal(balloon_batch, COUNT),
        balloon_clause.span,
    )
    subtotal = _sum(
        _signed(1, builder.literal(cake, MONEY), "cake"),
        _signed(
            1,
            _product(balloon_units, builder.literal(balloon_price, MONEY)),
            "balloons",
        ),
        _signed(
            1,
            _product(
                builder.literal(ice_count, COUNT),
                builder.literal(ice_price, PRICE_PER_COUNT),
            ),
            "ice_cream",
        ),
    )
    expression = QuotientExpr(
        subtotal,
        _cardinal_literal(builder, recipients),
        Span(0, len(source), source),
    )
    return builder.finish(expression, "equal_allowance_purchase_balance")


def _exact_packaging_capacity(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not (
        _contains(question.norms, "how", "many", "boxes")
        and "need" in question.norms
        and "apiece" in question.norms
    ):
        return None
    unit_clause = _single_clause(
        clause_set,
        lambda clause: (
            "each" in clause.norms
            and "sleeve" in clause.norms
            and "smores" in clause.norms
            and len(_counts(clause)) == 1
        ),
        reason="per-sleeve capacity is not unique",
    )
    pack_clause = _single_clause(
        clause_set,
        lambda clause: (
            "sleeves" in clause.norms
            and "box" in clause.norms
            and len(_counts(clause)) == 1
        ),
        reason="sleeves-per-box capacity is not unique",
    )
    capacity = _one_token(_counts(unit_clause), "sleeve capacity is incomplete")
    per_box = _one_token(_counts(pack_clause), "box capacity is incomplete")
    numbers = _counts(question)
    _require(len(numbers) == 4, "consumer demand terms are incomplete")
    kids, per_kid, adults, per_adult = numbers
    _require(
        _contains(question.norms, "kids", "want")
        and _contains(question.norms, "smores", "apiece")
        and _contains(question.norms, "adults", "will", "eat")
        and _noun_after(question, per_kid) == "smore"
        and _noun_after(question, per_adult) == "smore"
        and _contains(unit_clause.norms, "sleeve", "of", "graham", "crackers")
        and _contains(pack_clause.norms, "sleeves", "in", "a", "box")
        and _contains(question.norms, "boxes", "of", "graham", "crackers")
        and all(
            token.number is not None and token.number > 0
            for token in (*numbers, capacity, per_box)
        ),
        "capacity item, consumer role, or positive unit binding differs",
        FrontendStatus.AMBIGUOUS,
    )
    boxes_word = next(token for token in question.tokens if token.norm == "boxes")
    builder = _Builder(source, clause_set)
    demand = _sum(
        _signed(
            1,
            _product(
                builder.literal(kids, SCALAR),
                builder.literal(per_kid, COUNT),
            ),
            "child_demand",
        ),
        _signed(
            1,
            _product(
                builder.literal(adults, SCALAR),
                builder.literal(per_adult, COUNT),
            ),
            "adult_demand",
        ),
    )
    sleeves = CeilingExpr(
        QuotientExpr(
            demand,
            builder.literal(capacity, COUNT),
            unit_clause.span,
        ),
        unit_clause.span,
    )
    boxes = CeilingExpr(
        QuotientExpr(
            sleeves,
            builder.literal(per_box, SCALAR),
            pack_clause.span,
        ),
        pack_clause.span,
    )
    expression = _product(
        boxes,
        builder.lexical_literal(boxes_word, Fraction(1), COUNT),
    )
    return builder.finish(expression, "exact_packaging_capacity")


def _exact_package_demand_cost(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not ("spend" in question.norms and "jello" in question.norms):
        return None
    package = _single_clause(
        clause_set,
        lambda clause: (
            "box" in clause.norms
            and "jello" in clause.norms
            and "cups" in clause.norms
            and len(_counts(clause)) == 1
        ),
        reason="jello package yield is not unique",
    )
    demand_clause = _single_clause(
        clause_set,
        lambda clause: (
            "kids" in clause.norms
            and "each" in clause.norms
            and "cups" in clause.norms
            and len(_counts(clause)) == 2
        ),
        reason="party cup demand is not unique",
    )
    price_clause = _single_clause(
        clause_set,
        lambda clause: (
            "jello" in clause.norms
            and "sale" in clause.norms
            and len(_money_tokens(clause)) == 1
        ),
        reason="jello package price is not unique",
    )
    yield_count = _one_token(_counts(package), "cups-per-box yield is incomplete")
    consumers, each_count = _counts(demand_clause)
    price = _one_token(_money_tokens(price_clause), "package price is incomplete")
    yield_index = package.tokens.index(yield_count)
    _require(
        _contains(package.norms, "box", "of", "flavored", "jello")
        and "makes" in package.norms[:yield_index]
        and package.norms[yield_index + 1 : yield_index + 4]
        == ("small", "jello", "cups")
        and _contains(demand_clause.norms, "each", "kid", "can", "have")
        and _contains(
            price_clause.norms, "jello", "is", "currently", "on", "sale", "for"
        )
        and all(
            token.number is not None and token.number > 0
            for token in (yield_count, consumers, each_count)
        ),
        "package item, demand, price scope, or positive capacity differs",
        FrontendStatus.AMBIGUOUS,
    )
    builder = _Builder(source, clause_set)
    demand = _product(
        builder.literal(consumers, SCALAR),
        builder.literal(each_count, COUNT),
    )
    boxes = CeilingExpr(
        QuotientExpr(
            demand,
            builder.literal(yield_count, COUNT),
            package.span,
        ),
        package.span,
    )
    expression = _product(boxes, builder.literal(price, MONEY))
    return builder.finish(expression, "exact_package_demand_cost")


def _typed_bowl_capacity_leftover(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not (
        "leftover" in question.norms
        and _contains(question.norms, "each", "child")
        and "lunch" in question.norms
    ):
        return None
    owner_clause = _single_clause(
        clause_set,
        lambda clause: (
            "soup" in clause.norms
            and "family" in clause.norms
            and "dinner" in clause.norms
        ),
        reason="soup owner and meal scope are not unique",
    )
    capacity = _single_clause(
        clause_set,
        lambda clause: (
            "pot" in clause.norms
            and "adult's" in clause.norms
            and "child's" in clause.norms
            and len(_word_cardinals(clause)) == 2
        ),
        reason="adult/child bowl capacity is not unique",
    )
    family = _single_clause(
        clause_set,
        lambda clause: (
            "wife" in clause.norms
            and "adult" in clause.norms
            and "children" in clause.norms
            and "two" in clause.norms
        ),
        reason="meal participant composition is not unique",
    )
    owner = owner_clause.norms[0]
    _require(
        owner_clause.tokens[0].text[:1].isupper()
        and capacity.norms[0] in {owner, "he", "she", "they", "it"}
        and family.norms[0] in {owner, "he", "she", "they"}
        and _contains(capacity.norms, "adult's", "bowls", "or")
        and _contains(capacity.norms, "child's", "bowls")
        and _contains(family.norms, "adult", "wife")
        and _contains(family.norms, "their", "two", "children")
        and not any(token.text[:1].isupper() for token in family.tokens[1:])
        and _contains(
            question.norms, "everyone", "eats", "one", "bowl", "at", "a", "meal"
        )
        and _contains(question.norms, "bowl", "of", "soup")
        and _contains(question.norms, "leftover", "soup"),
        "soup actor, bowl type, participant, or meal state differs",
        FrontendStatus.AMBIGUOUS,
    )
    adult_capacity = _unique_word_token(clause_set, "four", clause=capacity)
    child_capacity = _unique_word_token(clause_set, "eight", clause=capacity)
    children = _unique_word_token(clause_set, "two", clause=family)
    one_bowl = _unique_word_token(clause_set, "one", clause=question)
    adult_start = family.tokens[0].span.start
    adult_end = family.tokens[family.norms.index("wife")].span.end
    adult_span = Span(adult_start, adult_end, source)
    adults = Token(
        source[adult_span.start : adult_span.end],
        "two_explicit_adults",
        adult_span,
    )

    builder = _Builder(source, clause_set)
    child_capacity_key = SymbolKey(
        owner, "capacity", "child_bowl", "pot", "before_meal"
    )
    child_capacity_definition = Definition(
        child_capacity_key,
        _cardinal_literal(builder, child_capacity, COUNT),
        capacity.span,
    )
    adult_equivalent = QuotientExpr(
        RefExpr(child_capacity_key, capacity.span),
        _cardinal_literal(builder, adult_capacity),
        capacity.span,
    )
    adult_use = _product(
        adult_equivalent,
        builder.lexical_literal(adults, Fraction(2), SCALAR),
    )
    before_children = _sum(
        _signed(1, RefExpr(child_capacity_key, question.span), "pot_capacity"),
        _signed(-1, adult_use, "adult_dinner_use"),
    )
    per_child_before_dinner = QuotientExpr(
        before_children,
        _cardinal_literal(builder, children),
        family.span,
    )
    expression = _sum(
        _signed(1, per_child_before_dinner, "per_child_share"),
        _signed(-1, _cardinal_literal(builder, one_bowl, COUNT), "dinner_bowl"),
    )
    return builder.finish(
        expression,
        "typed_bowl_capacity_leftover",
        definitions=(child_capacity_definition,),
    )


def _fractional_group_consumption_remainder(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not (
        _contains(question.norms, "how", "many", "slices", "of", "pizza")
        and "left" in question.norms
    ):
        return None
    group_clause = _single_clause(
        clause_set,
        lambda clause: (
            "friends" in clause.norms
            and "pizzas" in clause.norms
            and "each" in clause.norms
            and "four" in clause.norms
        ),
        reason="pizza owner group is not unique",
    )
    owner = group_clause.norms[0]
    size_clause = _single_clause(
        clause_set,
        lambda clause: (
            "each" in clause.norms
            and "pizza" in clause.norms
            and "slices" in clause.norms
            and len(_counts(clause)) == 1
        ),
        reason="slices-per-pizza size is not unique",
    )
    first_group = _single_clause(
        clause_set,
        lambda clause: (
            owner in clause.norms
            and "friends" in clause.norms
            and "ate" in clause.norms
            and any(
                token.number is not None and token.number < 1
                for token in _counts(clause)
            )
        ),
        reason="first pizza-consumption group is not unique",
    )
    second_group = _single_clause(
        clause_set,
        lambda clause: (
            "remaining" in clause.norms
            and "friends" in clause.norms
            and "ate" in clause.norms
            and any(
                token.number is not None and token.number < 1
                for token in _counts(clause)
            )
        ),
        reason="remaining pizza-consumption group is not unique",
    )
    _require(
        group_clause.tokens[0].text[:1].isupper()
        and first_group.norms[0] == owner
        and _contains(group_clause.norms, "each", "ordered", "their", "own", "pizzas")
        and _contains(size_clause.norms, "each", "pizza", "had")
        and _contains(first_group.norms, "of", "their", "pizzas")
        and _contains(second_group.norms, "of", "their", "pizzas"),
        "pizza actor, ownership, item, or group scope differs",
        FrontendStatus.AMBIGUOUS,
    )
    four = _unique_word_token(clause_set, "four", clause=group_clause)
    group_one_two = _unique_word_token(clause_set, "two", clause=first_group)
    group_two = _unique_word_token(clause_set, "two", clause=second_group)
    fraction_one = _one_token(
        _counts(first_group), "first eaten fraction is incomplete"
    )
    fraction_two = _one_token(
        _counts(second_group), "second eaten fraction is incomplete"
    )
    slices = _one_token(_counts(size_clause), "pizza slice count is incomplete")
    _require(
        all(
            token.number is not None and 0 < token.number < 1
            for token in (fraction_one, fraction_two)
        )
        and _LOCAL_CARDINALS[group_one_two.norm] + 1 + _LOCAL_CARDINALS[group_two.norm]
        == _LOCAL_CARDINALS[four.norm] + 1,
        "pizza groups do not exhaust the declared owners",
        FrontendStatus.AMBIGUOUS,
    )
    group_one_span = Span(
        first_group.tokens[0].span.start,
        first_group.tokens[first_group.norms.index("friends")].span.end,
        source,
    )
    group_one = Token(
        source[group_one_span.start : group_one_span.end],
        "named_owner_plus_two_friends",
        group_one_span,
    )

    builder = _Builder(source, clause_set)
    remaining_pizzas = _sum(
        _signed(
            1,
            builder.lexical_literal(group_clause.tokens[0], Fraction(1), SCALAR),
            "named_owner",
        ),
        _signed(1, _cardinal_literal(builder, four), "friends"),
        _signed(
            -1,
            _product(
                builder.lexical_literal(group_one, Fraction(3), SCALAR),
                builder.literal(fraction_one, SCALAR),
            ),
            "first_group_eaten",
        ),
        _signed(
            -1,
            _product(
                _cardinal_literal(builder, group_two),
                builder.literal(fraction_two, SCALAR),
            ),
            "second_group_eaten",
        ),
    )
    expression = _product(
        remaining_pizzas,
        builder.literal(slices, COUNT),
    )
    return builder.finish(expression, "fractional_group_consumption_remainder")


def _typed_chair_capacity_deficit(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not (
        _contains(question.norms, "how", "many", "more", "chairs")
        and "get" in question.norms
    ):
        return None
    attendance = _single_clause(
        clause_set,
        lambda clause: (
            "tomorrow" in clause.norms
            and "adults" in clause.norms
            and "babies" in clause.norms
            and len(_counts(clause)) == 2
        ),
        reason="typed attendance demand is not unique",
    )
    ratio_clause = _single_clause(
        clause_set,
        lambda clause: (
            "regular" in clause.norms
            and "high" in clause.norms
            and "times" in clause.norms
            and len(_counts(clause)) == 1
        ),
        reason="regular/high chair ratio is not unique",
    )
    adults, babies = _counts(attendance)
    scale = _one_token(_counts(ratio_clause), "regular-chair scale is incomplete")
    high = _one_token(_counts(question), "high-chair assignment is incomplete")
    _require(
        _noun_after(attendance, adults) == "adult"
        and _noun_after(attendance, babies) == "baby"
        and _contains(ratio_clause.norms, "regular", "chairs", "as", "high", "chairs")
        and _contains(
            question.norms[: question.tokens.index(high)], "if", "there", "are"
        )
        and _contains(question.norms, "high", "chairs")
        and "restaurant" in attendance.norms
        and "restaurant" in ratio_clause.norms,
        "attendance type, chair type, owner, or state differs",
        FrontendStatus.AMBIGUOUS,
    )
    _require(
        all(
            token.number is not None and token.number > 0
            for token in (adults, babies, scale, high)
        )
        and adults.number >= scale.number * high.number
        and babies.number >= high.number,
        "typed deficit requires positive demands and nonnegative per-type shortages",
        FrontendStatus.INVALID,
    )
    builder = _Builder(source, clause_set)
    high_key = SymbolKey(
        "restaurant", "capacity", "high_chair", "tomorrow", "available"
    )
    regular_key = SymbolKey(
        "restaurant", "capacity", "regular_chair", "tomorrow", "available"
    )
    definitions = (
        Definition(high_key, builder.literal(high, COUNT), question.span),
        Definition(
            regular_key,
            _product(
                builder.literal(scale, SCALAR),
                RefExpr(high_key, ratio_clause.span),
            ),
            ratio_clause.span,
        ),
    )
    expression = _sum(
        _signed(1, builder.literal(adults, COUNT), "adult_demand"),
        _signed(-1, RefExpr(regular_key, question.span), "regular_capacity"),
        _signed(1, builder.literal(babies, COUNT), "baby_demand"),
        _signed(-1, RefExpr(high_key, question.span), "high_capacity"),
    )
    return builder.finish(
        expression,
        "typed_chair_capacity_deficit",
        definitions=definitions,
    )


def _typed_percentage_trade_transitions(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not (
        _contains(question.norms, "how", "many", "buttons") and "end" in question.norms
    ):
        return None
    intro = _single_clause(
        clause_set,
        lambda clause: (
            "trading" in clause.norms
            and "stickers" in clause.norms
            and "buttons" in clause.norms
        ),
        reason="trade actor and item scope are not unique",
    )
    large_sticker_rate = _single_clause(
        clause_set,
        lambda clause: (
            "large" in clause.norms
            and "sticker" in clause.norms
            and "button" in clause.norms
            and "three" in clause.norms
            and "or" in clause.norms
        ),
        reason="large-sticker conversion choices are not unique",
    )
    small_identity = _single_clause(
        clause_set,
        lambda clause: (
            "small" in clause.norms
            and "sticker" in clause.norms
            and "one" in clause.norms
            and "button" in clause.norms
        ),
        reason="small-sticker identity conversion is not unique",
    )
    large_button_rate = _single_clause(
        clause_set,
        lambda clause: (
            "large" in clause.norms
            and "button" in clause.norms
            and "three" in clause.norms
            and "small" in clause.norms
            and "stickers" in clause.norms
        ),
        reason="large-button inverse conversion is not unique",
    )
    initial = _single_clause(
        clause_set,
        lambda clause: (
            "starts" in clause.norms
            and "small" in clause.norms
            and "large" in clause.norms
            and len(_counts(clause)) == 2
        ),
        reason="initial sticker inventory is not unique",
    )
    small_trade = _single_clause(
        clause_set,
        lambda clause: (
            _contains(clause.norms, "small", "stickers")
            and _contains(clause.norms, "large", "buttons")
            and "rest" not in clause.norms
            and len(_counts(clause)) == 1
        ),
        reason="small-sticker percentage trade is not unique",
    )
    large_trade = _single_clause(
        clause_set,
        lambda clause: (
            "large" in clause.norms
            and "rest" in clause.norms
            and "small" in clause.norms
            and "buttons" in clause.norms
            and len(_counts(clause)) == 1
        ),
        reason="large-sticker split trade is not unique",
    )
    owner = initial.norms[0]
    _require(
        owner == intro.norms[0]
        and all(
            clause.norms[0] in {owner, "he", "she", "they"}
            for clause in (small_trade, large_trade)
        )
        and _contains(
            large_sticker_rate.norms,
            "large",
            "sticker",
            "is",
            "worth",
            "a",
            "large",
            "button",
        )
        and _contains(large_sticker_rate.norms, "or", "three", "small", "buttons")
        and _contains(
            small_identity.norms,
            "small",
            "sticker",
            "is",
            "worth",
            "one",
            "small",
            "button",
        )
        and _contains(
            large_button_rate.norms,
            "large",
            "button",
            "is",
            "worth",
            "three",
            "small",
            "stickers",
        )
        and _contains(small_trade.norms, "small", "stickers", "for", "large", "buttons")
        and _contains(large_trade.norms, "large", "stickers", "for", "large", "buttons")
        and _contains(
            large_trade.norms, "rest", "of", "them", "for", "small", "buttons"
        ),
        "trade actor, item direction, conversion, or exhaustive split differs",
        FrontendStatus.AMBIGUOUS,
    )
    initial_small, initial_large = _counts(initial)
    small_percent = _one_token(
        _counts(small_trade), "small trade percentage is incomplete"
    )
    large_percent = _one_token(
        _counts(large_trade), "large trade percentage is incomplete"
    )
    _require(
        all(
            token.number is not None
            and 0 < token.number < 100
            and source[token.span.end : token.span.end + 1] == "%"
            for token in (small_percent, large_percent)
        ),
        "trade percentages are invalid",
    )
    to_small_rate = _unique_word_token(clause_set, "three", clause=large_sticker_rate)
    to_large_divisor = _unique_word_token(clause_set, "three", clause=large_button_rate)
    complement_span = Span(large_percent.span.start, large_percent.span.end + 1, source)
    complement = Token(
        source[complement_span.start : complement_span.end],
        "large_trade_complement",
        complement_span,
    )
    zero_span = Span(small_percent.span.end, small_percent.span.end + 1, source)
    zero = Token("%", "count_scale_origin", zero_span)

    builder = _Builder(source, clause_set)
    large_key = SymbolKey(owner, "quantity", "large_sticker", "trade", "initial")
    large_definition = Definition(
        large_key,
        builder.literal(initial_large, COUNT),
        initial.span,
    )
    small_to_large = QuotientExpr(
        _product(
            builder.literal(initial_small, COUNT),
            builder.literal(small_percent, PERCENT),
        ),
        _cardinal_literal(builder, to_large_divisor),
        small_trade.span,
    )
    large_to_large = _product(
        RefExpr(large_key, large_trade.span),
        builder.literal(large_percent, PERCENT),
    )
    large_to_small = _product(
        RefExpr(large_key, large_trade.span),
        builder.lexical_literal(
            complement,
            Fraction(100) - large_percent.number,
            PERCENT,
        ),
        _cardinal_literal(builder, to_small_rate),
    )
    expression = _sum(
        _signed(1, builder.lexical_literal(zero, Fraction(0), COUNT), "count_origin"),
        _signed(1, small_to_large, "large_buttons_from_small"),
        _signed(1, large_to_large, "large_buttons_from_large"),
        _signed(1, large_to_small, "small_buttons_from_large"),
    )
    return builder.finish(
        expression,
        "typed_percentage_trade_transitions",
        definitions=(large_definition,),
    )


def _exhaustive_unit_rate_ledger(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    """Resolve one owner pronoun into a fully exhaustive per-item ledger."""

    question = _question(clause_set)
    sale_mode = (
        _contains(question.norms, "sells", "everything") and "earn" in question.norms
    )
    commissioned_mode = "pay" in question.norms and not sale_mode
    if not (sale_mode or commissioned_mode):
        return None
    ledger_candidates = [
        clause
        for clause in clause_set
        if len(_money_tokens(clause)) >= 2
        and len(_money_tokens(clause)) == len(_counts(clause))
        and all(
            clause.tokens.index(token) + 1 < len(clause.tokens)
            and clause.norms[clause.tokens.index(token) + 1] == "each"
            for token in _money_tokens(clause)
        )
    ]
    if len(ledger_candidates) != 1:
        return None
    intro = _single_clause(
        clause_set,
        lambda clause: (
            (
                ("sell" in clause.norms and "planning" in clause.norms)
                or "carpenter" in clause.norms
            )
            and not _numeric((clause,))
        ),
        reason="unit-rate ledger owner is not uniquely introduced",
    )
    owner_names = [
        token.norm.rstrip("'s") for token in intro.tokens if token.text[:1].isupper()
    ]
    _require(
        len(owner_names) == 1,
        "unit-rate ledger requires one explicit owner",
        FrontendStatus.AMBIGUOUS,
    )
    owner = owner_names[0]
    ledger = ledger_candidates[0]

    subject_pronouns = {word for word in ledger.norms if word in {"he", "she", "they"}}
    possessives = {
        word
        for word in (*intro.norms, *ledger.norms)
        if word in {"his", "her", "their"}
    }
    subject_to_possessive = {"he": "his", "she": "her", "they": "their"}
    _require(
        len(subject_pronouns) == 1
        and possessives == {subject_to_possessive[next(iter(subject_pronouns))]},
        "unit-rate owner pronouns do not agree",
        FrontendStatus.AMBIGUOUS,
    )
    subject = next(iter(subject_pronouns))

    rows: list[tuple[Token, Token, str, frozenset[str]]] = []
    prior_money = -1
    for price in _money_tokens(ledger):
        price_index = ledger.tokens.index(price)
        local_counts = [
            token
            for token in _counts(ledger)
            if prior_money < ledger.tokens.index(token) < price_index
        ]
        _require(
            len(local_counts) == 1,
            "unit-rate price has no unique local quantity",
            FrontendStatus.AMBIGUOUS,
        )
        quantity = local_counts[0]
        quantity_index = ledger.tokens.index(quantity)
        relation_words = frozenset(ledger.norms[quantity_index + 1 : price_index])
        item_words = []
        for word in ledger.norms[quantity_index + 1 : price_index]:
            if word in {"at", "can", "cost", "for", "that", "which"}:
                break
            if word != ",":
                item_words.append(word)
        item = _item_id(item_words)
        rows.append((quantity, price, item, relation_words))
        prior_money = price_index
    _require(
        len({item for _, _, item, _ in rows}) == len(rows)
        and all(
            token.number is not None and token.number >= 0
            for row in rows
            for token in row[:2]
        ),
        "unit-rate ledger items or values are invalid",
        FrontendStatus.INVALID,
    )

    if sale_mode:
        question_subjects = [
            word for word in question.norms if word in {"he", "she", "they"}
        ]
        _require(
            ledger.norms[:2] == (subject, "has")
            and _contains(intro.norms, "planning", "to", "sell")
            and question_subjects == [subject, subject]
            and _contains(question.norms, "sells", "everything")
            and all(
                {"cost", "sold"}.intersection(relation) for _, _, _, relation in rows
            ),
            "sale owner, exhaustive scope, or rate relation differs",
            FrontendStatus.AMBIGUOUS,
        )
    else:
        friend_offsets = [
            index
            for index, word in enumerate(ledger.norms[:-1])
            if word == "friend" and ledger.tokens[index + 1].text[:1].isupper()
        ]
        friend_names = [ledger.norms[index + 1] for index in friend_offsets]
        _require(
            len(friend_names) == 1
            and "manufactured" in ledger.norms
            and all("for" in relation for _, _, _, relation in rows)
            and _contains(ledger.norms, "for", next(iter(possessives)), "friend")
            and _contains(
                question.norms, "does", friend_names[0], "have", "to", "pay", owner
            ),
            "commissioned item payer, maker, or payment direction differs",
            FrontendStatus.AMBIGUOUS,
        )

    builder = _Builder(source, clause_set)
    expression = _sum(
        *[
            _signed(
                1,
                _product(
                    builder.literal(quantity, COUNT),
                    builder.literal(price, PRICE_PER_COUNT),
                ),
                f"{item}_line_item",
            )
            for quantity, price, item, _ in rows
        ]
    )
    return builder.finish(expression, "exhaustive_unit_rate_ledger")


def _recurring_pronoun_rate_ledger(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    """Bind one possessive inventory to matching rates and a fixed recurrence."""

    question = _question(clause_set)
    if not (
        "charge" in question.norms
        and "spend" in question.norms
        and "weeks" in question.norms
        and "dry-cleaning" in question.norms
    ):
        return None
    intro = _single_clause(
        clause_set,
        lambda clause: (
            "clothes" in clause.norms
            and "dry" in clause.norms
            and "cleaners" in clause.norms
            and "weekly" in clause.norms
        ),
        reason="recurring service owner is not uniquely introduced",
    )
    owner_tokens = [
        token
        for token in intro.tokens
        if token.text[:1].isupper() and token.norm.endswith("'s")
    ]
    _require(
        len(owner_tokens) == 1,
        "recurring service requires one possessive owner",
        FrontendStatus.AMBIGUOUS,
    )
    inventory = _single_clause(
        clause_set,
        lambda clause: (
            clause.norms[:3] == ("her", "weekly", "drop-off")
            and "includes" in clause.norms
            and len(_counts(clause)) >= 2
        ),
        reason="weekly item inventory is not unique",
    )
    _require(
        _contains(question.norms, "they", "charge", "her")
        and _contains(question.norms, "does", "she", "spend"),
        "recurring service owner pronouns do not agree",
        FrontendStatus.AMBIGUOUS,
    )

    quantities: dict[str, Token] = {}
    for quantity in _counts(inventory):
        item = _noun_after(inventory, quantity)
        _require(
            item not in quantities,
            "duplicate recurring inventory item",
            FrontendStatus.AMBIGUOUS,
        )
        quantities[item] = quantity
    rates: dict[str, Token] = {}
    for price in _money_tokens(question):
        index = question.tokens.index(price)
        _require(
            index + 2 < len(question.tokens) and question.norms[index + 1] == "per",
            "recurring service rate lacks a per-item scope",
            FrontendStatus.AMBIGUOUS,
        )
        item = _noun_after_index(
            question,
            index + 1,
            skip=frozenset({"of", "pair", "the"}),
        )
        _require(
            item not in rates,
            "duplicate recurring service rate",
            FrontendStatus.AMBIGUOUS,
        )
        rates[item] = price
    horizon = _one_token(
        [
            token
            for token in _counts(question)
            if _noun_after(question, token) == "week"
        ],
        "recurring service horizon is incomplete",
    )
    _require(
        set(quantities) == set(rates)
        and horizon.number is not None
        and horizon.number > 0,
        "recurring inventory and service rates do not cover the same items",
        FrontendStatus.AMBIGUOUS,
    )

    builder = _Builder(source, clause_set)
    weekly = _sum(
        *[
            _signed(
                1,
                _product(
                    builder.literal(quantities[item], COUNT),
                    builder.literal(rates[item], PRICE_PER_COUNT),
                ),
                f"{item}_weekly_cost",
            )
            for item in sorted(quantities)
        ]
    )
    expression = _product(weekly, builder.literal(horizon, SCALAR))
    return builder.finish(expression, "recurring_pronoun_rate_ledger")


def _temporal_categorical_block_remainder(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    """Bind a closed categorical calendar from totals and ordered blocks."""

    question = _question(clause_set)
    if not (
        _contains(question.norms, "how", "many")
        and "left" in question.norms
        and "month" in question.norms
        and "next" in question.norms
        and "days" in question.norms
    ):
        return None
    many_index = question.norms.index("many")
    _require(
        many_index + 2 < len(question.tokens)
        and question.norms[many_index + 2] == "days",
        "temporal target category is not locally bound",
        FrontendStatus.AMBIGUOUS,
    )
    target_state = _singular(question.norms[many_index + 1])

    overview = _single_clause(
        clause_set,
        lambda clause: (
            "past" in clause.norms
            and "moods" in clause.norms
            and "had" in clause.norms
            and "rest" in clause.norms
            and "days" in clause.norms
        ),
        reason="categorical calendar totals are not unique",
    )
    blocks = _single_clause(
        clause_set,
        lambda clause: (
            all(ordinal in clause.norms for ordinal in ("first", "second", "third"))
            and clause.norms.count("days") == 3
        ),
        reason="ordered categorical blocks are not unique",
    )

    intro_names = {
        token.norm
        for clause in clause_set
        if clause.span.end <= overview.span.start
        for token in clause.tokens
        if token.text[:1].isupper()
    }
    overview_pronouns = {
        word for word in overview.norms if word in {"he", "she", "they"}
    }
    block_possessives = {
        word for word in blocks.norms if word in {"his", "her", "their"}
    }
    agreement = {"he": "his", "she": "her", "they": "their"}
    _require(
        len(intro_names) == 1
        and len(overview_pronouns) == 1
        and block_possessives == {agreement[next(iter(overview_pronouns))]},
        "calendar owner pronouns have no unique antecedent",
        FrontendStatus.AMBIGUOUS,
    )

    horizon_rows = []
    for index, token in enumerate(overview.tokens):
        value = _LOCAL_CARDINALS.get(token.norm)
        if (
            value is not None
            and index > 0
            and index + 1 < len(overview.tokens)
            and overview.norms[index - 1] == "past"
            and overview.norms[index + 1] == "days"
        ):
            horizon_rows.append((token, value))
    _require(
        len(horizon_rows) == 1,
        "calendar horizon is not one local word cardinal",
        FrontendStatus.AMBIGUOUS,
    )
    horizon_token, horizon = horizon_rows[0]

    category_rows: dict[str, tuple[Token, Fraction]] = {}
    for index in range(1, len(overview.tokens) - 1):
        token = overview.tokens[index - 1]
        value = _LOCAL_CARDINALS.get(token.norm)
        if value is None or overview.norms[index + 1] != "days":
            continue
        state = _singular(overview.norms[index])
        if state in category_rows:
            raise _Reject(FrontendStatus.AMBIGUOUS, "duplicate calendar category total")
        category_rows[state] = (token, value)
    rest_rows = [
        _singular(overview.norms[index + 2])
        for index in range(len(overview.tokens) - 2)
        if overview.norms[index : index + 2] == ("rest", "were")
    ]
    _require(
        len(category_rows) == 2
        and target_state in category_rows
        and len(rest_rows) == 1
        and rest_rows[0] not in category_rows,
        "calendar category totals or remainder state are incomplete",
        FrontendStatus.AMBIGUOUS,
    )
    explicit_total = sum(value for _, value in category_rows.values())
    rest_total = horizon - explicit_total
    _require(
        horizon > 0 and rest_total >= 0,
        "calendar category totals exceed the horizon",
        FrontendStatus.INVALID,
    )
    category_totals = {state: value for state, (_, value) in category_rows.items()}
    category_totals[rest_rows[0]] = rest_total

    parsed_blocks: list[tuple[str, Token, Fraction]] = []
    for ordinal in ("first", "second", "third"):
        ordinal_index = blocks.norms.index(ordinal)
        _require(
            ordinal_index + 4 < len(blocks.tokens)
            and blocks.norms[ordinal_index + 2 : ordinal_index + 4] == ("days", "were"),
            "ordered calendar block is malformed",
            FrontendStatus.AMBIGUOUS,
        )
        count_token = blocks.tokens[ordinal_index + 1]
        count = _LOCAL_CARDINALS.get(count_token.norm)
        _require(
            count is not None and count > 0,
            "calendar block count is not a positive word cardinal",
            FrontendStatus.INVALID,
        )
        state = _singular(blocks.norms[ordinal_index + 4])
        parsed_blocks.append((state, count_token, count))
    _require(
        {state for state, _, _ in parsed_blocks} == set(category_totals)
        and len({state for state, _, _ in parsed_blocks}) == 3
        and all(count <= category_totals[state] for state, _, count in parsed_blocks),
        "ordered blocks and category totals do not describe the same states",
        FrontendStatus.AMBIGUOUS,
    )

    next_index = question.norms.index("next")
    _require(
        next_index + 4 < len(question.tokens)
        and question.norms[next_index + 2 : next_index + 4] == ("days", "were"),
        "next categorical block is malformed",
        FrontendStatus.AMBIGUOUS,
    )
    next_count_token = question.tokens[next_index + 1]
    next_count = _LOCAL_CARDINALS.get(next_count_token.norm)
    how_index = question.norms.index("how")
    sequence_tokens = [
        token
        for token in question.tokens[next_index + 4 : how_index]
        if token.norm not in {",", "and"}
    ]
    _require(
        next_count is not None
        and next_count == len(sequence_tokens)
        and all(_singular(token.norm) in category_totals for token in sequence_tokens),
        "next block count and categorical sequence differ",
        FrontendStatus.AMBIGUOUS,
    )
    scheduled_days = sum(count for _, _, count in parsed_blocks) + next_count
    _require(
        scheduled_days <= horizon,
        "ordered calendar blocks exceed the horizon",
        FrontendStatus.INVALID,
    )

    target_total_token, target_total = category_rows[target_state]
    target_blocks = [
        (token, count) for state, token, count in parsed_blocks if state == target_state
    ]
    next_targets = [
        token for token in sequence_tokens if _singular(token.norm) == target_state
    ]
    known_target = sum(count for _, count in target_blocks) + len(next_targets)
    _require(
        target_total >= known_target,
        "scheduled target days exceed the declared target total",
        FrontendStatus.INVALID,
    )

    builder = _Builder(source, clause_set)
    terms = [
        _signed(
            1,
            builder.lexical_literal(target_total_token, target_total, COUNT),
            "declared_target_total",
        )
    ]
    terms.extend(
        _signed(
            -1,
            builder.lexical_literal(token, count, COUNT),
            "completed_target_block",
        )
        for token, count in target_blocks
    )
    terms.extend(
        _signed(
            -1,
            builder.lexical_literal(token, Fraction(1), COUNT),
            "scheduled_target_day",
        )
        for token in next_targets
    )
    return builder.finish(
        _sum(*terms),
        "temporal_categorical_block_remainder",
    )


def _absolute_weighted_score_difference(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    """Compile a shared score table and signed comparative deltas through Abs."""

    question = _question(clause_set)
    if not (
        _contains(question.norms, "what", "is", "the", "difference")
        and _contains(question.norms, "between", "their", "two", "scores")
    ):
        return None
    intro = _single_clause(
        clause_set,
        lambda clause: (
            "want" in clause.norms
            and "know" in clause.norms
            and "who" in clause.norms
            and "game" in clause.norms
        ),
        reason="score contestants are not uniquely introduced",
    )
    names = list(
        dict.fromkeys(token.norm for token in intro.tokens if token.text[:1].isupper())
    )
    _require(
        len(names) == 2 and len(set(names)) == 2,
        "score comparison requires exactly two named contestants",
        FrontendStatus.AMBIGUOUS,
    )
    contest = _single_clause(
        clause_set,
        lambda clause: (
            clause.norms[:1] == ("they",)
            and "each" in clause.norms
            and "score" in clause.norms
            and "wins" in clause.norms
        ),
        reason="shared score contest scope is not unique",
    )
    _require(
        _contains(contest.norms, "they", "are", "each", "going", "to", "play")
        and _contains(contest.norms, "highest", "score", "wins"),
        "score contest does not bind both contestants",
        FrontendStatus.AMBIGUOUS,
    )
    rates = _single_clause(
        clause_set,
        lambda clause: (
            clause.norms[:1] == ("they",)
            and "receive" in clause.norms
            and clause.norms.count("points") == 3
            and len(_counts(clause)) == 3
        ),
        reason="shared score-rate table is not unique",
    )

    rate_by_item: dict[str, tuple[Token, Unit]] = {}
    for rate in _counts(rates):
        index = rates.tokens.index(rate)
        _require(
            index + 1 < len(rates.tokens)
            and _singular(rates.norms[index + 1]) == "point",
            "score rate lacks a local point unit",
            FrontendStatus.AMBIGUOUS,
        )
        quantifiers = [
            offset
            for offset in range(index + 2, min(len(rates.tokens), index + 7))
            if rates.norms[offset] in {"each", "every"}
        ]
        _require(
            len(quantifiers) == 1 and quantifiers[0] + 1 < len(rates.tokens),
            "score rate lacks one local event scope",
            FrontendStatus.AMBIGUOUS,
        )
        item = _singular(rates.norms[quantifiers[0] + 1])
        _require(
            item not in rate_by_item,
            "duplicate score-rate event",
            FrontendStatus.AMBIGUOUS,
        )
        unit = POINT / (SECOND if item == "second" else COUNT)
        rate_by_item[item] = (rate, unit)

    deltas = _counts(question)
    _require(len(deltas) == 3, "comparative score deltas are incomplete")
    delta_by_item: dict[str, tuple[Token, int, Unit]] = {}
    for delta in deltas:
        index = question.tokens.index(delta)
        _require(index + 1 < len(question.tokens), "score delta item is missing")
        modifier = question.norms[index + 1]
        if modifier in {"more", "fewer", "less"}:
            _require(index + 2 < len(question.tokens), "score delta item is missing")
            item = _singular(question.norms[index + 2])
            sign = 1 if modifier == "more" else -1
        else:
            item = _singular(modifier)
            direction = (
                question.norms[index + 2] if index + 2 < len(question.tokens) else ""
            )
            _require(
                direction in {"faster", "slower"},
                "score delta has no explicit direction",
                FrontendStatus.AMBIGUOUS,
            )
            sign = 1 if direction == "faster" else -1
        _require(
            item not in delta_by_item,
            "duplicate comparative score event",
            FrontendStatus.AMBIGUOUS,
        )
        unit = SECOND if item == "second" else COUNT
        delta_by_item[item] = (delta, sign, unit)

    _require(
        set(rate_by_item) == set(delta_by_item)
        and question.norms[:2] == ("if", names[0])
        and _contains(question.norms, "than", names[1])
        and all(name in question.norms for name in names),
        "contestant, score event, or comparison scope differs",
        FrontendStatus.AMBIGUOUS,
    )
    first, second, third = deltas
    first_index = question.tokens.index(first)
    second_index = question.tokens.index(second)
    third_index = question.tokens.index(third)
    first_item = _singular(
        question.norms[first_index + 2]
        if question.norms[first_index + 1] in {"more", "fewer", "less"}
        else question.norms[first_index + 1]
    )
    second_item = _singular(
        question.norms[second_index + 2]
        if question.norms[second_index + 1] in {"more", "fewer", "less"}
        else question.norms[second_index + 1]
    )
    third_item = _singular(
        question.norms[third_index + 2]
        if question.norms[third_index + 1] in {"more", "fewer", "less"}
        else question.norms[third_index + 1]
    )
    _require(
        _contains(
            question.norms,
            "if",
            names[0],
            "jumps",
            "on",
            first.norm,
            "more",
            question.norms[first_index + 2],
            "than",
            names[1],
        )
        and _contains(
            question.norms,
            "than",
            names[1],
            "and",
            "collects",
            second.norm,
            "more",
            question.norms[second_index + 2],
        )
        and _contains(
            question.norms,
            "but",
            "finishes",
            "the",
            "level",
            third.norm,
            question.norms[third_index + 1],
            "slower",
        )
        and _contains(rates.norms, first_item, "they", "jump", "on")
        and _contains(rates.norms, second_item, "they", "collect")
        and third_item == "second"
        and _contains(rates.norms, "timer", "when", "they", "finish", "the", "level"),
        "score actions do not preserve one contestant's signed deltas",
        FrontendStatus.AMBIGUOUS,
    )
    builder = _Builder(source, clause_set)
    signed_rows = []
    for item in rate_by_item:
        rate_token, rate_unit = rate_by_item[item]
        delta_token, sign, delta_unit = delta_by_item[item]
        signed_rows.append(
            _signed(
                sign,
                _product(
                    builder.literal(rate_token, rate_unit),
                    builder.literal(delta_token, delta_unit),
                ),
                f"{item}_score_delta",
            )
        )
    raw_delta = _sum(*signed_rows)
    expression = AbsoluteExpr(raw_delta, Span(0, len(source), source))
    return builder.finish(expression, "absolute_weighted_score_difference")


def _surface_cardinal(token: Token) -> Fraction | None:
    if token.number is not None:
        return token.number
    return _LOCAL_CARDINALS.get(token.norm)


def _bound_cardinal(builder: _Builder, token: Token, unit: Unit) -> LiteralExpr:
    value = _surface_cardinal(token)
    if value is None:
        raise _Reject(FrontendStatus.UNSUPPORTED, "numeric relation is not exact")
    return builder.lexical_literal(token, value, unit)


def _leading_subject(clause: Clause) -> tuple[str, int]:
    index = 1 if clause.norms[:1] in {("and",), ("but",)} else 0
    _require(
        index < len(clause.tokens) and clause.tokens[index].text[:1].isupper(),
        "relation subject is not one explicit name",
        FrontendStatus.AMBIGUOUS,
    )
    return clause.norms[index], index


@_ssa_family
def _grounded_value_pipeline(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    """Compile an explicitly grounded sequence of named intermediate values."""

    question = _question(clause_set)
    if not _contains(question.norms, "what", "was", "the", "final", "value"):
        return None
    _require_single_target_marker(question)
    relation_rows = [
        clause
        for clause in clause_set
        if _contains(clause.norms, "this", "starting", "value", "plus", "half")
        and _contains(clause.norms, "the", "resulting", "value", "was", "multiplied")
    ]
    if not relation_rows:
        return None
    _require(
        len(relation_rows) == 1,
        "grounded value pipeline is not unique",
        FrontendStatus.AMBIGUOUS,
    )
    relation = relation_rows[0]
    intro = _single_clause(
        clause_set,
        lambda clause: (
            not clause.question
            and _contains(clause.norms, "starting", "value", "of")
            and "gave" in clause.norms
            and len(_counts(clause)) == 1
        ),
        reason="starting-value declaration is not unique",
    )
    owner, owner_index = _leading_subject(intro)
    gave_index = intro.norms.index("gave")
    start_index = intro.norms.index("starting")
    _require(
        owner_index == 0
        and gave_index == 1
        and start_index >= 6
        and intro.norms[start_index - 1] in {"a", "the"}
        and intro.norms[start_index : start_index + 3] == ("starting", "value", "of"),
        "starting value has no closed owner/object scope",
        FrontendStatus.AMBIGUOUS,
    )
    articles = [
        index
        for index in range(gave_index + 1, start_index)
        if intro.norms[index] in {"a", "an", "the"}
    ]
    _require(len(articles) >= 2, "value-stream object is not locally introduced")
    object_words = intro.norms[articles[0] + 1 : articles[-1]]
    object_id = _item_id(object_words)
    final_index = question.norms.index("final")
    _require(
        final_index + 4 < len(question.tokens)
        and question.norms[final_index + 2 : final_index + 4] == ("of", "the")
        and _item_id(question.norms[final_index + 4 :]) == object_id,
        "final value targets another value stream",
        FrontendStatus.AMBIGUOUS,
    )
    _require(
        relation.norms[:10]
        == (
            "this",
            "starting",
            "value",
            "plus",
            "half",
            "the",
            "number",
            "was",
            "divided",
            "by",
        )
        and _contains(
            relation.norms,
            "and",
            "the",
            "resulting",
            "value",
            "was",
            "multiplied",
            "by",
            "the",
            "starting",
            "value",
            "minus",
        ),
        "value pipeline operators or antecedents are incomplete",
        FrontendStatus.AMBIGUOUS,
    )
    half = relation.tokens[4]
    by_offsets = [index for index, word in enumerate(relation.norms) if word == "by"]
    minus_index = relation.norms.index("minus")
    _require(
        len(by_offsets) == 2
        and by_offsets[0] + 1 < len(relation.tokens)
        and minus_index + 1 < len(relation.tokens),
        "value pipeline operands are incomplete",
    )
    divisor = relation.tokens[by_offsets[0] + 1]
    offset = relation.tokens[minus_index + 1]
    _require(
        _surface_cardinal(half) == Fraction(1, 2)
        and divisor.number is not None
        and divisor.number != 0
        and offset.number is not None,
        "value pipeline operands are invalid",
        FrontendStatus.INVALID,
    )
    initial = _one_token(_counts(intro), "starting value is missing")

    builder = _Builder(source, clause_set)
    ssa = TypedDiscourseSSA(source)
    stream = ssa.entity(object_id, "value_stream", intro.span)

    def symbol(state: str, span: Span):
        return ssa.symbol(
            stream,
            property="value",
            item="scalar",
            scope="pipeline",
            state=state,
            role="state_value",
            unit=SCALAR,
            span=span,
        )

    starting = symbol("starting", intro.span)
    augmented = symbol("augmented", relation.span)
    divided = symbol("divided", relation.span)
    remainder = symbol("starting_remainder", relation.span)
    final = symbol("final", question.span)
    ssa.define(
        starting,
        builder.literal(initial, SCALAR),
        intro.span,
        relation_id="starting_value",
    )
    ssa.define(
        augmented,
        _sum(
            _signed(
                1,
                ssa.ref(starting, relation.span, role="state_value", unit=SCALAR),
                "starting_value",
            ),
            _signed(
                1,
                _product(
                    builder.literal(half, SCALAR),
                    ssa.ref(starting, relation.span, role="state_value", unit=SCALAR),
                ),
                "half_starting_value",
            ),
        ),
        relation.span,
        relation_id="starting_plus_half",
    )
    ssa.define(
        divided,
        QuotientExpr(
            ssa.ref(augmented, relation.span, role="state_value", unit=SCALAR),
            builder.literal(divisor, SCALAR),
            relation.span,
        ),
        relation.span,
        relation_id="divide_result",
    )
    ssa.define(
        remainder,
        _sum(
            _signed(
                1,
                ssa.ref(starting, relation.span, role="state_value", unit=SCALAR),
                "starting_value",
            ),
            _signed(-1, builder.literal(offset, SCALAR), "minus_offset"),
        ),
        relation.span,
        relation_id="starting_minus_offset",
    )
    ssa.define(
        final,
        GroundProductExpr(
            (
                ssa.ref(divided, relation.span, role="state_value", unit=SCALAR),
                ssa.ref(remainder, relation.span, role="state_value", unit=SCALAR),
            ),
            relation.span,
        ),
        relation.span,
        relation_id="grounded_final_product",
    )
    expression = ssa.ref(final, question.span, role="state_value", unit=SCALAR)
    definitions = ssa.finalize(expression)
    return builder.finish(
        expression, "grounded_value_pipeline", definitions=definitions
    )


@_ssa_family
def _shared_duration_affine_rates(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    """Bind two explicit production rates to one shared duration."""

    question = _question(clause_set)
    if not (
        "worked" in question.norms
        and "together" in question.norms
        and _contains(question.norms, "total", "amount")
    ):
        return None
    _require_single_target_marker(question)
    first_rows = [
        clause
        for clause in clause_set
        if not clause.question
        and _contains(clause.norms, "can", "make")
        and "in" in clause.norms
        and "times" not in clause.norms
        and len(_counts(clause)) == 1
    ]
    second_rows = [
        clause
        for clause in clause_set
        if not clause.question
        and _contains(clause.norms, "times", "as", "many")
        and "hour" in clause.norms
    ]
    if not first_rows or not second_rows:
        return None
    _require(
        len(first_rows) == len(second_rows) == 1,
        "production-rate clauses are not unique",
        FrontendStatus.AMBIGUOUS,
    )
    first, second = first_rows[0], second_rows[0]
    first_owner, first_owner_index = _leading_subject(first)
    second_owner, second_owner_index = _leading_subject(second)
    _require(
        first_owner_index == second_owner_index == 0 and first_owner != second_owner,
        "production owners are not two distinct names",
        FrontendStatus.AMBIGUOUS,
    )
    amount = _one_token(_counts(first), "production amount is missing")
    amount_index = first.tokens.index(amount)
    in_index = first.norms.index("in", amount_index)
    _require(
        amount_index + 3 <= in_index
        and _singular(first.norms[amount_index + 1]) == "pound"
        and first.norms[amount_index + 2] == "of",
        "production amount lacks pound/item scope",
    )
    item = _item_id(first.norms[amount_index + 3 : in_index])
    _require(in_index + 2 < len(first.tokens), "production duration is missing")
    first_duration = first.tokens[in_index + 1]
    duration_value = _surface_cardinal(first_duration)
    duration_unit = _singular(first.norms[in_index + 2])
    _require(
        duration_value is not None
        and duration_value > 0
        and duration_unit == "hour",
        "production duration is not a positive hour count",
    )

    times_index = second.norms.index("times")
    _require(times_index > 0, "relative production scale is missing")
    ratio = second.tokens[times_index - 1]
    ratio_value = _surface_cardinal(ratio)
    as_offsets = [index for index, word in enumerate(second.norms) if word == "as"]
    _require(
        ratio_value is not None
        and ratio_value > 0
        and len(as_offsets) == 2
        and as_offsets[-1] + 1 < len(second.tokens)
        and second.norms[as_offsets[-1] + 1] == first_owner,
        "relative production source or scale is incomplete",
        FrontendStatus.AMBIGUOUS,
    )
    second_singular = tuple(_singular(word) for word in second.norms)
    _require(
        _contains(second_singular, "pound", "of", item)
        and _contains(second.norms, "in", "an", "hour")
        and _contains(
            second.norms,
            "as",
            first_owner,
            "makes",
            "in",
            "the",
            first_duration.norm,
            "hours",
        ),
        "relative production item or reference duration differs",
        FrontendStatus.AMBIGUOUS,
    )
    horizon = _one_token(_counts(question), "shared work duration is missing")
    horizon_index = question.tokens.index(horizon)
    _require(
        _contains(question.norms, "if", "they", "worked", "for")
        and horizon_index + 1 < len(question.tokens)
        and _singular(question.norms[horizon_index + 1]) == "hour"
        and _contains(question.norms, "they", "made", "together")
        and {"pound", item}.issubset({_singular(word) for word in question.norms}),
        "shared duration target does not cover both producers and the same item",
        FrontendStatus.AMBIGUOUS,
    )
    _require(
        question.norms
        == (
            "if",
            "they",
            "worked",
            "for",
            horizon.norm,
            question.norms[horizon_index + 1],
            "in",
            "a",
            "day",
            ",",
            "calculate",
            "the",
            "total",
            "amount",
            "of",
            item,
            "pounds",
            "they",
            "made",
            "together",
        ),
        "production question contains another requested target",
        FrontendStatus.AMBIGUOUS,
    )

    builder = _Builder(source, clause_set)
    ssa = TypedDiscourseSSA(source)
    producer_a = ssa.entity(first_owner, "producer", first.tokens[0].span)
    producer_b = ssa.entity(second_owner, "producer", second.tokens[0].span)
    group = ssa.group(
        "producer_group", (producer_a, producer_b), "producer", question.span
    )
    ssa.resolve(
        "they",
        role="producer",
        number=DiscourseNumber.PLURAL,
        members=(first_owner, second_owner),
    )

    amount_symbol = ssa.symbol(
        producer_a,
        property="amount",
        item=item,
        scope="reference_period",
        state="declared",
        role="production_amount",
        unit=MASS,
        span=first.span,
    )
    rate_a = ssa.symbol(
        producer_a,
        property="rate",
        item=item,
        scope="hour",
        state="current",
        role="production_rate",
        unit=MASS / HOUR,
        span=first.span,
    )
    rate_b = ssa.symbol(
        producer_b,
        property="rate",
        item=item,
        scope="hour",
        state="current",
        role="production_rate",
        unit=MASS / HOUR,
        span=second.span,
    )
    group_rate = ssa.symbol(
        group,
        property="rate",
        item=item,
        scope="hour",
        state="combined",
        role="production_rate",
        unit=MASS / HOUR,
        span=question.span,
    )
    ssa.define(
        amount_symbol,
        builder.literal(amount, MASS),
        first.span,
        relation_id="reference_amount",
    )
    ssa.define(
        rate_a,
        QuotientExpr(
            ssa.ref(
                amount_symbol,
                first.span,
                role="production_amount",
                unit=MASS,
            ),
            _bound_cardinal(builder, first_duration, HOUR),
            first.span,
        ),
        first.span,
        relation_id="reference_rate",
    )
    one_hour = next(
        token
        for index, token in enumerate(second.tokens)
        if token.norm == "hour" and index > 0 and second.norms[index - 1] == "an"
    )
    ssa.define(
        rate_b,
        QuotientExpr(
            _product(
                _bound_cardinal(builder, ratio, SCALAR),
                ssa.ref(
                    amount_symbol,
                    second.span,
                    role="production_amount",
                    unit=MASS,
                ),
            ),
            builder.lexical_literal(one_hour, Fraction(1), HOUR),
            second.span,
        ),
        second.span,
        relation_id="relative_rate",
    )
    ssa.define(
        group_rate,
        _sum(
            _signed(
                1,
                ssa.ref(rate_a, question.span, role="production_rate"),
                "first_rate",
            ),
            _signed(
                1,
                ssa.ref(rate_b, question.span, role="production_rate"),
                "second_rate",
            ),
        ),
        question.span,
        relation_id="combined_rate",
    )
    expression = _product(
        ssa.ref(group_rate, question.span, role="production_rate"),
        builder.literal(horizon, HOUR),
    )
    definitions = ssa.finalize(expression)
    return builder.finish(
        expression, "shared_duration_affine_rates", definitions=definitions
    )


@_ssa_family
def _closed_collection_share_completion(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    """Complete one inventory until the new category has an exact final share."""

    question = _question(clause_set)
    if not (
        "percentage" in question.norms
        and _contains(question.norms, "how", "many")
        and "all" in question.norms
    ):
        return None
    _require_single_target_marker(question)
    intro = _single_clause(
        clause_set,
        lambda clause: (
            not clause.question
            and "collects" in clause.norms
            and "animals" in clause.norms
            and not _counts(clause)
        ),
        reason="collection owner is not uniquely introduced",
    )
    inventory = _single_clause(
        clause_set,
        lambda clause: (
            not clause.question
            and "has" in clause.norms
            and clause.norms.count("stuffed") >= 4
            and len(_word_cardinals(clause)) >= 2
        ),
        reason="collection inventory is not unique",
    )
    owner, _ = _leading_subject(intro)
    _require(
        intro.norms[:4] == (owner, "collects", "stuffed", "animals"),
        "collection type is not explicitly scoped",
        FrontendStatus.AMBIGUOUS,
    )
    ssa = TypedDiscourseSSA(source)
    collector = ssa.entity(owner, "collector", intro.tokens[0].span)
    inventory_pronoun = inventory.norms[0]
    ssa.resolve(
        inventory_pronoun, role="collector", number=DiscourseNumber.SINGULAR
    )

    count_tokens = [
        token
        for token in inventory.tokens
        if _surface_cardinal(token) is not None
        and token.norm not in {"half", "once", "twice"}
    ]
    _require(
        len(count_tokens) >= 2,
        "collection categories have no exhaustive exact counts",
    )
    categories: list[str] = []
    for count in count_tokens:
        index = inventory.tokens.index(count)
        _require(
            index + 2 < len(inventory.tokens)
            and inventory.norms[index + 1] == "stuffed",
            "collection count lacks one local category",
            FrontendStatus.AMBIGUOUS,
        )
        categories.append(_singular(inventory.norms[index + 2]))
    _require(
        len(set(categories)) == len(categories),
        "collection category is declared more than once",
        FrontendStatus.AMBIGUOUS,
    )

    how_index = question.norms.index("many")
    _require(
        how_index + 2 < len(question.tokens)
        and question.norms[how_index + 1] == "stuffed",
        "share-completion target category is missing",
    )
    target = _singular(question.norms[how_index + 2])
    percentage_index = question.norms.index("percentage")
    _require(
        percentage_index + 4 < len(question.tokens)
        and question.norms[percentage_index + 1 : percentage_index + 3]
        == ("of", "stuffed")
        and _singular(question.norms[percentage_index + 3]) == target
        and question.norms[percentage_index + 4] == "is"
        and _contains(question.norms, "of", "all", "of", "her", "stuffed", "animals")
        and target not in categories,
        "share target, owner, or final collection scope differs",
        FrontendStatus.AMBIGUOUS,
    )
    _require(
        question.norms[how_index:]
        == (
            "many",
            "stuffed",
            question.norms[how_index + 2],
            "should",
            question.norms[-2],
            "buy",
        )
        and question.norms[-2] in {"he", "she", "they"},
        "share question contains another requested target",
        FrontendStatus.AMBIGUOUS,
    )
    ssa.resolve("her", role="collector", number=DiscourseNumber.SINGULAR)
    shares = [
        token
        for token in _digit_tokens(question)
        if token.span.end < len(source)
        and source[token.span.end : token.span.end + 1] == "%"
    ]
    share = _one_token(shares, "final category share is missing")
    _require(
        share.number is not None and 0 < share.number < 100,
        "final category share must lie strictly inside 0..100",
        FrontendStatus.INVALID,
    )

    builder = _Builder(source, clause_set)
    existing_symbol = ssa.symbol(
        collector,
        property="quantity",
        item="stuffed_animal",
        scope="collection",
        state="existing",
        role="collection_total",
        unit=COUNT,
        span=inventory.span,
    )
    target_symbol = ssa.symbol(
        collector,
        property="quantity",
        item=target,
        scope="collection",
        state="purchase",
        role="category_completion",
        unit=COUNT,
        span=question.span,
    )
    existing_expr = _sum(
        *[
            _signed(
                1,
                _bound_cardinal(builder, token, COUNT),
                f"existing_{category}",
            )
            for token, category in zip(count_tokens, categories, strict=True)
        ]
    )
    ssa.define(
        existing_symbol,
        existing_expr,
        inventory.span,
        relation_id="exhaustive_existing_inventory",
    )
    ssa.define(
        target_symbol,
        ClosedShareExpr(
            ssa.ref(
                existing_symbol,
                question.span,
                role="collection_total",
                unit=COUNT,
            ),
            builder.literal(share, PERCENT),
            question.span,
        ),
        question.span,
        relation_id="closed_final_share",
    )
    expression = ssa.ref(
        target_symbol,
        question.span,
        role="category_completion",
        unit=COUNT,
    )
    definitions = ssa.finalize(expression)
    return builder.finish(
        expression, "closed_collection_share_completion", definitions=definitions
    )


@_ssa_family
def _typed_scale_chain_conversion(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    """Compile an order-independent named scale chain with explicit units."""

    question = _question(clause_set)
    if not (
        _contains(question.norms, "how", "high", "can")
        and "times" in tuple(word for clause in clause_set for word in clause.norms)
    ):
        return None
    _require_single_target_marker(question)
    relation_clauses = [
        clause
        for clause in clause_set
        if not clause.question
        and "times" in clause.norms
        and "higher" in clause.norms
        and "than" in clause.norms
    ]
    if not relation_clauses:
        return None
    parsed: list[tuple[str, str, Token, tuple[str, ...], Clause]] = []
    for clause in relation_clauses:
        target, start = _leading_subject(clause)
        _require(
            start + 1 < len(clause.tokens) and clause.norms[start + 1] == "can",
            "scale-chain target has no local predicate",
        )
        times_index = clause.norms.index("times")
        than_index = clause.norms.index("than")
        _require(
            times_index > start + 2
            and clause.norms[times_index + 1 : times_index + 2] == ("higher",)
            and than_index == times_index + 2
            and than_index + 2 < len(clause.tokens)
            and clause.tokens[than_index + 1].text[:1].isupper()
            and clause.norms[than_index + 2] == "can",
            "scale-chain roles or direction are incomplete",
            FrontendStatus.AMBIGUOUS,
        )
        factor = clause.tokens[times_index - 1]
        _require(
            _surface_cardinal(factor) is not None
            and _surface_cardinal(factor) > 0,
            "scale-chain factor is not positive and exact",
            FrontendStatus.INVALID,
        )
        predicate = tuple(
            word
            for word in clause.norms[start + 2 : times_index - 1]
            if word not in {"a", "an", "the"}
        )
        _require(predicate, "scale-chain predicate is missing")
        parsed.append(
            (target, clause.norms[than_index + 1], factor, predicate, clause)
        )
    predicates = {row[3] for row in parsed}
    _require(
        len(predicates) == 1,
        "scale-chain clauses describe different properties",
        FrontendStatus.AMBIGUOUS,
    )
    predicate = next(iter(predicates))

    _require(question.norms[:1] == ("if",), "scale-chain base is not explicit")
    _require(
        len(question.tokens) > 8 and question.tokens[1].text[:1].isupper(),
        "scale-chain base owner is missing",
    )
    base_owner = question.norms[1]
    _require(question.norms[2] == "can", "scale-chain base predicate is missing")
    base_numbers = _counts(question)
    base = _one_token(base_numbers, "scale-chain base magnitude is missing")
    base_index = question.tokens.index(base)
    base_predicate = tuple(
        word
        for word in question.norms[3:base_index]
        if word not in {"a", "an", "the"}
    )
    _require(
        base_predicate == predicate and base_index + 1 < len(question.tokens),
        "scale-chain base property differs",
        FrontendStatus.AMBIGUOUS,
    )
    base_unit_word = _singular(question.norms[base_index + 1])
    _require(base_unit_word == "inch", "scale-chain base unit is unsupported")
    how_index = question.norms.index("how")
    _require(
        question.norms[how_index : how_index + 3] == ("how", "high", "can")
        and how_index + 3 < len(question.tokens)
        and question.tokens[how_index + 3].text[:1].isupper(),
        "scale-chain target owner is missing",
    )
    target_owner = question.norms[how_index + 3]
    in_offsets = [
        index
        for index in range(how_index + 4, len(question.tokens))
        if question.norms[index] == "in"
    ]
    _require(len(in_offsets) == 1, "scale-chain output unit is not unique")
    output_index = in_offsets[0]
    query_predicate = tuple(
        word
        for word in question.norms[how_index + 4 : output_index]
        if word not in {"a", "an", "the", ","}
    )
    _require(
        query_predicate == predicate and output_index + 1 < len(question.tokens),
        "scale-chain query property differs",
        FrontendStatus.AMBIGUOUS,
    )
    output_word = _singular(question.norms[output_index + 1])
    output_unit = (
        FOOT if output_word == "foot" else INCH if output_word == "inch" else None
    )
    _require(output_unit is not None, "scale-chain output unit is unsupported")
    _require(
        output_index + 2 == len(question.tokens),
        "scale-chain question contains another requested target",
        FrontendStatus.AMBIGUOUS,
    )

    builder = _Builder(source, clause_set)
    ssa = TypedDiscourseSSA(source)
    names = sorted(
        {base_owner, target_owner}.union(
            {
                name
                for target, source_name, _, _, _ in parsed
                for name in (target, source_name)
            }
        )
    )
    referents = {
        name: ssa.entity(
            name,
            "scaled_actor",
            next(
                token.span
                for clause in clause_set
                for token in clause.tokens
                if token.norm == name and token.text[:1].isupper()
            ),
        )
        for name in names
    }
    symbols = {
        name: ssa.symbol(
            referents[name],
            property="height",
            item="_".join(predicate),
            scope="capacity",
            state="current",
            role="scaled_measure",
            unit=INCH,
            span=referents[name].span,
        )
        for name in names
    }
    ssa.resolve(base_owner, role="scaled_actor", number=DiscourseNumber.SINGULAR)
    ssa.resolve(target_owner, role="scaled_actor", number=DiscourseNumber.SINGULAR)
    ssa.define(
        symbols[base_owner],
        builder.literal(base, INCH),
        question.span,
        relation_id="scale_chain_base",
    )
    for target, source_name, factor, _, clause in parsed:
        ssa.define(
            symbols[target],
            _product(
                _bound_cardinal(builder, factor, SCALAR),
                ssa.ref(
                    symbols[source_name],
                    clause.span,
                    role="scaled_measure",
                    unit=INCH,
                ),
            ),
            clause.span,
            relation_id=f"scale_{target}_from_{source_name}",
        )
    expression = UnitConversionExpr(
        ssa.ref(
            symbols[target_owner],
            question.span,
            role="scaled_measure",
            unit=INCH,
        ),
        output_unit,
        question.span,
    )
    definitions = ssa.finalize(expression)
    return builder.finish(
        expression, "typed_scale_chain_conversion", definitions=definitions
    )


@_ssa_family
def _temporal_reader_affine_difference(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    """Bind explicit day states and one cross-entity affine antecedent."""

    question = _question(clause_set)
    if not (
        _contains(question.norms, "how", "many", "more", "pages")
        and question.norms.count("read") >= 1
    ):
        return None
    _require_single_target_marker(question)
    intro = _single_clause(
        clause_set,
        lambda clause: (
            not clause.question
            and "reading" in clause.norms
            and _contains(clause.norms, "the", "same", "book")
            and not _counts(clause)
        ),
        reason="reader group is not uniquely introduced",
    )
    intro_names = [
        token.norm for token in intro.tokens if token.text[:1].isupper()
    ]
    _require(
        len(intro_names) == 2 and len(set(intro_names)) == 2,
        "reader group needs exactly two names",
        FrontendStatus.AMBIGUOUS,
    )
    yesterday = _single_clause(
        clause_set,
        lambda clause: (
            not clause.question
            and clause.norms[:1] == ("yesterday",)
            and "while" in clause.norms
            and clause.norms.count("read") == 2
            and len(_counts(clause)) == 2
        ),
        reason="yesterday reader assignments are not unique",
    )
    today = _single_clause(
        clause_set,
        lambda clause: (
            not clause.question
            and clause.norms[:1] == ("today",)
            and "while" in clause.norms
            and "yesterday" in clause.norms
            and "not" in clause.norms
            and len(_counts(clause)) == 1
        ),
        reason="today reader relations are not unique",
    )

    yesterday_owner_index = 2 if yesterday.norms[1:2] == (",",) else 1
    today_owner_index = 2 if today.norms[1:2] == (",",) else 1
    while_index = yesterday.norms.index("while")
    first_read = yesterday.norms.index("read")
    second_read = yesterday.norms.index("read", while_index)
    _require(
        first_read > yesterday_owner_index
        and second_read > while_index
        and yesterday.tokens[yesterday_owner_index].text[:1].isupper()
        and yesterday.tokens[while_index + 1].text[:1].isupper(),
        "yesterday assignments lack explicit owners",
    )
    yesterday_rows = {
        yesterday.norms[yesterday_owner_index]: next(
            token
            for token in _counts(yesterday)
            if first_read < yesterday.tokens.index(token) < while_index
        ),
        yesterday.norms[while_index + 1]: next(
            token
            for token in _counts(yesterday)
            if yesterday.tokens.index(token) > second_read
        ),
    }
    _require(
        set(yesterday_rows) == set(intro_names)
        and all(
            _singular(yesterday.norms[yesterday.tokens.index(token) + 1]) == "page"
            for token in yesterday_rows.values()
        ),
        "yesterday owners or page units differ",
        FrontendStatus.AMBIGUOUS,
    )

    today_while = today.norms.index("while")
    _require(
        len(today.tokens) > today_owner_index + 1
        and today.tokens[today_owner_index].text[:1].isupper()
        and today.norms[today_owner_index + 1] == "read"
        and _contains(today.norms, "more", "than", "as", "many", "pages", "as", "what")
        and today_while + 1 < len(today.tokens)
        and today.tokens[today_while + 1].text[:1].isupper(),
        "today affine relation is malformed",
        FrontendStatus.AMBIGUOUS,
    )
    target_owner = today.norms[today_owner_index]
    offset = _one_token(_counts(today), "today affine offset is missing")
    what_index = today.norms.index("what")
    _require(
        what_index + 3 < len(today.tokens)
        and today.tokens[what_index + 1].text[:1].isupper()
        and today.norms[what_index + 2 : what_index + 4] == ("read", "yesterday"),
        "today source antecedent is not explicit",
    )
    source_owner = today.norms[what_index + 1]
    zero_owner = today.norms[today_while + 1]
    _require(
        {target_owner, source_owner} == set(intro_names)
        and source_owner == zero_owner
        and _contains(
            today.norms[today_while:],
            "was",
            "not",
            "able",
            "to",
            "read",
            "any",
            "pages",
            "today",
        ),
        "today owner roles or zero-state evidence differ",
        FrontendStatus.AMBIGUOUS,
    )
    did_index = question.norms.index("did")
    than_offsets = [
        index for index, word in enumerate(question.norms) if word == "than"
    ]
    _require(
        did_index + 1 < len(question.tokens)
        and question.norms[did_index + 1] == target_owner
        and len(than_offsets) == 1
        and than_offsets[0] + 1 < len(question.tokens)
        and question.norms[than_offsets[0] + 1] == source_owner,
        "reader difference target or direction differs",
        FrontendStatus.AMBIGUOUS,
    )
    _require(
        question.norms
        == (
            "how",
            "many",
            "more",
            "pages",
            "did",
            target_owner,
            "read",
            "more",
            "than",
            source_owner,
        ),
        "reader question changes the cumulative time scope",
        FrontendStatus.AMBIGUOUS,
    )

    builder = _Builder(source, clause_set)
    ssa = TypedDiscourseSSA(source)
    readers = {
        name: ssa.entity(
            name,
            "reader",
            next(token.span for token in intro.tokens if token.norm == name),
        )
        for name in intro_names
    }
    ssa.group(
        "reader_group",
        tuple(readers[name] for name in intro_names),
        "reader",
        intro.span,
    )

    def symbol(owner: str, day: str, span: Span):
        return ssa.symbol(
            readers[owner],
            property="quantity",
            item="page",
            scope="reading",
            state=day,
            role="daily_pages",
            unit=COUNT,
            span=span,
        )

    yesterday_symbols = {
        name: symbol(name, "yesterday", yesterday.span) for name in intro_names
    }
    today_symbols = {name: symbol(name, "today", today.span) for name in intro_names}
    for name in intro_names:
        ssa.define(
            yesterday_symbols[name],
            builder.literal(yesterday_rows[name], COUNT),
            yesterday.span,
            relation_id=f"yesterday_{name}",
        )
    ssa.define(
        today_symbols[target_owner],
        _sum(
            _signed(
                1,
                ssa.ref(
                    yesterday_symbols[source_owner],
                    today.span,
                    role="daily_pages",
                    unit=COUNT,
                ),
                "source_yesterday",
            ),
            _signed(1, builder.literal(offset, COUNT), "more_pages"),
        ),
        today.span,
        relation_id="today_affine_reading",
    )
    not_token = next(token for token in today.tokens if token.norm == "not")
    ssa.define(
        today_symbols[source_owner],
        builder.lexical_literal(not_token, Fraction(0), COUNT),
        today.span,
        relation_id="today_explicit_zero",
    )
    expression = _sum(
        _signed(
            1,
            ssa.ref(
                yesterday_symbols[target_owner],
                question.span,
                role="daily_pages",
            ),
            "target_yesterday",
        ),
        _signed(
            1,
            ssa.ref(today_symbols[target_owner], question.span, role="daily_pages"),
            "target_today",
        ),
        _signed(
            -1,
            ssa.ref(
                yesterday_symbols[source_owner],
                question.span,
                role="daily_pages",
            ),
            "source_yesterday",
        ),
        _signed(
            -1,
            ssa.ref(today_symbols[source_owner], question.span, role="daily_pages"),
            "source_today",
        ),
    )
    definitions = ssa.finalize(expression)
    return builder.finish(
        expression, "temporal_reader_affine_difference", definitions=definitions
    )


@_ssa_family
def _closed_named_scale_group_total(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    """Recover an exhaustive named group from reversible scale relations."""

    question = _question(clause_set)
    if not (
        _contains(question.norms, "total", "number", "of", "balls")
        and "coach" in question.norms
        and "practice" in question.norms
    ):
        return None
    _require_single_target_marker(question)
    relation = _single_clause(
        clause_set,
        lambda clause: (
            not clause.question
            and clause.norms.count("twice") == 2
            and clause.norms.count("carried") == 2
            and clause.norms.count("balls") == 2
        ),
        reason="carrier scale relations are not unique",
    )
    twice_offsets = [
        index for index, word in enumerate(relation.norms) if word == "twice"
    ]
    and_index = relation.norms.index("and")
    _require(
        twice_offsets[0] == 2
        and relation.norms[1] == "carried"
        and relation.norms[twice_offsets[0] : twice_offsets[0] + 6]
        == ("twice", "as", "many", "balls", "as", relation.norms[7])
        and and_index + 2 < len(relation.tokens)
        and relation.norms[and_index + 2] == "carried"
        and relation.norms[twice_offsets[1] : twice_offsets[1] + 6]
        == ("twice", "as", "many", "balls", "as", relation.norms[-1]),
        "carrier scale directions are incomplete",
        FrontendStatus.AMBIGUOUS,
    )
    first_target = relation.norms[0]
    middle = relation.norms[7]
    second_target = relation.norms[and_index + 1]
    last = relation.norms[-1]
    _require(
        middle == second_target
        and len({first_target, middle, last}) == 3
        and all(
            token.text[:1].isupper()
            for token in (
                relation.tokens[0],
                relation.tokens[7],
                relation.tokens[and_index + 1],
                relation.tokens[-1],
            )
        ),
        "carrier relation has name drift or multiple antecedents",
        FrontendStatus.AMBIGUOUS,
    )
    group_clause = _single_clause(
        clause_set,
        lambda clause: (
            not clause.question
            and "boys" in clause.norms
            and _contains(clause.norms, "carried", "all", "of", "the", "balls")
        ),
        reason="carrier group is not declared exhaustive",
    )
    three = next(
        (token for token in group_clause.tokens if token.norm == "three"), None
    )
    _require(
        three is not None
        and _surface_cardinal(three) == len({first_target, middle, last}),
        "carrier group size differs from named members",
        FrontendStatus.AMBIGUOUS,
    )
    assignment = _one_token(_counts(question), "carrier base count is missing")
    assignment_index = question.tokens.index(assignment)
    _require(
        question.norms[:2] == ("if", middle)
        and assignment_index + 1 < len(question.tokens)
        and _singular(question.norms[assignment_index + 1]) == "ball",
        "carrier base count belongs to another member or item",
        FrontendStatus.AMBIGUOUS,
    )
    _require(
        question.norms[question.norms.index("what") :]
        == (
            "what",
            "is",
            "the",
            "total",
            "number",
            "of",
            "balls",
            "that",
            "the",
            "coach",
            "brought",
            "to",
            "practice",
        ),
        "carrier question contains another requested target",
        FrontendStatus.AMBIGUOUS,
    )
    introduction = _single_clause(
        clause_set,
        lambda clause: (
            not clause.question
            and "asked" in clause.norms
            and all(name in clause.norms for name in (first_target, middle, last))
            and _contains(clause.norms, "pick", "up", "the", "balls")
        ),
        reason="carrier names are not uniquely introduced",
    )

    builder = _Builder(source, clause_set)
    ssa = TypedDiscourseSSA(source)
    owners = {
        name: ssa.entity(
            name,
            "carrier",
            next(token.span for token in introduction.tokens if token.norm == name),
        )
        for name in (first_target, middle, last)
    }
    ssa.group(
        "carrier_group",
        tuple(owners[name] for name in (first_target, middle, last)),
        "carrier",
        group_clause.span,
    )
    symbols = {
        name: ssa.symbol(
            owners[name],
            property="quantity",
            item="ball",
            scope="practice",
            state="carried",
            role="carried_count",
            unit=COUNT,
            span=relation.span,
        )
        for name in owners
    }
    ssa.define(
        symbols[middle],
        builder.literal(assignment, COUNT),
        question.span,
        relation_id="middle_carrier_count",
    )
    first_scale = relation.tokens[twice_offsets[0]]
    second_scale = relation.tokens[twice_offsets[1]]
    ssa.define(
        symbols[first_target],
        _product(
            builder.literal(first_scale, SCALAR),
            ssa.ref(symbols[middle], relation.span, role="carried_count"),
        ),
        relation.span,
        relation_id="first_from_middle",
    )
    ssa.define(
        symbols[last],
        QuotientExpr(
            ssa.ref(symbols[middle], relation.span, role="carried_count"),
            builder.literal(second_scale, SCALAR),
            relation.span,
        ),
        relation.span,
        relation_id="last_from_middle",
    )
    expression = _sum(
        *[
            _signed(
                1,
                ssa.ref(symbols[name], question.span, role="carried_count"),
                "exhaustive_carrier",
            )
            for name in (first_target, middle, last)
        ]
    )
    definitions = ssa.finalize(expression)
    return builder.finish(
        expression, "closed_named_scale_group_total", definitions=definitions
    )


@_ssa_family
def _ordered_affine_category_ledger(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    """Compile a closed five-category ledger with one shared scale antecedent."""

    question = _question(clause_set)
    if not (
        question.norms[:2] == ("in", "all")
        and _contains(question.norms, "how", "many")
        and "visited" in question.norms
    ):
        return None
    _require_single_target_marker(question)
    intro = _single_clause(
        clause_set,
        lambda clause: (
            not clause.question
            and _contains(clause.norms, "monday", "through", "friday")
            and "hosted" in clause.norms
        ),
        reason="category horizon is not uniquely declared",
    )
    monday = _single_clause(
        clause_set,
        lambda clause: (
            not clause.question
            and clause.norms[:2] == ("on", "monday")
            and len(_counts(clause)) == 1
            and "visited" in clause.norms
        ),
        reason="Monday category assignment is not unique",
    )
    scaled = _single_clause(
        clause_set,
        lambda clause: (
            not clause.question
            and "tuesday" in clause.norms
            and "wednesday" in clause.norms
            and "twice" in clause.norms
            and "times" in clause.norms
        ),
        reason="scaled weekday categories are not unique",
    )
    trailing = _single_clause(
        clause_set,
        lambda clause: (
            not clause.question
            and "thursday" in clause.norms
            and "friday" in clause.norms
            and len(_counts(clause)) == 2
        ),
        reason="trailing weekday categories are not unique",
    )
    _require(
        clause_set.index(scaled) == clause_set.index(monday) + 1
        and scaled.norms
        == (
            "twice",
            "as",
            "many",
            "visited",
            "on",
            "tuesday",
            "and",
            "three",
            "times",
            "as",
            "many",
            "visited",
            "on",
            "wednesday",
        ),
        "omitted scale antecedent is not the unique prior category",
        FrontendStatus.AMBIGUOUS,
    )
    monday_value = _one_token(_counts(monday), "Monday count is missing")
    monday_index = monday.tokens.index(monday_value)
    _require(monday_index + 1 < len(monday.tokens), "weekday item is missing")
    item = _singular(monday.norms[monday_index + 1])
    item_surface = monday.norms[monday_index + 1]
    many_index = question.norms.index("many")
    trailing_values = _counts(trailing)
    _require(
        many_index + 1 < len(question.tokens)
        and _singular(question.norms[many_index + 1]) == item
        and trailing.norms[:2] == ("another", trailing_values[0].norm)
        and _singular(trailing.norms[2]) == item
        and _contains(trailing.norms, "visited", "on", "thursday")
        and _contains(
            trailing.norms,
            "and",
            trailing_values[1].norm,
            "visited",
            "on",
            "friday",
        )
        and all(day in intro.norms for day in ("monday", "friday")),
        "weekday ledger roles are incomplete",
        FrontendStatus.AMBIGUOUS,
    )
    hosted_index = intro.norms.index("hosted")
    venue_words = intro.norms[1:hosted_index] if intro.norms[:1] == ("the",) else ()
    _require(
        venue_words
        and question.norms
        == (
            "in",
            "all",
            ",",
            "how",
            "many",
            item_surface,
            "visited",
            "the",
            *venue_words,
            "last",
            "week",
        ),
        "weekday question contains another requested target",
        FrontendStatus.AMBIGUOUS,
    )
    twice = scaled.tokens[0]
    three = scaled.tokens[7]

    builder = _Builder(source, clause_set)
    ssa = TypedDiscourseSSA(source)
    days = ("monday", "tuesday", "wednesday", "thursday", "friday")
    referents = {
        day: ssa.entity(day, "weekday", intro.span) for day in days
    }
    symbols = {
        day: ssa.symbol(
            referents[day],
            property="quantity",
            item=item,
            scope="field_trip",
            state="visited",
            role="category_count",
            unit=COUNT,
            span=intro.span,
        )
        for day in days
    }
    ssa.define(
        symbols["monday"],
        builder.literal(monday_value, COUNT),
        monday.span,
        relation_id="monday_count",
    )
    for day, scale_token in (("tuesday", twice), ("wednesday", three)):
        ssa.define(
            symbols[day],
            _product(
                _bound_cardinal(builder, scale_token, SCALAR),
                ssa.ref(symbols["monday"], scaled.span, role="category_count"),
            ),
            scaled.span,
            relation_id=f"{day}_from_monday",
        )
    for day, token in zip(("thursday", "friday"), trailing_values, strict=True):
        ssa.define(
            symbols[day],
            builder.literal(token, COUNT),
            trailing.span,
            relation_id=f"{day}_count",
        )
    expression = _sum(
        *[
            _signed(
                1,
                ssa.ref(symbols[day], question.span, role="category_count"),
                f"{day}_ledger",
            )
            for day in days
        ]
    )
    definitions = ssa.finalize(expression)
    return builder.finish(
        expression, "ordered_affine_category_ledger", definitions=definitions
    )


@_ssa_family
def _typed_species_scale_total(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    """Scale one per-member measure and apply an explicit target multiplicity."""

    question = _question(clause_set)
    if not (
        "gallons" in question.norms
        and "blood" in question.norms
        and "sharks" in question.norms
    ):
        return None
    _require_single_target_marker(question)
    base = _single_clause(
        clause_set,
        lambda clause: (
            not clause.question
            and "each" in clause.norms
            and "whale" in clause.norms
            and "gallons" in clause.norms
            and len(_counts(clause)) == 1
        ),
        reason="species base measure is not unique",
    )
    scale_clause = _single_clause(
        clause_set,
        lambda clause: (
            not clause.question
            and _contains(clause.norms, "shark", "has")
            and _contains(clause.norms, "times", "as", "much", "blood", "as")
        ),
        reason="species scale relation is not unique",
    )
    learned_index = base.norms.index("learned")
    owner_token = base.tokens[learned_index - 1]
    owner_names = [owner_token.norm] if owner_token.text[:1].isupper() else []
    _require(
        len(owner_names) == 1 and scale_clause.norms[:1] == ("she",),
        "science owner or pronoun antecedent is not unique",
        FrontendStatus.AMBIGUOUS,
    )
    factor_rows = [
        token
        for token in scale_clause.tokens
        if _surface_cardinal(token) is not None
    ]
    factor = _one_token(factor_rows, "species scale is missing")
    members = [
        token for token in question.tokens if _surface_cardinal(token) is not None
    ]
    member_count = _one_token(members, "target species multiplicity is missing")
    member_index = question.tokens.index(member_count)
    _require(
        _surface_cardinal(factor) is not None
        and _surface_cardinal(factor) > 0
        and member_index + 1 < len(question.tokens)
        and _singular(question.norms[member_index + 1]) == "shark"
        and _contains(scale_clause.norms, "as", "a", "whale"),
        "species relation target, source, or multiplicity differs",
        FrontendStatus.AMBIGUOUS,
    )
    _require(
        question.norms
        == (
            "calculate",
            "the",
            "number",
            "of",
            "gallons",
            "of",
            "blood",
            "that",
            member_count.norm,
            question.norms[member_index + 1],
            "swimming",
            "in",
            "the",
            "sea",
            "have",
        ),
        "species question contains another requested target",
        FrontendStatus.AMBIGUOUS,
    )
    amount = _one_token(_counts(base), "species base amount is missing")
    amount_index = base.tokens.index(amount)
    _require(
        amount_index + 3 < len(base.tokens)
        and _singular(base.norms[amount_index + 1]) == "gallon"
        and base.norms[amount_index + 2 : amount_index + 4] == ("of", "blood"),
        "species base unit or property differs",
    )

    builder = _Builder(source, clause_set)
    ssa = TypedDiscourseSSA(source)
    scientist = ssa.entity(owner_names[0], "scientist", base.span)
    ssa.resolve("she", role="scientist", number=DiscourseNumber.SINGULAR)
    whale = ssa.entity("whale", "species", base.span)
    shark = ssa.entity("shark", "species", scale_clause.span)
    whale_amount = ssa.symbol(
        whale,
        property="volume",
        item="blood",
        scope="per_member",
        state="current",
        role="member_measure",
        unit=VOLUME,
        span=base.span,
    )
    shark_amount = ssa.symbol(
        shark,
        property="volume",
        item="blood",
        scope="per_member",
        state="current",
        role="member_measure",
        unit=VOLUME,
        span=scale_clause.span,
    )
    ssa.define(
        whale_amount,
        builder.literal(amount, VOLUME),
        base.span,
        relation_id="whale_blood",
    )
    ssa.define(
        shark_amount,
        _product(
            _bound_cardinal(builder, factor, SCALAR),
            ssa.ref(whale_amount, scale_clause.span, role="member_measure"),
        ),
        scale_clause.span,
        relation_id="shark_from_whale",
    )
    expression = _product(
        _bound_cardinal(builder, member_count, SCALAR),
        ssa.ref(shark_amount, question.span, role="member_measure"),
    )
    definitions = ssa.finalize(expression)
    return builder.finish(
        expression, "typed_species_scale_total", definitions=definitions
    )


@_ssa_family
def _typed_ratio_property_chain_total(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    """Bind a reversible three-property ratio chain to one explicit base."""

    question = _question(clause_set)
    if not (
        _contains(question.norms, "total", "number", "of", "items")
        and "saw" in question.norms
    ):
        return None
    _require_single_target_marker(question)
    relation = _single_clause(
        clause_set,
        lambda clause: (
            not clause.question
            and clause.norms.count("half") == 2
            and clause.norms.count("as") == 4
            and "saw" in clause.norms
        ),
        reason="property ratio chain is not unique",
    )
    owner, _ = _leading_subject(relation)
    half_offsets = [
        index for index, word in enumerate(relation.norms) if word == "half"
    ]
    parsed: list[tuple[str, str, Token]] = []
    for index in half_offsets:
        _require(
            relation.norms[index : index + 3] == ("half", "as", "many")
            and index + 5 < len(relation.tokens)
            and relation.norms[index + 4] == "as",
            "property ratio roles are incomplete",
            FrontendStatus.AMBIGUOUS,
        )
        parsed.append(
            (
                _singular(relation.norms[index + 3]),
                _singular(relation.norms[index + 5]),
                relation.tokens[index],
            )
        )
    first_target, middle, first_half = parsed[0]
    second_target, last, second_half = parsed[1]
    _require(
        middle == second_target and len({first_target, middle, last}) == 3,
        "property ratio chain has no unique middle antecedent",
        FrontendStatus.AMBIGUOUS,
    )
    base = _one_token(_counts(question), "property-chain base is missing")
    base_index = question.tokens.index(base)
    _require(
        base_index + 1 < len(question.tokens)
        and _singular(question.norms[base_index + 1]) == middle
        and owner in question.norms
        and _contains(
            question.norms,
            "calculate",
            "the",
            "total",
            "number",
            "of",
            "items",
        ),
        "property-chain base, owner, or target differs",
        FrontendStatus.AMBIGUOUS,
    )
    _require(
        question.norms[question.norms.index("calculate") :]
        == (
            "calculate",
            "the",
            "total",
            "number",
            "of",
            "items",
            owner,
            "saw",
        ),
        "property-chain question contains another requested target",
        FrontendStatus.AMBIGUOUS,
    )

    builder = _Builder(source, clause_set)
    ssa = TypedDiscourseSSA(source)
    observer = ssa.entity(owner, "observer", relation.tokens[0].span)
    symbols = {
        item: ssa.symbol(
            observer,
            property="quantity",
            item=item,
            scope="changing_room",
            state="observed",
            role="observed_count",
            unit=COUNT,
            span=relation.span,
        )
        for item in (first_target, middle, last)
    }
    ssa.define(
        symbols[middle],
        builder.literal(base, COUNT),
        question.span,
        relation_id="middle_property_base",
    )
    ssa.define(
        symbols[first_target],
        _product(
            builder.literal(first_half, SCALAR),
            ssa.ref(symbols[middle], relation.span, role="observed_count"),
        ),
        relation.span,
        relation_id="first_half_middle",
    )
    ssa.define(
        symbols[last],
        QuotientExpr(
            ssa.ref(symbols[middle], relation.span, role="observed_count"),
            builder.literal(second_half, SCALAR),
            relation.span,
        ),
        relation.span,
        relation_id="middle_half_last",
    )
    expression = _sum(
        *[
            _signed(
                1,
                ssa.ref(symbols[item], question.span, role="observed_count"),
                "observed_item",
            )
            for item in (first_target, middle, last)
        ]
    )
    definitions = ssa.finalize(expression)
    return builder.finish(
        expression, "typed_ratio_property_chain_total", definitions=definitions
    )


@_ssa_family
def _typed_scaled_measure_difference(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    """Compile one named base, one scaled peer, and an explicit difference."""

    question = _question(clause_set)
    if not (
        _contains(question.norms, "how", "much", "greater")
        and question.norms.count("enrollment") == 2
    ):
        return None
    _require_single_target_marker(question)
    base = _single_clause(
        clause_set,
        lambda clause: (
            not clause.question
            and "enrolls" in clause.norms
            and "times" not in clause.norms
            and "as" not in clause.norms
            and len(_counts(clause)) == 1
        ),
        reason="base enrollment is not unique",
    )
    scaled = _single_clause(
        clause_set,
        lambda clause: (
            not clause.question
            and "enrolls" in clause.norms
            and _contains(clause.norms, "as", "many", "students", "as")
            and len(_counts(clause)) == 1
        ),
        reason="scaled enrollment is not unique",
    )
    base_enrolls = base.norms.index("enrolls")
    scaled_enrolls = scaled.norms.index("enrolls")
    base_words = base.norms[:base_enrolls]
    first_comma = scaled.norms.index(",") if "," in scaled.norms else scaled_enrolls
    scaled_words = scaled.norms[:first_comma]
    as_offsets = [index for index, word in enumerate(scaled.norms) if word == "as"]
    _require(
        base_words
        and scaled_words
        and base_words != scaled_words
        and len(as_offsets) == 2
        and tuple(scaled.norms[as_offsets[-1] + 1 :]) == tuple(base_words)
        and all(
            token.text[:1].isupper()
            for token in base.tokens[:base_enrolls]
        )
        and all(
            token.text[:1].isupper()
            for token in scaled.tokens[:first_comma]
        ),
        "enrollment names or scale source drifted",
        FrontendStatus.AMBIGUOUS,
    )
    base_value = _one_token(_counts(base), "base enrollment amount is missing")
    ratio = _one_token(_counts(scaled), "scaled enrollment ratio is missing")
    _require(
        ratio.number is not None and 0 < ratio.number < 1,
        "scaled enrollment ratio must lie inside 0..1",
        FrontendStatus.INVALID,
    )
    _require(
        _contains(question.norms, "at", *base_words)
        and _contains(question.norms, "than", "the", "enrollment", "at", *scaled_words),
        "enrollment difference direction or owners differ",
        FrontendStatus.AMBIGUOUS,
    )
    _require(
        question.norms
        == (
            "how",
            "much",
            "greater",
            "is",
            "the",
            "average",
            "enrollment",
            "at",
            *base_words,
            "than",
            "the",
            "enrollment",
            "at",
            *scaled_words,
        ),
        "enrollment question contains another requested target",
        FrontendStatus.AMBIGUOUS,
    )

    builder = _Builder(source, clause_set)
    ssa = TypedDiscourseSSA(source)
    base_owner = ssa.entity(" ".join(base_words), "school", base.span)
    scaled_owner = ssa.entity(" ".join(scaled_words), "school", scaled.span)
    base_symbol = ssa.symbol(
        base_owner,
        property="quantity",
        item="student",
        scope="annual_enrollment",
        state="average",
        role="enrollment_count",
        unit=COUNT,
        span=base.span,
    )
    scaled_symbol = ssa.symbol(
        scaled_owner,
        property="quantity",
        item="student",
        scope="annual_enrollment",
        state="average",
        role="enrollment_count",
        unit=COUNT,
        span=scaled.span,
    )
    ssa.define(
        base_symbol,
        builder.literal(base_value, COUNT),
        base.span,
        relation_id="base_enrollment",
    )
    ssa.define(
        scaled_symbol,
        _product(
            builder.literal(ratio, SCALAR),
            ssa.ref(base_symbol, scaled.span, role="enrollment_count"),
        ),
        scaled.span,
        relation_id="scaled_enrollment",
    )
    expression = _sum(
        _signed(
            1,
            ssa.ref(base_symbol, question.span, role="enrollment_count"),
            "greater_base",
        ),
        _signed(
            -1,
            ssa.ref(scaled_symbol, question.span, role="enrollment_count"),
            "smaller_peer",
        ),
    )
    definitions = ssa.finalize(expression)
    return builder.finish(
        expression, "typed_scaled_measure_difference", definitions=definitions
    )


_PLANNERS = (
    _grounded_value_pipeline,
    _shared_duration_affine_rates,
    _closed_collection_share_completion,
    _typed_scale_chain_conversion,
    _temporal_reader_affine_difference,
    _closed_named_scale_group_total,
    _ordered_affine_category_ledger,
    _typed_species_scale_total,
    _typed_ratio_property_chain_total,
    _typed_scaled_measure_difference,
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
    _discounted_purchase_ledger,
    _batch_sale_profit,
    _bundle_relative_price_dag,
    _group_seat_purchase,
    _funding_balance_residual,
    _mean_participant_totals,
    _balanced_percent_category_difference,
    _affine_price_chain_total,
    _ordinal_ratio_partition,
    _temporal_affine_score_chain,
    _entity_affine_chain_total,
    _cross_entity_property_dag,
    _inverse_rate_time_difference,
    _chained_inventory_residual,
    _part_scaled_period_total,
    _fractional_remnant_total,
    _reverse_affine_state_duration,
    _exact_trip_capacity_minimum,
    _weighted_bundle_residual_count,
    _equal_allowance_purchase_balance,
    _exact_packaging_capacity,
    _exact_package_demand_cost,
    _typed_bowl_capacity_leftover,
    _fractional_group_consumption_remainder,
    _typed_chair_capacity_deficit,
    _typed_percentage_trade_transitions,
    _exhaustive_unit_rate_ledger,
    _recurring_pronoun_rate_ledger,
    _temporal_categorical_block_remainder,
    _absolute_weighted_score_difference,
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
