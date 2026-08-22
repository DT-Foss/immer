"""Evidence-closed structural frontend for a small arithmetic IR.

This module intentionally recognises relation families, not benchmark
sentences.  Every accepted numeric clause becomes a typed IR constraint and
retains its source span.  Numeric clauses outside the grammar make the parse
fail closed so a plausible-looking partial graph can never become an answer.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from fractions import Fraction

from .arithmetic_ir import (
    Affine,
    Assign,
    Balance,
    Mean,
    Part,
    Problem,
    Quantity,
    Rate,
    Span,
    Sum,
    Unit,
    Variable,
)


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
_NUMBER = r"(?:\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?|[\u00bc-\u00be\u2150-\u215e])"
_MONEY = rf"(?:\$\s*{_NUMBER}|{_NUMBER}\s*(?:dollars?|USD))"
_NAME = r"(?-i:[A-Z][A-Za-z'\-]*(?:\s+[A-Z][A-Za-z'\-]*)*)"
_NOUN = r"[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){0,3}"
_PRONOUN = re.compile(
    r"\b(?:he|she|they|them|him|her|his|their|it|its|this|that|these|those)\b",
    re.IGNORECASE,
)
_NUMERIC = re.compile(rf"(?<![\w])(?:{_MONEY}|{_NUMBER})(?![\w])", re.IGNORECASE)

_AGE_ASSIGN = re.compile(
    rf"^(?P<entity>{_NAME})\s+(?:is|was)\s+(?P<value>{_NUMBER})\s+"
    r"(?P<unit>years?|months?)\s+old$",
    re.IGNORECASE,
)
_COUNT_ASSIGN = re.compile(
    rf"^(?P<entity>{_NAME})\s+(?:has|had|owns?|keeps?)\s+"
    rf"(?P<value>{_NUMBER})\s+(?P<noun>{_NOUN})$",
    re.IGNORECASE,
)
_AGE_AFFINE = re.compile(
    rf"^(?P<target>{_NAME})\s+(?:is|was)\s+(?P<offset>{_NUMBER})\s+"
    rf"(?P<unit>years?|months?)\s+(?P<direction>older|younger)\s+than\s+"
    rf"(?P<source>{_NAME})$",
    re.IGNORECASE,
)
_AGE_QUERY_AFFINE = re.compile(
    rf"^how\s+old\s+(?:is|was)\s+(?P<target>{_NAME})\s+if\s+"
    rf"(?P<pronoun>he|she)\s+(?:is|was)\s+(?P<offset>{_NUMBER})\s+"
    rf"(?P<unit>years?|months?)\s+(?P<direction>older|younger)\s+than\s+"
    rf"(?P<source>{_NAME})$",
    re.IGNORECASE,
)
_WEIGHT_ASSIGN = re.compile(
    rf"^(?P<entity>{_NAME})\s*[\u2019']s\s+weight\s+(?:is|was)\s+"
    rf"(?P<value>{_NUMBER})\s+(?P<unit>kg|kilograms?)$",
    re.IGNORECASE,
)
_WEIGHT_AFFINE = re.compile(
    rf"^(?P<target>{_NAME})(?:\s*[\u2019']s\s+weight)?\s+(?:is|was)\s+"
    rf"(?P<offset>{_NUMBER})\s+(?P<unit>kg|kilograms?)\s+"
    rf"(?P<direction>more|less)\s+than\s+(?P<source>{_NAME})"
    r"(?:\s*[\u2019']s\s+weight)?$",
    re.IGNORECASE,
)
_COUNT_AFFINE = re.compile(
    rf"^(?P<target>{_NAME})\s+(?:has|had|owns?|keeps?)\s+"
    rf"(?P<offset>{_NUMBER})\s+(?P<direction>more|fewer)\s+"
    rf"(?P<noun>{_NOUN}?)\s+than\s+(?P<source>{_NAME})$",
    re.IGNORECASE,
)
_SCALE_AFFINE = re.compile(
    rf"^(?P<target>{_NAME})\s+(?:has|had|owns?|keeps?)\s+"
    rf"(?P<scale>{_NUMBER})\s+times\s+as\s+many\s+(?P<noun>{_NOUN})\s+"
    rf"as\s+(?P<source>{_NAME})$",
    re.IGNORECASE,
)
_PRICE = re.compile(
    rf"^(?P<article>a|an|one|each)\s+(?P<item>{_NOUN}?)\s+"
    rf"(?:costs?|is\s+priced\s+at|sells?\s+for)\s+(?P<price>{_MONEY})"
    r"(?:\s+each)?$",
    re.IGNORECASE,
)
_AGE_QUERY = re.compile(
    rf"^how\s+old\s+(?:is|was)\s+(?P<entity>{_NAME})$", re.IGNORECASE
)
_WEIGHT_QUERY = re.compile(
    rf"^what\s+is\s+the\s+weight\s+of\s+(?P<entity>{_NAME})"
    r"(?:\s*,\s*in\s+(?P<unit>kg|kilograms?))?$",
    re.IGNORECASE,
)
_COUNT_QUERY = re.compile(
    rf"^how\s+many\s+(?P<noun>{_NOUN}?)\s+(?:does|did)\s+"
    rf"(?P<entity>{_NAME})\s+(?:have|own|keep)$",
    re.IGNORECASE,
)
_LEDGER_QUERY = re.compile(
    r"^how\s+much\s+(?:do|does|did|will)\s+(?P<items>.+?)\s+cost$",
    re.IGNORECASE,
)
_LEDGER_TERM = re.compile(
    rf"(?P<quantity>{_NUMBER})\s+(?P<item>{_NOUN}?)"
    r"(?=\s*(?:,|\band\b|$))",
    re.IGNORECASE,
)
_COORDINATED_PRICE = re.compile(r"\s+and\s+(?=(?:a|an|one|each)\s+)", re.IGNORECASE)

_SIGNED_NUMBER = rf"[+-]?{_NUMBER}"
_WORD_FRACTIONS = {
    "half": Fraction(1, 2),
    "third": Fraction(1, 3),
    "quarter": Fraction(1, 4),
    "fourth": Fraction(1, 4),
    "fifth": Fraction(1, 5),
    "sixth": Fraction(1, 6),
    "seventh": Fraction(1, 7),
    "eighth": Fraction(1, 8),
    "ninth": Fraction(1, 9),
    "tenth": Fraction(1, 10),
}
_TIME_UNITS = {
    "second": Unit("second", (("time", 1),), Fraction(1)),
    "minute": Unit("minute", (("time", 1),), Fraction(60)),
    "hour": Unit("hour", (("time", 1),), Fraction(3600)),
    "day": Unit("day", (("time", 1),), Fraction(86400)),
    "week": Unit("week", (("time", 1),), Fraction(604800)),
    # Calendar units are admitted only when the rate and duration name the
    # same unit.  Their absolute scale is therefore conventional but exact.
    "month": Unit("month", (("time", 1),), Fraction(2629800)),
    "year": Unit("year", (("time", 1),), Fraction(31557600)),
}

_TIME_WORD = r"seconds?|minutes?|hours?|days?|weeks?|months?|years?"
_LENGTH_WORD = r"cm|centimeters?|inches?|meters?"
_SIMPLE_NOUN = r"[A-Za-z][A-Za-z'\-]*"
_EXPLICIT_NUMERIC_FRACTION = r"(?:\d[\d,]*/\d[\d,]*|[\u00bc-\u00be\u2150-\u215e])"
_FRACTION_TEXT = (
    rf"(?:{_NUMBER}\s*%|{_EXPLICIT_NUMERIC_FRACTION}|half|"
    r"one\s+third|one\s+quarter|"
    r"one\s+fourth|third|quarter|fourth)"
)
_NUMBER_LIST = (
    rf"{_SIGNED_NUMBER}(?:\s*,\s*{_SIGNED_NUMBER})*"
    rf"(?:\s*,?\s+and\s+{_SIGNED_NUMBER})?"
)

_DIRECT_RATE = re.compile(
    rf"^\s*(?P<worker>{_NAME})\s+"
    rf"(?P<rate_verb>earns|makes|is\s+paid|gets\s+paid)\s+"
    rf"(?P<rate>{_MONEY})\s+(?:per|an|each)\s+"
    rf"(?P<rate_unit>{_TIME_WORD})\.\s+"
    rf"(?P<duration_worker>{_NAME})\s+(?:works|worked)\s+(?:for\s+)?"
    rf"(?P<duration>{_NUMBER})\s+(?P<duration_unit>{_TIME_WORD})\.\s+"
    rf"How\s+much(?:\s+money)?\s+(?:does|did|will)\s+"
    rf"(?P<query_worker>{_NAME})\s+(?:earn|make|get\s+paid)\s*\?\s*$",
    re.IGNORECASE,
)

_PART_INVENTORY = re.compile(
    rf"^\s*(?P<owner>{_NAME})\s+has\s+(?P<whole>{_NUMBER})\s+"
    rf"(?P<noun>{_SIMPLE_NOUN})\.\s+(?P<fraction>{_FRACTION_TEXT})\s+"
    rf"of\s+(?P<basis>the|those|these)\s+(?P<basis_noun>{_SIMPLE_NOUN})\s+"
    rf"(?:is|are)\s+(?P<label>{_SIMPLE_NOUN})\.\s+How\s+many\s+"
    rf"(?P<query_label>{_SIMPLE_NOUN})\s+(?P<query_noun>{_SIMPLE_NOUN})\s+"
    rf"does\s+(?P<query_owner>{_NAME})\s+have\s*\?\s*$",
    re.IGNORECASE,
)

_ORIGINAL_LENGTH_PART = re.compile(
    rf"^\s*(?P<owner>{_NAME})\s+extends\s+(?:a|an|the)\s+"
    rf"(?P<object>{_SIMPLE_NOUN})\s+by\s+(?P<percent>{_NUMBER})\s*%\s+"
    rf"of\s+its\s+original\s+length\s+and\s+adds\s+"
    rf"(?P<addition>{_NUMBER})\s*(?P<addition_unit>{_LENGTH_WORD})\.\s+"
    rf"The\s+final\s+length\s+is\s+(?P<final>{_NUMBER})\s*"
    rf"(?P<final_unit>{_LENGTH_WORD})\.\s+What\s+was\s+the\s+original\s+"
    rf"length\s+of\s+the\s+(?P<query_object>{_SIMPLE_NOUN})\s+in\s+"
    rf"(?P<query_unit>{_LENGTH_WORD})\s*\?\s*$",
    re.IGNORECASE,
)

_INVENTORY_BALANCE = re.compile(
    rf"^\s*(?P<owner>{_NAME})\s+had\s+some\s+(?P<noun>{_SIMPLE_NOUN})\.\s+"
    rf"(?P<gain_owner>{_NAME})\s+(?:received|found|bought)\s+"
    rf"(?P<gain>{_NUMBER})\s+more\s+(?P<gain_noun>{_SIMPLE_NOUN})\s+and\s+"
    rf"(?:used|lost|gave\s+away)\s+(?P<loss>{_NUMBER})\s+"
    rf"(?P<loss_noun>{_SIMPLE_NOUN})\.\s+(?P<final_owner>{_NAME})\s+now\s+"
    rf"has\s+(?P<final>{_NUMBER})\s+(?P<final_noun>{_SIMPLE_NOUN})\.\s+"
    rf"How\s+many\s+(?P<query_noun>{_SIMPLE_NOUN})\s+did\s+"
    rf"(?P<query_owner>{_NAME})\s+have\s+(?:at\s+first|originally)\s*\?\s*$",
    re.IGNORECASE,
)

_BUS_BALANCE = re.compile(
    rf"^\s*Some\s+(?P<noun>people|passengers)\s+got\s+on\s+a\s+bus\s+at\s+"
    rf"the\s+terminal\.\s+At\s+the\s+first\s+bus\s+stop,\s+"
    rf"(?P<gain_one>{_NUMBER})\s+more\s+(?P<gain_one_noun>people|passengers)\s+"
    rf"got\s+in\.\s+Then\s+at\s+the\s+second\s+bus\s+stop,\s+"
    rf"(?P<loss>{_NUMBER})\s+(?P<loss_noun>people|passengers)\s+got\s+down\s+"
    rf"and\s+(?P<gain_two>{_NUMBER})\s+more\s+"
    rf"(?P<gain_two_noun>people|passengers)?\s*got\s+in\.\s+If\s+there\s+were\s+"
    rf"a\s+total\s+of\s+(?P<final>{_NUMBER})\s+"
    rf"(?P<final_noun>people|passengers)\s+heading\s+to\s+the\s+third\s+stop,\s+"
    rf"how\s+many\s+(?P<query_noun>people|passengers)\s+got\s+on\s+the\s+bus\s+"
    rf"at\s+the\s+terminal\s*\?\s*$",
    re.IGNORECASE,
)

_SCORE_MEAN = re.compile(
    rf"^\s*(?P<owner>{_NAME})\s+(?:received|got|recorded)\s+"
    rf"(?:the\s+following\s+)?(?:scores|measurements)"
    rf"(?:\s+on\s+(?:his|her|their)\s+[A-Za-z][A-Za-z'\- ]*?)?\s*:\s*"
    rf"(?P<values>{_NUMBER_LIST})\.\s+(?:Find|What\s+is)\s+"
    rf"(?:(?:his|her|their)\s+|{_NAME}[\u2019']s\s+)?"
    rf"(?P<kind>mean|average)(?:\s+score|\s+measurement)?\s*[.?]?\s*$",
    re.IGNORECASE,
)


class ParseStatus(str, Enum):
    """Closed outcome set for structural parsing."""

    PARSED = "parsed"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"
    INVALID = "invalid"


@dataclass(frozen=True)
class ParseResult:
    status: ParseStatus
    problem: Problem | None = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status is ParseStatus.PARSED and self.problem is not None


@dataclass(frozen=True)
class _Clause:
    text: str
    start: int
    end: int
    question: bool

    def span(self, source: str) -> Span:
        return Span(self.start, self.end, source)

    def group_span(self, match: re.Match[str], group: str, source: str) -> Span:
        return Span(
            self.start + match.start(group),
            self.start + match.end(group),
            source,
        )


class _Abort(Exception):
    def __init__(self, status: ParseStatus, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


def _sentences(source: str) -> tuple[_Clause, ...]:
    """Split prose without treating decimal points as sentence boundaries."""

    clauses: list[_Clause] = []
    start = 0
    for index, char in enumerate(source):
        boundary = char in "!?"
        if char == ".":
            previous_digit = index > 0 and source[index - 1].isdigit()
            next_digit = index + 1 < len(source) and source[index + 1].isdigit()
            boundary = not (previous_digit and next_digit)
        if not boundary:
            continue
        left = start
        while left < index and source[left].isspace():
            left += 1
        right = index
        while right > left and source[right - 1].isspace():
            right -= 1
        if right > left:
            clauses.append(_Clause(source[left:right], left, right, char == "?"))
        start = index + 1
    left = start
    while left < len(source) and source[left].isspace():
        left += 1
    right = len(source)
    while right > left and source[right - 1].isspace():
        right -= 1
    if right > left:
        clauses.append(_Clause(source[left:right], left, right, False))
    return tuple(clauses)


def _number(token: str) -> Fraction:
    raw = unicodedata.normalize("NFKC", token.strip()).replace(",", "")
    if token.strip() in _UNICODE_FRACTIONS:
        return _UNICODE_FRACTIONS[token.strip()]
    if "/" in raw:
        numerator, denominator = raw.split("/", 1)
        value = Fraction(int(numerator), int(denominator))
    else:
        try:
            value = Fraction(Decimal(raw))
        except InvalidOperation as exc:
            raise _Abort(ParseStatus.INVALID, f"invalid number: {token!r}") from exc
    if value < 0:
        raise _Abort(
            ParseStatus.INVALID, "negative literals are not admitted by this grammar"
        )
    return value


def _money(token: str) -> Fraction:
    cleaned = re.sub(r"(?i)\s*(?:dollars?|USD)\s*$", "", token.strip())
    cleaned = cleaned.removeprefix("$").strip()
    return _number(cleaned)


def _singular(noun: str) -> str:
    value = re.sub(r"\s+", " ", noun.strip().lower())
    if value.endswith("ies") and len(value) > 3:
        return value[:-3] + "y"
    if value.endswith("ses") and len(value) > 3:
        return value[:-2]
    if value.endswith("s") and not value.endswith(("ss", "us")):
        return value[:-1]
    return value


def _entity(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip()).casefold()


def _currency_unit(token: str) -> Unit:
    if "$" in token or re.search(r"(?i)\b(?:dollars?|USD)\b", token):
        return Unit.base("USD")
    raise _Abort(ParseStatus.INVALID, f"unsupported currency: {token!r}")


def _weight_unit(token: str) -> Unit:
    if token.casefold() in {"kg", "kilogram", "kilograms"}:
        return Unit.base("mass", symbol="kg")
    raise _Abort(ParseStatus.INVALID, f"unsupported weight unit: {token!r}")


def _time_unit(token: str) -> Unit:
    key = _singular(token)
    try:
        return _TIME_UNITS[key]
    except KeyError as exc:
        raise _Abort(ParseStatus.INVALID, f"unsupported time unit: {token!r}") from exc


def _length_unit(token: str) -> Unit:
    key = _singular(token)
    if key in {"cm", "centimeter"}:
        return Unit("cm", (("length", 1),), Fraction(1, 100))
    if key == "inch":
        return Unit("inch", (("length", 1),), Fraction(127, 5000))
    if key == "meter":
        return Unit("meter", (("length", 1),), Fraction(1))
    raise _Abort(ParseStatus.INVALID, f"unsupported length unit: {token!r}")


def _fraction(token: str) -> Fraction:
    value = token.strip().casefold()
    if value.endswith("%"):
        return _number(value[:-1].strip()) / 100
    if value.startswith("one "):
        value = value[4:]
    if value in _WORD_FRACTIONS:
        return _WORD_FRACTIONS[value]
    return _number(value)


def _signed_number(token: str) -> Fraction:
    value = token.strip()
    if value.startswith("-"):
        return -_number(value[1:])
    if value.startswith("+"):
        value = value[1:]
    return _number(value)


def _same(*values: str) -> bool:
    return len({_entity(value) for value in values}) == 1


class StructuralParser:
    """Compile supported relation families to a provenance-carrying problem."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.variables: dict[str, Variable] = {}
        self.constraints: list[object] = []
        self.targets: list[Variable] = []
        self.price_variables: dict[str, Variable] = {}
        self.definitions: set[str] = set()

    def _variable(
        self,
        name: str,
        unit: Unit,
        *,
        span: Span,
        count: bool = False,
    ) -> Variable:
        previous = self.variables.get(name)
        if previous is not None:
            if not previous.unit.compatible(unit) or previous.count != count:
                raise _Abort(ParseStatus.AMBIGUOUS, f"unit conflict for {name}")
            return previous
        variable = Variable(name, unit, count=count, span=span)
        self.variables[name] = variable
        return variable

    def _quantity(
        self, match: re.Match[str], group: str, clause: _Clause, unit: Unit
    ) -> Quantity:
        return Quantity(
            _number(match.group(group)),
            unit,
            span=clause.group_span(match, group, self.source),
        )

    def _match_span(self, match: re.Match[str], group: str) -> Span:
        return Span(match.start(group), match.end(group), self.source)

    def _property_name(self, entity: str, property_name: str) -> str:
        return f"{_entity(entity)}.{property_name}"

    def _define(self, target: Variable, constraint: object) -> None:
        if target.name in self.definitions:
            raise _Abort(
                ParseStatus.AMBIGUOUS,
                f"multiple definitions for {target.name} require an explicit scope",
            )
        self.definitions.add(target.name)
        self.constraints.append(constraint)

    def _parse_direct_rate(self) -> bool:
        match = _DIRECT_RATE.fullmatch(self.source)
        if match is None:
            return False
        if not _same(
            match.group("worker"),
            match.group("duration_worker"),
            match.group("query_worker"),
        ):
            raise _Abort(ParseStatus.AMBIGUOUS, "rate subjects do not agree")
        rate_time = _time_unit(match.group("rate_unit"))
        duration_time = _time_unit(match.group("duration_unit"))
        if _singular(match.group("rate_unit")) != _singular(
            match.group("duration_unit")
        ):
            raise _Abort(
                ParseStatus.AMBIGUOUS,
                "cross-period work schedule is not explicitly stated",
            )
        money = _currency_unit(match.group("rate"))
        target = self._variable(
            f"earnings.{_entity(match.group('worker'))}",
            money,
            span=self._match_span(match, "query_worker"),
        )
        self._define(
            target,
            Rate(
                target,
                Quantity(
                    _money(match.group("rate")),
                    money / rate_time,
                    span=self._match_span(match, "rate"),
                ),
                Quantity(
                    _number(match.group("duration")),
                    duration_time,
                    span=self._match_span(match, "duration"),
                ),
                span=Span(match.start(), match.end(), self.source),
            ),
        )
        self.targets.append(target)
        return True

    def _parse_part_inventory(self) -> bool:
        match = _PART_INVENTORY.fullmatch(self.source)
        if match is None:
            return False
        if not _same(match.group("owner"), match.group("query_owner")):
            raise _Abort(ParseStatus.AMBIGUOUS, "part owner and target do not agree")
        nouns = tuple(
            _singular(match.group(group))
            for group in ("noun", "basis_noun", "query_noun")
        )
        if len(set(nouns)) != 1:
            raise _Abort(ParseStatus.AMBIGUOUS, "part basis noun is not explicit")
        if not _same(match.group("label"), match.group("query_label")):
            raise _Abort(ParseStatus.AMBIGUOUS, "part label and target do not agree")
        whole_value = _number(match.group("whole"))
        fraction = _fraction(match.group("fraction"))
        if whole_value.denominator != 1:
            raise _Abort(ParseStatus.INVALID, "whole count must be an integer")
        if fraction > 1:
            raise _Abort(ParseStatus.INVALID, "part fraction exceeds the whole")
        result_value = whole_value * fraction
        if result_value.denominator != 1:
            raise _Abort(ParseStatus.INVALID, "part does not produce an exact count")

        unit = Unit.count(symbol=nouns[0])
        target = self._variable(
            f"{_entity(match.group('owner'))}.{_entity(match.group('label'))}.{nouns[0]}",
            unit,
            span=self._match_span(match, "query_label"),
            count=True,
        )
        self._define(
            target,
            Part(
                target,
                Quantity(
                    whole_value,
                    unit,
                    span=self._match_span(match, "whole"),
                ),
                fraction,
                span=Span(match.start(), match.end(), self.source),
            ),
        )
        self.targets.append(target)
        return True

    def _parse_original_length_part(self) -> bool:
        match = _ORIGINAL_LENGTH_PART.fullmatch(self.source)
        if match is None:
            return False

        object_groups = ["object", "query_object"]
        objects = tuple(_singular(match.group(group)) for group in object_groups)
        if len(set(objects)) != 1:
            raise _Abort(ParseStatus.AMBIGUOUS, "percent basis object is not explicit")

        addition_unit = _length_unit(match.group("addition_unit"))
        final_unit = _length_unit(match.group("final_unit"))
        query_unit = _length_unit(match.group("query_unit"))
        if not (
            addition_unit.compatible(final_unit) and final_unit.compatible(query_unit)
        ):
            raise _Abort(ParseStatus.INVALID, "length units conflict")
        fraction = _number(match.group("percent")) / 100
        addition_value = _number(match.group("addition"))
        final_value = _number(match.group("final"))
        canonical_addition = addition_value * addition_unit.scale
        canonical_final = final_value * final_unit.scale
        if canonical_final <= canonical_addition:
            raise _Abort(ParseStatus.INVALID, "final length does not exceed addition")

        prefix = f"length.{_entity(match.group('owner'))}.{objects[0]}"
        original = self._variable(
            f"{prefix}.original",
            query_unit,
            span=self._match_span(match, "query_object"),
        )
        extension = self._variable(
            f"{prefix}.extension",
            query_unit,
            span=self._match_span(match, "percent"),
        )
        whole_span = Span(match.start(), match.end(), self.source)
        self._define(
            extension,
            Part(extension, original, fraction, span=whole_span),
        )
        self.constraints.append(
            Sum(
                Quantity(
                    final_value,
                    final_unit,
                    span=self._match_span(match, "final"),
                ),
                (
                    original,
                    extension,
                    Quantity(
                        addition_value,
                        addition_unit,
                        span=self._match_span(match, "addition"),
                    ),
                ),
                span=whole_span,
            )
        )
        self.targets.append(original)
        return True

    def _parse_inventory_balance(self) -> bool:
        match = _INVENTORY_BALANCE.fullmatch(self.source)
        if match is None:
            return False
        if not _same(
            match.group("owner"),
            match.group("gain_owner"),
            match.group("final_owner"),
            match.group("query_owner"),
        ):
            raise _Abort(ParseStatus.AMBIGUOUS, "inventory owners do not agree")
        nouns = tuple(
            _singular(match.group(group))
            for group in (
                "noun",
                "gain_noun",
                "loss_noun",
                "final_noun",
                "query_noun",
            )
        )
        if len(set(nouns)) != 1:
            raise _Abort(ParseStatus.AMBIGUOUS, "inventory nouns do not agree")
        return self._build_count_balance(
            match,
            nouns[0],
            gains=("gain",),
            loss="loss",
            final="final",
            target_span="query_owner",
            target_name=f"inventory.{_entity(match.group('owner'))}.{nouns[0]}.initial",
        )

    def _parse_bus_balance(self) -> bool:
        match = _BUS_BALANCE.fullmatch(self.source)
        if match is None:
            return False
        base = _singular(match.group("noun"))
        noun_groups = (
            "gain_one_noun",
            "loss_noun",
            "final_noun",
            "query_noun",
        )
        if any(_singular(match.group(group)) != base for group in noun_groups):
            raise _Abort(ParseStatus.AMBIGUOUS, "bus inventory nouns do not agree")
        gain_two_noun = match.group("gain_two_noun")
        if gain_two_noun is not None and _singular(gain_two_noun) != base:
            raise _Abort(ParseStatus.AMBIGUOUS, "bus inventory nouns do not agree")
        return self._build_count_balance(
            match,
            base,
            gains=("gain_one", "gain_two"),
            loss="loss",
            final="final",
            target_span="query_noun",
            target_name=f"bus.{base}.terminal",
        )

    def _build_count_balance(
        self,
        match: re.Match[str],
        noun: str,
        *,
        gains: tuple[str, ...],
        loss: str,
        final: str,
        target_span: str,
        target_name: str,
    ) -> bool:
        gain_values = tuple(_number(match.group(group)) for group in gains)
        loss_value = _number(match.group(loss))
        final_value = _number(match.group(final))
        all_values = (*gain_values, loss_value, final_value)
        if any(value.denominator != 1 for value in all_values):
            raise _Abort(ParseStatus.INVALID, "inventory counts must be integers")
        if final_value + loss_value < sum(gain_values, Fraction(0)):
            raise _Abort(ParseStatus.INVALID, "inventory implies a negative start")

        unit = Unit.count(symbol=noun)
        target = self._variable(
            target_name,
            unit,
            span=self._match_span(match, target_span),
            count=True,
        )
        left: list[Variable | Quantity] = [target]
        left.extend(
            Quantity(
                value,
                unit,
                span=self._match_span(match, group),
            )
            for group, value in zip(gains, gain_values)
        )
        right = (
            Quantity(
                final_value,
                unit,
                span=self._match_span(match, final),
            ),
            Quantity(
                loss_value,
                unit,
                span=self._match_span(match, loss),
            ),
        )
        self.constraints.append(
            Balance(
                tuple(left), right, span=Span(match.start(), match.end(), self.source)
            )
        )
        self.targets.append(target)
        return True

    def _parse_score_mean(self) -> bool:
        match = _SCORE_MEAN.fullmatch(self.source)
        if match is None:
            return False
        values_text = match.group("values")
        values_start = match.start("values")
        tokens = tuple(re.finditer(_SIGNED_NUMBER, values_text))
        if len(tokens) < 2:
            raise _Abort(ParseStatus.INVALID, "mean requires at least two values")
        unit = Unit.scalar()
        values = tuple(
            Quantity(
                _signed_number(token.group()),
                unit,
                span=Span(
                    values_start + token.start(),
                    values_start + token.end(),
                    self.source,
                ),
            )
            for token in tokens
        )
        target = self._variable(
            f"mean.{_entity(match.group('owner'))}.score",
            unit,
            span=self._match_span(match, "kind"),
        )
        self._define(
            target,
            Mean(
                target,
                values,
                span=Span(match.start(), match.end(), self.source),
            ),
        )
        self.targets.append(target)
        return True

    def _parse_closed_family(self) -> bool:
        parsers = (
            self._parse_direct_rate,
            self._parse_part_inventory,
            self._parse_original_length_part,
            self._parse_inventory_balance,
            self._parse_bus_balance,
            self._parse_score_mean,
        )
        return any(parser() for parser in parsers)

    def _parse_declaration(self, clause: _Clause) -> bool:
        text = clause.text.strip()
        if _PRONOUN.search(text):
            raise _Abort(ParseStatus.AMBIGUOUS, "numeric pronoun binding is not proven")

        # Coordination is a grammar operation: accept it only when every
        # conjunct independently satisfies the same price predicate.
        separators = tuple(_COORDINATED_PRICE.finditer(text))
        if separators:
            starts = [0, *(separator.end() for separator in separators)]
            ends = [*(separator.start() for separator in separators), len(text)]
            parts = [text[start:end].strip() for start, end in zip(starts, ends)]
            if all(_PRICE.fullmatch(part) for part in parts):
                for start, end in zip(starts, ends):
                    while start < end and text[start].isspace():
                        start += 1
                    while end > start and text[end - 1].isspace():
                        end -= 1
                    child = _Clause(
                        text[start:end], clause.start + start, clause.start + end, False
                    )
                    if not self._parse_declaration(child):
                        raise _Abort(
                            ParseStatus.INVALID,
                            "coordinated price predicate failed after validation",
                        )
                return True
            return False

        match = _AGE_ASSIGN.fullmatch(text)
        if match:
            unit = Unit.base(_singular(match.group("unit")))
            variable = self._variable(
                self._property_name(match.group("entity"), "age"),
                unit,
                span=clause.group_span(match, "entity", self.source),
            )
            self._define(
                variable,
                Assign(
                    variable,
                    self._quantity(match, "value", clause, unit),
                    span=clause.span(self.source),
                ),
            )
            return True

        match = _AGE_AFFINE.fullmatch(text)
        if match:
            unit = Unit.base(_singular(match.group("unit")))
            target = self._variable(
                self._property_name(match.group("target"), "age"),
                unit,
                span=clause.group_span(match, "target", self.source),
            )
            source = self._variable(
                self._property_name(match.group("source"), "age"),
                unit,
                span=clause.group_span(match, "source", self.source),
            )
            offset = self._quantity(match, "offset", clause, unit)
            if match.group("direction").casefold() == "younger":
                offset = Quantity(-offset.value, unit, span=offset.span)
            self._define(
                target, Affine(target, source, 1, offset, span=clause.span(self.source))
            )
            return True

        match = _WEIGHT_ASSIGN.fullmatch(text)
        if match:
            unit = _weight_unit(match.group("unit"))
            variable = self._variable(
                self._property_name(match.group("entity"), "weight"),
                unit,
                span=clause.group_span(match, "entity", self.source),
            )
            self._define(
                variable,
                Assign(
                    variable,
                    self._quantity(match, "value", clause, unit),
                    span=clause.span(self.source),
                ),
            )
            return True

        match = _WEIGHT_AFFINE.fullmatch(text)
        if match:
            unit = _weight_unit(match.group("unit"))
            target = self._variable(
                self._property_name(match.group("target"), "weight"),
                unit,
                span=clause.group_span(match, "target", self.source),
            )
            source = self._variable(
                self._property_name(match.group("source"), "weight"),
                unit,
                span=clause.group_span(match, "source", self.source),
            )
            offset = self._quantity(match, "offset", clause, unit)
            if match.group("direction").casefold() == "less":
                offset = Quantity(-offset.value, unit, span=offset.span)
            self._define(
                target, Affine(target, source, 1, offset, span=clause.span(self.source))
            )
            return True

        match = _COUNT_ASSIGN.fullmatch(text)
        if match and not re.search(
            r"(?i)\b(?:more|fewer|times|less|greater)\b|\bthan\b",
            match.group("noun"),
        ):
            # A relation-shaped remainder must never be reinterpreted as an
            # exotic object name merely because its affine base is malformed.
            noun = _singular(match.group("noun"))
            unit = Unit.count(symbol=noun)
            variable = self._variable(
                self._property_name(match.group("entity"), noun),
                unit,
                span=clause.group_span(match, "entity", self.source),
                count=True,
            )
            self._define(
                variable,
                Assign(
                    variable,
                    self._quantity(match, "value", clause, unit),
                    span=clause.span(self.source),
                ),
            )
            return True

        match = _COUNT_AFFINE.fullmatch(text)
        if match:
            noun = _singular(match.group("noun"))
            unit = Unit.count(symbol=noun)
            target = self._variable(
                self._property_name(match.group("target"), noun),
                unit,
                span=clause.group_span(match, "target", self.source),
                count=True,
            )
            source = self._variable(
                self._property_name(match.group("source"), noun),
                unit,
                span=clause.group_span(match, "source", self.source),
                count=True,
            )
            offset = self._quantity(match, "offset", clause, unit)
            if match.group("direction").casefold() == "fewer":
                offset = Quantity(-offset.value, unit, span=offset.span)
            self._define(
                target, Affine(target, source, 1, offset, span=clause.span(self.source))
            )
            return True

        match = _SCALE_AFFINE.fullmatch(text)
        if match:
            noun = _singular(match.group("noun"))
            unit = Unit.count(symbol=noun)
            target = self._variable(
                self._property_name(match.group("target"), noun),
                unit,
                span=clause.group_span(match, "target", self.source),
                count=True,
            )
            source = self._variable(
                self._property_name(match.group("source"), noun),
                unit,
                span=clause.group_span(match, "source", self.source),
                count=True,
            )
            scale = _number(match.group("scale"))
            self._define(
                target,
                Affine(
                    target,
                    source,
                    scale,
                    Quantity(0, unit),
                    span=clause.span(self.source),
                ),
            )
            return True

        match = _PRICE.fullmatch(text)
        if match:
            item = _singular(match.group("item"))
            item_unit = Unit.count(symbol=item)
            money_unit = _currency_unit(match.group("price"))
            rate_unit = money_unit / item_unit
            price = self._variable(
                f"unit_price.{item}",
                rate_unit,
                span=clause.group_span(match, "item", self.source),
            )
            self.price_variables[item] = price
            self._define(
                price,
                Assign(
                    price,
                    Quantity(
                        _money(match.group("price")),
                        rate_unit,
                        span=clause.group_span(match, "price", self.source),
                    ),
                    span=clause.span(self.source),
                ),
            )
            return True
        return False

    def _parse_question(self, clause: _Clause) -> bool:
        text = clause.text.strip()

        match = _AGE_QUERY_AFFINE.fullmatch(text)
        if match:
            unit = Unit.base(_singular(match.group("unit")))
            target = self._variable(
                self._property_name(match.group("target"), "age"),
                unit,
                span=clause.group_span(match, "target", self.source),
            )
            source = self._variable(
                self._property_name(match.group("source"), "age"),
                unit,
                span=clause.group_span(match, "source", self.source),
            )
            offset = self._quantity(match, "offset", clause, unit)
            if match.group("direction").casefold() == "younger":
                offset = Quantity(-offset.value, unit, span=offset.span)
            self._define(
                target, Affine(target, source, 1, offset, span=clause.span(self.source))
            )
            self.targets.append(target)
            return True

        if _PRONOUN.search(text):
            raise _Abort(
                ParseStatus.AMBIGUOUS, "question pronoun binding is not proven"
            )

        match = _AGE_QUERY.fullmatch(text)
        if match:
            unit = Unit.base("year")
            self.targets.append(
                self._variable(
                    self._property_name(match.group("entity"), "age"),
                    unit,
                    span=clause.group_span(match, "entity", self.source),
                )
            )
            return True

        match = _COUNT_QUERY.fullmatch(text)
        if match:
            noun = _singular(match.group("noun"))
            self.targets.append(
                self._variable(
                    self._property_name(match.group("entity"), noun),
                    Unit.count(symbol=noun),
                    span=clause.group_span(match, "entity", self.source),
                    count=True,
                )
            )
            return True

        match = _WEIGHT_QUERY.fullmatch(text)
        if match:
            unit = _weight_unit(match.group("unit") or "kg")
            self.targets.append(
                self._variable(
                    self._property_name(match.group("entity"), "weight"),
                    unit,
                    span=clause.group_span(match, "entity", self.source),
                )
            )
            return True

        match = _LEDGER_QUERY.fullmatch(text)
        if not match:
            return False
        items_start = clause.start + match.start("items")
        items = match.group("items")
        cursor = 0
        terms: list[tuple[str, Fraction, Span]] = []
        for term in _LEDGER_TERM.finditer(items):
            gap = items[cursor : term.start()]
            if gap.strip(" ,") and gap.strip().casefold() != "and":
                raise _Abort(ParseStatus.AMBIGUOUS, "ledger item list is ambiguous")
            item = _singular(term.group("item"))
            number_span = Span(
                items_start + term.start("quantity"),
                items_start + term.end("quantity"),
                self.source,
            )
            terms.append((item, _number(term.group("quantity")), number_span))
            cursor = term.end()
        tail = items[cursor:]
        if not terms or tail.strip(" ,"):
            raise _Abort(ParseStatus.AMBIGUOUS, "ledger item list is ambiguous")
        if len({item for item, _, _ in terms}) != len(terms):
            raise _Abort(ParseStatus.AMBIGUOUS, "duplicate ledger item lacks scope")

        money_unit = Unit.base("USD")
        total = self._variable(
            f"ledger_total.{clause.start}",
            money_unit,
            span=clause.span(self.source),
        )
        contributions: list[Variable] = []
        for item, value, number_span in terms:
            price = self.price_variables.get(item)
            if price is None:
                raise _Abort(ParseStatus.UNSUPPORTED, f"missing unit price for {item}")
            item_unit = Unit.count(symbol=item)
            contribution = self._variable(
                f"ledger_contribution.{clause.start}.{item}",
                money_unit,
                span=number_span,
            )
            self._define(
                contribution,
                Rate(
                    contribution,
                    price,
                    Quantity(value, item_unit, span=number_span),
                    span=clause.span(self.source),
                ),
            )
            contributions.append(contribution)
        self._define(
            total, Sum(total, tuple(contributions), span=clause.span(self.source))
        )
        self.targets.append(total)
        return True

    def parse(self) -> ParseResult:
        if not isinstance(self.source, str) or not self.source.strip():
            return ParseResult(
                ParseStatus.INVALID, reason="source must be non-empty text"
            )
        try:
            if self._parse_closed_family():
                return self._finalize()
            clauses = _sentences(self.source)
            if not clauses:
                raise _Abort(ParseStatus.INVALID, "source has no clauses")

            # Prices must exist before a ledger question, independent of prose order.
            declarations = [clause for clause in clauses if not clause.question]
            questions = [clause for clause in clauses if clause.question]
            for clause in declarations:
                parsed = self._parse_declaration(clause)
                if not parsed and _NUMERIC.search(clause.text):
                    raise _Abort(
                        ParseStatus.UNSUPPORTED,
                        f"unparsed numeric clause at {clause.start}:{clause.end}",
                    )
            for clause in questions:
                parsed = self._parse_question(clause)
                if not parsed:
                    raise _Abort(
                        ParseStatus.UNSUPPORTED,
                        f"unsupported target at {clause.start}:{clause.end}",
                    )

            return self._finalize()
        except _Abort as exc:
            return ParseResult(exc.status, reason=exc.reason)
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            return ParseResult(ParseStatus.INVALID, reason=str(exc))

    def _finalize(self) -> ParseResult:
        if len(self.targets) != 1:
            status = ParseStatus.AMBIGUOUS if self.targets else ParseStatus.UNSUPPORTED
            raise _Abort(status, "exactly one explicit target is required")
        target = self.targets[0]
        mentioned = {
            variable
            for constraint in self.constraints
            for variable in _constraint_variables(constraint)
        }
        if target not in mentioned:
            raise _Abort(ParseStatus.UNSUPPORTED, "target has no structural evidence")
        problem = Problem(
            tuple(self.variables.values()), tuple(self.constraints), target
        )
        return ParseResult(ParseStatus.PARSED, problem)


def _constraint_variables(constraint: object) -> tuple[Variable, ...]:
    values: list[Variable] = []
    for name in (
        "target",
        "source",
        "amount",
        "rate",
        "duration",
        "total",
        "terms",
        "part",
        "whole",
        "left",
        "right",
        "mean",
        "values",
    ):
        item = getattr(constraint, name, None)
        if isinstance(item, Variable):
            values.append(item)
        elif isinstance(item, tuple):
            values.extend(value for value in item if isinstance(value, Variable))
    return tuple(values)


def parse_structural_problem(source: str) -> ParseResult:
    """Parse *source* without ever returning a partial numeric graph."""

    return StructuralParser(source).parse()


__all__ = [
    "ParseResult",
    "ParseStatus",
    "StructuralParser",
    "parse_structural_problem",
]
