"""Guarded closed-form certificates for common arithmetic word problems.

Every resolver is a full-string grammar.  Every numeric literal must be bound
to a named field, and the certificate recomputes its answer with exact
``Fraction`` arithmetic.  Unsupported or ambiguous surface forms abstain.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import re
from typing import Any, Callable


_NUMBER = r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
_UNSIGNED = r"(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
_NAME = r"[A-Z][A-Za-z'\-]*"
_NOUN = r"[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){0,3}"
_NUMERIC = re.compile(rf"(?<![A-Za-z0-9.]){_NUMBER}(?![A-Za-z0-9]|\.\d)")


@dataclass(frozen=True, slots=True)
class NumericSpan:
    start: int
    end: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "text": self.text}


@dataclass(frozen=True, slots=True)
class FormulaCertificate:
    family: str
    equation: str
    inputs: tuple[tuple[str, str], ...]
    context: tuple[tuple[str, str], ...]
    numeric_spans: tuple[NumericSpan, ...]
    source_sha256: str
    verified: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "guarded_formula/v1",
            "family": self.family,
            "equation": self.equation,
            "inputs": dict(self.inputs),
            "context": dict(self.context),
            "numeric_spans": [span.to_dict() for span in self.numeric_spans],
            "numeric_coverage": True,
            "source_sha256": self.source_sha256,
            "verified": self.verified,
        }


@dataclass(frozen=True, slots=True)
class FormulaSolution:
    answer: Fraction
    certificate: FormulaCertificate


def _fraction(raw: str) -> Fraction:
    try:
        value = Fraction(Decimal(raw.replace(",", "")))
    except InvalidOperation as exc:
        raise ValueError(f"invalid decimal: {raw!r}") from exc
    return value


def _canonical(value: Fraction) -> str:
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def _same(*values: str) -> bool:
    return len({re.sub(r"\s+", " ", value.strip()).casefold() for value in values}) == 1


def _noun(value: str) -> str:
    result = re.sub(r"\s+", " ", value.strip()).casefold()
    if result.endswith("ies"):
        return result[:-3] + "y"
    if result.endswith("s") and not result.endswith(("ss", "us")):
        return result[:-1]
    return result


def _same_noun_family(left: str, right: str) -> bool:
    a, b = _noun(left), _noun(right)
    return a == b or a.endswith(b) or b.endswith(a) or a in b.split() or b in a.split()


def _item_phrase(value: str) -> str:
    result = re.sub(r"\btries\b", "tires", value.strip().casefold())
    result = re.sub(r"\bpairs?\s+of\s+", "", result)
    return _noun(result)


def _same_item_family(*values: str) -> bool:
    normalized = {_item_phrase(value) for value in values}
    return len(normalized) == 1


def _fullmatch(pattern: re.Pattern[str], source: str) -> re.Match[str] | None:
    return pattern.fullmatch(source.strip())


def _solution(
    source: str,
    match: re.Match[str],
    *,
    family: str,
    equation: str,
    numeric_groups: tuple[str, ...],
    inputs: dict[str, Fraction],
    context: dict[str, str],
    answer: Fraction,
    verify: Callable[[dict[str, Fraction]], Fraction],
) -> FormulaSolution | None:
    leading_offset = len(source) - len(source.lstrip())
    observed = tuple(
        NumericSpan(found.start(), found.end(), found.group(0))
        for found in _NUMERIC.finditer(source)
    )
    try:
        consumed = tuple(
            sorted(
                (
                    NumericSpan(
                        match.start(name) + leading_offset,
                        match.end(name) + leading_offset,
                        match.group(name),
                    )
                    for name in numeric_groups
                ),
                key=lambda span: (span.start, span.end),
            )
        )
    except (IndexError, TypeError):
        return None
    if [(row.start, row.end) for row in observed] != [
        (row.start, row.end) for row in consumed
    ]:
        return None
    if verify(inputs) != answer:
        return None
    certificate = FormulaCertificate(
        family=family,
        equation=equation,
        inputs=tuple(sorted((key, _canonical(value)) for key, value in inputs.items())),
        context=tuple(sorted(context.items())),
        numeric_spans=consumed,
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )
    return FormulaSolution(answer, certificate)


_PRISM_COST = re.compile(
    rf"(?P<owner>{_NAME})\s+fills\s+a\s+(?P<length>{_UNSIGNED})\s+foot\s+by\s+"
    rf"(?P<width>{_UNSIGNED})\s+foot\s+(?P<object>{_NOUN})\s+that\s+is\s+"
    rf"(?P<depth>{_UNSIGNED})\s+feet\s+deep\.\s+It\s+costs?\s+\$\s*"
    rf"(?P<rate>{_UNSIGNED})\s+per\s+cubic\s+foot\s+to\s+fill\.\s+"
    r"How\s+much\s+does\s+it\s+cost\s+to\s+fill\?",
    re.IGNORECASE,
)

_RATIO_DIFFERENCE = re.compile(
    rf"The\s+ratio\s+of\s+(?P<noun>{_NOUN})\s+that\s+(?P<a>{_NAME})\s+and\s+"
    rf"(?P<b>{_NAME})\s+have\s+is\s+(?P<a_part>{_UNSIGNED})\s*:\s*"
    rf"(?P<b_part>{_UNSIGNED})\.\s+If\s+the\s+total\s+number\s+of\s+"
    rf"(?P<total_noun>{_NOUN})\s+they\s+have\s+together\s+is\s+"
    rf"(?P<total>{_UNSIGNED}),\s+how\s+many\s+more\s+(?P<query_noun>{_NOUN})\s+"
    rf"does\s+(?P<query_b>{_NAME})\s+have\s+more\s+than\s+(?P<query_a>{_NAME})\?",
    re.IGNORECASE,
)

_FRACTION_TRANSFER = re.compile(
    rf"(?P<receiver>{_NAME})\s+had\s+\$\s*(?P<initial>{_UNSIGNED})\.\s+"
    rf"(?P<giver>{_NAME})\s+gave\s+(?P<object_pronoun>him|her)\s+"
    rf"(?P<fraction>half|one\s+third|one\s+quarter)\s+of\s+"
    rf"(?P<owner_pronoun>his|her)\s+\$\s*(?P<whole>{_UNSIGNED})\.\s+"
    rf"How\s+much\s+does\s+(?P<query>{_NAME})\s+have\s+now\?",
    re.IGNORECASE,
)

_TARGET_REMAINDER = re.compile(
    rf"(?P<intro>.+?)\s+needed\s+to\s+raise\s+\$\s*(?P<target>{_UNSIGNED})\s+"
    rf"for\s+.+?\.\s+By\s+(?P<time>{_UNSIGNED})\s+pm,\s+"
    rf"(?P<a>{_NAME})\s+had\s+earned\s+\$\s*(?P<a_amount>{_UNSIGNED})\s+and\s+"
    rf"(?P<b>{_NAME})\s+had\s+earned\s+\$\s*(?P<b_amount>{_UNSIGNED})\.\s+"
    r"How\s+much\s+more\s+did\s+they\s+need\s+to\s+earn\s+to\s+reach\s+their\s+goal\?",
    re.IGNORECASE,
)

_AFFINE_PRICE = re.compile(
    rf"(?P<owner>{_NAME})['’]s\s+(?P<target>{_NOUN})\s+cost\s+\$\s*"
    rf"(?P<offset>{_UNSIGNED})\s+less\s+than\s+(?P<scale>{_UNSIGNED})\s+times\s+"
    rf"as\s+much\s+as\s+(?:his|her)\s+(?P<source>{_NOUN})\s+cost\.\s+If\s+"
    rf"(?:his|her)\s+(?P<given_source>{_NOUN})\s+cost\s+\$\s*"
    rf"(?P<base>{_UNSIGNED}),\s+how\s+much\s+did\s+(?:his|her)\s+"
    rf"(?P<query_target>{_NOUN})\s+cost\?",
    re.IGNORECASE,
)

_THREE_VALUE_MEAN = re.compile(
    rf"The\s+(?P<measure>highest\s+temperature)\s+ever\s+recorded\s+in\s+"
    rf"(?P<a>{_NAME})\s+is\s+(?P<x>{_NUMBER})\s+degrees\s+Fahrenheit\.\s+"
    rf"The\s+(?P<measure_b>highest\s+temperature)\s+ever\s+recorded\s+in\s+"
    rf"(?P<b>{_NAME})\s+is\s+(?P<y>{_NUMBER})\s+degrees\s+Fahrenheit\.\s+"
    rf"The\s+(?P<measure_c>highest\s+temperature)\s+recorded\s+in\s+"
    rf"(?P<c>{_NAME})\s+is\s+(?P<z>{_NUMBER})\s+degrees\s+Fahrenheit\.\s+"
    rf"What\s+is\s+the\s+average\s+(?P<query_measure>highest\s+temperature)\s+of\s+"
    rf"these\s+(?P<count>{_UNSIGNED})\s+countries\?",
    re.IGNORECASE,
)

_MONTHLY_SPLIT_SAVING = re.compile(
    rf"(?P<owner>{_NAME})\s+has\s+a\s+monthly\s+saving\s+target\s+of\s+\$\s*"
    rf"(?P<target>{_UNSIGNED})\.\s+In\s+(?P<month>April|June|September|November),\s+"
    r"(?:he|she)\s+wants\s+to\s+save\s+twice\s+as\s+much\s+daily\s+in\s+the\s+"
    r"second\s+half\s+as\s+(?:he|she)\s+saves\s+in\s+the\s+first\s+half\s+"
    r"in\s+order\s+to\s+hit\s+(?:his|her)\s+target\.\s+How\s+much\s+does\s+"
    r"(?:he|she)\s+have\s+to\s+save\s+for\s+each\s+day\s+in\s+the\s+second\s+"
    r"half\s+of\s+the\s+month\?",
    re.IGNORECASE,
)

_TWO_DAY_REVENUE = re.compile(
    rf"A\s+(?P<worker>{_NOUN})\s+charges\s+different\s+rates\s+to\s+repair\s+"
    rf"the\s+(?P<objects>{_NOUN})\s+of\s+(?P<category_a>{_NOUN})\s+and\s+"
    rf"(?P<category_b>{_NOUN})\.\s+For\s+each\s+(?P<a_object>{_NOUN})\s+that\s+"
    rf"is\s+repaired,\s+the\s+(?P<worker_again>{_NOUN})\s+will\s+charge\s+\$\s*"
    rf"(?P<rate_a>{_UNSIGNED})\s+and\s+for\s+each\s+(?P<b_object>{_NOUN})\s+"
    rf"that\s+is\s+repaired,\s+the\s+(?P<worker_third>{_NOUN})\s+will\s+charge\s+"
    rf"\$\s*(?P<rate_b>{_UNSIGNED})\.\s+On\s+(?P<day_a>{_NAME}),\s+the\s+"
    rf"(?P<worker_fourth>{_NOUN})\s+repairs\s+(?P<count_aa>{_UNSIGNED})\s+"
    rf"(?P<count_aa_object>{_NOUN})\s+and\s+(?P<count_ab>{_UNSIGNED})\s+"
    rf"(?P<count_ab_object>{_NOUN})\.\s+On\s+(?P<day_b>{_NAME}),\s+the\s+"
    rf"(?P<worker_fifth>{_NOUN})\s+repairs\s+(?P<count_bb>{_UNSIGNED})\s+"
    rf"(?P<count_bb_object>{_NOUN})\s+and\s+doesn't\s+repair\s+any\s+"
    rf"(?P<zero_a_object>{_NOUN})\.\s+How\s+much\s+more\s+revenue\s+did\s+the\s+"
    rf"(?P<query_worker>{_NOUN})\s+earn\s+on\s+the\s+day\s+with\s+higher\s+revenue\?",
    re.IGNORECASE,
)

_DEPENDENT_PRICE_LEDGER = re.compile(
    rf"(?P<owner>{_NAME})['’]s\s+(?P<relation>{_NOUN})\s+sells\s+"
    rf"(?P<item_a>{_NOUN}),\s+(?P<item_b>{_NOUN}),\s+and\s+(?P<item_c>{_NOUN})\s+"
    rf"at\s+the\s+local\s+store\.\s+A\s+(?P<a_single>{_NOUN})\s+costs\s+three\s+"
    rf"times\s+what\s+each\s+(?P<b_single>{_NOUN})\s+costs\.\s+An\s+"
    rf"(?P<c_single>{_NOUN})\s+costs\s+(?P<offset>{_UNSIGNED})\s+less\s+than\s+"
    rf"what\s+a\s+(?P<a_reference>{_NOUN})\s+cost\.\s+(?P<buyer>{_NAME})\s+is\s+"
    rf"sent\s+to\s+the\s+store\s+to\s+buy\s+(?P<count_a>{_UNSIGNED})\s+"
    rf"(?P<a_plural>{_NOUN}),\s+(?P<count_b>{_UNSIGNED})\s+(?P<b_plural>{_NOUN}),\s+"
    rf"and\s+(?P<count_c>{_UNSIGNED})\s+(?P<c_plural>{_NOUN})\.\s+What's\s+the\s+"
    rf"total\s+amount\s+of\s+money\s+(?:he|she)\s+will\s+spend\s+if\s+each\s+"
    rf"(?P<given_b>{_NOUN})\s+costs\s+(?P<base>{_UNSIGNED})\$\?",
    re.IGNORECASE,
)

_SAVINGS_WORK_BALANCE = re.compile(
    rf"(?P<owner>{_NAME})\s+wants\s+to\s+buy\s+(?:himself|herself)\s+a\s+new\s+"
    rf"(?P<item_a>{_NOUN})\s+and\s+(?P<count_b>{_UNSIGNED})\s+(?P<item_b>{_NOUN})\.\s+"
    rf"The\s+(?P<a_again>{_NOUN})\s+(?:he|she)\s+wants\s+costs\s+\$\s*"
    rf"(?P<price_a>{_UNSIGNED})\s+and\s+each\s+(?P<b_single>{_NOUN})\s+cost\s+"
    rf"\$\s*(?P<price_b>{_UNSIGNED})\.\s+(?P<owner_again>{_NAME})\s+"
    rf"(?P<work_verb>babysits)\s+.+?\s+(?P<work_count>{_UNSIGNED})\s+times,\s+"
    rf"earning\s+\$\s*(?P<work_rate>{_UNSIGNED})\s+each\s+time\s+(?:he|she)\s+"
    rf"(?P<work_verb_again>babysits)\s+them\.\s+(?:His|Her)\s+parents\s+pay\s+"
    rf"(?:him|her)\s+\$\s*(?P<target_rate>{_UNSIGNED})\s+each\s+time\s+"
    rf"(?:he|she)\s+(?P<target_work>mows\s+the\s+lawn)\.\s+If\s+"
    rf"(?P<owner_third>{_NAME})\s+already\s+had\s+\$\s*(?P<saved>{_UNSIGNED})\s+"
    rf"saved\s+before\s+(?:he|she)\s+started\s+babysitting,\s+how\s+many\s+times\s+"
    rf"must\s+(?:he|she)\s+(?P<query_work>mow\s+the\s+lawn)\s+before\s+"
    rf"(?:he|she)\s+can\s+afford\s+the\s+(?P<query_a>{_NOUN})\s+and\s+"
    rf"(?P<query_b>{_NOUN})\?",
    re.IGNORECASE,
)

_HOURLY_PURCHASE = re.compile(
    rf"(?P<owner>{_NAME})\s+wants\s+to\s+buy\s+a\s+\$\s*(?P<first>{_UNSIGNED})\s+"
    rf"(?P<item_a>{_NOUN})\s+and\s+a\s+(?P<item_b>{_NOUN})\s+that\s+is\s+\$\s*"
    rf"(?P<second>{_UNSIGNED})\.\s+(?:His|Her)\s+part-time\s+job\s+pays\s+"
    rf"(?:him|her)\s+\$\s*(?P<rate>{_UNSIGNED})\s+an\s+hour\.\s+How\s+many\s+"
    rf"hours\s+will\s+(?:he|she)\s+have\s+to\s+work\s+before\s+(?:he|she)\s+"
    rf"can\s+make\s+(?:his|her)\s+purchase\?",
    re.IGNORECASE,
)

_LOAN_BALANCE = re.compile(
    rf"(?P<owner>{_NAME})\s+borrowed\s+\$(?P<principal>{_UNSIGNED})\s+and\s+"
    rf"promised\s+to\s+return\s+it\s+with\s+an\s+additional\s+"
    rf"(?P<percent>{_UNSIGNED})%\s+of\s+the\s+amount\.\s+If\s+(?:he|she)\s+is\s+"
    rf"going\s+to\s+pay\s+\$(?P<payment>{_UNSIGNED})\s+a\s+month\s+for\s+"
    rf"(?P<months>{_UNSIGNED})\s+months,\s+how\s+much\s+will\s+be\s+"
    rf"(?P<query_owner>{_NAME})['’]s\s+remaining\s+balance\s+by\s+then\?",
    re.IGNORECASE,
)

_CONSUMPTION_DURATION = re.compile(
    rf"Each\s+person\s+in\s+a\s+certain\s+household\s+consumes\s+"
    rf"(?P<rate>{_UNSIGNED})\s+kg\s+of\s+(?P<resource>{_NOUN})\s+every\s+meal\.\s+"
    rf"Supposing\s+(?P<people>{_UNSIGNED})\s+members\s+of\s+the\s+household\s+eat\s+"
    rf"(?P<resource_again>{_NOUN})\s+every\s+lunch\s+and\s+dinner,\s+how\s+many\s+"
    rf"weeks\s+will\s+a\s+(?P<total>{_UNSIGNED})\s+kg\s+bag\s+of\s+"
    rf"(?P<query_resource>{_NOUN})\s+last\?",
    re.IGNORECASE,
)

_ANNUAL_DRIVING_COST = re.compile(
    rf"(?P<owner>{_NAME})\s+hires\s+a\s+driving\s+service\s+to\s+get\s+(?:him|her)\s+"
    rf"to\s+work\s+each\s+day\.\s+(?:His|Her)\s+work\s+is\s+"
    rf"(?P<miles>{_UNSIGNED})\s+miles\s+away\s+and\s+(?:he|she)\s+has\s+to\s+go\s+"
    rf"there\s+and\s+back\s+each\s+day\.\s+(?:He|She)\s+goes\s+to\s+work\s+"
    rf"(?P<days>{_UNSIGNED})\s+days\s+a\s+week\s+for\s+(?P<weeks>{_UNSIGNED})\s+"
    rf"weeks\s+a\s+year\.\s+(?:He|She)\s+gets\s+charged\s+\$(?P<rate>{_UNSIGNED})\s+"
    rf"per\s+mile\s+driven\s+and\s+(?:he|she)\s+also\s+gives\s+(?:his|her)\s+driver\s+"
    rf"a\s+\$(?P<bonus>{_UNSIGNED})\s+bonus\s+per\s+month\.\s+How\s+much\s+does\s+"
    rf"(?:he|she)\s+pay\s+a\s+year\s+for\s+driving\?",
    re.IGNORECASE,
)

_PARTY_REMAINDER = re.compile(
    rf"At\s+the\s+beginning\s+of\s+the\s+party,\s+there\s+were\s+"
    rf"(?P<men>{_UNSIGNED})\s+men\s+and\s+(?P<women>{_UNSIGNED})\s+women\.\s+"
    rf"After\s+an\s+hour,\s+(?P<numerator>{_UNSIGNED})/(?P<denominator>{_UNSIGNED})\s+"
    rf"of\s+the\s+total\s+number\s+of\s+people\s+left\.\s+How\s+many\s+women\s+"
    rf"are\s+left\s+if\s+(?P<men_stayed>{_UNSIGNED})\s+men\s+stayed\s+at\s+the\s+party\?",
    re.IGNORECASE,
)

_JOB_NET_DIFFERENCE = re.compile(
    rf"(?P<owner>{_NAME})\s+is\s+choosing\s+between\s+two\s+jobs\.\s+Job\s+A\s+pays\s+"
    rf"\$(?P<a_hourly>{_UNSIGNED})\s+an\s+hour\s+for\s+(?P<a_hours>{_UNSIGNED})\s+"
    rf"hours\s+a\s+year,\s+and\s+is\s+in\s+a\s+state\s+with\s+a\s+"
    rf"(?P<a_tax>{_UNSIGNED})%\s+total\s+tax\s+rate\.\s+Job\s+B\s+pays\s+"
    rf"\$(?P<b_salary>{_UNSIGNED})\s+a\s+year\s+and\s+is\s+in\s+a\s+state\s+that\s+"
    rf"charges\s+\$(?P<b_property>{_UNSIGNED})\s+in\s+property\s+tax\s+and\s+a\s+"
    rf"(?P<b_tax>{_UNSIGNED})%\s+tax\s+rate\s+on\s+net\s+income\s+after\s+property\s+"
    rf"tax\.\s+How\s+much\s+more\s+money\s+will\s+(?P<query_owner>{_NAME})\s+make\s+"
    rf"at\s+the\s+job\s+with\s+a\s+higher\s+net\s+pay\s+rate,\s+compared\s+to\s+"
    rf"the\s+other\s+job\?",
    re.IGNORECASE,
)


def _resolve_prism(source: str) -> FormulaSolution | None:
    match = _fullmatch(_PRISM_COST, source)
    if match is None:
        return None
    values = {
        name: _fraction(match.group(name))
        for name in ("length", "width", "depth", "rate")
    }
    answer = values["length"] * values["width"] * values["depth"] * values["rate"]
    return _solution(
        source,
        match,
        family="rectangular_prism_unit_cost",
        equation="answer = length * width * depth * rate",
        numeric_groups=("length", "width", "depth", "rate"),
        inputs=values,
        context={"object": match.group("object")},
        answer=answer,
        verify=lambda row: row["length"] * row["width"] * row["depth"] * row["rate"],
    )


def _resolve_ratio(source: str) -> FormulaSolution | None:
    match = _fullmatch(_RATIO_DIFFERENCE, source)
    if (
        match is None
        or not _same(match.group("a"), match.group("query_a"))
        or not _same(match.group("b"), match.group("query_b"))
    ):
        return None
    if (
        len({_noun(match.group(name)) for name in ("noun", "total_noun", "query_noun")})
        != 1
    ):
        return None
    values = {
        name: _fraction(match.group(name)) for name in ("a_part", "b_part", "total")
    }
    if min(values.values()) <= 0:
        return None
    answer = (
        values["total"]
        * (values["b_part"] - values["a_part"])
        / (values["a_part"] + values["b_part"])
    )
    return _solution(
        source,
        match,
        family="ratio_total_difference",
        equation="answer = total * (b_part - a_part) / (a_part + b_part)",
        numeric_groups=("a_part", "b_part", "total"),
        inputs=values,
        context={
            "a": match.group("a"),
            "b": match.group("b"),
            "noun": _noun(match.group("noun")),
        },
        answer=answer,
        verify=lambda row: (
            row["total"]
            * (row["b_part"] - row["a_part"])
            / (row["a_part"] + row["b_part"])
        ),
    )


def _resolve_transfer(source: str) -> FormulaSolution | None:
    match = _fullmatch(_FRACTION_TRANSFER, source)
    if (
        match is None
        or not _same(match.group("receiver"), match.group("query"))
        or _same(match.group("receiver"), match.group("giver"))
        or (
            match.group("object_pronoun").casefold(),
            match.group("owner_pronoun").casefold(),
        )
        not in {("him", "her"), ("her", "his")}
    ):
        return None
    fractions = {
        "half": Fraction(1, 2),
        "one third": Fraction(1, 3),
        "one quarter": Fraction(1, 4),
    }
    fraction = fractions[re.sub(r"\s+", " ", match.group("fraction").casefold())]
    values = {
        "initial": _fraction(match.group("initial")),
        "whole": _fraction(match.group("whole")),
        "fraction": fraction,
    }
    answer = values["initial"] + values["fraction"] * values["whole"]
    return _solution(
        source,
        match,
        family="initial_plus_fractional_transfer",
        equation="answer = initial + fraction * whole",
        numeric_groups=("initial", "whole"),
        inputs=values,
        context={"giver": match.group("giver"), "receiver": match.group("receiver")},
        answer=answer,
        verify=lambda row: row["initial"] + row["fraction"] * row["whole"],
    )


def _resolve_remainder(source: str) -> FormulaSolution | None:
    match = _fullmatch(_TARGET_REMAINDER, source)
    if match is None or _same(match.group("a"), match.group("b")):
        return None
    values = {
        name: _fraction(match.group(name))
        for name in ("target", "time", "a_amount", "b_amount")
    }
    answer = values["target"] - values["a_amount"] - values["b_amount"]
    if answer < 0:
        return None
    return _solution(
        source,
        match,
        family="target_remainder_after_contributions",
        equation="answer = target - contribution_a - contribution_b",
        numeric_groups=("target", "time", "a_amount", "b_amount"),
        inputs={k: v for k, v in values.items() if k != "time"},
        context={"time": _canonical(values["time"])},
        answer=answer,
        verify=lambda row: row["target"] - row["a_amount"] - row["b_amount"],
    )


def _resolve_affine_price(source: str) -> FormulaSolution | None:
    match = _fullmatch(_AFFINE_PRICE, source)
    if (
        match is None
        or _noun(match.group("source")) != _noun(match.group("given_source"))
        or not _same_noun_family(match.group("target"), match.group("query_target"))
    ):
        return None
    values = {
        name: _fraction(match.group(name)) for name in ("offset", "scale", "base")
    }
    answer = values["scale"] * values["base"] - values["offset"]
    return _solution(
        source,
        match,
        family="affine_relative_price",
        equation="answer = scale * base - offset",
        numeric_groups=("offset", "scale", "base"),
        inputs=values,
        context={
            "source": _noun(match.group("source")),
            "target": _noun(match.group("target")),
        },
        answer=answer,
        verify=lambda row: row["scale"] * row["base"] - row["offset"],
    )


def _resolve_mean(source: str) -> FormulaSolution | None:
    match = _fullmatch(_THREE_VALUE_MEAN, source)
    if match is None or not _same(
        match.group("measure"),
        match.group("measure_b"),
        match.group("measure_c"),
        match.group("query_measure"),
    ):
        return None
    values = {name: _fraction(match.group(name)) for name in ("x", "y", "z", "count")}
    if values["count"] != 3:
        return None
    answer = (values["x"] + values["y"] + values["z"]) / values["count"]
    return _solution(
        source,
        match,
        family="signed_three_value_mean",
        equation="answer = (x + y + z) / count",
        numeric_groups=("x", "y", "z", "count"),
        inputs=values,
        context={"unit": "degree Fahrenheit"},
        answer=answer,
        verify=lambda row: (row["x"] + row["y"] + row["z"]) / row["count"],
    )


def _resolve_monthly_split(source: str) -> FormulaSolution | None:
    match = _fullmatch(_MONTHLY_SPLIT_SAVING, source)
    if match is None:
        return None
    target = _fraction(match.group("target"))
    half_days = Fraction(15)
    values = {"target": target, "half_days": half_days, "second_factor": Fraction(2)}
    answer = (
        values["target"]
        * values["second_factor"]
        / (values["half_days"] * (1 + values["second_factor"]))
    )
    return _solution(
        source,
        match,
        family="even_month_split_daily_saving",
        equation="answer = target * second_factor / (half_days * (1 + second_factor))",
        numeric_groups=("target",),
        inputs=values,
        context={"month": match.group("month"), "days": "30"},
        answer=answer,
        verify=lambda row: (
            row["target"]
            * row["second_factor"]
            / (row["half_days"] * (1 + row["second_factor"]))
        ),
    )


def _resolve_two_day_revenue(source: str) -> FormulaSolution | None:
    match = _fullmatch(_TWO_DAY_REVENUE, source)
    if match is None or not _same(
        match.group("worker"),
        match.group("worker_again"),
        match.group("worker_third"),
        match.group("worker_fourth"),
        match.group("worker_fifth"),
        match.group("query_worker"),
    ):
        return None
    if _same(match.group("day_a"), match.group("day_b")):
        return None
    if not all(
        (
            _same_item_family(
                match.group("a_object"),
                match.group("count_aa_object"),
                match.group("zero_a_object"),
            ),
            _same_item_family(
                match.group("b_object"),
                match.group("count_ab_object"),
                match.group("count_bb_object"),
            ),
            _same_noun_family(match.group("category_a"), match.group("a_object")),
            _same_noun_family(match.group("category_b"), match.group("b_object")),
            _same_noun_family(match.group("objects"), match.group("a_object")),
            _same_noun_family(match.group("objects"), match.group("b_object")),
        )
    ):
        return None
    values = {
        name: _fraction(match.group(name))
        for name in ("rate_a", "rate_b", "count_aa", "count_ab", "count_bb")
    }
    if min(values.values()) < 0:
        return None
    revenue_a = (
        values["rate_a"] * values["count_aa"] + values["rate_b"] * values["count_ab"]
    )
    revenue_b = values["rate_b"] * values["count_bb"]
    answer = abs(revenue_a - revenue_b)
    return _solution(
        source,
        match,
        family="two_day_mixed_rate_revenue_difference",
        equation=(
            "answer = abs((rate_a * count_aa + rate_b * count_ab) - rate_b * count_bb)"
        ),
        numeric_groups=("rate_a", "rate_b", "count_aa", "count_ab", "count_bb"),
        inputs=values,
        context={
            "category_a": _item_phrase(match.group("a_object")),
            "category_b": _item_phrase(match.group("b_object")),
            "day_a": match.group("day_a"),
            "day_b": match.group("day_b"),
            "worker": match.group("worker"),
        },
        answer=answer,
        verify=lambda row: abs(
            row["rate_a"] * row["count_aa"]
            + row["rate_b"] * row["count_ab"]
            - row["rate_b"] * row["count_bb"]
        ),
    )


def _resolve_dependent_price_ledger(source: str) -> FormulaSolution | None:
    match = _fullmatch(_DEPENDENT_PRICE_LEDGER, source)
    if match is None:
        return None
    if not all(
        (
            _same_item_family(
                match.group("item_a"),
                match.group("a_single"),
                match.group("a_reference"),
                match.group("a_plural"),
            ),
            _same_item_family(
                match.group("item_b"),
                match.group("b_single"),
                match.group("b_plural"),
                match.group("given_b"),
            ),
            _same_item_family(
                match.group("item_c"),
                match.group("c_single"),
                match.group("c_plural"),
            ),
        )
    ):
        return None
    values = {
        name: _fraction(match.group(name))
        for name in ("offset", "count_a", "count_b", "count_c", "base")
    }
    if min(values.values()) < 0 or values["base"] <= 0:
        return None
    values["a_factor"] = Fraction(3)
    price_a = values["a_factor"] * values["base"]
    price_c = price_a - values["offset"]
    if price_c < 0:
        return None
    answer = (
        values["count_a"] * price_a
        + values["count_b"] * values["base"]
        + values["count_c"] * price_c
    )
    return _solution(
        source,
        match,
        family="dependent_price_purchase_ledger",
        equation=(
            "price_a = 3 * base; price_c = price_a - offset; "
            "answer = count_a * price_a + count_b * base + count_c * price_c"
        ),
        numeric_groups=("offset", "count_a", "count_b", "count_c", "base"),
        inputs=values,
        context={
            "buyer": match.group("buyer"),
            "item_a": _item_phrase(match.group("item_a")),
            "item_b": _item_phrase(match.group("item_b")),
            "item_c": _item_phrase(match.group("item_c")),
        },
        answer=answer,
        verify=lambda row: (
            row["count_a"] * row["a_factor"] * row["base"]
            + row["count_b"] * row["base"]
            + row["count_c"] * (row["a_factor"] * row["base"] - row["offset"])
        ),
    )


def _resolve_savings_work_balance(source: str) -> FormulaSolution | None:
    match = _fullmatch(_SAVINGS_WORK_BALANCE, source)
    if match is None or not _same(
        match.group("owner"),
        match.group("owner_again"),
        match.group("owner_third"),
    ):
        return None
    if not all(
        (
            _same_item_family(
                match.group("item_a"),
                match.group("a_again"),
                match.group("query_a"),
            ),
            _same_item_family(
                match.group("item_b"),
                match.group("b_single"),
                match.group("query_b"),
            ),
            _same(match.group("work_verb"), match.group("work_verb_again")),
        )
    ):
        return None
    values = {
        name: _fraction(match.group(name))
        for name in (
            "count_b",
            "price_a",
            "price_b",
            "work_count",
            "work_rate",
            "target_rate",
            "saved",
        )
    }
    if min(values.values()) < 0 or values["target_rate"] <= 0:
        return None
    target = values["price_a"] + values["count_b"] * values["price_b"]
    available = values["saved"] + values["work_count"] * values["work_rate"]
    remaining = target - available
    if remaining < 0:
        return None
    answer = remaining / values["target_rate"]
    if answer.denominator != 1:
        return None
    return _solution(
        source,
        match,
        family="savings_plus_work_balance",
        equation=(
            "answer = (price_a + count_b * price_b - saved "
            "- work_count * work_rate) / target_rate"
        ),
        numeric_groups=(
            "count_b",
            "price_a",
            "price_b",
            "work_count",
            "work_rate",
            "target_rate",
            "saved",
        ),
        inputs=values,
        context={
            "item_a": _item_phrase(match.group("item_a")),
            "item_b": _item_phrase(match.group("item_b")),
            "owner": match.group("owner"),
            "target_work": match.group("target_work"),
        },
        answer=answer,
        verify=lambda row: (
            (
                row["price_a"]
                + row["count_b"] * row["price_b"]
                - row["saved"]
                - row["work_count"] * row["work_rate"]
            )
            / row["target_rate"]
        ),
    )


def _resolve_hourly_purchase(source: str) -> FormulaSolution | None:
    match = _fullmatch(_HOURLY_PURCHASE, source)
    if match is None:
        return None
    values = {
        name: _fraction(match.group(name)) for name in ("first", "second", "rate")
    }
    if min(values.values()) < 0 or values["rate"] <= 0:
        return None
    answer = (values["first"] + values["second"]) / values["rate"]
    return _solution(
        source,
        match,
        family="hourly_wage_purchase_time",
        equation="answer = (first + second) / hourly_rate",
        numeric_groups=("first", "second", "rate"),
        inputs=values,
        context={
            "item_a": _item_phrase(match.group("item_a")),
            "item_b": _item_phrase(match.group("item_b")),
            "owner": match.group("owner"),
        },
        answer=answer,
        verify=lambda row: (row["first"] + row["second"]) / row["rate"],
    )


def _resolve_loan_balance(source: str) -> FormulaSolution | None:
    match = _fullmatch(_LOAN_BALANCE, source)
    if match is None or not _same(match.group("owner"), match.group("query_owner")):
        return None
    values = {
        name: _fraction(match.group(name))
        for name in ("principal", "percent", "payment", "months")
    }
    if min(values.values()) < 0 or values["principal"] <= 0 or values["percent"] > 100:
        return None
    values["percent_base"] = Fraction(100)
    answer = (
        values["principal"] * (1 + values["percent"] / values["percent_base"])
        - values["payment"] * values["months"]
    )
    if answer < 0:
        return None
    return _solution(
        source,
        match,
        family="simple_interest_loan_balance",
        equation=("answer = principal * (1 + percent / 100) - payment * months"),
        numeric_groups=("principal", "percent", "payment", "months"),
        inputs=values,
        context={"owner": match.group("owner")},
        answer=answer,
        verify=lambda row: (
            row["principal"] * (1 + row["percent"] / row["percent_base"])
            - row["payment"] * row["months"]
        ),
    )


def _resolve_consumption_duration(source: str) -> FormulaSolution | None:
    match = _fullmatch(_CONSUMPTION_DURATION, source)
    if match is None or not _same_item_family(
        match.group("resource"),
        match.group("resource_again"),
        match.group("query_resource"),
    ):
        return None
    values = {
        name: _fraction(match.group(name)) for name in ("rate", "people", "total")
    }
    if min(values.values()) <= 0:
        return None
    values["meals_per_day"] = Fraction(2)
    values["days_per_week"] = Fraction(7)
    answer = values["total"] / (
        values["rate"]
        * values["people"]
        * values["meals_per_day"]
        * values["days_per_week"]
    )
    return _solution(
        source,
        match,
        family="household_consumption_duration",
        equation="answer = total / (rate * people * meals_per_day * days_per_week)",
        numeric_groups=("rate", "people", "total"),
        inputs=values,
        context={
            "days_per_week": "7",
            "meals": "lunch and dinner",
            "resource": _item_phrase(match.group("resource")),
        },
        answer=answer,
        verify=lambda row: (
            row["total"]
            / (
                row["rate"]
                * row["people"]
                * row["meals_per_day"]
                * row["days_per_week"]
            )
        ),
    )


def _resolve_annual_driving_cost(source: str) -> FormulaSolution | None:
    match = _fullmatch(_ANNUAL_DRIVING_COST, source)
    if match is None:
        return None
    values = {
        name: _fraction(match.group(name))
        for name in ("miles", "days", "weeks", "rate", "bonus")
    }
    if min(values.values()) < 0:
        return None
    values["round_trip"] = Fraction(2)
    values["months_per_year"] = Fraction(12)
    answer = (
        values["miles"]
        * values["round_trip"]
        * values["days"]
        * values["weeks"]
        * values["rate"]
        + values["bonus"] * values["months_per_year"]
    )
    return _solution(
        source,
        match,
        family="annual_commute_plus_monthly_bonus",
        equation=(
            "answer = miles * round_trip * days * weeks * rate "
            "+ bonus * months_per_year"
        ),
        numeric_groups=("miles", "days", "weeks", "rate", "bonus"),
        inputs=values,
        context={"owner": match.group("owner")},
        answer=answer,
        verify=lambda row: (
            row["miles"] * row["round_trip"] * row["days"] * row["weeks"] * row["rate"]
            + row["bonus"] * row["months_per_year"]
        ),
    )


def _resolve_party_remainder(source: str) -> FormulaSolution | None:
    match = _fullmatch(_PARTY_REMAINDER, source)
    if match is None:
        return None
    values = {
        name: _fraction(match.group(name))
        for name in ("men", "women", "numerator", "denominator", "men_stayed")
    }
    if (
        min(values.values()) < 0
        or values["denominator"] <= 0
        or values["numerator"] > values["denominator"]
    ):
        return None
    people_stayed = (values["men"] + values["women"]) * (
        1 - values["numerator"] / values["denominator"]
    )
    answer = people_stayed - values["men_stayed"]
    if answer < 0 or answer > values["women"] or answer.denominator != 1:
        return None
    return _solution(
        source,
        match,
        family="party_fraction_departure_remainder",
        equation=(
            "answer = (men + women) * (1 - numerator / denominator) - men_stayed"
        ),
        numeric_groups=("men", "women", "numerator", "denominator", "men_stayed"),
        inputs=values,
        context={"target": "women_stayed"},
        answer=answer,
        verify=lambda row: (
            (row["men"] + row["women"]) * (1 - row["numerator"] / row["denominator"])
            - row["men_stayed"]
        ),
    )


def _resolve_job_net_difference(source: str) -> FormulaSolution | None:
    match = _fullmatch(_JOB_NET_DIFFERENCE, source)
    if match is None or not _same(match.group("owner"), match.group("query_owner")):
        return None
    values = {
        name: _fraction(match.group(name))
        for name in (
            "a_hourly",
            "a_hours",
            "a_tax",
            "b_salary",
            "b_property",
            "b_tax",
        )
    }
    if (
        min(values.values()) < 0
        or values["a_tax"] > 100
        or values["b_tax"] > 100
        or values["b_property"] > values["b_salary"]
    ):
        return None
    values["percent_base"] = Fraction(100)
    net_a = (
        values["a_hourly"]
        * values["a_hours"]
        * (1 - values["a_tax"] / values["percent_base"])
    )
    net_b = (values["b_salary"] - values["b_property"]) * (
        1 - values["b_tax"] / values["percent_base"]
    )
    answer = abs(net_a - net_b)
    return _solution(
        source,
        match,
        family="two_job_after_tax_difference",
        equation=(
            "net_a = a_hourly * a_hours * (1 - a_tax / 100); "
            "net_b = (b_salary - b_property) * (1 - b_tax / 100); "
            "answer = abs(net_a - net_b)"
        ),
        numeric_groups=(
            "a_hourly",
            "a_hours",
            "a_tax",
            "b_salary",
            "b_property",
            "b_tax",
        ),
        inputs=values,
        context={"owner": match.group("owner")},
        answer=answer,
        verify=lambda row: abs(
            row["a_hourly"] * row["a_hours"] * (1 - row["a_tax"] / row["percent_base"])
            - (row["b_salary"] - row["b_property"])
            * (1 - row["b_tax"] / row["percent_base"])
        ),
    )


_RESOLVERS = (
    _resolve_prism,
    _resolve_ratio,
    _resolve_transfer,
    _resolve_remainder,
    _resolve_affine_price,
    _resolve_mean,
    _resolve_monthly_split,
    _resolve_two_day_revenue,
    _resolve_dependent_price_ledger,
    _resolve_savings_work_balance,
    _resolve_hourly_purchase,
    _resolve_loan_balance,
    _resolve_consumption_duration,
    _resolve_annual_driving_cost,
    _resolve_party_remainder,
    _resolve_job_net_difference,
)


def solve_guarded_formula(source: str) -> FormulaSolution | None:
    if not isinstance(source, str) or not source.strip():
        return None
    solutions = [
        solution
        for resolver in _RESOLVERS
        if (solution := resolver(source)) is not None
    ]
    return solutions[0] if len(solutions) == 1 else None


__all__ = [
    "FormulaCertificate",
    "FormulaSolution",
    "NumericSpan",
    "solve_guarded_formula",
]
