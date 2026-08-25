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
    variables_in_constraint,
)
from .clause_compiler import SymbolKey, compile_clauses
from .signed_event_frontend import FrontendStatus, compile_signed_events


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
    r"\b(?:he|she|they|them|him|her|his|their|it|its|this|that|these|those|"
    r"himself|herself|itself|themselves)\b",
    re.IGNORECASE,
)
_SUBJECT_PRONOUN = r"(?:he|she|it|they)"
_OBJECT_PRONOUN = r"(?:him|her|it|them)"
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
_COUNT_AFFINE_PRONOUN_SOURCE = re.compile(
    rf"^(?P<target>{_NAME})\s+(?:has|had|owns?|keeps?)\s+"
    rf"(?P<offset>{_NUMBER})\s+(?P<direction>more|fewer)\s+"
    rf"(?P<noun>{_NOUN}?)\s+than\s+(?P<source_pronoun>{_OBJECT_PRONOUN})$",
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
_COUNT_PRONOUN_QUERY = re.compile(
    rf"^how\s+many\s+(?P<noun>{_NOUN}?)\s+"
    rf"(?P<auxiliary>do|does|did)\s+"
    rf"(?P<entity_pronoun>{_SUBJECT_PRONOUN})\s+(?:have|own|keep)$",
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
_MASCULINE_OWNER_WORDS = frozenset(
    {
        "boy",
        "brother",
        "dad",
        "father",
        "grandfather",
        "grandpa",
        "husband",
        "man",
        "son",
        "uncle",
    }
)
_FEMININE_OWNER_WORDS = frozenset(
    {
        "aunt",
        "daughter",
        "girl",
        "grandma",
        "grandmother",
        "mother",
        "mom",
        "sister",
        "wife",
        "woman",
    }
)
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
_DECORATION_WORD = (
    r"(?!(?:if|then|how|what|when|where|why|final|original|length|design)\b)"
    + _SIMPLE_NOUN
)
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

_ORIGINAL_LENGTH_POSSESSIVE = re.compile(
    rf"^\s*(?P<owner>{_NAME})\s+is\s+designing\s+"
    rf"(?P<possessive>his|her|their)\s+own\s+"
    rf"(?P<intro_object>{_SIMPLE_NOUN})\s*,\s+and\s+decides\s+to\s+make\s+it\s+"
    rf"a\s+longer\s+(?P<longer_object>{_SIMPLE_NOUN})\s+by\s+extending\s+the\s+"
    rf"(?P<object>{_SIMPLE_NOUN})\s+by\s+(?P<percent>{_NUMBER})\s*%\s+of\s+"
    rf"its\s+original\s+length\.\s+(?P<subject_pronoun>He|She|They)\s+also\s+"
    rf"adds\s+(?P<addition>{_NUMBER})\s*(?P<addition_unit>{_LENGTH_WORD})\s+"
    rf"to\s+the\s+bottom\s+of\s+the\s+(?P<addition_object>{_SIMPLE_NOUN})\s+"
    rf"with\s+(?:a|an|the)\s+"
    rf"(?P<decoration>{_DECORATION_WORD}(?:\s+{_DECORATION_WORD}){{0,3}})\.\s+"
    rf"If\s+the\s+final\s+design\s+is\s+"
    rf"(?P<final>{_NUMBER})\s*(?P<final_unit>{_LENGTH_WORD})\s+long\s+then\s+"
    rf"how\s+long\s*,\s+in\s+(?P<query_unit>{_LENGTH_WORD})\s*,\s+was\s+the\s+"
    rf"(?P<query_object>{_SIMPLE_NOUN})\s+in\s+its\s+original\s+design\s*\?\s*$",
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

_MEASURE_WORD = r"ounces?|liters?|milliliters?|gallons?|cups?"
_PRONOUN_WORD = r"he|she|they"
_PHRASE = r"[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){0,4}"
_LAZY_PHRASE = r"[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){0,4}?"
_RESOURCE_USE_BALANCE = re.compile(
    rf"^\s*(?P<owner>{_NAME})\s+wants\s+to\s+make\s+"
    rf"(?P<intro>{_PHRASE})\s+with\s+(?P<total>{_NUMBER})\s+"
    rf"(?P<total_unit>{_MEASURE_WORD})\s+of\s+(?P<material>{_PHRASE})\.\s+"
    rf"(?P<rates_pronoun>{_PRONOUN_WORD})\s+can\s+make\s+"
    rf"(?P<rates>.+?)\.\s+If\s+(?P<count_pronoun>{_PRONOUN_WORD})\s+"
    rf"makes?\s+(?P<counts>.+?),\s+how\s+many\s+"
    rf"(?P<query_unit>{_MEASURE_WORD})\s+of\s+(?P<query_material>{_PHRASE})\s+"
    rf"does\s+(?P<query_pronoun>{_PRONOUN_WORD})\s+have\s+left\s*\?\s*$",
    re.IGNORECASE,
)
_RESOURCE_RATE_TERM = re.compile(
    rf"(?P<label>[A-Za-z][A-Za-z'\-]*)\s+"
    rf"(?P<noun>{_SIMPLE_NOUN})\s+that\s+uses?\s+"
    rf"(?P<rate>{_NUMBER})\s+(?:an?\s+)?(?P<unit>{_MEASURE_WORD})"
    rf"(?:\s+per\s+(?P<per_noun>{_SIMPLE_NOUN}))?",
    re.IGNORECASE,
)
_RESOURCE_COUNT_TERM = re.compile(
    rf"(?P<count>{_NUMBER})\s+(?P<label>[A-Za-z][A-Za-z'\-]*)\s+"
    rf"(?P<noun>{_SIMPLE_NOUN})",
    re.IGNORECASE,
)

_COMPONENT_RATIO_SYSTEM = re.compile(
    rf"^\s*(?P<owner>{_NAME})\s+makes\s+(?:a|an)\s+(?P<product>{_PHRASE})\s+"
    rf"from\s+(?P<components>.+?)\.\s+"
    rf"(?P<total_pronoun>{_PRONOUN_WORD})\s+makes\s+enough\s+to\s+fill\s+"
    rf"(?:a|an)\s+(?P<total>{_NUMBER})\s*-\s*(?P<total_unit>{_MEASURE_WORD})\s+"
    rf"(?P<container>{_SIMPLE_NOUN})\s+each\s+time\.\s+"
    rf"(?P<equal_pronoun>{_PRONOUN_WORD})\s+uses\s+(?P<equalities>.+?)\.\s+"
    rf"(?P<scale_pronoun>{_PRONOUN_WORD})\s+uses\s+(?P<scales>.+?)\.\s+"
    rf"How\s+many\s+(?P<query_unit>{_MEASURE_WORD})\s+of\s+"
    rf"(?P<target>{_PHRASE})\s+does\s+(?P<query_pronoun>{_PRONOUN_WORD})\s+"
    rf"use\s*\?\s*$",
    re.IGNORECASE,
)
_SAME_AMOUNT_TERM = re.compile(
    rf"the\s+same\s+amount\s+of\s+(?P<target>{_PHRASE})\s+as\s+"
    rf"(?P<source>{_PHRASE})(?=\s*(?:,|\band\b|$))",
    re.IGNORECASE,
)
_SCALED_AMOUNT_TERM = re.compile(
    rf"(?P<factor>twice|half|{_NUMBER}\s+times)\s+as\s+much\s+"
    rf"(?P<target>{_LAZY_PHRASE})\s+as\s+(?P<source>{_PHRASE})"
    rf"(?=\s*(?:,|\band\b|$))",
    re.IGNORECASE,
)

_TEAM = r"team\s+(?:[A-Z]|[A-Za-z][A-Za-z'\-]*)"
_PERIOD_SCORE_SYSTEM = re.compile(
    rf"^\s*In\s+the\s+first\s+(?P<period>half|period|round)\s+of\s+"
    rf"(?:a|the)\s+(?:[A-Za-z][A-Za-z'\-]*\s+){{0,3}}"
    rf"(?P<event>match|game|contest),\s+(?P<a_first>{_TEAM})\s+scores\s+"
    rf"(?P<a_value>{_NUMBER})\s+(?P<a_noun>{_SIMPLE_NOUN})\s+while\s+"
    rf"(?P<b_first>{_TEAM})\s+scores\s+(?P<offset>{_NUMBER})\s+"
    rf"(?P<offset_noun>{_SIMPLE_NOUN})\s+(?P<direction>fewer|more)\s+than\s+"
    rf"(?P<a_reference>{_TEAM})\.\s+In\s+the\s+second\s+"
    rf"(?P<second_period>half|period|round),\s+(?P<a_second>{_TEAM})\s+"
    rf"scores\s+(?P<fraction>{_FRACTION_TEXT})\s+of\s+the\s+number\s+of\s+"
    rf"(?P<fraction_noun>{_SIMPLE_NOUN})\s+scored\s+by\s+"
    rf"(?P<b_second>{_TEAM}),\s+which\s+scores\s+(?P<scale>{_NUMBER})\s+"
    rf"times\s+the\s+number\s+of\s+(?P<scale_noun>{_SIMPLE_NOUN})\s+it\s+"
    rf"scored\s+in\s+the\s+first\s+(?P<reference_period>half|period|round)\.\s+"
    rf"What(?:\s+is|'s)\s+the\s+total\s+number\s+of\s+"
    rf"(?P<query_noun>{_SIMPLE_NOUN})\s+scored\s+in\s+the\s+"
    rf"(?P<query_event>match|game|contest)\s*\?\s*$",
    re.IGNORECASE,
)

_SIZE_LABEL = r"[A-Za-z][A-Za-z'\-]*"
_SATIETY_EQUIVALENCE_CHAIN = re.compile(
    rf"^\s*(?P<owner>{_NAME})\s+loves\s+to\s+eat\s+"
    rf"(?P<intro_item>{_PHRASE}),\s+but\s+how\s+many\s+"
    rf"(?P<intro_repeat>{_PHRASE})\s+(?P<intro_pronoun>he|she|they)\s+can\s+"
    rf"eat\s+depends\s+on\s+the\s+size\s+of\s+the\s+"
    rf"(?P<size_item>{_PHRASE})\.\s+It\s+takes\s+(?P<base_count>{_NUMBER})\s+"
    rf"(?P<base_label>{_SIZE_LABEL})\s+(?P<base_item>{_PHRASE})\s+to\s+fill\s+"
    rf"(?P<fill_owner>{_NAME})\s+up\.\s+"
    rf"(?P<scale_pronoun>he|she|they)\s+can\s+eat\s+"
    rf"(?P<scale>twice|{_NUMBER}\s+times)\s+as\s+many\s+"
    rf"(?P<scaled_label>{_SIZE_LABEL})\s+(?P<scaled_item>{_PHRASE})\s+as\s+"
    rf"(?P<scale_source_label>{_SIZE_LABEL})\s+"
    rf"(?P<scale_source_item>{_PHRASE})\.\s+"
    rf"(?:And\s+)?eating\s+(?P<equivalent_target_count>{_NUMBER})\s+"
    rf"(?P<equivalent_target_label>{_SIZE_LABEL})\s+"
    rf"(?P<equivalent_target_item>{_PHRASE})\s+is\s+the\s+same\s+as\s+"
    rf"eating\s+(?P<equivalent_source_count>{_NUMBER})\s+"
    rf"(?P<equivalent_source_label>{_SIZE_LABEL})\s+"
    rf"(?P<equivalent_source_item>{_PHRASE})\.\s+How\s+many\s+"
    rf"(?P<query_label>{_SIZE_LABEL})\s+(?P<query_item>{_PHRASE})\s+can\s+"
    rf"(?P<query_owner>{_NAME})\s+eat\s*\?\s*$",
    re.IGNORECASE,
)

_FLOW_TIMELINE = re.compile(
    rf"^\s*The\s+amount\s+of\s+(?P<initial_material>{_PHRASE})\s+passing\s+"
    rf"through\s+(?:a|an|the)\s+(?P<initial_channel>{_PHRASE})\s+at\s+"
    rf"(?P<initial_location>{_PHRASE})\s+is\s+(?P<initial>{_NUMBER})\s+"
    rf"(?P<initial_unit>{_MEASURE_WORD})\.\s+After\s+(?:a|one)\s+"
    rf"(?P<elapsed_unit>{_TIME_WORD})\s+of\s+(?P<event>{_PHRASE}),\s+the\s+"
    rf"amount\s+of\s+(?P<scaled_material>{_PHRASE})\s+passing\s+through\s+"
    rf"the\s+(?P<scaled_channel>{_PHRASE})\s+"
    rf"(?P<multiplier>doubles|triples|quadruples)\s+at\s+the\s+same\s+point\.\s+"
    rf"If\s+the\s+volume\s+of\s+(?P<increment_material>{_PHRASE})\s+passing\s+"
    rf"through\s+the\s+(?P<increment_channel>{_PHRASE})\s+at\s+that\s+point\s+"
    rf"increases\s+by\s+(?P<increment>{_NUMBER})\s+"
    rf"(?P<increment_unit>{_MEASURE_WORD})\s+on\s+the\s+third\s+"
    rf"(?P<third_unit>{_TIME_WORD}),\s+calculate\s+the\s+total\s+amount\s+of\s+"
    rf"(?P<query_material>{_PHRASE})\s+passing\s+through\s+the\s+"
    rf"(?P<query_channel>{_PHRASE})\s+at\s+that\s+point\s*\.\s*$",
    re.IGNORECASE,
)

# A deliberately small recurrence language.  The three clauses are matched as
# typed predicates and may appear in any sentence order; no keyword search or
# fuzzy binding participates.  Requiring every owner occurrence makes the
# recurrence basis explicit instead of resolving pronouns heuristically.
_RECURRENCE_START = re.compile(
    rf"^(?P<owner>{_NAME})\s*[\u2019']s\s+sequence\s+has\s+value\s+"
    rf"(?P<value>{_SIGNED_NUMBER})\s+at\s+step\s+(?P<index>{_NUMBER})$",
    re.IGNORECASE,
)
_RECURRENCE_AFFINE = re.compile(
    rf"^At\s+each\s+step,\s+the\s+next\s+value\s+in\s+"
    rf"(?P<next_owner>{_NAME})\s*[\u2019']s\s+sequence\s+(?:is|equals)\s+"
    rf"(?P<factor>{_SIGNED_NUMBER})\s+times\s+the\s+current\s+value\s+in\s+"
    rf"(?P<current_owner>{_NAME})\s*[\u2019']s\s+sequence\s+"
    rf"(?P<offset_direction>plus|minus)\s+(?P<offset>{_NUMBER})$",
    re.IGNORECASE,
)
_RECURRENCE_CURRENT_PERCENT = re.compile(
    rf"^At\s+each\s+step,\s+the\s+next\s+value\s+in\s+"
    rf"(?P<next_owner>{_NAME})\s*[\u2019']s\s+sequence\s+(?:is|equals)\s+"
    rf"the\s+current\s+value\s+in\s+"
    rf"(?P<base_owner>{_NAME})\s*[\u2019']s\s+sequence\s+plus\s+"
    rf"(?P<percent>{_NUMBER})\s*%\s+of\s+the\s+current\s+value\s+in\s+"
    rf"(?P<percent_owner>{_NAME})\s*[\u2019']s\s+sequence\s+"
    rf"(?P<offset_direction>plus|minus)\s+(?P<offset>{_NUMBER})$",
    re.IGNORECASE,
)
_RECURRENCE_ORIGINAL_PERCENT = re.compile(
    rf"^At\s+each\s+step,\s+the\s+next\s+value\s+in\s+"
    rf"(?P<next_owner>{_NAME})\s*[\u2019']s\s+sequence\s+(?:is|equals)\s+"
    rf"the\s+current\s+value\s+in\s+"
    rf"(?P<base_owner>{_NAME})\s*[\u2019']s\s+sequence\s+plus\s+"
    rf"(?P<percent>{_NUMBER})\s*%\s+of\s+the\s+original\s+value\s+in\s+"
    rf"(?P<percent_owner>{_NAME})\s*[\u2019']s\s+sequence\s+"
    rf"(?P<offset_direction>plus|minus)\s+(?P<offset>{_NUMBER})$",
    re.IGNORECASE,
)
_RECURRENCE_BARE_PERCENT = re.compile(
    rf"^At\s+each\s+step,\s+the\s+next\s+value\s+in\s+"
    rf"(?P<next_owner>{_NAME})\s*[\u2019']s\s+sequence\s+(?:is|equals)\s+"
    rf"the\s+current\s+value\s+in\s+"
    rf"(?P<base_owner>{_NAME})\s*[\u2019']s\s+sequence\s+plus\s+"
    rf"(?P<percent>{_NUMBER})\s*%\s+"
    rf"(?P<offset_direction>plus|minus)\s+(?P<offset>{_NUMBER})$",
    re.IGNORECASE,
)
_RECURRENCE_VALUE_QUERY = re.compile(
    rf"^What\s+is\s+the\s+value\s+in\s+(?P<owner>{_NAME})\s*[\u2019']s\s+"
    rf"sequence\s+at\s+step\s+(?P<end>{_NUMBER})$",
    re.IGNORECASE,
)
_RECURRENCE_CHANGE_QUERY = re.compile(
    rf"^What\s+is\s+the\s+net\s+change\s+in\s+"
    rf"(?P<owner>{_NAME})\s*[\u2019']s\s+sequence\s+from\s+step\s+"
    rf"(?P<start>{_NUMBER})\s+to\s+step\s+(?P<end>{_NUMBER})$",
    re.IGNORECASE,
)
_RECURRENCE_SUM_QUERY = re.compile(
    rf"^What\s+is\s+the\s+cumulative\s+sum\s+of\s+the\s+values\s+in\s+"
    rf"(?P<owner>{_NAME})\s*[\u2019']s\s+sequence\s+from\s+step\s+"
    rf"(?P<start>{_NUMBER})\s+through\s+step\s+(?P<end>{_NUMBER})$",
    re.IGNORECASE,
)

_MAX_RECURRENCE_STEPS = 64


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


@dataclass(frozen=True, slots=True)
class _ActiveReferent:
    """One explicitly named entity in one typed semantic scope."""

    symbol: SymbolKey
    variable: Variable
    span: Span
    number: str


class _ActiveReferentLedger:
    """Conservative, document-local antecedents for typed pronoun positions.

    Entries are keyed by :class:`SymbolKey`, so a count of shells can never be
    reused as a count of marbles.  Resolution is based only on grammatical
    number, role, source order, and caller-supplied binding exclusions.  Names
    and pronouns carry no inferred gender semantics.
    """

    _SINGULAR_BY_ROLE = {
        "subject": frozenset({"he", "she", "it"}),
        "object": frozenset({"him", "her", "it"}),
    }
    _PLURAL_BY_ROLE = {
        "subject": frozenset({"they"}),
        "object": frozenset({"them"}),
    }

    def __init__(self) -> None:
        self._entries: dict[SymbolKey, _ActiveReferent] = {}

    def register(
        self,
        symbol: SymbolKey,
        variable: Variable,
        span: Span,
        *,
        number: str = "singular",
    ) -> None:
        if number not in {"singular", "plural"}:
            raise ValueError("referent number must be singular or plural")
        previous = self._entries.get(symbol)
        if previous is not None:
            if previous.variable != variable or previous.number != number:
                raise _Abort(
                    ParseStatus.AMBIGUOUS,
                    "explicit referent has conflicting structural identity",
                )
            # Keep the first explicit mention.  A later repetition must not turn
            # a forward reference into a retrospectively available antecedent.
            return
        self._entries[symbol] = _ActiveReferent(symbol, variable, span, number)

    def resolve(
        self,
        pronoun: str,
        expected: SymbolKey,
        expected_unit: Unit,
        pronoun_span: Span,
        *,
        role: str,
        excluded: frozenset[SymbolKey] = frozenset(),
    ) -> Variable | None:
        normalized = _entity(pronoun)
        singular = self._SINGULAR_BY_ROLE.get(role)
        plural = self._PLURAL_BY_ROLE.get(role)
        if singular is None or plural is None:
            raise ValueError(f"unsupported pronoun role: {role}")
        if normalized in singular:
            number = "singular"
        elif normalized in plural:
            number = "plural"
        else:
            return None

        candidates = tuple(
            entry.variable
            for entry in self._entries.values()
            if entry.number == number
            and entry.span.end <= pronoun_span.start
            and entry.symbol not in excluded
            and entry.symbol.property == expected.property
            and entry.symbol.item == expected.item
            and entry.symbol.scope == expected.scope
            and entry.symbol.state == expected.state
            and entry.variable.unit.compatible(expected_unit)
        )
        if len(candidates) != 1:
            return None
        return candidates[0]


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


def _step_index(token: str) -> int:
    value = _number(token)
    if value.denominator != 1:
        raise _Abort(ParseStatus.INVALID, "recurrence step indices must be integers")
    return value.numerator


def _same(*values: str) -> bool:
    return len({_entity(value) for value in values}) == 1


def _same_item_family(*values: str) -> bool:
    """Require exact noun identity or one unambiguous compound-noun suffix."""

    normalized = tuple(_singular(value) for value in values)
    canonical = max(normalized, key=lambda value: (len(value.split()), len(value)))
    return all(
        value == canonical or canonical.endswith(f" {value}") for value in normalized
    )


def _pronouns_match_explicit_owner(owner: str, *pronouns: str) -> bool:
    normalized = {_entity(pronoun) for pronoun in pronouns}
    if len(normalized) != 1:
        return False
    pronoun = next(iter(normalized))
    if pronoun == "they":
        return False
    owner_words = frozenset(re.findall(r"[a-z]+", _entity(owner)))
    if pronoun == "he":
        return bool(owner_words & _MASCULINE_OWNER_WORDS) and not bool(
            owner_words & _FEMININE_OWNER_WORDS
        )
    if pronoun == "she":
        return bool(owner_words & _FEMININE_OWNER_WORDS) and not bool(
            owner_words & _MASCULINE_OWNER_WORDS
        )
    return False


def _coordinated_matches(
    text: str,
    pattern: re.Pattern[str],
    *,
    label: str,
    minimum: int = 2,
) -> tuple[re.Match[str], ...]:
    """Parse a comma/and list without silently dropping any surface text."""

    matches: list[re.Match[str]] = []
    cursor = 0
    while cursor < len(text):
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        match = pattern.match(text, cursor)
        if match is None:
            raise _Abort(ParseStatus.UNSUPPORTED, f"unparsed {label} list")
        matches.append(match)
        cursor = match.end()
        if cursor == len(text):
            break
        separator = re.match(r"\s*(?:,\s*(?:and\s+)?|and\s+)", text[cursor:], re.I)
        if separator is None or separator.end() == 0:
            raise _Abort(ParseStatus.UNSUPPORTED, f"unparsed {label} coordination")
        cursor += separator.end()
    if len(matches) < minimum:
        raise _Abort(ParseStatus.UNSUPPORTED, f"{label} requires coordination")
    return tuple(matches)


def _coordinated_phrases(text: str, *, label: str) -> tuple[tuple[str, int, int], ...]:
    pieces: list[tuple[str, int, int]] = []
    cursor = 0
    separator = re.compile(r"\s*(?:,\s*(?:and\s+)?|and\s+)", re.I)
    for match in separator.finditer(text):
        raw = text[cursor : match.start()]
        left = cursor + len(raw) - len(raw.lstrip())
        right = match.start() - len(raw) + len(raw.rstrip())
        if right <= left:
            raise _Abort(ParseStatus.UNSUPPORTED, f"empty {label} list item")
        pieces.append((text[left:right], left, right))
        cursor = match.end()
    raw = text[cursor:]
    left = cursor + len(raw) - len(raw.lstrip())
    right = len(text) - len(raw) + len(raw.rstrip())
    if right <= left:
        raise _Abort(ParseStatus.UNSUPPORTED, f"empty {label} list item")
    pieces.append((text[left:right], left, right))
    if len(pieces) < 2 or any(
        re.fullmatch(_PHRASE, item) is None for item, _, _ in pieces
    ):
        raise _Abort(ParseStatus.UNSUPPORTED, f"invalid {label} list")
    return tuple(pieces)


class StructuralParser:
    """Compile supported relation families to a provenance-carrying problem."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.variables: dict[str, Variable] = {}
        self.constraints: list[object] = []
        self.targets: list[Variable] = []
        self.price_variables: dict[str, Variable] = {}
        self.definitions: set[str] = set()
        self.referents = _ActiveReferentLedger()

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

    def _count_symbol(self, entity: str, item: str) -> SymbolKey:
        return SymbolKey(_entity(entity), "count", item, "collection", "current")

    def _register_named_count(
        self,
        entity: str,
        item: str,
        variable: Variable,
        span: Span,
    ) -> SymbolKey:
        symbol = self._count_symbol(entity, item)
        self.referents.register(symbol, variable, span)
        return symbol

    def _resolve_typed_pronoun(
        self,
        pronoun: str,
        item: str,
        unit: Unit,
        span: Span,
        *,
        role: str,
        excluded: frozenset[SymbolKey] = frozenset(),
        question: bool = False,
    ) -> Variable:
        expected = self._count_symbol("pronoun", item)
        variable = self.referents.resolve(
            pronoun,
            expected,
            unit,
            span,
            role=role,
            excluded=excluded,
        )
        if variable is None:
            reason = (
                "question pronoun binding is not proven"
                if question
                else "numeric pronoun binding is not proven"
            )
            raise _Abort(ParseStatus.AMBIGUOUS, reason)
        return variable

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
            match = _ORIGINAL_LENGTH_POSSESSIVE.fullmatch(self.source)
        if match is None:
            return False

        expanded = "intro_object" in match.groupdict()
        object_groups = (
            (
                "intro_object",
                "longer_object",
                "object",
                "addition_object",
                "query_object",
            )
            if expanded
            else ("object", "query_object")
        )
        objects = tuple(_singular(match.group(group)) for group in object_groups)
        if len(set(objects)) != 1:
            raise _Abort(ParseStatus.AMBIGUOUS, "percent basis object is not explicit")
        if expanded:
            possessive_subject = {
                "his": "he",
                "her": "she",
                "their": "they",
            }
            possessive = _entity(match.group("possessive"))
            subject = _entity(match.group("subject_pronoun"))
            if possessive_subject[possessive] != subject:
                raise _Abort(
                    ParseStatus.AMBIGUOUS,
                    "possessive owner and subject pronoun do not agree",
                )

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

    def _parse_resource_use_balance(self) -> bool:
        """Compile coordinated per-item use and a remaining resource balance."""

        match = _RESOURCE_USE_BALANCE.fullmatch(self.source)
        if match is None:
            return False
        if not _same(
            match.group("rates_pronoun"),
            match.group("count_pronoun"),
            match.group("query_pronoun"),
        ):
            raise _Abort(ParseStatus.AMBIGUOUS, "resource pronouns do not agree")
        if not _same(match.group("material"), match.group("query_material")):
            raise _Abort(ParseStatus.AMBIGUOUS, "resource target does not agree")

        total_unit_name = _singular(match.group("total_unit"))
        if total_unit_name != _singular(match.group("query_unit")):
            raise _Abort(ParseStatus.AMBIGUOUS, "resource units do not agree")

        rate_matches = _coordinated_matches(
            match.group("rates"), _RESOURCE_RATE_TERM, label="resource rates"
        )
        count_matches = _coordinated_matches(
            match.group("counts"), _RESOURCE_COUNT_TERM, label="resource counts"
        )
        rate_labels = tuple(_entity(term.group("label")) for term in rate_matches)
        count_labels = tuple(_entity(term.group("label")) for term in count_matches)
        if len(set(rate_labels)) != len(rate_labels) or len(set(count_labels)) != len(
            count_labels
        ):
            raise _Abort(ParseStatus.AMBIGUOUS, "duplicate resource size lacks scope")
        if set(rate_labels) != set(count_labels):
            raise _Abort(ParseStatus.AMBIGUOUS, "resource rate and count labels differ")

        nouns = tuple(
            _singular(term.group("noun")) for term in (*rate_matches, *count_matches)
        )
        if len(set(nouns)) != 1:
            raise _Abort(ParseStatus.AMBIGUOUS, "resource item nouns do not agree")
        item_noun = nouns[0]
        if _singular(match.group("intro").split()[-1]) != item_noun:
            raise _Abort(ParseStatus.AMBIGUOUS, "resource introduction item differs")

        per_nouns = tuple(term.group("per_noun") for term in rate_matches)
        if per_nouns[0] is None:
            raise _Abort(ParseStatus.UNSUPPORTED, "first resource rate lacks a basis")
        if any(
            value is not None and _singular(value) != item_noun for value in per_nouns
        ):
            raise _Abort(ParseStatus.AMBIGUOUS, "resource rate bases do not agree")
        if any(
            _singular(term.group("unit")) != total_unit_name for term in rate_matches
        ):
            raise _Abort(ParseStatus.AMBIGUOUS, "resource rate units do not agree")

        total_value = _number(match.group("total"))
        rate_values = {
            label: _number(term.group("rate"))
            for label, term in zip(rate_labels, rate_matches, strict=True)
        }
        count_values = {
            label: _number(term.group("count"))
            for label, term in zip(count_labels, count_matches, strict=True)
        }
        if total_value <= 0 or any(value <= 0 for value in rate_values.values()):
            raise _Abort(ParseStatus.INVALID, "resource quantities must be positive")
        if any(value.denominator != 1 for value in count_values.values()):
            raise _Abort(ParseStatus.INVALID, "resource item counts must be integers")
        consumed = sum(
            (rate_values[label] * count_values[label] for label in rate_labels),
            Fraction(0),
        )
        if consumed > total_value:
            raise _Abort(
                ParseStatus.INVALID, "resource use exceeds the available total"
            )

        amount_unit = Unit.base("volume", symbol=total_unit_name)
        item_unit = Unit.count(symbol=item_noun)
        rate_start = match.start("rates")
        count_start = match.start("counts")
        counts_by_label = {_entity(term.group("label")): term for term in count_matches}
        prefix = (
            f"resource.{_entity(match.group('owner'))}."
            f"{_entity(match.group('material'))}"
        )
        used: list[Variable] = []
        whole_span = Span(match.start(), match.end(), self.source)
        for label, rate_match in zip(rate_labels, rate_matches, strict=True):
            count_match = counts_by_label[label]
            count_span = Span(
                count_start + count_match.start("count"),
                count_start + count_match.end("count"),
                self.source,
            )
            rate_span = Span(
                rate_start + rate_match.start("rate"),
                rate_start + rate_match.end("rate"),
                self.source,
            )
            amount = self._variable(
                f"{prefix}.used.{label}",
                amount_unit,
                span=Span(
                    rate_start + rate_match.start(),
                    rate_start + rate_match.end(),
                    self.source,
                ),
            )
            self._define(
                amount,
                Rate(
                    amount,
                    Quantity(rate_values[label], amount_unit / item_unit, rate_span),
                    Quantity(count_values[label], item_unit, count_span),
                    span=whole_span,
                ),
            )
            used.append(amount)

        total_used = self._variable(
            f"{prefix}.used.total",
            amount_unit,
            span=self._match_span(match, "rates"),
        )
        self._define(total_used, Sum(total_used, tuple(used), span=whole_span))
        left = self._variable(
            f"{prefix}.left",
            amount_unit,
            span=self._match_span(match, "query_material"),
        )
        self._define(
            left,
            Balance(
                (left, total_used),
                (
                    Quantity(
                        total_value,
                        amount_unit,
                        span=self._match_span(match, "total"),
                    ),
                ),
                span=whole_span,
            ),
        )
        self.targets.append(left)
        return True

    def _parse_component_ratio_system(self) -> bool:
        """Compile an exhaustive component list and its equality/ratio graph."""

        match = _COMPONENT_RATIO_SYSTEM.fullmatch(self.source)
        if match is None:
            return False
        if not _same(
            match.group("total_pronoun"),
            match.group("equal_pronoun"),
            match.group("scale_pronoun"),
            match.group("query_pronoun"),
        ):
            raise _Abort(ParseStatus.AMBIGUOUS, "recipe pronouns do not agree")
        unit_name = _singular(match.group("total_unit"))
        if unit_name != _singular(match.group("query_unit")):
            raise _Abort(ParseStatus.AMBIGUOUS, "recipe units do not agree")

        component_items = _coordinated_phrases(
            match.group("components"), label="recipe components"
        )
        component_names = tuple(_entity(item) for item, _, _ in component_items)
        if len(component_names) < 3:
            raise _Abort(ParseStatus.UNSUPPORTED, "ratio system needs three components")
        if len(set(component_names)) != len(component_names):
            raise _Abort(
                ParseStatus.AMBIGUOUS, "duplicate recipe component lacks scope"
            )

        def resolve(surface: str) -> str:
            reference = _entity(surface)
            if reference in component_names:
                return reference
            suffix_matches = tuple(
                component
                for component in component_names
                if component.endswith(f" {reference}")
            )
            if len(suffix_matches) == 1:
                return suffix_matches[0]
            if len(suffix_matches) > 1:
                raise _Abort(
                    ParseStatus.AMBIGUOUS,
                    f"ambiguous component suffix: {surface}",
                )
            raise _Abort(ParseStatus.AMBIGUOUS, f"unknown component: {surface}")

        unit = Unit.base("volume", symbol=unit_name)
        components_start = match.start("components")
        prefix = (
            f"recipe.{_entity(match.group('owner'))}.{_entity(match.group('product'))}"
        )
        variables = {
            name: self._variable(
                f"{prefix}.{name}",
                unit,
                span=Span(
                    components_start + start,
                    components_start + end,
                    self.source,
                ),
            )
            for name, (_, start, end) in zip(
                component_names, component_items, strict=True
            )
        }

        equalities = _coordinated_matches(
            match.group("equalities"),
            _SAME_AMOUNT_TERM,
            label="recipe equalities",
            minimum=1,
        )
        scales = _coordinated_matches(
            match.group("scales"),
            _SCALED_AMOUNT_TERM,
            label="recipe scales",
            minimum=1,
        )
        if len(equalities) + len(scales) != len(component_names) - 1:
            raise _Abort(
                ParseStatus.UNSUPPORTED,
                "ratio graph is not a closed component tree",
            )

        seen_edges: set[frozenset[str]] = set()
        whole_span = Span(match.start(), match.end(), self.source)

        def relation_span(local: re.Match[str], group: str) -> Span:
            base = match.start(group)
            return Span(base + local.start(), base + local.end(), self.source)

        equality_relations: list[tuple[str, str, Span]] = []
        for equality in equalities:
            named_target = resolve(equality.group("target"))
            named_source = resolve(equality.group("source"))
            edge = frozenset((named_target, named_source))
            if len(edge) != 2 or edge in seen_edges:
                raise _Abort(ParseStatus.AMBIGUOUS, "duplicate recipe relation")
            seen_edges.add(edge)
            equality_relations.append(
                (
                    named_target,
                    named_source,
                    relation_span(equality, "equalities"),
                )
            )

        scale_relations: list[tuple[str, str, Fraction, Span]] = []
        for scale_match in scales:
            named_target = resolve(scale_match.group("target"))
            named_source = resolve(scale_match.group("source"))
            edge = frozenset((named_target, named_source))
            if len(edge) != 2 or edge in seen_edges:
                raise _Abort(ParseStatus.AMBIGUOUS, "duplicate recipe relation")
            seen_edges.add(edge)
            factor_text = scale_match.group("factor").casefold()
            if factor_text == "twice":
                factor = Fraction(2)
            elif factor_text == "half":
                factor = Fraction(1, 2)
            else:
                factor = _number(re.sub(r"(?i)\s+times$", "", factor_text))
            if factor <= 0:
                raise _Abort(ParseStatus.INVALID, "recipe scale must be positive")
            scale_relations.append(
                (
                    named_target,
                    named_source,
                    factor,
                    relation_span(scale_match, "scales"),
                )
            )

        connected = {component_names[0]}
        while True:
            expanded = connected.union(
                *(edge for edge in seen_edges if edge.intersection(connected))
            )
            if expanded == connected:
                break
            connected = expanded
        if connected != set(component_names):
            raise _Abort(ParseStatus.UNSUPPORTED, "ratio graph is disconnected")

        # Scale clauses are directed.  Equalities are symmetric, so orient
        # their forest through an exact edge-to-endpoint matching that leaves
        # every IR variable with at most one functional definition.
        for named_target, named_source, factor, span in scale_relations:
            self._define(
                variables[named_target],
                Affine(
                    variables[named_target],
                    variables[named_source],
                    factor,
                    Quantity(0, unit, span=span),
                    span=span,
                ),
            )

        owner_by_component: dict[str, int] = {}
        oriented_target: dict[int, str] = {}

        def orient(edge_index: int, visited: set[str]) -> bool:
            explicit_target, explicit_source, _ = equality_relations[edge_index]
            for candidate in (explicit_target, explicit_source):
                if (
                    variables[candidate].name in self.definitions
                    or candidate in visited
                ):
                    continue
                visited.add(candidate)
                previous = owner_by_component.get(candidate)
                if previous is None or orient(previous, visited):
                    owner_by_component[candidate] = edge_index
                    oriented_target[edge_index] = candidate
                    return True
            return False

        for edge_index in range(len(equality_relations)):
            if not orient(edge_index, set()):
                raise _Abort(
                    ParseStatus.AMBIGUOUS,
                    "recipe relations require multiple definitions of one component",
                )
        for edge_index, (left, right, span) in enumerate(equality_relations):
            target = oriented_target[edge_index]
            source = right if target == left else left
            self._define(
                variables[target],
                Assign(variables[target], variables[source], span=span),
            )

        total_value = _number(match.group("total"))
        if total_value <= 0:
            raise _Abort(ParseStatus.INVALID, "recipe total must be positive")
        self.constraints.append(
            Sum(
                Quantity(
                    total_value,
                    unit,
                    span=self._match_span(match, "total"),
                ),
                tuple(variables.values()),
                span=whole_span,
            )
        )
        target_name = resolve(match.group("target"))
        self.targets.append(variables[target_name])
        return True

    def _parse_period_score_system(self) -> bool:
        """Compile two explicitly scoped periods and a total score target."""

        match = _PERIOD_SCORE_SYSTEM.fullmatch(self.source)
        if match is None:
            return False
        if not _same(match.group("event"), match.group("query_event")):
            raise _Abort(ParseStatus.AMBIGUOUS, "score events do not agree")
        if not _same(
            match.group("period"),
            match.group("second_period"),
            match.group("reference_period"),
        ):
            raise _Abort(ParseStatus.AMBIGUOUS, "score periods do not agree")
        if not _same(
            match.group("a_first"),
            match.group("a_reference"),
            match.group("a_second"),
        ):
            raise _Abort(ParseStatus.AMBIGUOUS, "first score team references drift")
        if not _same(match.group("b_first"), match.group("b_second")):
            raise _Abort(ParseStatus.AMBIGUOUS, "second score team references drift")
        if _same(match.group("a_first"), match.group("b_first")):
            raise _Abort(ParseStatus.AMBIGUOUS, "score teams must be distinct")
        nouns = tuple(
            _singular(match.group(group))
            for group in (
                "a_noun",
                "offset_noun",
                "fraction_noun",
                "scale_noun",
                "query_noun",
            )
        )
        if len(set(nouns)) != 1:
            raise _Abort(ParseStatus.AMBIGUOUS, "score nouns do not agree")

        first_a_value = _number(match.group("a_value"))
        offset_value = _number(match.group("offset"))
        scale = _number(match.group("scale"))
        fraction = _fraction(match.group("fraction"))
        if any(
            value.denominator != 1 for value in (first_a_value, offset_value, scale)
        ):
            raise _Abort(
                ParseStatus.INVALID, "score counts and multiplier must be integers"
            )
        if not 0 < fraction <= 1 or scale <= 0:
            raise _Abort(ParseStatus.INVALID, "score ratios must be positive parts")

        signed_offset = (
            -offset_value
            if match.group("direction").casefold() == "fewer"
            else offset_value
        )
        first_b_value = first_a_value + signed_offset
        second_b_value = scale * first_b_value
        second_a_value = fraction * second_b_value
        if any(
            value < 0 or value.denominator != 1
            for value in (
                first_a_value,
                first_b_value,
                second_a_value,
                second_b_value,
            )
        ):
            raise _Abort(ParseStatus.INVALID, "score relations imply invalid counts")

        noun = nouns[0]
        unit = Unit.count(symbol=noun)
        prefix = f"score.{_entity(match.group('event'))}"
        first_a = self._variable(
            f"{prefix}.{_entity(match.group('a_first'))}.first",
            unit,
            span=self._match_span(match, "a_first"),
            count=True,
        )
        first_b = self._variable(
            f"{prefix}.{_entity(match.group('b_first'))}.first",
            unit,
            span=self._match_span(match, "b_first"),
            count=True,
        )
        second_a = self._variable(
            f"{prefix}.{_entity(match.group('a_second'))}.second",
            unit,
            span=self._match_span(match, "a_second"),
            count=True,
        )
        second_b = self._variable(
            f"{prefix}.{_entity(match.group('b_second'))}.second",
            unit,
            span=self._match_span(match, "b_second"),
            count=True,
        )
        total = self._variable(
            f"{prefix}.total",
            unit,
            span=self._match_span(match, "query_noun"),
            count=True,
        )
        whole_span = Span(match.start(), match.end(), self.source)
        self._define(
            first_a,
            Assign(
                first_a,
                Quantity(
                    first_a_value,
                    unit,
                    span=self._match_span(match, "a_value"),
                ),
                span=whole_span,
            ),
        )
        self._define(
            first_b,
            Affine(
                first_b,
                first_a,
                Fraction(1),
                Quantity(
                    signed_offset,
                    unit,
                    span=self._match_span(match, "offset"),
                ),
                span=whole_span,
            ),
        )
        self._define(
            second_b,
            Affine(
                second_b,
                first_b,
                scale,
                Quantity(0, unit, span=self._match_span(match, "scale")),
                span=whole_span,
            ),
        )
        self._define(
            second_a,
            Part(second_a, second_b, fraction, span=whole_span),
        )
        self._define(
            total,
            Sum(total, (first_a, first_b, second_a, second_b), span=whole_span),
        )
        self.targets.append(total)
        return True

    def _parse_satiety_equivalence_chain(self) -> bool:
        """Compile category counts linked by one scale and one equivalence."""

        match = _SATIETY_EQUIVALENCE_CHAIN.fullmatch(self.source)
        if match is None:
            return False
        if not _same(
            match.group("owner"), match.group("fill_owner"), match.group("query_owner")
        ):
            raise _Abort(ParseStatus.AMBIGUOUS, "satiety owners do not agree")
        if not _pronouns_match_explicit_owner(
            match.group("owner"),
            match.group("intro_pronoun"),
            match.group("scale_pronoun"),
        ):
            raise _Abort(
                ParseStatus.AMBIGUOUS,
                "satiety pronouns are not bound to an explicit owner role",
            )

        item_groups = (
            "intro_item",
            "intro_repeat",
            "size_item",
            "base_item",
            "scaled_item",
            "scale_source_item",
            "equivalent_target_item",
            "equivalent_source_item",
            "query_item",
        )
        item_surfaces = tuple(match.group(group) for group in item_groups)
        if not _same_item_family(*item_surfaces):
            raise _Abort(ParseStatus.AMBIGUOUS, "satiety item nouns do not agree")
        canonical_item = _singular(
            max(item_surfaces, key=lambda value: (len(value.split()), len(value)))
        )

        base_label = _entity(match.group("base_label"))
        scaled_label = _entity(match.group("scaled_label"))
        target_label = _entity(match.group("equivalent_target_label"))
        if not _same(match.group("base_label"), match.group("scale_source_label")):
            raise _Abort(ParseStatus.AMBIGUOUS, "satiety scale basis is not explicit")
        if not _same(
            match.group("scaled_label"), match.group("equivalent_source_label")
        ):
            raise _Abort(
                ParseStatus.AMBIGUOUS, "satiety equivalence source is not explicit"
            )
        if not _same(
            match.group("equivalent_target_label"), match.group("query_label")
        ):
            raise _Abort(ParseStatus.AMBIGUOUS, "satiety target is not explicit")
        if len({base_label, scaled_label, target_label}) != 3:
            raise _Abort(ParseStatus.AMBIGUOUS, "satiety categories must be distinct")

        base_count = _number(match.group("base_count"))
        scale_text = match.group("scale").casefold()
        scale = (
            Fraction(2)
            if scale_text == "twice"
            else _number(re.sub(r"(?i)\s+times$", "", scale_text))
        )
        target_equivalent = _number(match.group("equivalent_target_count"))
        source_equivalent = _number(match.group("equivalent_source_count"))
        if (
            any(
                value <= 0 or value.denominator != 1
                for value in (base_count, target_equivalent, source_equivalent)
            )
            or scale <= 0
        ):
            raise _Abort(
                ParseStatus.INVALID,
                "satiety counts must be positive integers and scale positive",
            )
        final_count = base_count * scale * target_equivalent / source_equivalent
        if final_count.denominator != 1:
            raise _Abort(
                ParseStatus.INVALID, "satiety chain implies a fractional count"
            )

        unit = Unit.count(symbol=canonical_item)
        prefix = f"satiety.{_entity(match.group('owner'))}.{canonical_item}"
        base = self._variable(
            f"{prefix}.{base_label}",
            unit,
            span=self._match_span(match, "base_label"),
            count=True,
        )
        scaled = self._variable(
            f"{prefix}.{scaled_label}",
            unit,
            span=self._match_span(match, "scaled_label"),
            count=True,
        )
        target = self._variable(
            f"{prefix}.{target_label}",
            unit,
            span=self._match_span(match, "query_label"),
            count=True,
        )
        whole_span = Span(match.start(), match.end(), self.source)
        self._define(
            base,
            Assign(
                base,
                Quantity(
                    base_count,
                    unit,
                    span=self._match_span(match, "base_count"),
                ),
                span=whole_span,
            ),
        )
        self._define(
            scaled,
            Affine(
                scaled,
                base,
                scale,
                Quantity(0, unit, span=self._match_span(match, "scale")),
                span=whole_span,
            ),
        )
        self._define(
            target,
            Affine(
                target,
                scaled,
                target_equivalent / source_equivalent,
                Quantity(
                    0,
                    unit,
                    span=self._match_span(match, "equivalent_target_count"),
                ),
                span=whole_span,
            ),
        )
        self.targets.append(target)
        return True

    def _parse_flow_timeline(self) -> bool:
        """Compile one explicitly ordered multiply-then-increment state chain."""

        match = _FLOW_TIMELINE.fullmatch(self.source)
        if match is None:
            return False
        if not _same(
            match.group("initial_material"),
            match.group("scaled_material"),
            match.group("increment_material"),
            match.group("query_material"),
        ):
            raise _Abort(ParseStatus.AMBIGUOUS, "flow materials do not agree")
        if not _same(
            match.group("initial_channel"),
            match.group("scaled_channel"),
            match.group("increment_channel"),
            match.group("query_channel"),
        ):
            raise _Abort(ParseStatus.AMBIGUOUS, "flow channels do not agree")
        unit_name = _singular(match.group("initial_unit"))
        if unit_name != _singular(match.group("increment_unit")):
            raise _Abort(ParseStatus.AMBIGUOUS, "flow units do not agree")
        if _singular(match.group("elapsed_unit")) != _singular(
            match.group("third_unit")
        ):
            raise _Abort(ParseStatus.AMBIGUOUS, "flow timeline units do not agree")

        multiplier = {
            "doubles": Fraction(2),
            "triples": Fraction(3),
            "quadruples": Fraction(4),
        }[match.group("multiplier").casefold()]
        initial_value = _number(match.group("initial"))
        increment_value = _number(match.group("increment"))
        if initial_value <= 0:
            raise _Abort(ParseStatus.INVALID, "initial flow must be positive")

        unit = Unit.base("volume", symbol=unit_name)
        prefix = (
            f"flow.{_entity(match.group('initial_material'))}."
            f"{_entity(match.group('initial_channel'))}"
        )
        initial = self._variable(
            f"{prefix}.initial",
            unit,
            span=self._match_span(match, "initial"),
        )
        after_event = self._variable(
            f"{prefix}.after_event",
            unit,
            span=self._match_span(match, "multiplier"),
        )
        final = self._variable(
            f"{prefix}.final",
            unit,
            span=self._match_span(match, "query_material"),
        )
        whole_span = Span(match.start(), match.end(), self.source)
        self._define(
            initial,
            Assign(
                initial,
                Quantity(
                    initial_value,
                    unit,
                    span=self._match_span(match, "initial"),
                ),
                span=whole_span,
            ),
        )
        self._define(
            after_event,
            Affine(
                after_event,
                initial,
                multiplier,
                Quantity(0, unit, span=self._match_span(match, "multiplier")),
                span=whole_span,
            ),
        )
        self._define(
            final,
            Affine(
                final,
                after_event,
                Fraction(1),
                Quantity(
                    increment_value,
                    unit,
                    span=self._match_span(match, "increment"),
                ),
                span=whole_span,
            ),
        )
        self.targets.append(final)
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

    def _parse_recurrence(self) -> bool:
        """Compile one closed finite affine recurrence into exact linear IR."""

        clauses = _sentences(self.source)
        starts: list[tuple[_Clause, re.Match[str]]] = []
        rules: list[tuple[str, _Clause, re.Match[str]]] = []
        queries: list[tuple[str, _Clause, re.Match[str]]] = []
        matched_clause_count = 0

        for clause in clauses:
            text = clause.text.strip()
            match = _RECURRENCE_START.fullmatch(text)
            if match is not None:
                starts.append((clause, match))
                matched_clause_count += 1
                continue

            for kind, pattern in (
                ("affine", _RECURRENCE_AFFINE),
                ("current_percent", _RECURRENCE_CURRENT_PERCENT),
                ("original_percent", _RECURRENCE_ORIGINAL_PERCENT),
            ):
                match = pattern.fullmatch(text)
                if match is not None:
                    rules.append((kind, clause, match))
                    matched_clause_count += 1
                    break
            else:
                match = _RECURRENCE_BARE_PERCENT.fullmatch(text)
                if match is not None:
                    raise _Abort(
                        ParseStatus.AMBIGUOUS,
                        "percentage recurrence must name current or original basis",
                    )
                for kind, pattern in (
                    ("value", _RECURRENCE_VALUE_QUERY),
                    ("change", _RECURRENCE_CHANGE_QUERY),
                    ("sum", _RECURRENCE_SUM_QUERY),
                ):
                    match = pattern.fullmatch(text)
                    if match is not None:
                        queries.append((kind, clause, match))
                        matched_clause_count += 1
                        break

        if matched_clause_count == 0:
            return False
        if matched_clause_count != len(clauses):
            raise _Abort(
                ParseStatus.UNSUPPORTED,
                "recurrence contains a clause outside the closed grammar",
            )
        if len(starts) != 1 or len(rules) != 1 or len(queries) != 1:
            status = (
                ParseStatus.AMBIGUOUS
                if any(len(group) > 1 for group in (starts, rules, queries))
                else ParseStatus.UNSUPPORTED
            )
            raise _Abort(
                status,
                "recurrence requires exactly one start, rule, and target clause",
            )

        start_clause, start_match = starts[0]
        rule_kind, rule_clause, rule_match = rules[0]
        query_kind, query_clause, query_match = queries[0]
        owner = start_match.group("owner")
        rule_owner_groups = {
            "affine": ("next_owner", "current_owner"),
            "current_percent": ("next_owner", "base_owner", "percent_owner"),
            "original_percent": ("next_owner", "base_owner", "percent_owner"),
        }[rule_kind]
        owners = (
            owner,
            *(rule_match.group(group) for group in rule_owner_groups),
            query_match.group("owner"),
        )
        if not _same(*owners):
            raise _Abort(ParseStatus.AMBIGUOUS, "recurrence owners do not agree")

        start_index = _step_index(start_match.group("index"))
        end_index = _step_index(query_match.group("end"))
        if query_kind != "value":
            query_start = _step_index(query_match.group("start"))
            if query_start != start_index:
                raise _Abort(
                    ParseStatus.AMBIGUOUS,
                    "recurrence target range does not start at the declared index",
                )
        if end_index < start_index:
            raise _Abort(ParseStatus.INVALID, "recurrence target precedes its start")
        step_count = end_index - start_index
        if step_count > _MAX_RECURRENCE_STEPS:
            raise _Abort(
                ParseStatus.UNSUPPORTED,
                f"recurrence exceeds {_MAX_RECURRENCE_STEPS} exact steps",
            )

        unit = Unit.scalar()
        prefix = f"recurrence.{_entity(owner)}"
        start_value = _signed_number(start_match.group("value"))
        start_value_span = start_clause.group_span(start_match, "value", self.source)

        def state(index: int, span: Span) -> Variable:
            return self._variable(f"{prefix}.x[{index}]", unit, span=span)

        states = [state(start_index, start_value_span)]
        self._define(
            states[0],
            Assign(
                states[0],
                Quantity(start_value, unit, span=start_value_span),
                span=start_clause.span(self.source),
            ),
        )

        offset = _number(rule_match.group("offset"))
        if rule_match.group("offset_direction").casefold() == "minus":
            offset = -offset
        offset_quantity = Quantity(
            offset,
            unit,
            span=rule_clause.group_span(rule_match, "offset", self.source),
        )
        rule_span = rule_clause.span(self.source)

        original_growth: Variable | None = None
        if rule_kind == "original_percent":
            percent = _number(rule_match.group("percent")) / 100
            percent_span = rule_clause.group_span(rule_match, "percent", self.source)
            original_growth = self._variable(
                f"{prefix}.original_growth",
                unit,
                span=percent_span,
            )
            self._define(
                original_growth,
                Part(
                    original_growth,
                    states[0],
                    percent,
                    span=rule_span,
                ),
            )

        for index in range(start_index, end_index):
            current = states[-1]
            following = state(index + 1, rule_span)
            if rule_kind == "affine":
                self._define(
                    following,
                    Affine(
                        following,
                        current,
                        _signed_number(rule_match.group("factor")),
                        offset_quantity,
                        span=rule_span,
                    ),
                )
            elif rule_kind == "current_percent":
                percent = _number(rule_match.group("percent")) / 100
                growth = self._variable(
                    f"{prefix}.current_growth[{index}]",
                    unit,
                    span=rule_clause.group_span(rule_match, "percent", self.source),
                )
                self._define(
                    growth,
                    Part(growth, current, percent, span=rule_span),
                )
                self._define(
                    following,
                    Sum(
                        following,
                        (current, growth, offset_quantity),
                        span=rule_span,
                    ),
                )
            else:
                assert original_growth is not None
                self._define(
                    following,
                    Sum(
                        following,
                        (current, original_growth, offset_quantity),
                        span=rule_span,
                    ),
                )
            states.append(following)

        if query_kind == "value":
            target = states[-1]
        elif query_kind == "change":
            target = self._variable(
                f"{prefix}.change[{start_index}:{end_index}]",
                unit,
                span=query_clause.span(self.source),
            )
            self._define(
                target,
                Balance(
                    (target, states[0]),
                    (states[-1],),
                    span=query_clause.span(self.source),
                ),
            )
        else:
            target = self._variable(
                f"{prefix}.sum[{start_index}:{end_index}]",
                unit,
                span=query_clause.span(self.source),
            )
            self._define(
                target,
                Sum(target, tuple(states), span=query_clause.span(self.source)),
            )
        self.targets.append(target)
        return True

    def _parse_closed_family(self) -> bool:
        parsers = (
            self._parse_resource_use_balance,
            self._parse_component_ratio_system,
            self._parse_period_score_system,
            self._parse_satiety_equivalence_chain,
            self._parse_flow_timeline,
            self._parse_recurrence,
            self._parse_direct_rate,
            self._parse_part_inventory,
            self._parse_original_length_part,
            self._parse_inventory_balance,
            self._parse_bus_balance,
            self._parse_score_mean,
        )
        return any(parser() for parser in parsers)

    def _parse_pronominal_declaration(self, clause: _Clause) -> bool:
        """Parse only typed object-pronoun relations with a unique antecedent."""

        text = clause.text.strip()

        match = _COUNT_AFFINE_PRONOUN_SOURCE.fullmatch(text)
        if match:
            noun = _singular(match.group("noun"))
            unit = Unit.count(symbol=noun)
            target_span = clause.group_span(match, "target", self.source)
            target = self._variable(
                self._property_name(match.group("target"), noun),
                unit,
                span=target_span,
                count=True,
            )
            target_symbol = self._register_named_count(
                match.group("target"), noun, target, target_span
            )
            # A non-reflexive comparative object cannot denote this clause's
            # subject; that reading would require a reflexive surface form.
            source = self._resolve_typed_pronoun(
                match.group("source_pronoun"),
                noun,
                unit,
                clause.group_span(match, "source_pronoun", self.source),
                role="object",
                excluded=frozenset({target_symbol}),
            )
            offset = self._quantity(match, "offset", clause, unit)
            if match.group("direction").casefold() == "fewer":
                offset = Quantity(-offset.value, unit, span=offset.span)
            self._define(
                target, Affine(target, source, 1, offset, span=clause.span(self.source))
            )
            return True

        raise _Abort(ParseStatus.AMBIGUOUS, "numeric pronoun binding is not proven")

    def _parse_declaration(self, clause: _Clause) -> bool:
        text = clause.text.strip()
        if _COUNT_AFFINE_PRONOUN_SOURCE.fullmatch(text):
            return self._parse_pronominal_declaration(clause)
        if _PRONOUN.search(text) and _NUMERIC.search(text):
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
            entity_span = clause.group_span(match, "entity", self.source)
            variable = self._variable(
                self._property_name(match.group("entity"), noun),
                unit,
                span=entity_span,
                count=True,
            )
            self._register_named_count(
                match.group("entity"), noun, variable, entity_span
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
            target_span = clause.group_span(match, "target", self.source)
            target = self._variable(
                self._property_name(match.group("target"), noun),
                unit,
                span=target_span,
                count=True,
            )
            source_span = clause.group_span(match, "source", self.source)
            source = self._variable(
                self._property_name(match.group("source"), noun),
                unit,
                span=source_span,
                count=True,
            )
            self._register_named_count(match.group("target"), noun, target, target_span)
            self._register_named_count(match.group("source"), noun, source, source_span)
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
            target_span = clause.group_span(match, "target", self.source)
            target = self._variable(
                self._property_name(match.group("target"), noun),
                unit,
                span=target_span,
                count=True,
            )
            source_span = clause.group_span(match, "source", self.source)
            source = self._variable(
                self._property_name(match.group("source"), noun),
                unit,
                span=source_span,
                count=True,
            )
            self._register_named_count(match.group("target"), noun, target, target_span)
            self._register_named_count(match.group("source"), noun, source, source_span)
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

    def _parse_pronominal_question(self, clause: _Clause) -> bool:
        """Bind a typed singular question target only when it is unique."""

        text = clause.text.strip()

        match = _COUNT_PRONOUN_QUERY.fullmatch(text)
        if match:
            pronoun = _entity(match.group("entity_pronoun"))
            auxiliary = _entity(match.group("auxiliary"))
            expected_auxiliaries = (
                {"do", "did"}
                if pronoun == "they"
                else {
                    "does",
                    "did",
                }
            )
            if auxiliary not in expected_auxiliaries:
                raise _Abort(
                    ParseStatus.AMBIGUOUS,
                    "question pronoun binding is not proven",
                )
            noun = _singular(match.group("noun"))
            unit = Unit.count(symbol=noun)
            self.targets.append(
                self._resolve_typed_pronoun(
                    match.group("entity_pronoun"),
                    noun,
                    unit,
                    clause.group_span(match, "entity_pronoun", self.source),
                    role="subject",
                    question=True,
                )
            )
            return True

        raise _Abort(ParseStatus.AMBIGUOUS, "question pronoun binding is not proven")

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
            return self._parse_pronominal_question(clause)

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
            for variable in variables_in_constraint(constraint)
        }
        if target not in mentioned:
            raise _Abort(ParseStatus.UNSUPPORTED, "target has no structural evidence")
        problem = Problem(
            tuple(self.variables.values()), tuple(self.constraints), target
        )
        return ParseResult(ParseStatus.PARSED, problem)


def parse_structural_problem(source: str) -> ParseResult:
    """Parse *source* without ever returning a partial numeric graph."""

    primary = StructuralParser(source).parse()
    if primary.ok:
        return primary
    signed = compile_signed_events(source)
    if signed.ok and signed.compiled is not None:
        return ParseResult(
            ParseStatus.PARSED,
            signed.compiled.problem,
            reason=f"evidence-closed signed event grammar: {signed.family}",
        )
    if signed.family_matched and signed.status in {
        FrontendStatus.AMBIGUOUS,
        FrontendStatus.INVALID,
    }:
        status = (
            ParseStatus.AMBIGUOUS
            if signed.status is FrontendStatus.AMBIGUOUS
            else ParseStatus.INVALID
        )
        return ParseResult(status, reason=signed.reason)
    if primary.status is not ParseStatus.UNSUPPORTED:
        return primary
    compiled = compile_clauses(source)
    if not compiled.ok or compiled.problem is None:
        return primary
    return ParseResult(
        ParseStatus.PARSED,
        compiled.problem,
        reason="evidence-closed clause compiler",
    )


__all__ = [
    "ParseResult",
    "ParseStatus",
    "StructuralParser",
    "parse_structural_problem",
]
