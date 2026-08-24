"""Evidence-closed clause compiler for exact affine entity systems.

The compiler recognises local clauses and then lowers their typed relations to
the prompt-independent :mod:`arithmetic_ir`.  It deliberately does not match a
whole benchmark prompt.  Every digit-based numeric surface is entered in an
evidence ledger and a problem is returned only when each surface is consumed by
exactly one accepted relation.

The complete families cover affine entity counts and typed rate ledgers.  A
rate ledger binds local unit-price declarations to local acquisitions, lowers
each price-times-quantity line through :class:`~arithmetic_ir.Rate`, and sums
the line amounts into one explicit cost target.  Unsupported or ambiguous
language produces a typed abstention result rather than partial IR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from fractions import Fraction
import re
import unicodedata

from .arithmetic_ir import (
    Affine,
    ArithmeticProblem,
    Assign,
    Constraint,
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
_INTEGER = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
_DECIMAL = rf"(?:{_INTEGER}(?:\.\d+)?|\.\d+)"
_NUMBER = (
    rf"[+-]?(?:{_INTEGER}\s*/\s*{_INTEGER}|{_DECIMAL}|[\u00bc-\u00be\u2150-\u215e])"
)
_NUMERIC_SURFACE = re.compile(
    rf"(?<![\w.])(?P<number>{_NUMBER})(?!\w|[.,]\d|\s*/\s*\d)"
)
_WORD = r"[A-Za-z][A-Za-z'\-’]*"
_NOUN = rf"{_WORD}(?:\s+{_WORD}){{0,5}}"
_NAME = r"(?-i:[A-Z][A-Za-z]*(?:['\-’][A-Za-z]+)*)"
_NAME_LIST = rf"{_NAME}(?:\s*,\s*{_NAME})*(?:\s*,?\s+(?:and|&)\s+{_NAME})?"


class CompileStatus(str, Enum):
    """Closed outcome set for clause compilation."""

    COMPILED = "compiled"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class SymbolKey:
    """Semantic identity of one unknown, independent of its surface name."""

    owner: str
    property: str
    item: str
    scope: str
    state: str

    def __post_init__(self) -> None:
        for field_name in ("owner", "property", "item", "scope", "state"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"symbol {field_name} must be non-empty text")

    @property
    def variable_name(self) -> str:
        parts = (self.owner, self.property, self.item, self.scope, self.state)
        return ".".join(_identifier(part) for part in parts)


@dataclass(frozen=True, slots=True)
class NumericMention:
    """One exact numeric surface and its adjacent lexical unit text."""

    value: Fraction
    unit_text: str
    span: Span

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, Fraction)):
            raise TypeError("numeric mention value must be int or Fraction")
        if not isinstance(self.unit_text, str):
            raise TypeError("numeric mention unit_text must be text")
        if not isinstance(self.span, Span):
            raise TypeError("numeric mention span must be a Span")
        object.__setattr__(self, "value", Fraction(self.value))

    @property
    def unit(self) -> str:
        """Compatibility alias for callers that use the shorter field name."""

        return self.unit_text

    @property
    def text(self) -> str:
        return self.span.source[self.span.start : self.span.end]


@dataclass(frozen=True, slots=True)
class RateScope:
    """Lexically attested scope shared by one local block of unit prices."""

    group: str | None
    label: str | None
    venue: str | None
    span: Span = field(compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.group is not None and not self.group.strip():
            raise ValueError("rate-scope group must be non-empty when present")
        if self.label is not None and not self.label.strip():
            raise ValueError("rate-scope label must be non-empty when present")
        if self.venue is not None and not self.venue.strip():
            raise ValueError("rate-scope venue must be non-empty when present")
        if not isinstance(self.span, Span):
            raise TypeError("rate-scope span must be a Span")


RelationArgument = SymbolKey | RateScope | Fraction | str | tuple[SymbolKey, ...]


@dataclass(frozen=True, slots=True)
class Relation:
    """A typed local proposition with complete numeric provenance."""

    kind: str
    args: tuple[RelationArgument, ...]
    span: Span
    numeric_mentions: tuple[NumericMention, ...] = ()

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("relation kind must not be empty")
        if not isinstance(self.span, Span):
            raise TypeError("relation span must be a Span")
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "numeric_mentions", tuple(self.numeric_mentions))


@dataclass(frozen=True, slots=True)
class TargetRelation:
    """The single explicit question target."""

    symbol: SymbolKey
    span: Span
    numeric_mentions: tuple[NumericMention, ...] = ()

    @property
    def target(self) -> SymbolKey:
        return self.symbol


@dataclass(frozen=True, slots=True)
class CompileDiagnostics:
    """Evidence and reason attached to either compilation or abstention."""

    status: CompileStatus
    reason: str = ""
    messages: tuple[str, ...] = ()
    observed_numeric: tuple[NumericMention, ...] = ()
    consumed_numeric: tuple[NumericMention, ...] = ()
    unconsumed_numeric: tuple[NumericMention, ...] = ()
    duplicate_numeric: tuple[NumericMention, ...] = ()
    unsupported_spans: tuple[Span, ...] = ()
    ambiguous_spans: tuple[Span, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status is CompileStatus.COMPILED


@dataclass(frozen=True, slots=True)
class CompileResult:
    """Clause relations and their lowered arithmetic problem, if accepted."""

    relations: tuple[Relation, ...]
    target: TargetRelation | None
    problem: ArithmeticProblem | None
    diagnostics: CompileDiagnostics

    @property
    def status(self) -> CompileStatus:
        return self.diagnostics.status

    @property
    def reason(self) -> str:
        return self.diagnostics.reason

    @property
    def ok(self) -> bool:
        return self.diagnostics.ok and self.problem is not None

    @property
    def compiled(self) -> bool:
        return self.ok

    @property
    def abstained(self) -> bool:
        return not self.ok


def _number(raw: str) -> Fraction:
    text = raw.strip().replace(",", "")
    sign = Fraction(1)
    if text[:1] in {"+", "-"}:
        if text[0] == "-":
            sign = -sign
        text = text[1:].strip()
    if text in _UNICODE_FRACTIONS:
        return sign * _UNICODE_FRACTIONS[text]
    text = unicodedata.normalize("NFKC", text)
    try:
        if "/" in text:
            numerator, denominator = (part.strip() for part in text.split("/", 1))
            return sign * Fraction(Decimal(numerator)) / int(denominator)
        return sign * Fraction(Decimal(text))
    except (InvalidOperation, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid exact number: {raw!r}") from exc


def _surface_unit(source: str, start: int, end: int) -> str:
    prefix = source[max(0, start - 3) : start]
    currency = re.search(r"([$€£])\s*$", prefix)
    if currency:
        return currency.group(1)
    suffix = source[end:]
    percent = re.match(r"\s*(%)", suffix)
    if percent:
        return percent.group(1)
    word = re.match(rf"\s*({_WORD})", suffix)
    return word.group(1) if word else ""


class EvidenceLedger:
    """Discover and enforce exact one-time numeric-surface consumption."""

    def __init__(self, source: str) -> None:
        if not isinstance(source, str):
            raise TypeError("evidence source must be text")
        self.source = source
        mentions: list[NumericMention] = []
        invalid: list[Span] = []
        for match in _NUMERIC_SURFACE.finditer(source):
            span = Span(match.start("number"), match.end("number"), source)
            try:
                value = _number(match.group("number"))
            except ValueError:
                invalid.append(span)
                continue
            mentions.append(
                NumericMention(
                    value,
                    _surface_unit(source, span.start, span.end),
                    span,
                )
            )
        self.mentions = tuple(mentions)
        self.invalid_spans = tuple(invalid)
        self._by_span = {
            (mention.span.start, mention.span.end): mention for mention in self.mentions
        }
        self._counts = {key: 0 for key in self._by_span}
        self._unknown_consumptions: list[Span] = []

    def mention_at(self, start: int, end: int) -> NumericMention | None:
        return self._by_span.get((start, end))

    def consume(self, mention: NumericMention | Span, *, consumer: str = "") -> bool:
        """Consume one known surface; return ``False`` for duplicates/unknowns."""

        del consumer  # Reserved for richer audit labels without changing the API.
        span = mention.span if isinstance(mention, NumericMention) else mention
        key = (span.start, span.end)
        if key not in self._counts:
            self._unknown_consumptions.append(span)
            return False
        self._counts[key] += 1
        return self._counts[key] == 1

    def consume_span(
        self, start: int, end: int, *, consumer: str = ""
    ) -> NumericMention | None:
        mention = self.mention_at(start, end)
        if mention is None or not self.consume(mention, consumer=consumer):
            return None
        return mention

    def consumption_count(self, mention: NumericMention | Span) -> int:
        span = mention.span if isinstance(mention, NumericMention) else mention
        return self._counts.get((span.start, span.end), 0)

    @property
    def consumed(self) -> tuple[NumericMention, ...]:
        return tuple(
            mention for mention in self.mentions if self.consumption_count(mention) >= 1
        )

    @property
    def unconsumed(self) -> tuple[NumericMention, ...]:
        return tuple(
            mention for mention in self.mentions if self.consumption_count(mention) == 0
        )

    @property
    def duplicates(self) -> tuple[NumericMention, ...]:
        return tuple(
            mention for mention in self.mentions if self.consumption_count(mention) > 1
        )

    @property
    def closed(self) -> bool:
        return (
            not self.invalid_spans
            and not self._unknown_consumptions
            and not self.unconsumed
            and not self.duplicates
        )


@dataclass(frozen=True, slots=True)
class _Clause:
    text: str
    start: int
    end: int
    question: bool

    @property
    def key(self) -> tuple[int, int]:
        return (self.start, self.end)

    def span(self, source: str) -> Span:
        return Span(self.start, self.end, source)


class _CompileAbort(Exception):
    def __init__(
        self, status: CompileStatus, reason: str, span: Span | None = None
    ) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason
        self.span = span


def _identifier(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    result = re.sub(r"[^a-z0-9]+", "_", normalized.casefold()).strip("_")
    return result or "symbol"


def _singular_word(word: str) -> str:
    lowered = word.casefold()
    if lowered.endswith("ies") and len(lowered) > 3:
        return lowered[:-3] + "y"
    if lowered.endswith(("ches", "shes", "xes", "zes")):
        return lowered[:-2]
    if lowered.endswith("s") and not lowered.endswith(("ss", "us", "is")):
        return lowered[:-1]
    return lowered


def _normal_item(raw: str) -> str:
    words = re.findall(_WORD, raw.casefold())
    while words and words[0] in {"a", "an", "the", "each"}:
        words.pop(0)
    if not words:
        raise ValueError("item phrase must contain a noun")
    words[-1] = _singular_word(words[-1])
    return " ".join(words)


def _item_head(item: str) -> str:
    return item.rsplit(" ", 1)[-1]


def _normal_unit(raw: str) -> str:
    words = re.findall(_WORD, raw.casefold())
    if len(words) != 1:
        raise ValueError("rate unit must be one explicit unit word")
    return _singular_word(words[0])


_GENERIC_RATE_CONTEXT_WORDS = {
    "downtown",
    "large",
    "local",
    "nearby",
    "neighborhood",
    "new",
    "old",
    "small",
}


def _rate_context_group(raw: str) -> str | None:
    """Return only the explicit noun immediately modifying a venue."""

    words = re.findall(_WORD, raw.casefold())
    if not words:
        return None
    candidate = _singular_word(words[-1])
    if candidate in _GENERIC_RATE_CONTEXT_WORDS:
        return None
    return candidate


def _rate_context_label(raw: str) -> str | None:
    words = re.findall(_WORD, raw.casefold())
    return " ".join(words) or None


def _sentences(source: str) -> tuple[_Clause, ...]:
    clauses: list[_Clause] = []
    start = 0
    index = 0
    while index < len(source):
        char = source[index]
        decimal_point = (
            char == "."
            and index > 0
            and index + 1 < len(source)
            and source[index - 1].isdigit()
            and source[index + 1].isdigit()
        )
        if char not in "!?" and (char != "." or decimal_point):
            index += 1
            continue
        raw_start = start
        raw_end = index
        while raw_start < raw_end and source[raw_start].isspace():
            raw_start += 1
        while raw_end > raw_start and source[raw_end - 1].isspace():
            raw_end -= 1
        if raw_start < raw_end:
            clauses.extend(
                _split_coordinated(
                    _Clause(
                        source[raw_start:raw_end],
                        raw_start,
                        raw_end,
                        char == "?",
                    ),
                    source,
                )
            )
        start = index + 1
        index += 1
    raw_start = start
    while raw_start < len(source) and source[raw_start].isspace():
        raw_start += 1
    raw_end = len(source)
    while raw_end > raw_start and source[raw_end - 1].isspace():
        raw_end -= 1
    if raw_start < raw_end:
        clauses.extend(
            _split_coordinated(
                _Clause(source[raw_start:raw_end], raw_start, raw_end, False), source
            )
        )
    return tuple(clauses)


_COORDINATOR = re.compile(
    rf",\s+and\s+(?={_NAME}\s+(?:has|owns?|holds?|keeps?)\b)",
    re.IGNORECASE,
)
_RELATION_VERB = re.compile(r"\b(?:has|owns?|holds?|keeps?)\b", re.IGNORECASE)


def _split_coordinated(clause: _Clause, source: str) -> tuple[_Clause, ...]:
    if clause.question:
        return (clause,)
    pieces: list[_Clause] = []
    cursor = 0
    for match in _COORDINATOR.finditer(clause.text):
        prefix = clause.text[cursor : match.start()]
        if not _RELATION_VERB.search(prefix):
            continue
        left_start = clause.start + cursor
        left_end = clause.start + match.start()
        while left_end > left_start and source[left_end - 1].isspace():
            left_end -= 1
        pieces.append(_Clause(source[left_start:left_end], left_start, left_end, False))
        cursor = match.end()
    if not pieces:
        return (clause,)
    tail_start = clause.start + cursor
    while tail_start < clause.end and source[tail_start].isspace():
        tail_start += 1
    pieces.append(
        _Clause(source[tail_start : clause.end], tail_start, clause.end, False)
    )
    return tuple(pieces)


_CONDITIONAL_QUESTION = re.compile(
    r"^if\s+(?P<fact>.+),\s*(?P<question>how\s+much\b.+)$",
    re.IGNORECASE,
)


def _split_conditional_question(clause: _Clause, source: str) -> tuple[_Clause, ...]:
    """Split a local ``If <fact>, <target>?`` without parsing either side."""

    if not clause.question:
        return (clause,)
    match = _CONDITIONAL_QUESTION.fullmatch(clause.text)
    if match is None:
        return (clause,)
    fact_start = clause.start + match.start("fact")
    fact_end = clause.start + match.end("fact")
    target_start = clause.start + match.start("question")
    target_end = clause.start + match.end("question")
    return (
        _Clause(source[fact_start:fact_end], fact_start, fact_end, False),
        _Clause(source[target_start:target_end], target_start, target_end, True),
    )


_ENTITY_SCOPE = re.compile(
    rf"^(?P<owners>{_NAME_LIST})\s+(?:all\s+)?(?:have|own)\s+"
    rf"(?:(?:a|their)\s+)?(?:(?:collections?|sets?)\s+of\s+)?"
    rf"(?P<item>{_NOUN})$",
    re.IGNORECASE,
)
_ASSIGNMENT = re.compile(
    rf"^(?P<target>{_NAME})\s+(?:has|owns?|holds?|keeps?)\s+"
    rf"(?P<value>{_NUMBER})\s+"
    rf"(?!(?:more|fewer|less|times)\b)(?P<item>{_NOUN})$",
    re.IGNORECASE,
)
_ADDITIVE = re.compile(
    rf"^(?P<target>{_NAME})\s+(?:has|owns?|holds?|keeps?)\s+"
    rf"(?P<offset>{_NUMBER})\s+(?:"
    rf"(?P<direction_first>more|fewer|less)(?:\s+(?P<item_after>{_NOUN}))?"
    rf"|(?P<item_before>{_NOUN})\s+(?P<direction_last>more|fewer|less)"
    rf")\s+than\s+(?P<source>{_NAME})(?:\s+(?:does|has))?$",
    re.IGNORECASE,
)
_SCALED_AMOUNT = re.compile(
    rf"^(?P<target>{_NAME})\s+(?:has|owns?|holds?|keeps?)\s+"
    rf"(?P<scale>{_NUMBER})\s+times\s+(?:the\s+)?(?:amount|number)\s+of\s+"
    rf"(?P<item>{_NOUN})\s+that\s+(?P<source>{_NAME})\s+(?:has|owns?)$",
    re.IGNORECASE,
)
_SCALED_MANY = re.compile(
    rf"^(?P<target>{_NAME})\s+(?:has|owns?|holds?|keeps?)\s+"
    rf"(?P<scale>{_NUMBER})\s+times\s+as\s+many\s+(?P<item>{_NOUN})\s+"
    rf"as\s+(?P<source>{_NAME})(?:\s+(?:does|has))?$",
    re.IGNORECASE,
)
_SCALED_POSSESSIVE = re.compile(
    rf"^(?P<target>{_NAME})\s+(?:has|owns?|holds?|keeps?)\s+"
    rf"(?P<scale>{_NUMBER})\s+times\s+(?P<source>{_NAME})['’]s\s+"
    rf"(?P<item>{_NOUN})$",
    re.IGNORECASE,
)
_DIVIDED_AMOUNT = re.compile(
    rf"^(?P<target>{_NAME})\s+(?:has|owns?|holds?|keeps?)\s+"
    rf"(?:the\s+)?(?:amount|number)\s+of\s+(?P<item>{_NOUN})\s+that\s+"
    rf"(?P<source>{_NAME})\s+(?:has|owns?)\s+divided\s+by\s+"
    rf"(?P<divisor>{_NUMBER})$",
    re.IGNORECASE,
)
_DIVIDED_POSSESSIVE = re.compile(
    rf"^(?P<target>{_NAME})\s+(?:has|owns?|holds?|keeps?)\s+"
    rf"(?P<source>{_NAME})['’]s\s+(?P<item>{_NOUN})\s+divided\s+by\s+"
    rf"(?P<divisor>{_NUMBER})$",
    re.IGNORECASE,
)
_TOTAL_THERE = re.compile(
    rf"^there\s+(?:is|are)\s+(?:a\s+)?total\s+of\s+"
    rf"(?P<total>{_NUMBER})\s+(?P<item>{_NOUN})$",
    re.IGNORECASE,
)
_TOTAL_NUMBER = re.compile(
    rf"^the\s+total\s+(?:number|amount)\s+of\s+(?P<item>{_NOUN})\s+"
    rf"(?:is|equals)\s+(?P<total>{_NUMBER})$",
    re.IGNORECASE,
)
_TOTAL_OWNERS = re.compile(
    rf"^(?:(?:together|in\s+total)\s*,\s*)?(?P<owners>{_NAME_LIST})\s+"
    rf"(?:have|own)\s+(?P<total>{_NUMBER})\s+(?P<item>{_NOUN})\s+"
    rf"(?:in\s+total|altogether|combined)$",
    re.IGNORECASE,
)
_TARGET_COUNT = re.compile(
    rf"^how\s+many\s+(?P<item>{_NOUN})\s+(?:does|do)\s+"
    rf"(?P<owner>{_NAME})\s+(?:have|own|hold|keep)$",
    re.IGNORECASE,
)
_TARGET_NUMBER = re.compile(
    rf"^what\s+is\s+the\s+(?:number|amount)\s+of\s+(?P<item>{_NOUN})\s+"
    rf"(?:that\s+)?(?P<owner>{_NAME})\s+(?:has|owns|holds|keeps)$",
    re.IGNORECASE,
)
_RATE_CONTEXT_PREFIX = re.compile(
    r"^at\s+(?:the\s+)?(?P<label>[^,]{0,100}?)"
    r"(?P<venue>\b(?:orchard|market|store|shop))\s*,\s*",
    re.IGNORECASE,
)
_RATE_PICK_INTRO = re.compile(
    r"^(?:you|customers?|shoppers?)\s+could\s+(?:pick|buy)\s+"
    r"(?:(?:your|their)\s+own\s+)?",
    re.IGNORECASE,
)
_RATE_TERM = re.compile(
    rf"(?<![A-Za-z'\-’])(?!and\b)"
    rf"(?P<item>{_WORD}(?:\s+{_WORD}){{0,2}}?)\s+"
    r"(?:for|(?:were|was|are|is)(?:\s+priced\s+at)?|"
    r"(?:costs?|sells?)(?:\s+for)?)\s+"
    rf"(?:\$\s*(?P<price>{_NUMBER})|"
    rf"(?P<price_words>{_NUMBER})\s+dollars?)\s+per\s+"
    rf"(?P<unit>{_WORD})",
    re.IGNORECASE,
)
_ACQUISITION = re.compile(
    rf"^(?P<buyer>{_NAME})\s+(?:picked|bought|purchased)\s+"
    r"(?P<terms>.+)$",
    re.IGNORECASE,
)
_ACQUISITION_TERM = re.compile(
    rf"(?P<quantity>{_NUMBER})\s+(?P<unit>{_WORD})\s+of\s+"
    rf"(?P<item>{_WORD}(?:\s+{_WORD}){{0,2}}?)"
    r"(?=\s*(?:,|\band\b|$))",
    re.IGNORECASE,
)
_COST_TARGET = re.compile(
    rf"^how\s+much(?:\s+money)?\s+did\s+"
    rf"(?P<owner>he|she|they|{_NAME})\s+spend\s+(?:on|for)\s+"
    rf"(?P<item>{_NOUN})$",
    re.IGNORECASE,
)


def _is_complete_list(text: str, matches: tuple[re.Match[str], ...]) -> bool:
    if not matches:
        return False
    cursor = 0
    for index, match in enumerate(matches):
        gap = text[cursor : match.start()]
        if index == 0:
            if gap.strip():
                return False
        elif re.fullmatch(r"\s*(?:,|,?\s+and)\s*", gap, re.IGNORECASE) is None:
            return False
        cursor = match.end()
    return not text[cursor:].strip()


class ClauseCompiler:
    """Compile one source text using independent local clause predicates."""

    def __init__(self, source: str) -> None:
        if not isinstance(source, str):
            raise TypeError("source must be text")
        self.source = source
        self.ledger = EvidenceLedger(source)
        self.relations: list[Relation] = []
        self.targets: list[TargetRelation] = []
        self._owners: dict[str, str] = {}
        self._items: set[str] = set()
        self._scopes: list[tuple[tuple[str, ...], str]] = []
        self._scope_clause_keys: set[tuple[int, int]] = set()

    def compile(self) -> CompileResult:
        if not self.source.strip():
            return self._failure(CompileStatus.INVALID, "source must be non-empty text")
        if self.ledger.invalid_spans:
            return self._failure(
                CompileStatus.INVALID, "source contains an invalid number"
            )
        try:
            clauses = tuple(
                piece
                for clause in _sentences(self.source)
                for piece in _split_conditional_question(clause, self.source)
            )
            if not clauses:
                raise _CompileAbort(CompileStatus.INVALID, "source has no clauses")
            self._discover_entity_scopes(clauses)
            for clause in clauses:
                if clause.key in self._scope_clause_keys:
                    continue
                if clause.question:
                    targets = tuple(
                        target
                        for target in (
                            self._parse_target(clause),
                            self._parse_cost_target(clause),
                        )
                        if target is not None
                    )
                    if not targets:
                        raise _CompileAbort(
                            CompileStatus.UNSUPPORTED,
                            f"unsupported target at {clause.start}:{clause.end}",
                            clause.span(self.source),
                        )
                    if len(targets) != 1:
                        raise _CompileAbort(
                            CompileStatus.AMBIGUOUS,
                            f"target has {len(targets)} interpretations",
                            clause.span(self.source),
                        )
                    self.targets.append(targets[0])
                    continue
                candidate_sets: list[tuple[Relation, ...]] = []
                for relation in (
                    self._parse_assignment(clause),
                    self._parse_additive(clause),
                    self._parse_scaled(clause),
                    self._parse_divided(clause),
                    self._parse_total(clause),
                ):
                    if relation is not None:
                        candidate_sets.append((relation,))
                for relations in (
                    self._parse_rate_declarations(clause),
                    self._parse_acquisition(clause),
                ):
                    if relations is not None:
                        candidate_sets.append(relations)
                if not candidate_sets:
                    raise _CompileAbort(
                        CompileStatus.UNSUPPORTED,
                        f"unsupported clause at {clause.start}:{clause.end}",
                        clause.span(self.source),
                    )
                if len(candidate_sets) != 1:
                    raise _CompileAbort(
                        CompileStatus.AMBIGUOUS,
                        f"clause has {len(candidate_sets)} interpretations",
                        clause.span(self.source),
                    )
                for relation in candidate_sets[0]:
                    self._accept(relation)

            if len(self.targets) != 1:
                status = (
                    CompileStatus.AMBIGUOUS
                    if self.targets
                    else CompileStatus.UNSUPPORTED
                )
                raise _CompileAbort(status, "exactly one explicit target is required")
            if self.ledger.duplicates:
                raise _CompileAbort(
                    CompileStatus.INVALID,
                    "a numeric surface was consumed more than once",
                )
            if self.ledger.unconsumed:
                raise _CompileAbort(
                    CompileStatus.UNSUPPORTED,
                    "one or more numeric surfaces are unconsumed",
                )
            relations = self._materialize_totals(tuple(self.relations))
            target = self.targets[0]
            problem = self._lower(relations, target)
            diagnostics = self._diagnostics(CompileStatus.COMPILED, "")
            return CompileResult(relations, target, problem, diagnostics)
        except _CompileAbort as exc:
            return self._failure(exc.status, exc.reason, exc.span)
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            return self._failure(CompileStatus.INVALID, str(exc))

    def _discover_entity_scopes(self, clauses: tuple[_Clause, ...]) -> None:
        for clause in clauses:
            if clause.question:
                continue
            match = _ENTITY_SCOPE.fullmatch(clause.text)
            if match is None:
                continue
            owners = self._name_list(match.group("owners"), clause)
            item = self._resolve_item(match.group("item"), clause)
            owner_ids = tuple(
                self._owner(owner, clause, allow_new=True) for owner in owners
            )
            if len(owner_ids) < 2:
                raise _CompileAbort(
                    CompileStatus.AMBIGUOUS,
                    "entity scope must name at least two owners",
                    clause.span(self.source),
                )
            self._scopes.append((owner_ids, item))
            symbols = tuple(self._symbol(owner, item) for owner in owner_ids)
            self.relations.append(
                Relation("entity_scope", (symbols,), clause.span(self.source))
            )
            self._scope_clause_keys.add(clause.key)

    def _name_list(self, raw: str, clause: _Clause) -> tuple[str, ...]:
        parts = tuple(
            part.strip()
            for part in re.split(r"\s*,\s*(?:and\s+)?|\s+(?:and|&)\s+", raw)
            if part.strip()
        )
        if len({part.casefold() for part in parts}) != len(parts):
            raise _CompileAbort(
                CompileStatus.AMBIGUOUS,
                "entity list contains a duplicate owner",
                clause.span(self.source),
            )
        return parts

    def _owner(self, raw: str, clause: _Clause, *, allow_new: bool) -> str:
        key = raw.strip().casefold()
        existing = self._owners.get(key)
        if existing is not None:
            return existing
        if self._scopes and not allow_new:
            raise _CompileAbort(
                CompileStatus.AMBIGUOUS,
                f"owner {raw!r} is outside the declared entity scope",
                clause.span(self.source),
            )
        owner = raw.strip()
        self._owners[key] = owner
        return owner

    def _resolve_item(self, raw: str | None, clause: _Clause) -> str:
        if raw is None or not raw.strip():
            if len(self._items) != 1:
                raise _CompileAbort(
                    CompileStatus.AMBIGUOUS,
                    "clause omits an item in a multi-item or unknown scope",
                    clause.span(self.source),
                )
            return next(iter(self._items))
        candidate = _normal_item(raw)
        exact = candidate if candidate in self._items else None
        if exact is not None:
            return exact
        aliases = tuple(
            item
            for item in self._items
            if candidate == _item_head(candidate)
            and _item_head(item) == _item_head(candidate)
        )
        if len(aliases) == 1:
            return aliases[0]
        if len(aliases) > 1:
            raise _CompileAbort(
                CompileStatus.AMBIGUOUS,
                f"item phrase {raw!r} has multiple scoped antecedents",
                clause.span(self.source),
            )
        self._items.add(candidate)
        return candidate

    def _symbol(self, owner: str, item: str) -> SymbolKey:
        return SymbolKey(owner, "count", item, "collection", "current")

    def _mention(
        self, match: re.Match[str], group: str, clause: _Clause
    ) -> NumericMention:
        start = clause.start + match.start(group)
        end = clause.start + match.end(group)
        mention = self.ledger.mention_at(start, end)
        if mention is None:
            raise _CompileAbort(
                CompileStatus.INVALID,
                f"numeric group {group!r} does not match the evidence ledger",
                clause.span(self.source),
            )
        return mention

    def _mention_at(self, start: int, end: int, evidence_span: Span) -> NumericMention:
        mention = self.ledger.mention_at(start, end)
        if mention is None:
            raise _CompileAbort(
                CompileStatus.INVALID,
                "numeric term does not match the evidence ledger",
                evidence_span,
            )
        return mention

    def _relation(
        self,
        kind: str,
        args: tuple[RelationArgument, ...],
        clause: _Clause,
        match: re.Match[str],
        numeric_groups: tuple[str, ...],
    ) -> Relation:
        return Relation(
            kind,
            args,
            clause.span(self.source),
            tuple(self._mention(match, group, clause) for group in numeric_groups),
        )

    def _parse_assignment(self, clause: _Clause) -> Relation | None:
        match = _ASSIGNMENT.fullmatch(clause.text)
        if match is None:
            return None
        item = self._resolve_item(match.group("item"), clause)
        owner = self._owner(match.group("target"), clause, allow_new=not self._scopes)
        mention = self._mention(match, "value", clause)
        if mention.value < 0:
            raise _CompileAbort(
                CompileStatus.INVALID,
                "count assignments must be non-negative",
                clause.span(self.source),
            )
        return self._relation(
            "assign",
            (self._symbol(owner, item), mention.value),
            clause,
            match,
            ("value",),
        )

    def _parse_additive(self, clause: _Clause) -> Relation | None:
        match = _ADDITIVE.fullmatch(clause.text)
        if match is None:
            return None
        raw_item = match.group("item_after") or match.group("item_before")
        item = self._resolve_item(raw_item, clause)
        target_owner = self._owner(
            match.group("target"), clause, allow_new=not self._scopes
        )
        source_owner = self._owner(
            match.group("source"), clause, allow_new=not self._scopes
        )
        mention = self._mention(match, "offset", clause)
        if mention.value < 0:
            raise _CompileAbort(
                CompileStatus.INVALID,
                "comparison magnitudes must be non-negative",
                clause.span(self.source),
            )
        direction = match.group("direction_first") or match.group("direction_last")
        offset = mention.value if direction.casefold() == "more" else -mention.value
        return self._relation(
            "affine",
            (
                self._symbol(target_owner, item),
                self._symbol(source_owner, item),
                Fraction(1),
                offset,
            ),
            clause,
            match,
            ("offset",),
        )

    def _parse_scaled(self, clause: _Clause) -> Relation | None:
        matches = tuple(
            match
            for pattern in (_SCALED_AMOUNT, _SCALED_MANY, _SCALED_POSSESSIVE)
            if (match := pattern.fullmatch(clause.text)) is not None
        )
        if not matches:
            return None
        if len(matches) != 1:
            raise _CompileAbort(
                CompileStatus.AMBIGUOUS,
                "scalar clause has multiple local parses",
                clause.span(self.source),
            )
        match = matches[0]
        item = self._resolve_item(match.group("item"), clause)
        target_owner = self._owner(
            match.group("target"), clause, allow_new=not self._scopes
        )
        source_owner = self._owner(
            match.group("source"), clause, allow_new=not self._scopes
        )
        mention = self._mention(match, "scale", clause)
        if mention.value <= 0:
            raise _CompileAbort(
                CompileStatus.INVALID,
                "affine scale must be positive",
                clause.span(self.source),
            )
        return self._relation(
            "affine",
            (
                self._symbol(target_owner, item),
                self._symbol(source_owner, item),
                mention.value,
                Fraction(0),
            ),
            clause,
            match,
            ("scale",),
        )

    def _parse_divided(self, clause: _Clause) -> Relation | None:
        matches = tuple(
            match
            for pattern in (_DIVIDED_AMOUNT, _DIVIDED_POSSESSIVE)
            if (match := pattern.fullmatch(clause.text)) is not None
        )
        if not matches:
            return None
        if len(matches) != 1:
            raise _CompileAbort(
                CompileStatus.AMBIGUOUS,
                "division clause has multiple local parses",
                clause.span(self.source),
            )
        match = matches[0]
        item = self._resolve_item(match.group("item"), clause)
        target_owner = self._owner(
            match.group("target"), clause, allow_new=not self._scopes
        )
        source_owner = self._owner(
            match.group("source"), clause, allow_new=not self._scopes
        )
        mention = self._mention(match, "divisor", clause)
        if mention.value <= 0:
            raise _CompileAbort(
                CompileStatus.INVALID,
                "affine divisor must be positive",
                clause.span(self.source),
            )
        return self._relation(
            "affine",
            (
                self._symbol(target_owner, item),
                self._symbol(source_owner, item),
                Fraction(1, 1) / mention.value,
                Fraction(0),
            ),
            clause,
            match,
            ("divisor",),
        )

    def _parse_total(self, clause: _Clause) -> Relation | None:
        matches = tuple(
            match
            for pattern in (_TOTAL_THERE, _TOTAL_NUMBER, _TOTAL_OWNERS)
            if (match := pattern.fullmatch(clause.text)) is not None
        )
        if not matches:
            return None
        if len(matches) != 1:
            raise _CompileAbort(
                CompileStatus.AMBIGUOUS,
                "total clause has multiple local parses",
                clause.span(self.source),
            )
        match = matches[0]
        item = self._resolve_item(match.group("item"), clause)
        raw_owners = match.groupdict().get("owners")
        if raw_owners:
            owners = self._name_list(raw_owners, clause)
            members = tuple(
                self._symbol(
                    self._owner(owner, clause, allow_new=not self._scopes), item
                )
                for owner in owners
            )
        else:
            members = tuple(
                self._symbol(owner, item)
                for owner_ids, scoped_item in self._scopes
                if scoped_item == item
                for owner in owner_ids
            )
        mention = self._mention(match, "total", clause)
        if mention.value < 0:
            raise _CompileAbort(
                CompileStatus.INVALID,
                "explicit total must be non-negative",
                clause.span(self.source),
            )
        return self._relation(
            "total", (members, item, mention.value), clause, match, ("total",)
        )

    def _parse_target(self, clause: _Clause) -> TargetRelation | None:
        matches = tuple(
            match
            for pattern in (_TARGET_COUNT, _TARGET_NUMBER)
            if (match := pattern.fullmatch(clause.text)) is not None
        )
        if not matches:
            return None
        if len(matches) != 1:
            raise _CompileAbort(
                CompileStatus.AMBIGUOUS,
                "target clause has multiple local parses",
                clause.span(self.source),
            )
        match = matches[0]
        item = self._resolve_item(match.group("item"), clause)
        owner = self._owner(match.group("owner"), clause, allow_new=not self._scopes)
        return TargetRelation(self._symbol(owner, item), clause.span(self.source))

    def _parse_rate_declarations(self, clause: _Clause) -> tuple[Relation, ...] | None:
        body_offset = 0
        context = _RATE_CONTEXT_PREFIX.match(clause.text)
        group: str | None = None
        label: str | None = None
        venue: str | None = None
        if context is not None:
            body_offset = context.end()
            group = _rate_context_group(context.group("label"))
            label = _rate_context_label(context.group("label"))
            venue = _singular_word(context.group("venue"))
        body = clause.text[body_offset:]
        intro = _RATE_PICK_INTRO.match(body)
        if intro is not None:
            body_offset += intro.end()
            body = clause.text[body_offset:]
        matches = tuple(_RATE_TERM.finditer(body))
        if not _is_complete_list(body, matches):
            return None

        scope_end = clause.start + (
            context.end() if context is not None else clause.end
        )
        scope = RateScope(
            group,
            label,
            venue,
            Span(clause.start, scope_end, self.source),
        )

        relations: list[Relation] = []
        seen: set[str] = set()
        for match in matches:
            term_start = clause.start + body_offset + match.start()
            term_end = clause.start + body_offset + match.end()
            term_span = Span(term_start, term_end, self.source)
            item = self._resolve_item(match.group("item"), clause)
            if item in seen:
                raise _CompileAbort(
                    CompileStatus.AMBIGUOUS,
                    f"duplicate unit price for {item}",
                    term_span,
                )
            seen.add(item)
            price_group = "price" if match.group("price") is not None else "price_words"
            price_start = clause.start + body_offset + match.start(price_group)
            price_end = clause.start + body_offset + match.end(price_group)
            mention = self._mention_at(price_start, price_end, term_span)
            if mention.value <= 0:
                raise _CompileAbort(
                    CompileStatus.INVALID,
                    "unit prices must be positive",
                    term_span,
                )
            unit = _normal_unit(match.group("unit"))
            symbol = SymbolKey("vendor", "unit_price", item, "purchase", "offered")
            relations.append(
                Relation(
                    "unit_rate",
                    (symbol, mention.value, unit, "USD", scope),
                    term_span,
                    (mention,),
                )
            )
        return tuple(relations)

    def _parse_acquisition(self, clause: _Clause) -> tuple[Relation, ...] | None:
        match = _ACQUISITION.fullmatch(clause.text)
        if match is None:
            return None
        terms = match.group("terms")
        term_matches = tuple(_ACQUISITION_TERM.finditer(terms))
        if not _is_complete_list(terms, term_matches):
            return None
        buyer = self._owner(match.group("buyer"), clause, allow_new=not self._scopes)
        terms_start = clause.start + match.start("terms")
        relations: list[Relation] = []
        seen: set[str] = set()
        for term_match in term_matches:
            term_start = terms_start + term_match.start()
            term_end = terms_start + term_match.end()
            term_span = Span(term_start, term_end, self.source)
            item = self._resolve_item(term_match.group("item"), clause)
            if item in seen:
                raise _CompileAbort(
                    CompileStatus.AMBIGUOUS,
                    f"duplicate acquisition item {item}",
                    term_span,
                )
            seen.add(item)
            quantity_start = terms_start + term_match.start("quantity")
            quantity_end = terms_start + term_match.end("quantity")
            mention = self._mention_at(quantity_start, quantity_end, term_span)
            if mention.value < 0:
                raise _CompileAbort(
                    CompileStatus.INVALID,
                    "acquisition quantities must be non-negative",
                    term_span,
                )
            unit = _normal_unit(term_match.group("unit"))
            symbol = SymbolKey(buyer, "quantity", item, "purchase", "acquired")
            relations.append(
                Relation(
                    "acquire",
                    (symbol, mention.value, unit),
                    term_span,
                    (mention,),
                )
            )
        return tuple(relations)

    def _parse_cost_target(self, clause: _Clause) -> TargetRelation | None:
        match = _COST_TARGET.fullmatch(clause.text)
        if match is None:
            return None
        buyers = {
            _as_symbol(relation.args[0]).owner
            for relation in self.relations
            if relation.kind == "acquire"
        }
        raw_owner = match.group("owner")
        if raw_owner.casefold() in {"he", "she", "they"}:
            if len(buyers) != 1:
                raise _CompileAbort(
                    CompileStatus.AMBIGUOUS,
                    "cost-target pronoun requires one local acquisition buyer",
                    clause.span(self.source),
                )
            owner = next(iter(buyers))
        else:
            owner = self._owner(raw_owner, clause, allow_new=not self._scopes)
            if buyers and owner not in buyers:
                raise _CompileAbort(
                    CompileStatus.AMBIGUOUS,
                    "cost target does not match the acquisition buyer",
                    clause.span(self.source),
                )
        item = _normal_item(match.group("item"))
        symbol = SymbolKey(owner, "cost", item, "purchase", "total")
        return TargetRelation(symbol, clause.span(self.source))

    def _accept(self, relation: Relation) -> None:
        for mention in relation.numeric_mentions:
            if not self.ledger.consume(mention, consumer=relation.kind):
                raise _CompileAbort(
                    CompileStatus.INVALID,
                    "numeric evidence was consumed more than once",
                    relation.span,
                )
        self.relations.append(relation)

    def _materialize_totals(
        self, relations: tuple[Relation, ...]
    ) -> tuple[Relation, ...]:
        fact_symbols: set[SymbolKey] = set()
        for relation in relations:
            if relation.kind == "assign":
                fact_symbols.add(_as_symbol(relation.args[0]))
            elif relation.kind == "affine":
                fact_symbols.update(
                    (_as_symbol(relation.args[0]), _as_symbol(relation.args[1]))
                )
        materialized: list[Relation] = []
        for relation in relations:
            if relation.kind != "total":
                materialized.append(relation)
                continue
            members = _as_symbols(relation.args[0])
            item = _as_text(relation.args[1])
            if not members:
                members = tuple(
                    sorted(
                        (symbol for symbol in fact_symbols if symbol.item == item),
                        key=lambda symbol: symbol.variable_name,
                    )
                )
            if len(members) < 2 or len(set(members)) != len(members):
                raise _CompileAbort(
                    CompileStatus.AMBIGUOUS,
                    "an explicit total requires at least two distinct scoped entities",
                    relation.span,
                )
            materialized.append(
                Relation(
                    relation.kind,
                    (members, item, relation.args[2]),
                    relation.span,
                    relation.numeric_mentions,
                )
            )
        return tuple(materialized)

    def _lower(
        self, relations: tuple[Relation, ...], target: TargetRelation
    ) -> Problem:
        if target.symbol.property == "cost":
            return self._lower_rate_ledger(relations, target)
        if any(relation.kind in {"unit_rate", "acquire"} for relation in relations):
            raise _CompileAbort(
                CompileStatus.AMBIGUOUS,
                "affine and rate-ledger families cannot share one target",
                target.span,
            )
        referenced: set[SymbolKey] = set()
        for relation in relations:
            if relation.kind == "assign":
                referenced.add(_as_symbol(relation.args[0]))
            elif relation.kind == "affine":
                referenced.update(
                    (_as_symbol(relation.args[0]), _as_symbol(relation.args[1]))
                )
            elif relation.kind == "total":
                referenced.update(_as_symbols(relation.args[0]))
        if target.symbol not in referenced:
            raise _CompileAbort(
                CompileStatus.AMBIGUOUS,
                "target entity/item has no matching structural evidence",
                target.span,
            )
        variables = {
            symbol: Variable(
                symbol.variable_name,
                Unit.count(symbol=_identifier(symbol.item)),
                count=True,
            )
            for symbol in sorted(referenced, key=lambda value: value.variable_name)
        }
        constraints: list[Constraint] = []
        for relation in relations:
            if relation.kind == "entity_scope":
                continue
            if relation.kind == "assign":
                symbol = _as_symbol(relation.args[0])
                value = _as_fraction(relation.args[1])
                constraints.append(
                    Assign(
                        variables[symbol],
                        Quantity(value, variables[symbol].unit),
                        span=relation.span,
                    )
                )
            elif relation.kind == "affine":
                target_symbol = _as_symbol(relation.args[0])
                source_symbol = _as_symbol(relation.args[1])
                scale = _as_fraction(relation.args[2])
                offset = _as_fraction(relation.args[3])
                constraints.append(
                    Affine(
                        variables[target_symbol],
                        variables[source_symbol],
                        scale,
                        Quantity(offset, variables[target_symbol].unit),
                        span=relation.span,
                    )
                )
            elif relation.kind == "total":
                members = _as_symbols(relation.args[0])
                value = _as_fraction(relation.args[2])
                unit = variables[members[0]].unit
                constraints.append(
                    Sum(
                        Quantity(value, unit),
                        tuple(variables[member] for member in members),
                        span=relation.span,
                    )
                )
            else:
                raise ValueError(f"unsupported lowered relation: {relation.kind}")
        if not constraints:
            raise _CompileAbort(
                CompileStatus.UNSUPPORTED, "source contains no affine constraints"
            )
        return Problem(
            tuple(variables.values()), tuple(constraints), variables[target.symbol]
        )

    def _lower_rate_ledger(
        self, relations: tuple[Relation, ...], target: TargetRelation
    ) -> Problem:
        allowed = {"unit_rate", "acquire"}
        unexpected = tuple(
            relation for relation in relations if relation.kind not in allowed
        )
        if unexpected:
            raise _CompileAbort(
                CompileStatus.AMBIGUOUS,
                "rate ledger contains relations from another family",
                unexpected[0].span,
            )

        rates: dict[str, Relation] = {}
        acquisitions: dict[str, Relation] = {}
        for relation in relations:
            symbol = _as_symbol(relation.args[0])
            destination = rates if relation.kind == "unit_rate" else acquisitions
            if symbol.item in destination:
                label = "unit price" if relation.kind == "unit_rate" else "acquisition"
                raise _CompileAbort(
                    CompileStatus.AMBIGUOUS,
                    f"duplicate {label} for {symbol.item}",
                    relation.span,
                )
            destination[symbol.item] = relation

        if not rates or not acquisitions:
            raise _CompileAbort(
                CompileStatus.UNSUPPORTED,
                "rate ledger requires local prices and acquisitions",
                target.span,
            )
        missing_prices = sorted(acquisitions.keys() - rates.keys())
        missing_quantities = sorted(rates.keys() - acquisitions.keys())
        if missing_prices:
            raise _CompileAbort(
                CompileStatus.UNSUPPORTED,
                f"missing unit price for {', '.join(missing_prices)}",
                acquisitions[missing_prices[0]].span,
            )
        if missing_quantities:
            raise _CompileAbort(
                CompileStatus.UNSUPPORTED,
                f"missing acquisition quantity for {', '.join(missing_quantities)}",
                rates[missing_quantities[0]].span,
            )

        buyers = {
            _as_symbol(relation.args[0]).owner for relation in acquisitions.values()
        }
        if buyers != {target.symbol.owner}:
            raise _CompileAbort(
                CompileStatus.AMBIGUOUS,
                "all acquisition terms must match the explicit cost target owner",
                target.span,
            )

        currencies = {_as_text(relation.args[3]) for relation in rates.values()}
        if len(currencies) != 1:
            raise _CompileAbort(
                CompileStatus.INVALID,
                "rate ledger contains incompatible currencies",
                target.span,
            )
        currency = next(iter(currencies))
        scopes = {_as_rate_scope(relation.args[4]) for relation in rates.values()}
        if len(scopes) != 1:
            raise _CompileAbort(
                CompileStatus.AMBIGUOUS,
                "unit prices come from multiple local rate scopes",
                target.span,
            )
        scope = next(iter(scopes))
        scope_spans = {
            (scope_value.span.start, scope_value.span.end)
            for scope_value in (
                _as_rate_scope(relation.args[4]) for relation in rates.values()
            )
        }
        if len(scope_spans) > 1 and scope.label is None:
            raise _CompileAbort(
                CompileStatus.AMBIGUOUS,
                "unlabelled price declarations cannot span multiple clauses",
                target.span,
            )
        target_item = target.symbol.item
        exact_item_target = target_item in rates
        group_target = scope.group == target_item
        if exact_item_target and group_target and len(rates) > 1:
            raise _CompileAbort(
                CompileStatus.AMBIGUOUS,
                "cost target names both a line item and the enclosing rate group",
                target.span,
            )
        if not exact_item_target and not group_target:
            raise _CompileAbort(
                CompileStatus.AMBIGUOUS,
                "cost target item is not bound to a line item or explicit rate group",
                target.span,
            )
        money_unit = Unit.base("money", symbol=currency)
        variables: list[Variable] = []
        constraints: list[Constraint] = []
        contributions: dict[str, Variable] = {}

        for item in sorted(rates):
            rate_relation = rates[item]
            acquisition = acquisitions[item]
            rate_unit_text = _as_text(rate_relation.args[2])
            quantity_unit_text = _as_text(acquisition.args[2])
            if rate_unit_text != quantity_unit_text:
                raise _CompileAbort(
                    CompileStatus.INVALID,
                    f"incompatible units for {item}: "
                    f"{rate_unit_text} and {quantity_unit_text}",
                    acquisition.span,
                )
            measure_unit = Unit.base(f"measure:{rate_unit_text}", symbol=rate_unit_text)
            price_unit = money_unit / measure_unit
            price_symbol = _as_symbol(rate_relation.args[0])
            price_variable = Variable(price_symbol.variable_name, price_unit)
            contribution_symbol = (
                target.symbol
                if exact_item_target and item == target_item
                else SymbolKey(
                    target.symbol.owner,
                    "cost",
                    item,
                    "purchase",
                    "line_item",
                )
            )
            contribution = Variable(contribution_symbol.variable_name, money_unit)
            variables.extend((price_variable, contribution))
            contributions[item] = contribution
            constraints.append(
                Assign(
                    price_variable,
                    Quantity(_as_fraction(rate_relation.args[1]), price_unit),
                    span=rate_relation.span,
                )
            )
            constraints.append(
                Rate(
                    contribution,
                    price_variable,
                    Quantity(_as_fraction(acquisition.args[1]), measure_unit),
                    span=acquisition.span,
                )
            )

        if exact_item_target:
            return Problem(
                tuple(variables),
                tuple(constraints),
                contributions[target_item],
            )

        total = Variable(target.symbol.variable_name, money_unit)
        variables.append(total)
        constraints.append(Sum(total, tuple(contributions.values()), span=target.span))
        return Problem(tuple(variables), tuple(constraints), total)

    def _diagnostics(
        self,
        status: CompileStatus,
        reason: str,
        span: Span | None = None,
    ) -> CompileDiagnostics:
        unsupported = (span,) if status is CompileStatus.UNSUPPORTED and span else ()
        ambiguous = (span,) if status is CompileStatus.AMBIGUOUS and span else ()
        return CompileDiagnostics(
            status=status,
            reason=reason,
            messages=(reason,) if reason else (),
            observed_numeric=self.ledger.mentions,
            consumed_numeric=self.ledger.consumed,
            unconsumed_numeric=self.ledger.unconsumed,
            duplicate_numeric=self.ledger.duplicates,
            unsupported_spans=unsupported,
            ambiguous_spans=ambiguous,
        )

    def _failure(
        self,
        status: CompileStatus,
        reason: str,
        span: Span | None = None,
    ) -> CompileResult:
        return CompileResult(
            tuple(self.relations),
            self.targets[0] if len(self.targets) == 1 else None,
            None,
            self._diagnostics(status, reason, span),
        )


def _as_symbol(value: object) -> SymbolKey:
    if not isinstance(value, SymbolKey):
        raise TypeError("relation argument must be a SymbolKey")
    return value


def _as_symbols(value: object) -> tuple[SymbolKey, ...]:
    if not isinstance(value, tuple) or not all(
        isinstance(symbol, SymbolKey) for symbol in value
    ):
        raise TypeError("relation argument must be a SymbolKey tuple")
    return value


def _as_fraction(value: object) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError("relation argument must be an exact number")
    return Fraction(value)


def _as_text(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("relation argument must be text")
    return value


def _as_rate_scope(value: object) -> RateScope:
    if not isinstance(value, RateScope):
        raise TypeError("relation argument must be a RateScope")
    return value


def compile_clauses(source: str) -> CompileResult:
    """Compile one evidence-closed supported problem from local clauses."""

    if not isinstance(source, str):
        diagnostics = CompileDiagnostics(
            CompileStatus.INVALID, "source must be text", ("source must be text",)
        )
        return CompileResult((), None, None, diagnostics)
    return ClauseCompiler(source).compile()


# Descriptive aliases keep callers independent of the frontend's eventual breadth.
compile_affine_entity_problem = compile_clauses
compile_clause_problem = compile_clauses


__all__ = [
    "ClauseCompiler",
    "CompileDiagnostics",
    "CompileResult",
    "CompileStatus",
    "EvidenceLedger",
    "NumericMention",
    "Relation",
    "SymbolKey",
    "TargetRelation",
    "compile_affine_entity_problem",
    "compile_clause_problem",
    "compile_clauses",
]
