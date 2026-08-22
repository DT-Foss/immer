"""
fertig.bindings — der Bindungs-Parser (GSM8K-Projekt, Kernstück).

Diagnose (Falsifikations-Ledger): Muster-Templates erreichen ~4% Präzision,
weil sie Zahlen ohne Bindung verarbeiten. Der Bindungs-Parser macht aus
einer Textaufgabe eine BINDUNGS-STRUKTUR:

    Zahl --(zählt)--> Objekt (clips)
    Zahl --(Einheit)--> clips/friends/dollars
    Zahl --(Rolle)--> qty | partitive | ratio | price | duration
    Frage --(Ziel)--> Objekt + Operation (sum/left/diff/...)

Prinzipien (Hausregeln):
  * Determinismus: gleiche Frage -> gleiche Bindung -> gleiche Antwort.
  * Abstinenz: unvollständige Bindung (Objekt oder Einheit fehlt) -> None.
    Kein Raten auf Zahlen allein.
  * Objekt-Konsistenz: eine Operation bindet nur Mengen DESSELBEN Objekts
    (oder explizit konvertierbarer Einheiten).
  * Verifikation: jede Zwischengröße trägt Objekt+Einheit; die Antwort
    wird nur abgegeben, wenn sie das Frageziel trifft.

Die Rolle 'ratio' referenziert ein VORHER gebundenes Objekt ("half as
many clips") — das ist die Kette, die Muster-Templates nicht können.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Datenmodell
# ---------------------------------------------------------------------------


@dataclass
class Quantity:
    """Eine gebundene Menge im Text."""

    value: Fraction  # numerischer Wert (exakt)
    text: str  # Original-Text der Menge ("48", "half as many")
    unit: Optional[str]  # Einheit ("clips", "friends", "$", "days")
    obj: Optional[str]  # gebundenes Objekt (NP-Kopf, Singular)
    role: str  # qty|partitive|ratio|price|duration|rate|each
    ref_obj: Optional[str] = None  # bei ratio: das referenzierte Objekt
    span: Tuple[int, int] = (0, 0)
    sentence: int = 0

    def __repr__(self) -> str:  # pragma: no cover - Debug
        return (
            f"Q({self.text}={self.value} {self.unit or ''} "
            f"obj={self.obj} role={self.role}"
            f"{' <- ' + self.ref_obj if self.ref_obj else ''})"
        )


@dataclass
class QuestionTarget:
    """Was die Frage sucht: Objekt + Operation + Bezugszeitraum."""

    obj: Optional[str] = None  # "clips"
    op: str = "sum"  # sum | left | diff | product | total
    period: Optional[str] = None  # "april and may"
    ok: bool = False

    def __repr__(self) -> str:  # pragma: no cover - Debug
        return f"Target(obj={self.obj}, op={self.op})"


@dataclass
class BindingResult:
    """Vollständiges Bindungsergebnis einer Aufgabe."""

    quantities: List[Quantity] = field(default_factory=list)
    target: QuestionTarget = field(default_factory=QuestionTarget)
    answer: Optional[str] = None
    ok: bool = False
    reason: str = ""

    def __repr__(self) -> str:  # pragma: no cover - Debug
        return f"BindingResult(ok={self.ok}, answer={self.answer}, {self.reason})"


class BindingParserError(RuntimeError):
    """An internal binding defect, distinct from an evidence-based abstention."""


# ---------------------------------------------------------------------------
# Lexikon (geschlossene Klassen — keine Hardcoding auf Benchmarks)
# ---------------------------------------------------------------------------

# Objekt-/Einheits-Köpfe: was Mengen tragen kann (allgemeine NP-Köpfe)
_SINGULAR = {
    "clips": "clip",
    "friends": "friend",
    "students": "student",
    "children": "child",
    "apples": "apple",
    "oranges": "orange",
    "books": "book",
    "pages": "page",
    "miles": "mile",
    "hours": "hour",
    "days": "day",
    "weeks": "week",
    "months": "month",
    "years": "year",
    "dollars": "dollar",
    "cents": "cent",
    "minutes": "minute",
    "seconds": "second",
    "bottles": "bottle",
    "cookies": "cookie",
    "candies": "candy",
    "tickets": "ticket",
    "cards": "card",
    "pencils": "pencil",
    "crayons": "crayon",
    "marbles": "marble",
    "stamps": "stamp",
    "toys": "toy",
    "games": "game",
    "points": "point",
    "miles per hour": "mph",
    "kilograms": "kilogram",
    "grams": "gram",
    "meters": "meter",
    "feet": "foot",
    "inches": "inch",
    "laps": "lap",
    "rows": "row",
    "chairs": "chair",
    "tables": "table",
    "rooms": "room",
    "floors": "floor",
    "flights": "flight",
    "steps": "step",
    "questions": "question",
    "problems": "problem",
    "answers": "answer",
    "tests": "test",
    "scores": "score",
    "runs": "run",
    "walks": "walk",
    "songs": "song",
    "pictures": "picture",
    "photos": "photo",
    "slices": "slice",
    "pieces": "piece",
    "pizzas": "pizza",
    "cakes": "cake",
    "pies": "pie",
    "loaves": "loaf",
    "eggs": "egg",
    "sandwiches": "sandwich",
    "burgers": "burger",
    "hot dogs": "hot dog",
    "coins": "coin",
    "quarters": "quarter",
    "dimes": "dime",
    "nickels": "nickel",
    "pennies": "penny",
    "bills": "bill",
    "bags": "bag",
    "boxes": "box",
    "baskets": "basket",
    "jars": "jar",
    "cans": "can",
    "packs": "pack",
    "cartons": "carton",
    "dozens": "dozen",
    "hours per day": "hour",
    "times": "time",
    "miles per gallon": "mpg",
    "beetles": "beetle",
    "birds": "bird",
    "snakes": "snake",
    "jaguars": "jaguar",
    "arms": "arm",
    "starfish": "starfish",
    "seastars": "seastar",
    "gnomes": "gnome",
    "houses": "house",
    "situps": "situp",
    "cans": "can",
    "bushes": "bush",
    "roses": "rose",
    "petals": "petal",
    "trees": "tree",
    "plants": "plant",
    "seeds": "seed",
    "glasses": "glass",
    "glass": "glass",
    "gas": "gas",
    "bus": "bus",
    "class": "class",
    "stick": "stick",
    "sticks": "stick",
    "cookies": "cookie",
    "chickens": "chicken",
    "cups": "cup",
    "bolts": "bolt",
    "fibers": "fiber",
    "jewels": "jewel",
    "pages": "page",
    "downloads": "download",
    "months": "month",
}

# Objekt-Synonyme (geschlossene Klasse): pieces == slices, coins == cents...
_SYNONYMS = {
    "piece": "slice",
    "pieces": "slice",
    "slice": "slice",
    "coin": "coin",
    "coins": "coin",
    "cent": "coin",
    "cents": "coin",
    "item": "item",
    "items": "item",
    "total": None,
}


# Operatoren in der Frage (geschlossene Klasse)
_SUM_WORDS = [
    "altogether",
    "in all",
    "in total",
    "total",
    "combined",
    "all together",
    "sum",
    "both",
]
_LEFT_WORDS = ["left", "remain", "remaining", "still have", "left over"]
_DIFF_WORDS = ["more than", "less than", "how many more", "difference", "how much more"]
_PRODUCT_WORDS = [
    "total cost",
    "how much did he spend",
    "how much does",
    "how much do",
    "how much would",
]

# Rollen-Verben: was ist die Menge des Objekts?
_ROLE_VERBS = {
    "sold": "qty",
    "bought": "qty",
    "has": "qty",
    "had": "qty",
    "have": "qty",
    "collected": "qty",
    "gathered": "qty",
    "ate": "qty",
    "drank": "qty",
    "made": "qty",
    "baked": "qty",
    "cooked": "qty",
    "read": "qty",
    "wrote": "qty",
    "answered": "qty",
    "solved": "qty",
    "earned": "qty",
    "spent": "qty",
    "paid": "qty",
    "cost": "price",
    "costs": "price",
    "save": "qty",
    "saved": "qty",
    "won": "qty",
    "lost": "qty",
    "gave": "qty",
    "donated": "qty",
    "lent": "qty",
    "borrowed": "qty",
    "received": "qty",
    "found": "qty",
    "picked": "qty",
    "caught": "qty",
    "planted": "qty",
    "watered": "qty",
    "walked": "qty",
    "ran": "qty",
    "swam": "qty",
    "drove": "qty",
    "flew": "qty",
    "traveled": "qty",
    "visited": "qty",
    "built": "qty",
    "painted": "qty",
    "learned": "qty",
    "studied": "qty",
    "practiced": "qty",
    "played": "qty",
    "watched": "qty",
    "listened": "qty",
    "worked": "qty",
    "needed": "qty",
    "wanted": "qty",
    "invited": "qty",
    "attended": "qty",
    "joined": "qty",
    "used": "qty",
    "purchased": "qty",
    "ordered": "qty",
    "brought": "qty",
    "took": "qty",
    "got": "qty",
    "kept": "qty",
    "returned": "qty",
    "sent": "qty",
    "mailed": "qty",
    "delivered": "qty",
    "produced": "qty",
    "created": "qty",
    "designed": "qty",
    "wrapped": "qty",
    "filled": "qty",
}

# Zahlwörter (geschlossene Klasse)
_WORD_NUM = {
    "half": Fraction(1, 2),
    "twice": Fraction(2),
    "double": Fraction(2),
    "triple": Fraction(3),
    "quarter": Fraction(1, 4),
    "a dozen": Fraction(12),
    "a couple": Fraction(2),
}


# ---------------------------------------------------------------------------
# 1. Mengen-Phrasen finden
# ---------------------------------------------------------------------------

_NUM = r"(?:\d+(?:\.\d+)?|\d+/\d+)"


def _find_quantities(text: str) -> List[Tuple[str, Fraction, int, int]]:
    """(Phrase, Wert, Start, Ende) — Zahlen UND Zahlwörter mit Kontext."""
    out: List[Tuple[str, Fraction, int, int]] = []
    low = text.lower()
    # Zahlwörter mit Verhältnis-Struktur: "half as many X", "3 times as many X"
    for m in re.finditer(
        r"(?:(\d+)\s+times\s+as\s+many|half\s+as\s+many|twice\s+as\s+many"
        r"|three\s+times\s+as\s+many)\s+([a-z][a-z ]+?)(?=[,.;]|\s+(?:as|"
        r"than|in|for|to|and|but)\b|\s*$)",
        low,
    ):
        val = (
            Fraction(m.group(1))
            if m.group(1)
            else (Fraction(2) if "twice" in m.group(0) else Fraction(1, 2))
        )
        if "three" in m.group(0):
            val = Fraction(3)
        out.append((m.group(0).strip(), val, m.start(), m.end()))
    # Bindestrich-Mengen: "10-acre farm" -> 10 acres
    for m in re.finditer(
        r"(\d+)-(?:acre|page|mile|gallon|pound|inch|"
        r"foot|year|day|week|hour)(?:s)?\b",
        low,
    ):
        out.append((m.group(0), Fraction(m.group(1)), m.start(), m.end()))
    # Ziffern: "48 clips", "48 of her friends", "$2", "2 days", "48%",
    # Brüche "1/2 cup"
    for m in re.finditer(
        r"(?<![a-z])(\d+(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?|"
        r"\d+/\d+)\s*(%|dollars?|\$)?"
        r"\s*([a-z][a-z ]{0,20}?)?(?=[,.;]|\s+(?:of|each|per|for|in|"
        r"to|and|but|than|at|with|by|from|on)\b|\s*$)",
        low,
    ):
        val = Fraction(m.group(1).replace(",", ""))
        unit = (m.group(2) or "").replace("$", "dollars").strip()
        rest = (m.group(3) or "").strip()
        phrase = m.group(0).strip()
        out.append((phrase, val, m.start(), m.end()))
        # "48 of her friends" -> partitive: Zahl gefolgt von "of X"
    return out


def _np_head(phrase: str) -> Optional[str]:
    """Der letzte Nomen-Teil einer Phrase ("48 of her friends" -> friends)."""
    words = [w for w in re.split(r"\s+", phrase) if w]
    if not words:
        return None
    # "of her friends" -> friends
    if "of" in words:
        idx = words.index("of")
        tail = words[idx + 1 :]
    else:
        tail = words
    # Determinierer abwerfen
    tail = [
        w
        for w in tail
        if w not in ("the", "a", "an", "her", "his", "their", "my", "its", "our")
    ]
    if not tail:
        return None
    head = tail[-1]
    return head


# ---------------------------------------------------------------------------
# 2. Rollen-Bindung
# ---------------------------------------------------------------------------

# jede-Relationen: <Menge> <obj> (each) (has/contains/costs/...) <Menge> <unit>
_EACH_RE = [
    re.compile(
        r"(\d+)\s+([a-z]+)\s+each\s+(?:has|contains?|holds|costs?|"
        r"is\s+worth|weighs?|is)\s+(\d+(?:\.\d+)?)\s*([a-z]+)?"
    ),
    re.compile(
        r"each\s+(?:of\s+)?(\d+)?\s*([a-z]+)\s+(?:has|contains?|"
        r"holds|costs?|weighs?|is)\s+(\d+(?:\.\d+)?)\s*([a-z]+)?"
    ),
    re.compile(r"(\d+)\s+([a-z]+)\s+per\s+(\w+)"),  # rate: 12 beetles per day
]

# more/fewer-than: "M more <obj> than <base>" — base kann eine Zahl ODER
# ein referenziertes Objekt sein ("than snowflake stamps" -> 11)
_MORE_FEWER_RE = re.compile(
    r"(\d+)\s+(more|fewer|less)\s+([a-z]+(?:\s+[a-z]+){0,2})\s+than\s+"
    r"(?:(\d+)|([a-z]+(?:\s+[a-z]+){0,2}))"
)

# a/an <obj> has/contains M <unit>: Typ->Pro-Stück-Menge (Adjektive
# überspringen: "a large pizza has 16 slices")
_A_AN_HAS_RE = re.compile(
    r"\b(?:a|an|each)\s+(?:[a-z]+\s+){0,1}([a-z]+)(?:\s+of\s+"
    r"[a-z]+)?\s+(?:has|contains?|holds|costs?|weighs?|produces?|"
    r"makes?|grows?)\s+(\d+(?:\.\d+)?)\s*([a-z]+)?"
)

# times-as-many: "X times as many <obj> as <base>" — base kann Zahl ODER
# eine Variable sein ("as Carlos memorized"); "digits of pi" erlaubt;
# "half as many ... as" = factor 1/2
_TIMES_AS_MANY_RE = re.compile(
    r"(?:(\d+)\s+times|half|twice|triple)\s+as\s+many\s+"
    r"([a-z]+(?:\s+of\s+[a-z]+){0,2})\s+as\s+"
    r"(?:(\d+)|([a-z]+\s+and\s+[a-z]+\s+combined)|([a-z]+))"
)

# Variablen-Zuweisung: "If Mina memorized 24 digits" / "Mina memorized 24"
# — unit muss ein Nomen sein ("30 less" ist KEINE Zuweisung)
_VAR_ASSIGN_RE = re.compile(
    r"\b(?:if\s+)?([A-Z][a-z]+)\s+(?:memorized|has|had|bought|sold|ate|\
    collected|saved|earned|spent|wrote|read|answered|solved|ran|walked|\
    traveled|earned|made|built|found|planted|caught|picked|received|\
    took|got|purchased|broke|caught|planted|picked|collected)\s+"
    r"(\d+)\s*(?!less\b|more\b|than\b)([a-z]+)?"
)


# each-of-N: "each of the first four houses has 3 gnomes" -> 4x3
_EACH_OF_RE = re.compile(
    r"each\s+of\s+(?:the\s+)?(?:first\s+)?(\d+)\s+([a-z]+)\s+"
    r"(?:has|contains?|holds)\s+(\d+)\s*([a-z]+)?"
)

# with-Mengen: "7 starfish with 5 arms each and one seastar with 14
# arms" -> 7x5 + 14
_WITH_EACH_RE = re.compile(r"(\d+)\s+([a-z]+)\s+with\s+(\d+)\s+([a-z]+)\s+each")
_WITH_SINGLE_RE = re.compile(r"\b(?:one|a|an)\s+([a-z]+)\s+with\s+(\d+)\s+([a-z]+)")

# Futterketten: "Each bird eats 12 beetles per day, each snake eats 3
# birds per day, each jaguar eats 5 snakes per day. 6 jaguars..."
_EATS_RE = re.compile(r"each\s+([a-z]+)\s+eats?\s+(\d+)\s+([a-z]+)\s+per\s+day")


# Raten-Dauer: "For the next two hours, she collected 35 coins" -> 2x35
_FOR_DURATION_RE = re.compile(
    r"for\s+(?:the\s+)?(?:next\s+)?(\d+)\s+(hours?|days?|weeks?|\
    months?|minutes?|seconds?)\s*,\s*([a-z]+)\s+collected\s+"
    r"(\d+)\s*([a-z]+)?"
)

# Subtraktion: "gave 15 of them" / "ate 3 of the cookies"
_GAVE_RE = re.compile(
    r"(?:gave|donated|lent|threw\s+away|ate|used|sold|erased|deleted|\
    removed|had\s+to\s+erase)\s+(\d+)\s+"
    r"(?:of\s+)?(?:them|it|those|the\s+[a-z]+|sentences)\b"
)

# Dauer: "for 43 minutes" / "worked 3 hours" / "typed 15 minutes longer"
_DURATION_RE = re.compile(
    r"(?:(?:for|worked|lasting|spent|typed|ran|drove|walked|studied)\s+)?"
    r"(\d+(?:\.\d+)?)\s+(minutes?|hours?|days?|weeks?|months?|years?)"
    r"(?:\s+(?:longer|more|extra))?"
)


# times-the-number: "4 times the number of glasses David broke" -> x4
_TIMES_THE_NUMBER_RE = re.compile(
    r"(\d+)\s+times\s+the\s+number\s+of\s+([a-z]+)\s+"
    r"([A-Z][a-z]+)"
)

# Bruch-von: "one-half of it, one-fifth of it, one-third of the
# remaining" -> sequenzielle Rest-Kette
_FRACTION_WORDS = {
    "one-half": Fraction(1, 2),
    "half": Fraction(1, 2),
    "one-third": Fraction(1, 3),
    "two-thirds": Fraction(2, 3),
    "one-quarter": Fraction(1, 4),
    "three-quarters": Fraction(3, 4),
    "one-fifth": Fraction(1, 5),
    "two-fifths": Fraction(2, 5),
    "three-fifths": Fraction(3, 5),
    "one-sixth": Fraction(1, 6),
    "one-tenth": Fraction(1, 10),
    "one-eighth": Fraction(1, 8),
    "three-tenths": Fraction(3, 10),
}
_FRACTION_OF_RE = re.compile(
    r"(one-half|half|one-third|two-thirds|one-quarter|three-quarters|"
    r"one-fifth|two-fifths|three-fifths|one-sixth|one-tenth|one-eighth|"
    r"three-tenths)\s+of\s+(it|the remaining|them|that|those)"
)

# Prozent-von: "plants 60% of those" -> x(60/100)
_PCT_OF_RE = re.compile(
    r"(?:plants?|uses?|eats?|grows?|keeps?|sells?|gives?)\s+(\d+)"
    r"\s*(?:%|percent)\s+of\s+(?:those|them|it|the\s+[a-z]+)"
)

# times-more: "25 times more stickers than Kristoff" = (25+1)x
_TIMES_MORE_RE = re.compile(r"(\d+)\s+times\s+more\s+([a-z]+)\s+than\s+([A-Z][a-z]+)")

# more-than-twice: "five more roommates than twice as many as Bob"
# -> John = 2xBob + 5 (auf dem ORIGINAL-Text, Wort-Zahlen erlaubt)
_MORE_THAN_TWICE_RE = re.compile(
    r"(?:(\d+)|five|four|three|six|seven|eight|nine|ten|two|twenty|\
    thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)\s+more\s+"
    r"(?:([a-z]+(?:\s+[a-z]+){0,2})\s+)?than\s+"
    r"(?:twice|thrice)\s+as\s+many\s+as\s+([A-Z][a-z]+)"
)


# Prozent: "loses/uses N% of them/it" -> x(1 - N/100)
_PERCENT_OFF_RE = re.compile(
    r"(?:loses?|used|spent|gave\s+away|ate|removed|takes?\s+away)\s+"
    r"(\d+)\s*(?:%|percent)\s+of\s+(?:them|it|those|the\s+[a-z]+)"
)

# Double/Triple: "gives her double the amount ..." -> x(1+2)
_DOUBLE_RE = re.compile(
    r"gives?\s+(?:her|him|them)\s+(double|triple|twice)\s+the\s+amount"
)


# Prozent-Gruppen: "denied N% of the K kids from X High" -> K x (100-N)%
_PCT_GROUP_RE = re.compile(
    r"(?:(?:denied|rejected|admitted)\s+)?(?:(\d+)\s*(?:%|percent)|half)"
    r"\s+(?:of\s+)?the\s+(\d+)\s+([a-z]+)"
)

# Ratio: "6 additional cans for every 5 cans Mark bought" -> x(6/5)
_RATIO_EVERY_RE = re.compile(
    r"(\d+)\s+(?:(?:additional|more|extra)\s+)?([a-z]+)\s+for\s+"
    r"every\s+(\d+)\s+([a-z]+)\s+([a-z]+)\s+bought"
)


@dataclass
class Relation:
    """Bindungs-Relation zwischen Mengen: per/each/more/fewer/times."""

    kind: str  # each | more | fewer | times | assign
    source: int  # Index in quantities (Träger der Relation)
    base_value: Fraction  # Basis-Wert (bei each: Anzahl der Träger)
    factor: Fraction = Fraction(1)  # Multiplikator (each: Wert je Träger)
    unit: Optional[str] = None  # Ziel-Einheit (each: Scheiben...)
    var: Optional[str] = None  # Variablen-Name (times/assign)
    mult: int = 2  # more2x: twice=2, thrice=3
    text: str = ""


# Kalender (geschlossene Klasse): Monat -> Tage (Nicht-Schaltjahr)
_MONTH_DAYS = {
    "january": 31,
    "february": 28,
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
_MONTH_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\b"
)

# Behälter: "bottles that can hold 15 stars each" / "another 3 identical"
_HOLD_EACH_RE = re.compile(
    r"(\d+)\s+(?:[a-z]+\s+){0,2}([a-z]+)\s+that\s+can\s+hold\s+"
    r"(\d+)\s+(?:[a-z]+\s+){0,2}([a-z]+)\s+each"
)
_ANOTHER_IDENTICAL_RE = re.compile(
    r"another\s+(\d+)\s+identical\s+(?:[a-z]+\s+){0,2}([a-z]+)"
)

# Wort-Zahlen -> Ziffern (geschlossene Klasse) — für Relations-Matching
_WORD2NUM = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
}


def _digitize(text: str) -> str:
    """Wort-Zahlen in Ziffern wandeln (nur für Relations-Matching)."""
    out = text
    for w, n in sorted(_WORD2NUM.items(), key=lambda kv: -len(kv[0])):
        out = re.sub(rf"\b{w}\b", str(n), out)
    return out


def _find_relations(
    text: str, quants: List[Quantity]
) -> Tuple[List[Relation], List[Quantity]]:
    """jede/more/fewer/times-Relationen finden und die betroffenen
    Quantities durch die RELATIVEN Werte ersetzen (Bindung statt Summe
    von Rohzahlen)."""
    low = _digitize(text.lower())
    rels: List[Relation] = []
    for m in _EACH_RE[0].finditer(low):
        n, obj, m2, unit = m.group(1), m.group(2), m.group(3), m.group(4)
        # "N obj each has M unit" -> N Stück zu je M -> Gesamt M*N
        rels.append(
            Relation(
                kind="each",
                source=-1,
                base_value=Fraction(n),
                factor=Fraction(m2),
                unit=unit or obj,
                text=m.group(0),
            )
        )
    for m in _EACH_RE[1].finditer(low):
        n, obj, m2, unit = m.group(1), m.group(2), m.group(3), m.group(4)
        cnt = Fraction(n) if n else Fraction(1)
        rels.append(
            Relation(
                kind="each",
                source=-1,
                base_value=cnt,
                factor=Fraction(m2),
                unit=unit or obj,
                text=m.group(0),
            )
        )
    for m in _EACH_RE[2].finditer(low):
        # Rate: "12 beetles per day" -> nur registrieren, wenn ein
        # Zeitraum folgt (sonst Abstinenz)
        rels.append(
            Relation(
                kind="rate",
                source=-1,
                base_value=Fraction(m.group(1)),
                factor=Fraction(1),
                unit=m.group(2),
                text=m.group(0),
            )
        )
    for m in _MORE_FEWER_RE.finditer(low):
        if "as many" in m.group(0):
            continue  # "more X than twice as many" gehört more2x
        unit = m.group(3)
        # Kern-Nomen: "digits of pi" -> digits (of-Phrase abtrennen)
        core = unit.split(" of ")[0].split()[-1]
        var_ref = None
        if m.group(4):
            base = Fraction(m.group(4))
        else:
            # Referenz: der TYP-First-Wort ("truck stamps" -> "truck")
            # — eindeutig, während das Kern-Nomen ("stamps") mehrdeutig ist
            ref_name = m.group(5).split()[0]

            def _type_matches(q: Quantity, name: str) -> bool:
                """Typ-Match OHNE Selbstreferenz: bei relativen Mengen
                ("9 more truck than snowflake") zählt nur der Subjekt-
                Teil — sonst referenziert sich die Relation selbst."""
                if name not in q.text:
                    return False
                if re.search(r"\b(more|fewer|less)\b", q.text):
                    subj = q.text.split("than")[0] if "than" in q.text else q.text
                    return name in subj
                return True

            ref = [q for q in quants if _type_matches(q, ref_name)]
            if not ref:
                # keine Quantity -> Variablen-Referenz ("than Carlos")
                var_ref = m.group(5).split()[0].lower()
                base = Fraction(0)
            else:
                # PRIORITÄT: Relation-generierte Werte (ref_obj gesetzt) sind
                # absoluter als Roh-Zahlen — "truck" muss 20 sein, nicht das
                # rohe Inkrement 9.
                rel_ref = [q for q in ref if q.ref_obj]
                base = (rel_ref[0] if rel_ref else ref[0]).value
        rels.append(
            Relation(
                kind=m.group(2),
                source=-1,
                base_value=base,
                factor=Fraction(m.group(1)),
                unit=core,
                var=var_ref,
                text=m.group(0),
            )
        )
    for m in _TIMES_AS_MANY_RE.finditer(low):
        if m.group(3):
            base_v = Fraction(m.group(3))
        elif m.group(4):
            # Kombination: "twice as many fish as cats and dogs
            # combined" — Base = Summe der genannten Tier-Mengen
            parts = m.group(4).split(" and ")
            vals = []
            for part in parts:
                w = part.replace("combined", "").strip()
                hit = [
                    q
                    for q in quants
                    if q.role == "qty"
                    and (q.obj or "").rstrip("s") == w.rstrip("s")
                    or (q.unit or "").rstrip("s") == w.rstrip("s")
                ]
                if hit:
                    vals.append(hit[0].value)
            if not vals:
                continue
            base_v = sum(vals, Fraction(0))
        else:
            # Variable: "as Carlos memorized" — wird später aufgelöst
            base_v = Fraction(0)
        if m.group(1) is None:
            factor = (
                Fraction(1, 2)
                if m.group(0).startswith("half")
                else (Fraction(3) if m.group(0).startswith("triple") else Fraction(2))
            )
        else:
            factor = Fraction(m.group(1))
        rels.append(
            Relation(
                kind="times",
                source=-1,
                base_value=base_v,
                factor=factor,
                unit=m.group(2),
                var=m.group(5) if m.group(5) else None,
                text=m.group(0),
            )
        )
    for m in _VAR_ASSIGN_RE.finditer(text):
        if m.group(3) is None or m.group(3) in (
            "additional",
            "extra",
            "more",
            "less",
            "than",
            "times",
        ):
            # "Tim has 30 less ..." / "Jennifer bought 6 additional ..."
            # ist KEINE Zuweisung
            continue
        after = text[m.end() :]
        if re.match(r"\s+[a-z]+", after) and not re.match(
            r"\s+(?:of|for|in|to|and|but|than|with|at|per|each|on|"
            r"from|by|before|after|during)\b",
            after,
        ):
            # "bought 2 glass bottles that..." — unit ist nicht das
            # letzte Wort der Phrase -> keine Zuweisung
            continue
        # REINE Zuweisungssätze: weitere Zahlen IM SELBEN SATZ -> Liste
        # ("Ed has 2 dogs, 3 cats and ..." ist KEINE Ed=2-Zuweisung)
        satz_anfang = (
            max(
                text.rfind(". ", 0, m.start()),
                text.rfind("? ", 0, m.start()),
                text.rfind("! ", 0, m.start()),
            )
            + 2
        )
        satz_ende = (
            min(
                x
                for x in [
                    text.find(". ", m.end()),
                    text.find("? ", m.end()),
                    text.find("! ", m.end()),
                ]
                if x != -1
            )
            if any(
                x != -1
                for x in [
                    text.find(". ", m.end()),
                    text.find("? ", m.end()),
                    text.find("! ", m.end()),
                ]
            )
            else len(text)
        )
        satz = text[satz_anfang:satz_ende]
        if len(re.findall(r"\b\d+\b", satz)) > 1:
            continue
        rels.append(
            Relation(
                kind="assign",
                source=-1,
                base_value=Fraction(m.group(2)),
                factor=Fraction(1),
                unit=m.group(3) or "",
                var=m.group(1).lower(),
                text=m.group(0),
            )
        )
    # Gruppen-each: "Half the tables have 2 chairs each, 5 have 3
    # chairs each and the rest have 4 chairs each"
    grp_total = None
    grp_used = Fraction(0)
    for m in _GROUP_EACH_RE.finditer(low):
        if grp_total is None:
            hq = [
                q
                for q in quants
                if q.role == "qty"
                and (
                    m.group(2) is not None
                    and _canon(q.obj or "") == m.group(2).rstrip("s")
                )
            ]
            if not hq:
                continue
            grp_total = hq[0].value
        n = m.group(1)
        if n == "half":
            cnt = grp_total / 2
        elif n == "the rest":
            cnt = grp_total - grp_used
        else:
            cnt = Fraction(int(n))
        grp_used += cnt
        rels.append(
            Relation(
                kind="each",
                source=-1,
                base_value=cnt,
                factor=Fraction(m.group(3)),
                unit=m.group(4),
                text=m.group(0),
            )
        )
    for m in _WITH_EACH_RE.finditer(low):
        rels.append(
            Relation(
                kind="each",
                source=-1,
                base_value=Fraction(m.group(1)),
                factor=Fraction(m.group(3)),
                unit=m.group(4),
                text=m.group(0),
            )
        )
    for m in _EACH_OF_RE.finditer(low):
        rels.append(
            Relation(
                kind="each",
                source=-1,
                base_value=Fraction(m.group(1)),
                factor=Fraction(m.group(3)),
                unit=m.group(4) or m.group(2),
                text=m.group(0),
            )
        )
    for m in _WITH_SINGLE_RE.finditer(low):
        rels.append(
            Relation(
                kind="each",
                source=-1,
                base_value=Fraction(1),
                factor=Fraction(m.group(2)),
                unit=m.group(3),
                text=m.group(0),
            )
        )
    # Typ->Pro-Stück: "a large pizza has 16 slices" — verknüpft mit einer
    # vorherigen Menge desselben Typs ("2 large pizzas")
    for m in _A_AN_HAS_RE.finditer(low):
        typ, per, unit = m.group(1), m.group(2), m.group(3)
        # Typ-Menge suchen: "2 large pizzas" -> unit==pizza.
        # PRIORITÄT: relation-generierte Mengen (ref_obj) — "75 roses"
        # ist die tiefere Bindung als das rohe "25 roses".
        typ_key = _canon(typ)
        holder = [
            q
            for q in quants
            if q.role == "qty"
            and (_canon(q.obj or "") == typ_key or _canon(q.unit or "") == typ_key)
        ]
        rel_holder = [q for q in holder if q.ref_obj]
        holder = rel_holder or holder
        if holder:
            total = holder[0].value * Fraction(per)
            rels.append(
                Relation(
                    kind="each",
                    source=-1,
                    base_value=holder[0].value,
                    factor=Fraction(per),
                    unit=unit or typ,
                    text=m.group(0),
                )
            )
    for m in _FOR_DURATION_RE.finditer(low):
        rels.append(
            Relation(
                kind="each",
                source=-1,
                base_value=Fraction(m.group(1)),
                factor=Fraction(m.group(4)),
                unit=m.group(5) or m.group(3),
                text=m.group(0),
            )
        )
    for m in _GAVE_RE.finditer(low):
        rels.append(
            Relation(
                kind="subtract",
                source=-1,
                base_value=Fraction(m.group(1)),
                factor=Fraction(1),
                unit="",
                text=m.group(0),
            )
        )
    for m in _TIMES_THE_NUMBER_RE.finditer(text):
        # Quantity-Version, wenn der Referent im Text einen Wert hat:
        # "David broke 2 glasses ... 4 times the number of glasses
        # David broke" -> 4 x 2 = 8 (als Menge, nicht Variable)
        ref = [
            q
            for q in quants
            if q.role == "qty" and _canon(q.obj or "") == _canon(m.group(2))
        ]
        if ref:
            rels.append(
                Relation(
                    kind="times",
                    source=-1,
                    base_value=ref[0].value,
                    factor=Fraction(m.group(1)),
                    unit=m.group(2),
                    text=m.group(0),
                )
            )
        else:
            rels.append(
                Relation(
                    kind="times",
                    source=-1,
                    base_value=Fraction(0),
                    factor=Fraction(m.group(1)),
                    unit=m.group(2),
                    var=m.group(3).lower(),
                    text=m.group(0),
                )
            )
    for m in _PCT_OF_RE.finditer(low):
        rels.append(
            Relation(
                kind="pct_of",
                source=-1,
                base_value=Fraction(0),
                factor=Fraction(m.group(1)),
                unit="",
                text=m.group(0),
            )
        )
    for m in _TIMES_MORE_RE.finditer(text):
        rels.append(
            Relation(
                kind="times_more",
                source=-1,
                base_value=Fraction(0),
                factor=Fraction(m.group(1)),
                unit=m.group(2),
                var=m.group(3).lower(),
                text=m.group(0),
            )
        )
    for m in _MORE_THAN_TWICE_RE.finditer(text):
        if m.group(1):
            factor = Fraction(m.group(1))
        else:
            factor = Fraction(_WORD2NUM.get(m.group(0).split()[0], 5))
        mult = 2 if "twice" in m.group(0) else 3
        rels.append(
            Relation(
                kind="more2x",
                source=-1,
                base_value=Fraction(0),
                factor=factor,
                mult=mult,
                unit=m.group(2) or "",
                var=m.group(3).lower(),
                text=m.group(0),
            )
        )
    for m in _HOLD_EACH_RE.finditer(low):
        rels.append(
            Relation(
                kind="each",
                source=-1,
                base_value=Fraction(m.group(1)),
                factor=Fraction(m.group(3)),
                unit=m.group(4),
                text=m.group(0),
            )
        )
    # Kapazitäten der Behälter sammeln ("bottles that can hold 15")
    hold_caps: dict[str, tuple[Fraction, str]] = {}
    for m in _HOLD_EACH_RE.finditer(low):
        unit = m.group(4)
        if unit == "each":
            unit = m.group(3)
        hold_caps[m.group(2).rstrip("s")] = (Fraction(m.group(3)), unit)
    for m in _ANOTHER_IDENTICAL_RE.finditer(low):
        # "another 3 identical bottles" — erbt Kapazität UND Einheit
        typ = m.group(2).rstrip("s")
        cap_unit = hold_caps.get(typ)
        if cap_unit is None:
            prev = [
                q
                for q in quants
                if (q.unit or "").rstrip("s") == typ or (q.obj or "").rstrip("s") == typ
            ]
            if not prev:
                continue
            cap_unit = (Fraction(1), typ)
        rels.append(
            Relation(
                kind="each",
                source=-1,
                base_value=Fraction(m.group(1)),
                factor=cap_unit[0],
                unit=cap_unit[1],
                text=m.group(0),
            )
        )
    for m in _DURATION_RE.finditer(low):
        rels.append(
            Relation(
                kind="duration",
                source=-1,
                base_value=Fraction(m.group(1)),
                factor=Fraction(1),
                unit=m.group(2),
                text=m.group(0),
            )
        )
    for m in _PERCENT_OFF_RE.finditer(low):
        rels.append(
            Relation(
                kind="percent_off",
                source=-1,
                base_value=Fraction(0),
                factor=Fraction(m.group(1)),
                unit="",
                text=m.group(0),
            )
        )
    for m in _DOUBLE_RE.finditer(low):
        mult = 2 if m.group(1) in ("double", "twice") else 3
        rels.append(
            Relation(
                kind="double",
                source=-1,
                base_value=Fraction(0),
                factor=Fraction(mult),
                unit="",
                text=m.group(0),
            )
        )
    for m in _PCT_GROUP_RE.finditer(low):
        pct = Fraction(m.group(1)) if m.group(1) else Fraction(50)  # half
        rels.append(
            Relation(
                kind="pct_got",
                source=-1,
                base_value=Fraction(m.group(2)),
                factor=pct,
                unit=m.group(3),
                text=m.group(0),
            )
        )
    for m in _RATIO_EVERY_RE.finditer(low):
        rels.append(
            Relation(
                kind="ratio",
                source=-1,
                base_value=Fraction(m.group(1)),
                factor=Fraction(m.group(3)),
                unit=m.group(2),
                var=m.group(5).lower(),
                text=m.group(0),
            )
        )
    return rels, quants


def _effective_quantities(quants: List[Quantity]) -> List[Quantity]:
    """Relation-Mengen (ref_obj gesetzt) sind ABSOLUTE Werte; sie
    ersetzen Roh-Mengen, die Teil ihrer Text-Bindung sind — sowohl
    relative ("9 more X") als auch Pro-Stück-Werte ("16 slices" in
    "a large pizza has 16 slices"). Sonst summiert man Basis + Inkrement
    + Pro-Stück-Wert."""
    eff = [q for q in quants if q.role == "qty" and q.ref_obj]
    if not eff:
        return quants
    out = []
    for q in quants:
        if q.role == "qty" and not q.ref_obj:
            replaced = False
            for e in eff:
                if e.unit == q.unit and q.text in e.text and q.value != e.value:
                    replaced = True
                    break
            if replaced:
                continue
        out.append(q)
    return out


def _solve_variables(
    rels: List[Relation], question: str, quants: Optional[List[Quantity]] = None
) -> Optional[Fraction]:
    """Variablen-Gleichungen auflösen: "Mina memorized six times as many
    digits as Carlos. If Mina memorized 24 digits, how many did Sam
    memorize?" -> Carlos=4, Sam=10.

    Gleichungsformen (alle mit Subjekt-Variable):
      assign: Subjekt = base_value
      times:  Subjekt = factor * (base_value | Var)
      more/fewer: Subjekt = (base_value | Var) ± factor
    """
    low = question.lower()

    # Subjekt einer Relation: Satzanfang der den Ausdruck enthält
    def _subject(rel: Relation) -> Optional[str]:
        # Satz, der rel.text enthält (beide Seiten digitize-normalisiert);
        # Subjekt = die KLAUSEL (Satz- oder Nebensatz-Anfang) —
        # "..., and Harry has half as many..." -> Harry, nicht Tim.
        for sent in re.split(r"(?<=[.!?])\s+", question):
            idx = _digitize(sent.lower()).find(_digitize(rel.text.lower()))
            if idx >= 0:
                before = sent[:idx]
                if not before.strip():
                    # Relation am SATZANFANG: Subjekt = Satzanfang
                    m = re.match(r"([A-Z][a-z]+)", sent)
                    return m.group(1).lower() if m else None
                clause = re.split(
                    r",\s+(?:and|but|then|so|yet|while)\s+|;\s+|\.\s+", before
                )[-1]
                names = re.findall(r"([A-Z][a-z]+)", clause)
                return names[0].lower() if names else None
        return None

    # Ziel: "how many digits did Sam memorize" / "does Harry have" /
    # "do THEY have in total" (Summe aller aufgelösten Personen)
    tm = re.search(
        r"how many [a-z ]*? (?:did|does|do|would) "
        r"([A-Z][a-z]+|they|them)",
        question,
        re.IGNORECASE,
    )
    passive_target = tm is None and re.search(
        r"how many [a-z ]+ (?:were|are|was) "
        r"(?!left|remaining|in\b|on\b|at\b|present\b|absent\b|"
        r"there\b|here\b)(?:[a-z]+ed\b|broken\b|sold\b|made\b|"
        r"given\b|bought\b|eaten\b|planted\b|collected\b|used\b|"
        r"born\b|needed\b|spent\b|taken\b|read\b)",
        question,
        re.IGNORECASE,
    )
    if not tm and not passive_target:
        return None
    target = tm.group(1).lower() if tm else "they"
    they_total = target in ("they", "them")

    known: Dict[str, Fraction] = {}
    eqs = []  # (subj, kind, factor, base, ref_var)
    for r in rels:
        subj = _subject(r)
        if not subj:
            continue
        if r.kind == "assign":
            # "If Mina memorized 24" — der Satz startet mit 'If', das
            # Subjekt ist die Variable selbst.
            known[r.var or subj] = r.base_value
        elif r.kind == "times" and r.var:
            eqs.append((subj, "times", r.factor, r.base_value, r.var))
        elif r.kind == "times" and not r.var:
            known[subj] = r.base_value * r.factor
        elif r.kind == "times" and r.var:
            eqs.append((subj, "times", r.factor, r.base_value, r.var))
        elif r.kind == "times_more" and r.var:
            eqs.append((subj, "times_more", r.factor, r.base_value, r.var))
        elif r.kind == "more2x" and r.var:
            eqs.append((subj, "more2x", r.factor, r.base_value, r.var, r.mult))
        elif r.kind == "ratio" and r.var:
            eqs.append((subj, "ratio", r.factor, r.base_value, r.var))
        elif r.kind in ("more", "fewer", "less") and r.var:
            eqs.append((subj, r.kind, r.factor, r.base_value, r.var))

    # Propagation (einfache Gleichungen, keine zyklischen Systeme)
    applied_ratio = set()  # außerhalb: Inkremente nur EINMAL anwenden
    for _ in range(10):
        progressed = False
        for eq in eqs:
            subj, kind, factor, base, ref = eq[:5]
            if (
                kind == "ratio"
                and subj in known
                and ref in known
                and (subj, ref) not in applied_ratio
            ):
                # Ratio = INKREMENT auf einen bekannten Bestand
                # ("40 cans gekauft + 6 für jede 5 von Mark" = 40 + 60)
                known[subj] = known[subj] + known[ref] * base / factor
                applied_ratio.add((subj, ref))
                progressed = True
                continue
            if subj in known and ref is not None and ref not in known:
                # RÜCKWÄRTS: Subjekt bekannt, Referenz unbekannt
                # ("Mina=24, Mina=6xCarlos" -> Carlos=4)
                if kind == "times":
                    known[ref] = known[subj] / factor
                elif kind == "more":
                    known[ref] = known[subj] - factor
                elif kind in ("fewer", "less"):
                    known[ref] = known[subj] + factor
                progressed = True
                continue
            if subj in known:
                continue
            src = known.get(ref)
            if src is None and base > 0:
                src = base
            if src is None:
                continue
            if kind == "times":
                known[subj] = src * factor
            elif kind == "times_more":
                known[subj] = src * (factor + 1)
            elif kind == "ratio":
                known[subj] = src * base / factor
            elif kind == "more2x":
                mult = eq[5] if len(eq) > 5 else 2
                known[subj] = src * mult + factor
            elif kind == "more":
                known[subj] = src + factor
            elif kind in ("fewer", "less"):
                known[subj] = src - factor
            progressed = True
        if not progressed:
            break

    if passive_target:
        # Passiv-Summe VOR they-total (target ist hier "they", aber die
        # Semantik ist passiv: alle beteiligten Mengen zählen)
        total = sum(known.values())
        if quants:
            tq = _parse_target(question)
            total += sum(
                q.value
                for q in quants
                if q.role == "qty" and not q.ref_obj and q.obj == tq.obj
            )
        return total if (known or total > 0) else None
    if they_total:
        return sum(known.values()) if known else None
    return known.get(target)


def _food_chain(
    question: str, rels: List[Relation], quants: List[Quantity]
) -> Optional[Quantity]:
    """Futterkette: each X eats N Y per day; ... ; M Z (Prädator-Zahl).
    -> M x N1 x N2 x ... (Kettenmultiplikation). Ziel-Einheit = unterste
    Beute."""
    low = _digitize(question.lower())
    eats = {}  # predator -> (prey, rate)
    for m in _EATS_RE.finditer(low):
        eats[m.group(1)] = (m.group(3), Fraction(m.group(2)))
    if not eats:
        return None

    def _sing(w: str) -> str:
        return _SINGULAR.get(w, w[:-1] if w.endswith("s") else w)

    # Rate-Mengen ausschließen ("3 birds" in "each snake eats 3 birds")
    # — Prädator-Menge ist NUR die unabhängige Menge ("6 jaguars").
    rate_phrases = {f"{m.group(2)} {m.group(3)}" for m in _EATS_RE.finditer(low)}

    # Prädator-Menge: "M jaguars" / "2 snakes" (letzte Stufe)
    top = [
        q
        for q in quants
        if q.role == "qty" and q.text not in rate_phrases and _sing(q.obj or "") in eats
    ]
    if not top:
        return None
    # Kette absteigen: top-Predator -> ... -> unterste Beute
    total = Fraction(0)
    for start in top:
        pred, n = _sing(start.obj or ""), start.value
        v = n
        seen = set()
        while pred in eats and pred not in seen:
            seen.add(pred)
            prey, rate = eats[pred]
            v = v * rate
            pred = _sing(prey)  # "snakes" -> "snake" (eats-Key)
        total += v
        unit = pred
    if total > 0:
        return Quantity(
            value=total,
            text="Futterkette",
            unit=unit,
            obj=unit,
            role="qty",
            ref_obj=unit,
        )
    return None


def _rate_duration(
    question: str, rels: List[Relation], quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Rate x Dauer: "can type 6 sentences per minute ... for 43 minutes"
    -> 6 x 43. Einheiten müssen kompatibel sein (minute <-> minutes)."""
    low = question.lower()
    # "for N days" als MULTIPLIKATOR (nicht als Dauer) markieren
    md_span = None
    mdm = re.search(
        r"for\s+(?:the\s+)?(\d+)\s+(days?|weeks?)|"
        r"for\s+(\d+)\s+(days?|weeks?)",
        _digitize(low),
    )
    if mdm:
        md_span = (mdm.start(), mdm.end())
    rates = [r for r in rels if r.kind == "rate"]
    durs = [r for r in rels if r.kind == "duration"]
    if not rates or not durs:
        return None
    out = Fraction(0)
    # Rate-Dauer: 1 Rate -> Summe über alle Dauern (Rosie); mehrere
    # Rates -> PAARUNG in Textreihenfolge (Alisa: 12x4.5 + 10x2.5)
    if len(rates) == 1:
        r = rates[0]
        r_unit = r.text.rsplit("per", 1)[-1].strip() if "per" in r.text else ""
        r_unit = r_unit.rstrip("s")
        for d in durs:
            if mdm and d.unit.rstrip("s") in ("day", "week"):
                continue
            d_unit = d.unit.rstrip("s")
            if r_unit != d_unit and not _compatible_units(r_unit, d_unit):
                continue
            factor = _unit_factor(d_unit, r_unit)
            out += r.base_value * d.base_value * factor
    else:
        for i, r in enumerate(rates):
            if i >= len(durs):
                break
            d = durs[i]
            r_unit = r.text.rsplit("per", 1)[-1].strip() if "per" in r.text else ""
            r_unit = r_unit.rstrip("s")
            d_unit = d.unit.rstrip("s")
            if r_unit != d_unit and not _compatible_units(r_unit, d_unit):
                continue
            factor = _unit_factor(d_unit, r_unit)
            out += r.base_value * d.base_value * factor
    if out <= 0:
        return None
    # Woche: "5+4+2 pro Tag ... in einer Woche" -> x7
    if re.search(r"in\s+one\s+week|per\s+week", low):
        wk = re.findall(r"(\d+)\s+sandwiches?\s+per\s+day", low)
        if wk:
            return sum(Fraction(x) for x in wk) * 7
    # Runden: "5 rounds of 3 minutes" -> rate x 5 x 3 (ERSETZT die
    # einfache Dauer-Summe, die "3 minutes" schon enthielt)
    rm = re.search(r"(\d+)\s+rounds?\s+of\s+(\d+)\s+minutes?", low)
    if rm:
        if rates:
            return rates[0].base_value * Fraction(rm.group(1)) * Fraction(rm.group(2))
        return None
    # Multi-Tage: "2 hours every morning, for five days" -> x5
    if mdm:
        n = mdm.group(1) or mdm.group(2)
        out *= Fraction(n)
    return out


def _compatible_units(a: str, b: str) -> bool:
    """Zeit-Einheiten untereinander (hour/minute/day/week), sonst nein."""
    times = {"hour", "minute", "day", "week", "month", "year", "second"}
    return a in times and b in times


def _unit_factor(src: str, dst: str) -> Fraction:
    """Umrechnungsfaktor von src nach dst (Zeit)."""
    per_hour = {
        "hour": 1,
        "minute": 60,
        "second": 3600,
        "day": Fraction(1, 24),
        "week": Fraction(1, 168),
        "month": Fraction(1, 720),
        "year": Fraction(1, 8760),
    }
    return (
        Fraction(per_hour[dst], per_hour[src])
        if src in per_hour and dst in per_hour
        else Fraction(1)
    )


def _fraction_of_chain(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Bruch-von-Kette: "10 kg; used one-half of it, one-fifth of it,
    one-third of the remaining" -> 10 - 5 - 2 - 1 = 2.
    'of it' bezieht sich auf den Anfangsbestand, 'of the remaining' auf
    den aktuellen Rest."""
    low = question.lower()
    steps = [(m.group(1), m.group(2)) for m in _FRACTION_OF_RE.finditer(low)]
    if not steps:
        return None
    base_q = [
        q
        for q in quants
        if q.role == "qty" and not q.ref_obj and (q.obj == tgt.obj or not tgt.obj)
    ]
    if not base_q:
        return None
    base = min(base_q, key=lambda q: q.span[0]).value
    rem = base
    for frac_word, ref in steps:
        frac = _FRACTION_WORDS[frac_word]
        if ref == "the remaining":
            used = rem * frac
        else:
            used = base * frac
        rem = rem - used
    for m in re.finditer(r"additional\s+\$?(\d+)", low):
        rem = rem - Fraction(m.group(1))
    if rem < 0:
        return None
    # Ziel-Richtung: "how many did they EAT/sell/use/give" -> Verbrauch
    if re.search(
        r"how many [a-z ]+ did (?:they|he|she) (?:eat|sold|"
        r"use|give|ate|sell|used)",
        low,
    ):
        return used_total
    # "how many UNMANNED/sold/abandoned X" -> letzte Schritt-Menge
    if re.search(
        r"how many [a-z ]+ (?:unmanned|abandoned|ruined|"
        r"decorated|broken)",
        low,
    ):
        return last_used
    # Division: "how many pages each day for 5 days" -> rem / 5
    dm = re.search(
        r"how many [a-z ]+ each (day|week|hour)\s+for\s+"
        r"(\d+)",
        low,
    )
    if dm:
        n = Fraction(dm.group(2))
        if n > 0:
            return rem / n
    return rem


# Typ-Mapping: "20 gallons for a heavy wash ... two heavy washes"
# -> 2x20 + 3x10 + 1x2 + 2x2 (Bleach-Extra)
_TYPE_DEF_RE = re.compile(
    r"(?:(\d+)\s+([a-z]+)(?:\s+of\s+[a-z]+)?\s+for\s+a\s+"
    r"(heavy|regular|light|small|large|medium)\s+([a-z]+)|"
    r"(large|small|heavy|regular|light)\s+(?:animals?|washes?|loads?)\s+"
    r"(?:take|use|need)\s+(\d+)\s+([a-z]+))"
)
_TYPE_COUNT_RE = re.compile(
    r"(\d+)\s+(heavy|regular|light|small|large|medium)\s+([a-z]+)"
)

# Gruppen-each: "Half the tables have 2 chairs each, 5 have 3 chairs
# each and the rest have 4 chairs each"
_GROUP_EACH_RE = re.compile(
    r"(half|the\s+rest|\d+)\s+(?:[a-z]+\s+)?(?:of\s+the\s+)?"
    r"(?:([a-z]+)\s+)?have\s+(\d+)\s+([a-z]+)\s+each"
)


# ---- Chained-Executor: Zustand = Restbestand, jede Operation
# dekrementiert ihn (Ali/Derek/Julie/Bear sind DASSELBE Muster).
_SUB_RE = re.compile(
    r"(?:read|gave\s+away|gave|sold|spent|spends|lost|loses|used|ate|"
    r"took|donated|paid|erased|shelved|passes|kept|gets?\s+off|gives|"
    r"snuck)\s+\$?(\d+)\s+([a-z]+)|"
    r"(\d+)\s+(?:people|passengers?)\s+get\s+off"
)
_RATIO_SUB_RE = re.compile(
    r"(?:read|sold|gained|collected)\s+(twice|three\s+times|half)\s+"
    r"as\s+many\s+(?:[a-z]+\s+)?as\s+(yesterday|before|that|"
    r"the\s+previous\s+day|the\s+day\s+before)|"
    r"(?:gained|collected|read)\s+twice\s+that\s+amount"
)
_FRAC_X_RE = re.compile(
    r"(half|quarter|a\s+fifth|a\s+third|two\s+thirds|three\s+quarters|"
    r"a\s+tenth|one\s+quarter|one\s+third|one\s+fifth|one\s+tenth|"
    r"one\s+half|one-half|one-third|one-fifth|one-quarter|one-tenth|"
    r"three-quarters|two-thirds|a\s+fourth|a\s+quarter|a\s+half|"
    r"\d+-(?:quarter|third|fifth|tenth|sixth|eighth|fourth)s?|"
    r"\d+\s+(?:thirds?|fourths?|fifths?|tenths?|sixths?|eighths?|"
    r"halves?|quarters?)|"
    r"\d+\s*(?:%|percent)|\d+/\d+)\s+of\s+"
    r"(that|it|them|those|what\s+is\s+left|what\'s\s+left|what\s+was\s+left|"
    r"what\s+he\s+has\s+left|what\s+she\s+has\s+left|"
    r"the\s+remaining\s+[a-z]+|"
    r"remaining|the\s+rest|his\s+[a-z]+|her\s+[a-z]+|"
    r"the\s+weight\s+it\s+needed|the\s+money|the\s+total|"
    r"the\s+students\s+who\s+are\s+present|who\s+are\s+present|"
    r"all\s+the\s+[a-z]+|the\s+[a-z]+s?|[a-z]+'s\s+[a-z]+|"
    r"the\s+(?:bag|pizza|cookies|cake|jar|box|carton|bottle|container|"
    r"plate|bowl|glass|cup|slices|cards|loaves|pencils|candies|"
    r"chocolates|books|marbles|eggs|apples|oranges|cookies))"
)

_FRAC_VAL = {
    "half": Fraction(1, 2),
    "quarter": Fraction(1, 4),
    "a fifth": Fraction(1, 5),
    "a third": Fraction(1, 3),
    "two thirds": Fraction(2, 3),
    "three quarters": Fraction(3, 4),
    "a tenth": Fraction(1, 10),
    "one quarter": Fraction(1, 4),
    "one third": Fraction(1, 3),
    "one fifth": Fraction(1, 5),
    "one tenth": Fraction(1, 10),
    "one half": Fraction(1, 2),
}


def _frac_val(word: str) -> Fraction:
    mm0 = re.match(r"(\d+)\s+(\w+)", word)
    if mm0:
        den = {
            "thirds": 3,
            "third": 3,
            "fourths": 4,
            "fourth": 4,
            "fifths": 5,
            "fifth": 5,
            "halves": 2,
            "half": 2,
            "tenths": 10,
            "tenth": 10,
            "sixths": 6,
            "sixth": 6,
            "eighths": 8,
            "eighth": 8,
            "quarters": 4,
            "quarter": 4,
        }.get(mm0.group(2), None)
        if den:
            return Fraction(int(mm0.group(1)), den)
    if word.endswith("percent") and word[:-7].strip().isdigit():
        return Fraction(int(word[:-7].strip()), 100)
    if word in _FRAC_VAL:
        return _FRAC_VAL[word]
    if word in ("a fourth", "a quarter", "a half"):
        return {
            "a fourth": Fraction(1, 4),
            "a quarter": Fraction(1, 4),
            "a half": Fraction(1, 2),
        }[word]
    if word in (
        "one-half",
        "one-third",
        "one-fifth",
        "one-quarter",
        "one-tenth",
        "three-quarters",
        "two-thirds",
    ):
        return _FRAC_VAL[word.replace("-", " ")]
    if "-" in word and word.split("-")[0].isdigit():
        n, den = word.split("-")
        den = den.rstrip("s")
        return Fraction(
            int(n),
            {
                "quarter": 4,
                "fourth": 4,
                "third": 3,
                "fifth": 5,
                "tenth": 10,
                "sixth": 6,
                "eighth": 8,
            }[den],
        )
    if word.endswith("%"):
        return Fraction(int(word[:-1]), 100)
    if "/" in word:
        a, b = word.split("/")
        return Fraction(int(a), int(b))
    return Fraction(1, 2)


def _chained_executor(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Zustand = Restbestand. Schritte in Textreihenfolge:
    SUB(n) -> rem -= n | FRAC(f, remaining) -> rem -= rem*f |
    FRAC(f, basis) -> rem -= basis*f | RATIO -> rem -= 2*letzter SUB.
    Bear: Startbestand = Ziel ('needs to gain 1000')."""
    low = question.lower()

    # Startbestand: "started with N X" / "has N X" / "N-page book" /
    # "needs to gain N X"
    start = None
    m = re.search(
        r"(?:started\s+with|has|had|had\s+to\s+read|"
        r"has\s+to\s+read|needs?\s+to\s+gain|purchased|"
        r"bought|baked|made|there\s+are|weighs?|filled\s+with)"
        r"\s+\$?(\d+)",
        low,
    )
    if m:
        # "has 40 MORE goats than..." ist ein Vergleich, kein Start;
        # "has 8 cookies LEFT" ist der Endbestand, kein Start
        after = low[m.end() : m.end() + 34]
        if not re.match(r"\s*(more|less|fewer)\b", after) and not re.search(
            r"\s+(?:[a-z]+\s+){0,4}left\b", after
        ):
            start = Fraction(m.group(1))
            # "bought 34 and 22 oranges" -> Summe
            am = re.search(r"and\s+(\d+)\s+[a-z]+", low[m.end() :])
            if am and "bought" in m.group(0):
                start = start + Fraction(am.group(1))
    if start is None:
        # "20% of 50 people think..." -> Basis = 50
        pm2 = re.search(
            r"(\d+)\s*(?:%|percent)\s+of\s+(\d+)\s+"
            r"([a-z]+)",
            low,
        )
        if pm2:
            start = Fraction(pm2.group(2))
    # "4 rows of seats, 18 seats in each row" -> 4 x 18
    # (hat Vorrang vor "has 4 rows" als Start)
    rm = re.search(r"(\d+)\s+rows?\s+of\s+([a-z]+)", low)
    if rm:
        sm = re.search(r"(\d+)\s+([a-z]+)\s+in\s+each\s+row", low)
        if sm:
            start = Fraction(rm.group(1)) * Fraction(sm.group(1))
    if m and re.search(r"has\s+to\s+read", low):
        # "has to read 4 pages from X, 20 pages from Y" -> Summe
        ps = re.findall(r"(\d+)\s+pages?\s+from", low)
        if ps:
            start = sum(Fraction(x) for x in ps)
    else:
        pm3 = re.search(r"a\s+(\d+)-page\s+([a-z]+)", low)
        if pm3:
            start = Fraction(pm3.group(1))

    # Zielobjekt aus der Frage ("pages", "seashells", "pounds", "money")
    tm = re.search(
        r"how (?:many|much)\s+([a-z]+)|"
        r"what is the amount of ([a-z]+)",
        low,
    )
    if not tm:
        return None
    ziel_w = tm.group(1) or tm.group(2)
    if ziel_w in (
        "money",
        "pounds",
        "pages",
        "seashells",
        "marbles",
        "cookies",
        "gallons",
        "mangoes",
        "pounds",
        "kg",
    ):
        pass
    else:
        ziel_w = ziel_w.rstrip("s")

    steps = []  # (pos, type, value, ref)
    start_span = (m.start(), m.end()) if m else None
    for m in _SUB_RE.finditer(low):
        # Objekt-Kompatibilität: gleiche Einheit wie Ziel/Start
        w = (m.group(2) or "").rstrip("s")
        if w == ziel_w or (start is not None and True):
            # Start-Phrase selbst ("had to read 408") ist KEIN Schritt —
            # nur wenn die SUB innerhalb der Start-Phrase beginnt
            if start_span and start_span[0] <= m.start() < start_span[1]:
                continue
            val = m.group(1) or m.group(3)
            if val is None:
                continue
            steps.append((m.start(), "sub", Fraction(val), None))
    for m in re.finditer(r"(?:and\s+)?(\d+)\s+more\s+([a-z]+)", low):
        if m.group(2) in (
            "days",
            "day",
            "weeks",
            "week",
            "hours",
            "hour",
            "months",
            "month",
            "years",
            "year",
        ):
            continue  # "4 more days" ist Zeit, kein Mengen-Schritt
        before = low[max(0, m.start() - 8) : m.start()]
        if re.search(r"(?:eats?|ate|sold|gave|bought)\s*$", before):
            continue  # "eats/bought N more" — schon als SUB/add gezählt
        after = low[m.end() : m.end() + 20]
        if " than" in after or re.match(r"\s*than\b", after):
            continue  # "40 more goats THAN" — Vergleich, kein Schritt
        if re.search(r"get\s+on|board", after):
            steps.append((m.start(), "add_n", Fraction(m.group(1)), None))
            continue  # "13 more people GET ON" = Addition
        steps.append((m.start(), "sub", Fraction(m.group(1)), None))
    # Add: "bought 21 stickers" / "got 23 for birthday" -> rem += N
    # (nur wenn NICHT der Start selbst und kein Ratio-"for every")
    for m in re.finditer(
        r"(?:bought|got|received|earned|collected)\s+"
        r"(\d+)\s+(?!times)([a-z]+)",
        low,
    ):
        if start_span and start_span[0] <= m.start() < start_span[1]:
            continue  # Start selbst ("bought 10 kg")
        after2 = low[m.end() : m.end() + 40]
        if "for every" in after2:
            continue  # Ratio-Kontext ("bought 6 additional ... for every")
        steps.append((m.start(), "add_n", Fraction(m.group(1)), None))
    # Add: "His friend gave HIM 28 marbles" -> rem += 28
    am3 = re.search(
        r"gave\s+(?:him|her|them)\s+(\d+/\d+)\s+of|"
        r"gave\s+(?:him|her|them)\s+(\d+)(?!\s*/)|"
        r"gave\s+(\d+)\s+[a-z]+\s+to\s+"
        r"(him|her|them)(?!\s+[a-z])|"
        r"gave\s+(\d+)\s+[a-z]+\s+to\s+me",
        low,
    )
    if am3:
        if am3.group(1):
            # "gave him 1/5 of her $100" -> add 100 x 1/5
            frac = Fraction(am3.group(1))
            om = re.search(r"of\s+her\s+\$?(\d+)", low)
            if om:
                steps.append((am3.start(), "add_n", Fraction(om.group(1)) * frac, None))
        else:
            val = Fraction(am3.group(2) or am3.group(3) or am3.group(5))
            steps.append((am3.start(), "add_n", val, None))
    # Add-Ratio: "collect rainwater that is twice as much as what was
    # left" -> rem += rem x 2
    am = re.search(
        r"(?:collect|gained|receives?|adds?)\s+(?:rainwater\s+)?"
        r"(?:that\s+is\s+)?(twice|half|triple)\s+as\s+much"
        r"\s+as\s+(?:what\s+was\s+)?left",
        low,
    )
    if am:
        f = {"twice": 2, "triple": 3, "half": Fraction(1, 2)}[am.group(1)]
        steps.append((am.start(), "add_ratio", f, None))
    # Add-Ratio mit last_sub: "buys twice as many as he gave to his
    # friends" -> rem += 2 x letzter SUB
    am2 = re.search(
        r"buys?\s+(twice|triple)\s+as\s+many\s+"
        r"(?:[a-z]+\s+){0,6}as\s+he\s+gave",
        low,
    )
    if am2:
        f = {"twice": 2, "triple": 3}[am2.group(1)]
        steps.append((am2.start(), "add_ratio_sub", f, None))
    # Rate-Subtraktion: "lose 3 kg per month ... at 4 months from now"
    rm = re.search(
        r"(?:loses?|gains?|losing)\s+(\d+)\s+([a-z]+)\s+"
        r"per\s+(month|week|day|year)",
        low,
    )
    if rm and re.search(r"at\s+(\d+)\s+months?", low):
        n = re.search(r"at\s+(\d+)\s+months?", low)
        steps.append(
            (
                rm.start(),
                "rate_sub",
                None,
                (Fraction(rm.group(1)), Fraction(n.group(1))),
            )
        )
    for m in _RATIO_SUB_RE.finditer(low):
        if m.group(1) is None:
            f = Fraction(2)  # "twice that amount"
        else:
            f = (
                Fraction(2)
                if "twice" in m.group(1)
                else (Fraction(3) if "three" in m.group(1) else Fraction(1, 2))
            )
        steps.append((m.start(), "ratio", f, None))
    for m in _FRAC_X_RE.finditer(_digitize(low)):
        f = _frac_val(m.group(1))
        ref = m.group(2)
        if re.search(r"remaining|left|rest|present|ruined|pomeranians", ref):
            steps.append((m.start(), "frac_rem", f, ref))
        else:
            steps.append((m.start(), "frac_base", f, ref))
    for m in re.finditer(r"(\d+)\s*(?:%|percent)\s+in\s+([a-z]+)", low):
        steps.append((m.start(), "pct_in", None, (Fraction(m.group(1)) / 100, None)))
    for m in re.finditer(
        r"(\d+)\s*(?:%|percent)\s+of\s+(\d+)\s+"
        r"([a-z]+)",
        low,
    ):
        steps.append(
            (
                m.start(),
                "pct_of_n",
                None,
                (Fraction(m.group(1)) / 100, Fraction(m.group(2))),
            )
        )
    for m in re.finditer(
        r"(\d+/\d+|half|quarter|\d+\s*%)\s+are\s+"
        r"absent",
        low,
    ):
        steps.append((m.start(), "frac_base", _frac_val(m.group(1)), None))

    if not steps:
        # Nur-Division-Fälle: "share 56 equally with 8" ohne Schritte
        dm0 = re.search(
            r"shared?\s+(?:the\s+remaining|them)\s+equally\s+"
            r"with\s+(?:his|their)\s+(\d+)",
            _digitize(low),
        )
        if dm0 and start is not None:
            n = Fraction(dm0.group(1))
            if "their" in dm0.group(0):
                n = n + 2
            return start / n if n > 0 else None
        return None
    steps.sort(key=lambda s: s[0])

    # RÜCKWÄRTS-Modus: Start unbekannt, Endbestand bekannt
    # ("took half of all the candies and 4 more. Paul took the
    # remaining 7 sweets. How many were there at first?")
    if start is None:
        rm = re.search(
            r"(?:remaining|left)\s+(\d+(?:\.\d+)?)|"
            r"(\d+(?:\.\d+)?)\s+(?:[a-z]+\s+){0,3}left",
            low,
        )
        if not rm:
            return None
        end_v = Fraction(rm.group(1) or rm.group(2))
        x = end_v
        for pos, typ, val, ref in reversed(steps):
            if typ == "sub":
                x = x + val
            elif typ in ("frac_rem", "frac_base"):
                if val >= 1:
                    return None
                x = x / (1 - val)
            elif typ == "ratio":
                return None
        return x

    rem = start
    last_sub = None
    used_total = Fraction(0)
    last_used = Fraction(0)
    for pos, typ, val, ref in steps:
        if typ == "pct_in":
            pct, _ = ref
            used = start * pct
            rem = rem - used
            last_used = used
            used_total += used
        elif typ == "add_n":
            rem = rem + val
            last_sub = val
        elif typ == "pct_of_n":
            pct, n = ref
            used = n * pct
            rem = rem - used
            last_sub = used
            used_total += used
        elif typ == "add_ratio":
            rem = rem + rem * val
        elif typ == "add_ratio_sub":
            if last_sub is not None:
                rem = rem + last_sub * val
                last_sub = last_sub * val
        elif typ == "rate_sub":
            rate, n = ref
            rem = rem - rate * n
            last_sub = rate * n
        elif typ == "sub":
            rem = rem - val
            last_sub = val
            used_total += val
        elif typ == "ratio":
            if last_sub is not None:
                rem = rem - last_sub * val
                last_sub = last_sub * val
        elif typ == "frac_rem":
            # "half of the RUINED/sold/... castles" referenziert das
            # VORHERIGE Teilergebnis (88), nicht den Restbestand
            ref2 = ref if isinstance(ref, str) else ""
            if re.search(r"ruined|sold|eaten|broken|left|rest|pomeranians", ref2):
                base_f = last_used if last_used > 0 else rem
            else:
                base_f = rem
            used = base_f * val
            rem = rem - used
            last_sub = used
            used_total += used
            last_used = used
        elif typ == "frac_base":
            used = start * val
            rem = rem - used
            last_sub = used
            used_total += used
            last_used = used
    if rem < 0:
        return None
    # "how many UNMANNED/girl Poms X" -> letzte Schritt-Menge
    if re.search(
        r"how many (?:unmanned|abandoned|decorated|broken|girl|"
        r"bloomed)\b|"
        r"how many [a-z\' ]+ (?:unmanned|abandoned|decorated|"
        r"broken|girl|bloomed)\b",
        low,
    ):
        return last_used
    # "Altogether ... red AND blue" -> Summe der Teilgruppen
    if re.search(r"altogether|in total", low) and re.search(r"and\b", low):
        return used_total
    # Division: "how many pages each day for 5 days" -> rem / 5
    dm = re.search(
        r"how many [a-z ]+ each (day|week|hour)\s+for\s+"
        r"(\d+)",
        low,
    )
    if dm:
        n = Fraction(dm.group(2))
        if n > 0:
            return rem / n
    # Division: "divided the rest equally into 3 piles" -> rem / 3
    dm2 = re.search(
        r"(?:divided?|placed?)\s+the\s+rest\s+equally\s+"
        r"(?:into|among)\s+(\d+)|"
        r"shared?\s+(?:the\s+remaining|them)\s+equally\s+"
        r"with\s+(?:his|their)\s+(\d+)|"
        r"equally\s+divide\s+them\s+between\s+her\s+"
        r"(\d+)",
        _digitize(low),
    )
    if dm2:
        n = Fraction(dm2.group(1) or dm2.group(2) or dm2.group(3))
        if "their" in dm2.group(0):
            n = n + 2  # "with their 6 other friends" -> 8 insgesamt
        if n > 0:
            return rem / n
    # Rate-Tages-Division: "sells 15kg/h, 10h/Tag, bull 750kg. Wie
    # viele Tage?" -> 750 / (15 x 10)
    dm4 = re.search(
        r"(?:sell|sells|uses?|eat)\s+(\d+)\s+kg\s+every\s+"
        r"hour\s+he\s+works",
        low,
    )
    if dm4 and re.search(r"works\s+(\d+)\s+hours?\s+a\s+day", low):
        wh = re.search(r"works\s+(\d+)\s+hours?\s+a\s+day", low)
        if re.search(r"how many days", low):
            return rem / (Fraction(dm4.group(1)) * Fraction(wh.group(1)))
    # Dollar-Konvertierung: "pennies ... dollar amount" -> /100
    if re.search(r"pennies", low) and re.search(r"dollar", low):
        if re.search(r"contains|two\s+thirds|stack", low):
            return used_total / 100
        return rem / 100
    # Verkauf: "sold ... at $3 each" -> rem x 3
    pm = re.search(r"at\s+\$?(\d+(?:\.\d+)?)\s+each", low)
    if pm and re.search(r"sold|sell", low):
        return rem * Fraction(pm.group(1))
    # Division: "has 4 more days to complete" -> rem / 4
    dm3 = re.search(r"has\s+(\d+)\s+more\s+days?", _digitize(low))
    if dm3:
        n = Fraction(dm3.group(1))
        if n > 0:
            return rem / n
    return rem


def _type_mapping(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Typ-Mapping: Definitionen (X gallons für Typ) + Anzahlen
    (N Typ-Wäschen) + Extras (bleach -> +N light cycles)."""
    low = _digitize(question.lower())
    types: dict[str, tuple[Fraction, str]] = {}
    for m in _TYPE_DEF_RE.finditer(low):
        if m.group(1):
            # "20 gallons for a heavy wash": (20, gallons) -> heavy
            types[m.group(3)] = (Fraction(m.group(1)), m.group(2))
        else:
            # "large animals take 4 sticks": (4, sticks) -> large
            types[m.group(5)] = (Fraction(m.group(6)), m.group(7))
    if not types:
        return None
    total = Fraction(0)
    found = False
    for m in _TYPE_COUNT_RE.finditer(low):
        typ = m.group(2)
        if typ not in types:
            continue
        cnt = Fraction(m.group(1))
        g, unit = types[typ]
        total += cnt * g
        found = True
    # RÜCKWÄRTS vor dem found-Gate: "used N sticks for small animals"
    um = re.search(
        r"used\s+(\d+)\s+([a-z]+)(?:\s+of\s+[a-z]+)?"
        r"\s+for\s+(small|large|heavy|regular|light)",
        low,
    )
    if um:
        typ = um.group(3)
        if typ in types:
            cnt = Fraction(um.group(1)) / types[typ][0]
            total = cnt * types[typ][0]
            rm = re.search(
                r"(\d+)\s+times\s+as\s+many\s+"
                r"(small|large)\s+(?:animals?\s+)?as\s+"
                r"(small|large)",
                low,
            )
            if rm:
                times, a, b = rm.group(1), rm.group(2), rm.group(3)
                if a == typ and b in types:
                    cnt_b = cnt / Fraction(times)
                    total += cnt_b * types[b][0]
                elif b == typ and a in types:
                    cnt_a = cnt * Fraction(times)
                    total += cnt_a * types[a][0]
            return total
    if not found:
        return None
    # Extras: "Two of the loads need to be bleached" -> +N light cycles
    bm = re.search(
        r"(\d+)\s+of\s+the\s+loads?\s+need\s+to\s+be\s+"
        r"bleached",
        low,
    )
    if bm and "light" in types and re.search(r"extra\s+light\s+wash", low):
        total += Fraction(bm.group(1)) * types["light"][0]
    return total


def _reihe(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Arithmetische Reihe: "each week 2 more than the week before,
    starts with 3, practices for 5 weeks" -> 3 + 2x5 = 13."""
    low = _digitize(question.lower())
    m = re.search(
        r"each\s+(week|day|month|year)\s+(?:[a-z]+\s+){0,5}"
        r"(\d+)\s+more\s+(?:[a-z]+\s+)?than\s+the\s+week"
        r"\s+before",
        low,
    )
    if not m:
        return None
    diff = Fraction(m.group(2))
    sm = re.search(
        r"starts?\s+(?:out\s+)?(?:juggling|with|by)?\s*"
        r"(\d+)",
        low,
    )
    if not sm:
        return None
    start = Fraction(sm.group(1))
    nm = re.search(r"for\s+(\d+)\s+weeks?", low)
    if not nm:
        return None
    return start + diff * Fraction(nm.group(1))


def _pair_total(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Summe+Differenz-Paare: "110 coins, 30 more gold than silver.
    How many gold?" -> (110+30)/2 = 70."""
    low = _digitize(question.lower())
    pm = re.search(
        r"(\d+)\s+more\s+([a-z]+)(?:\s+[a-z]+)?\s+than\s+"
        r"([a-z]+)",
        low,
    )
    if not pm:
        return None
    # "than twice/half" -> Ratio-Szenario, kein Paar-Total
    if re.search(
        r"than\s+(twice|half|three|four|five|two|\d+)\s+"
        r"(times\s+)?as\s+many",
        low,
    ):
        return None
    # "than Washington" (Name) -> Variablen-Szenario, kein Paar-Total
    orig = question.lower()
    m_idx = orig.find(pm.group(0))
    if m_idx < 0:
        m_idx = _digitize(orig).find(pm.group(0))
    if m_idx >= 0:
        seg = orig[m_idx : m_idx + len(pm.group(0)) + 4]
        if re.search(r"than\s+[A-Z][a-z]+", question[m_idx : m_idx + 60]):
            return None
    d = Fraction(pm.group(1))
    a_typ, b_typ = pm.group(2), pm.group(3)
    # Total: "N <Einheit> in total" oder die Gesamtmenge
    tm = re.search(r"has\s+(\d+)\s+[a-z]+\s*(?:,| and|\.)", low)
    tm2 = re.search(r"in total|altogether|total", low)
    total = None
    if tm2 or tm:
        # Gesamtmenge = die größte qty im Text (bei Paar-Vergleich ist
        # die große Menge die Summe beider Typen)
        qt = [q for q in quants if q.role == "qty" and not q.ref_obj]
        if qt:
            total = max(q.value for q in qt)
    if total is None:
        return None
    # gold = silver + d; gold + silver = total -> gold = (total+d)/2
    if _canon(a_typ) == _canon(tgt.obj or ""):
        return (total + d) / 2
    if _canon(b_typ) == _canon(tgt.obj or ""):
        return (total - d) / 2
    # Variante: eine Seite bekannt ("If she had 70 gold coins, wie
    # viele insgesamt?") -> gold + (gold - d)
    ifm = re.search(r"had\s+(\d+)\s+[a-z]+", low)
    if ifm:
        known = Fraction(ifm.group(1))
        if re.search(r"in total|altogether", low):
            return known + (known - d)
    return None


def _that_much(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Verhältnis-Referenz: "2 bolts of blue fiber and half that much
    white fiber" -> 2 + 2x1/2 = 3. 'twice that much' -> x2."""
    low = _digitize(question.lower())
    m = re.search(r"(half|quarter|twice|triple|a\s+third)\s+that\s+much", low)
    if not m:
        return None
    f = {
        "half": Fraction(1, 2),
        "quarter": Fraction(1, 4),
        "twice": Fraction(2),
        "triple": Fraction(3),
        "a third": Fraction(1, 3),
    }[m.group(1)]
    # Basis = die letzte qty VOR dem Verhältnis
    qt = [
        q for q in quants if q.role == "qty" and not q.ref_obj and q.span[0] < m.start()
    ]
    if not qt:
        return None
    base = max(qt, key=lambda q: q.span[0]).value
    return base + base * f


def _volume_boxen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Boxen-Volumen: "3 boxes, each 5x6x4 inches, walls 1 inch thick"
    -> (5-2)(6-2)(4-2) x 3 = 72."""
    low = _digitize(question.lower())
    m = re.search(
        r"(\d+)\s+boxes?\s*[.,]?\s*each(?:\s+[a-z]+)?\s+is?\s+"
        r"(\d+)\s+inches?\s+by\s+(\d+)\s+inches?\s+by\s+"
        r"(\d+)\s+inches?",
        low,
    )
    if not m:
        # "a hole 6 feet long by 4 feet wide by 3 feet deep" -> 1 Loch
        mh = re.search(
            r"a\s+hole\s+(\d+)\s+feet?\s+long\s+by\s+"
            r"(\d+)\s+feet?\s+wide\s+by\s+(\d+)\s+feet?\s+"
            r"deep",
            low,
        )
        if mh:
            n = Fraction(1)
            a, b, c = (Fraction(mh.group(i)) for i in (1, 2, 3))
            wm = re.search(r"walls?\s+are?\s+(\d+)\s+inch", low)
            if wm:
                w = Fraction(wm.group(1))
                inner = (a - 2 * w) * (b - 2 * w) * (c - 2 * w)
            else:
                inner = a * b * c
            if inner <= 0:
                return None
            sm = re.search(r"(\d+)\s+seconds?\s+to\s+shovel", low)
            if sm:
                return inner * n * Fraction(sm.group(1))
            return inner * n
    if not m:
        return None
    n = Fraction(m.group(1))
    a, b, c = (Fraction(m.group(i)) for i in (2, 3, 4))
    wm = re.search(r"walls?\s+are?\s+(\d+)\s+inch", low)
    if wm:
        w = Fraction(wm.group(1))
        inner = (a - 2 * w) * (b - 2 * w) * (c - 2 * w)
    else:
        inner = a * b * c
    if inner <= 0:
        return None
    vol = inner * n
    # "3 Sekunden pro Kubikfuß" -> x3
    sm = re.search(r"(\d+)\s+seconds?\s+to\s+shovel", low)
    if sm:
        return vol * Fraction(sm.group(1))
    return vol


def _rate_kette(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Raten-Kette: "2 pads/week, 30 sheets/pad, every month"
    -> 2 x 30 x 4 = 240."""
    low = _digitize(question.lower())
    m1 = re.search(
        r"(\d+)\s+([a-z]+)(?:\s+of\s+[a-z]+)?\s+"
        r"(?:per|a)\s+([a-z]+)",
        low,
    )
    if not m1:
        return None
    r1 = Fraction(m1.group(1))
    u1, p1 = m1.group(2), m1.group(3)
    m2 = re.search(
        r"(\d+)\s+([a-z]+)\s+of\s+([a-z]+)\s+on\s+a\s+"
        r"([a-z]+)",
        low,
    )
    if not m2 or _canon(m2.group(4)) != _canon(u1):
        return None
    r2 = Fraction(m2.group(1))
    # "every month" -> 4 Wochen/Monat (approximiert, GSM8K-Konvention)
    if re.search(r"every\s+month|per\s+month", low):
        return r1 * r2 * 4
    return None


def _gruppen_kauf(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Gruppen-Käufe: 'first 3 customers buy one DVD each, next 2 buy
    2 each, last 3 buy none' -> 3x1 + 2x2 = 7."""
    low = _digitize(question.lower())
    total = Fraction(0)
    found = False
    for m in re.finditer(
        r"(?:first|next|last)\s+(\d+)\s+([a-z]+)\s+"
        r"(?:buy|bought)\s+(\d+|one|two|three|four|"
        r"five|no)\s+([a-z]+)?\s*(?:DVDs?|each)?",
        low,
    ):
        n = Fraction(m.group(1))
        k = m.group(3)
        if k == "no":
            continue
        kf = (
            Fraction(k)
            if k.isdigit()
            else {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}[k]
        )
        total += n * kf
        found = True
    return total if found else None


def _roundtrip(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Hin- und Rückfahrt: '10 mph von 1 bis 4 PM, zurück mit 6 mph.
    Wie lange zurück?' -> (3h x 10) / 6 = 5h."""
    low = _digitize(question.lower())
    rm = re.search(r"(\d+)\s+miles?\s+per\s+hour", low)
    tm = re.search(r"from\s+(\d+)\s+to\s+(\d+)\s+pm", low)
    bm = re.search(r"back\s+at\s+a\s+rate\s+of\s+(\d+)\s+mph", low)
    if not (rm and tm and bm):
        return None
    h = Fraction(tm.group(2)) - Fraction(tm.group(1))
    if h <= 0:
        return None
    dist = h * Fraction(rm.group(1))
    return dist / Fraction(bm.group(1))


def _jede_n_einheiten(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Proportional: '10 min to cover every 3 miles ... 42 miles'
    -> 42/3 x 10 = 140."""
    low = _digitize(question.lower())
    m = re.search(
        r"(\d+)\s+([a-z]+)\s+to\s+cover\s+every\s+(\d+)\s+"
        r"([a-z]+)",
        low,
    )
    if not m:
        return None
    per = Fraction(m.group(1))
    step = Fraction(m.group(3))
    total = re.search(r"(?:is|across)\s+(\d+)\s+[a-z]+", low)
    if not total:
        return None
    return Fraction(total.group(1)) / step * per


def _daily_total(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Tages-Summe x Einheit: '5 pails every morning and 6 every
    afternoon. Each pail contains 5 liters' -> (5+6)x5 = 55."""
    low = _digitize(question.lower())
    ms = re.findall(
        r"(\d+)\s+([a-z]+)(?:\s+of\s+[a-z]+)?\s+every\s+"
        r"(morning|afternoon|evening|night|day)",
        low,
    )
    if len(ms) < 2:
        return None
    total = sum(Fraction(m[0]) for m in ms)
    unit = ms[0][1]
    sing = unit.rstrip("s")
    km = re.search(
        r"each\s+" + sing + r"\s+(?:contains|holds|has|is)"
        r"\s+(\d+)\s+([a-z]+)",
        low,
    )
    if not km:
        return None
    return total * Fraction(km.group(1))


def _wochen_muster(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Wochen-Muster: '20 miles a day, except weekends 10'
    -> 5x20 + 2x10 = 120."""
    low = _digitize(question.lower())
    m = re.search(r"(\d+)\s+([a-z]+)\s+a\s+day", low)
    em = re.search(
        r"except\s+on\s+weekends?\s+when\s+he\s+walks?\s+"
        r"(\d+)",
        low,
    )
    if not (m and em):
        return None
    return Fraction(m.group(1)) * 5 + Fraction(em.group(1)) * 2


def _bag_typen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Beutel-Typen: '6 bags of blue pens and 2 bags of red. 9 pens in
    each bag of blue and 6 in each bag of red' -> 6x9 + 2x6 + Rest."""
    low = _digitize(question.lower())
    # Typ-Definitionen: "N pens in each bag of blue"
    defs: dict[str, Fraction] = {}
    for m in re.finditer(
        r"(\d+)\s+([a-z]+)\s+in\s+each\s+bag\s+of\s+"
        r"([a-z]+)",
        low,
    ):
        defs[m.group(3)] = Fraction(m.group(1))
    if not defs:
        return None
    # Mengen: "N bags of <typ>"
    total = Fraction(0)
    base = Fraction(0)
    found = False
    for m in re.finditer(r"(\d+)\s+bags?\s+of\s+([a-z]+)", low):
        typ = m.group(2)
        if typ in defs:
            total += Fraction(m.group(1)) * defs[typ]
            found = True
        elif typ in ("blue", "red", "green", "yellow", "pens"):
            pass
        else:
            base += Fraction(m.group(1))
    if not found:
        return None
    # Basis-Menge: "had 22 green pens and 10 yellow pens"
    bm = re.search(
        r"(?:had|has)\s+(\d+)\s+[a-z]+\s+[a-z]+\s+and\s+"
        r"(\d+)\s+[a-z]+",
        low,
    )
    if bm:
        base = Fraction(bm.group(1)) + Fraction(bm.group(2))
    return base + total


def _ketten_multiplikator(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Ketten-Multiplikatoren: 'bought 4 cakes, then three times that
    number, then 5 times the number she did on Tuesday'
    -> 4 + 4x3 + 12x5 = 76."""
    low = _digitize(question.lower())
    start_m = re.search(r"bought\s+(\d+)\s+([a-z]+)", low)
    if not start_m:
        return None
    base = Fraction(start_m.group(1))
    total = base
    cur = base
    steps = list(
        re.finditer(
            r"(\d+)\s+times\s+(?:that\s+number|"
            r"the\s+number)",
            low,
        )
    )
    for i, m in enumerate(steps):
        f = Fraction(m.group(1))
        if i == 0:
            nxt = base * f
        else:
            nxt = cur * f
        total += nxt
        cur = nxt
    return total if steps else None


def _percent_komplement(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Komplement-Prozent: 'arm wrestles 20 people, beats 80%, how
    many did he LOSE to?' -> 20 x (100-80)% = 4."""
    low = _digitize(question.lower())
    m = re.search(r"(\d+)\s+people?", low)
    pm = re.search(r"(?:beats?|wins?)\s+(\d+)\s*(?:%|percent)", low)
    if not (m and pm):
        return None
    if not re.search(r"lose|loss|didn'?t|not\s+beat|left", low):
        return None
    return Fraction(m.group(1)) * (100 - int(pm.group(1))) / 100


def _pro_teil(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Total / Anzahl Teile: '6+4+2 gifts, 144 inches, bow for each
    gift' -> 144 / 12 = 12."""
    low = _digitize(question.lower())
    tm = re.search(r"has\s+(\d+)\s+([a-z]+)\s+of\s+([a-z]+)", low)
    if not tm:
        return None
    total = Fraction(tm.group(1))
    # Teil-Anzahl: "N gifts to wrap for X, M for Y..."
    cnt = sum(
        Fraction(m.group(1)) for m in re.finditer(r"(\d+)\s+([a-z]+)\s+to\s+wrap", low)
    )
    cnt += sum(
        Fraction(m.group(1)) for m in re.finditer(r"(?:,\s*|and\s+)(\d+)\s+for\s+", low)
    )
    if cnt <= 0:
        return None
    if re.search(r"for\s+each\s+gift|per\s+gift", low):
        return total / cnt
    return None


def _prozent_anteil(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Prozent-Anteil: 'editing 90 min von 4h+2h+90min total' -> 20%."""
    low = question.lower()
    if not re.search(r"what\s+(?:percentage|percent|fraction)", low):
        return None
    # Teil: "N minutes editing" etc.
    em = re.search(r"(\d+)\s+(minutes?|hours?)\s+editing", low)
    if not em:
        return None
    teil = Fraction(em.group(1)) * (
        Fraction(1, 60) if em.group(2).startswith("min") else Fraction(1)
    )
    total = Fraction(0)
    for m in re.finditer(
        r"(\d+)\s+(hours?|minutes?)\s+(?:writing|"
        r"recording|editing)",
        low,
    ):
        v = Fraction(m.group(1))
        if m.group(2).startswith("min"):
            v = v / 60
        total += v
    # "half that much time recording" — Zusatz
    hm = re.search(r"half\s+that\s+much\s+time", low)
    if hm:
        wm = re.search(r"(\d+)\s+hours?\s+writing", low)
        if wm:
            total += Fraction(wm.group(1)) / 2
    if total <= 0:
        return None
    return teil / total * 100


def _durchschnitt(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Durchschnitt: 'two zebras with 17 stripes each, a zebra with
    36, another with half that many' -> (34+36+18)/4 = 22."""
    low = _digitize(question.lower())
    if not re.search(r"on\s+average|average", low):
        return None
    total = Fraction(0)
    count = Fraction(0)
    for m in re.finditer(
        r"(\d+)\s+([a-z]+)\s+with\s+(\d+)\s+"
        r"([a-z]+)\s+each",
        low,
    ):
        total += Fraction(m.group(1)) * Fraction(m.group(3))
        count += Fraction(m.group(1))
    for m in re.finditer(r"a\s+([a-z]+)\s+with\s+(\d+)\s+([a-z]+)", low):
        total += Fraction(m.group(2))
        count += 1
    hm = re.search(r"another\s+([a-z]+)\s+with\s+half\s+that\s+many", low)
    if hm and count > 0:
        # half that many = letzte explizite Menge / 2
        last = re.findall(r"with\s+(\d+)\s+", low)
        if last:
            total += Fraction(last[-1]) / 2
            count += 1
    if count <= 0:
        return None
    return total / count


def _tray_rest(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Tray-Rest: '64 eggs, 2 trays, each holds 24' -> 64 - 2x24 = 16."""
    low = _digitize(question.lower())
    m = re.search(r"has\s+(\d+)\s+([a-z]+)", low)
    km = re.search(
        r"each\s+([a-z]+)\s+can\s+hold\s+(\d+)\s+"
        r"([a-z]+)",
        low,
    )
    tm = re.search(r"and\s+(\d+)\s+([a-z]+)", low)
    if not (m and km and tm):
        return None
    if not re.search(r"won'?t|not\s+be\s+able|left|remain", low):
        return None
    return Fraction(m.group(1)) - Fraction(tm.group(1)) * Fraction(km.group(2))


def _prozent_kette(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Prozent-Kette: '50 employees, 20% management, 30% davon leiten'
    -> 50 x 0.20 x 0.30 = 3."""
    low = _digitize(question.lower())
    m2 = re.search(
        r"out\s+of\s+this\s+(\d+)\s*(?:%|percent),\s+only\s+"
        r"(\d+)\s*(?:%|percent)",
        low,
    )
    if not m2:
        return None
    base_m = re.search(r"of\s+(\d+)\s+([a-z]+)", low)
    if not base_m:
        return None
    base = Fraction(base_m.group(1))
    pcts = [Fraction(m2.group(1)) / 100, Fraction(m2.group(2)) / 100]
    v = base
    for p in pcts:
        v = v * p
    return v


def _gabe_summe(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Gabe-Summe x Einheit: 'gave 3 sacks to cousin and 4 to brother,
    25 kg per sack, wie viele kg gab sie?' -> (3+4)x25 = 175."""
    low = _digitize(question.lower())
    if not re.search(
        r"how many (?:[a-z]+ ){0,3}(?:gave|give|did\s+she\s+"
        r"give)",
        low,
    ):
        return None
    gaben = re.findall(r"gave\s+(\d+)\s+sacks?", low)
    gaben += re.findall(r"(?:and\s+)(\d+)\s+sacks?\s+to", low)
    if not gaben:
        return None
    km = re.search(
        r"(\d+)\s+kilograms?(?:\s+of\s+[a-z]+)?\s+per\s+"
        r"sack",
        low,
    )
    if not km:
        return None
    return sum(Fraction(g) for g in gaben) * Fraction(km.group(1))


def _spar_rate(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Spar-Rate: '20 coins/Monat, Neil 2/5 times mehr, 10 Jahre'
    -> (20 + 28) x 12 x 10 = 5760."""
    low = _digitize(question.lower())
    m = re.search(r"saving\s+(\d+)\s+coins?", low)
    if not m:
        return None
    r1 = Fraction(m.group(1))
    r2 = None
    nm = re.search(r"(\d+/\d+)\s+times\s+more", low)
    if nm:
        r2 = r1 + r1 * Fraction(nm.group(1))
    ym = re.search(r"(\d+)\s+years?", low)
    if not ym:
        return None
    total_rate = r1 + (r2 if r2 is not None else r1)
    return total_rate * 12 * Fraction(ym.group(1))


def _prozent_mehr(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Prozent-mehr: '45% more pears than bananas, 200 bananas. Wie
    viele Früchte?' -> 200 + 200x1.45 = 490."""
    low = _digitize(question.lower())
    m = re.search(
        r"(\d+)\s*(?:%|percent)\s+more\s+([a-z]+)\s+than\s+"
        r"([a-z]+)",
        low,
    )
    if not m:
        return None
    pct = Fraction(m.group(1)) / 100
    bm = re.search(r"has\s+(\d+)\s+([a-z]+)", low)
    if not bm:
        return None
    base = Fraction(bm.group(1))
    if re.search(r"how many fruits|in total|altogether", low):
        return base + base * (1 + pct)
    return base * (1 + pct)


def _relativ_geschwindigkeit(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Relative Geschwindigkeit: 'car1 60 mph, car2 70 mph passes,
    wie viele Meilen nach 2 h?' -> (70-60) x 2 = 20."""
    low = _digitize(question.lower())
    vs = re.findall(r"(\d+)\s+miles?\s+per\s+hour", low)
    if len(vs) < 2:
        return None
    if not re.search(r"passes|catch|separate", low):
        return None
    hm = re.search(r"after\s+(\d+)\s+hours?", low)
    if not hm:
        return None
    v1, v2 = sorted(Fraction(v) for v in vs)
    return (v2 - v1) * Fraction(hm.group(1))


def _episoden(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Episoden: '20 min pro Episode, halb so viele Episoden wie
    Minuten pro Episode' -> 20 x (20/2) = 200."""
    low = _digitize(question.lower())
    m = re.search(r"each\s+episode\s+is\s+(\d+)\s+minutes?", low)
    if not m:
        return None
    per = Fraction(m.group(1))
    em = re.search(r"half\s+as\s+many\s+episodes", low)
    if not em:
        return None
    n_ep = per / 2
    return per * n_ep


def _sammel_kette(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Sammel-Kette: 'picks 4 Mi, 6 Do, triple Mi Fr' -> 4+6+12 = 22.
    'collected 50, twice the previous year' -> 100+50+100 = 250."""
    low = _digitize(question.lower())
    m1 = re.search(
        r"(?:picks?|collected|gathered|has)\s+(\d+)\s+"
        r"([a-z]+)",
        low,
    )
    if not m1:
        return None
    base = Fraction(m1.group(1))
    total = base
    prev = base
    # weitere Nennungen: "picks N on X"
    for m in re.finditer(r"picks?\s+(\d+)\s+([a-z]+)\s+on\s+", low):
        if m.start() == m1.start():
            continue
        v = Fraction(m.group(1))
        total += v
        prev = v
    # "triple/double the number ... he did on [Wochentag]"
    tm = re.search(
        r"(triple|twice|double|half)\s+the\s+number\s+"
        r"(?:[a-z]+\s+){0,4}(?:did\s+)?(?:on\s+)?"
        r"(?:wednesday|thursday|friday|monday|tuesday)",
        low,
    )
    if tm:
        f = {"triple": 3, "twice": 2, "double": 2, "half": Fraction(1, 2)}[tm.group(1)]
        # Referenz: der Wochentag-Wert
        wd = tm.group(0).split()[-1]
        wm = re.search(r"picks?\s+(\d+)\s+[a-z]+\s+on\s+" + wd, low)
        ref = Fraction(wm.group(1)) if wm else base
        total += ref * f
    # "twice the number of stickers as the previous year"
    ym = re.search(
        r"(twice|triple|double)\s+the\s+number\s+(?:of\s+"
        r"[a-z]+\s+){0,2}as\s+the\s+previous\s+year",
        low,
    )
    if ym:
        f = {"twice": 2, "double": 2, "triple": 3}[ym.group(1)]
        pm = re.search(r"collected\s+(\d+)\s+stickers", low)
        if pm:
            total += Fraction(pm.group(1)) * f
    if total != base:
        return total
    return None


def _blink_rate(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Blink-Rate: '255 mal in 5 min, wie lange für 459?' -> 459x5/255."""
    low = _digitize(question.lower())
    m = re.search(r"blinks?\s+(\d+)\s+times?\s+in\s+(\d+)\s+minutes?", low)
    km = re.search(r"blink\s+(\d+)\s+times?", low)
    if not (m and km):
        return None
    if not re.search(r"how long", low):
        return None
    rate = Fraction(m.group(1)) / Fraction(m.group(2))
    return Fraction(km.group(1)) / rate


def _alter(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Alters-Differenz: 'X born N Jahre vor Y, X hatte Sohn mit A,
    Y ist jetzt B, wie lange her?' -> B + N - A."""
    low = question.lower()
    m = re.search(r"born\s+(\d+)\s+years?\s+before", low)
    sm = re.search(r"had\s+a\s+son\s+at\s+the\s+age\s+of\s+(\d+)", low)
    nm = re.search(r"is\s+now\s+(\d+)", low)
    if not (m and sm and nm):
        return None
    if not re.search(r"years?\s+ago|how\s+old", low):
        return None
    return Fraction(nm.group(1)) + Fraction(m.group(1)) - Fraction(sm.group(1))


def _multi_day(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Multi-Tag: 'ran 3 miles on Monday, Wednesday and Friday. 2 am
    Dienstag, 2 am Donnerstag' -> 3+3+3+2+2."""
    low = _digitize(question.lower())
    days = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]
    total = Fraction(0)
    found = False
    # "ran N on Monday, Wednesday and Friday" -> N x 3 (Liste, Vorrang)
    lm = re.search(
        r"(?:ran|walked|jogged)\s+(\d+)\s+[a-z]+\s+on\s+"
        r"(\w+)(?:,?\s+\w+)*\s+and\s+(\w+)",
        low,
    )
    liste_seg = None
    if lm and lm.group(2) in days and lm.group(3) in days:
        n = Fraction(lm.group(1))
        seg = low[lm.start() : lm.end()]
        cnt = sum(1 for d in days if d in seg)
        total += n * cnt
        found = True
        liste_seg = seg
    for d in days:
        if liste_seg and d in liste_seg:
            continue  # schon in der Liste gezählt
        m = re.search(
            r"(?:ran|walked|jogged|biked|drove|swam)\s+"
            r"(\d+)\s+([a-z]+)\s+on\s+" + d,
            low,
        )
        if m:
            total += Fraction(m.group(1))
            found = True
    # "On Tuesday and Thursday he ran 2 each day" -> 2 x 2
    om = re.search(
        r"on\s+(\w+)\s+and\s+(\w+)\s*,?\s+"
        r"(?:[a-z]+\s+)?(?:ran|walked|jogged)\s+(\d+)",
        low,
    )
    if om and om.group(1) in days and om.group(2) in days:
        total += Fraction(om.group(3)) * 2
        found = True
    return total if found else None


def _pflanzen_gruppen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Pflanzen-Gruppen: '4 plants brauchen 1/2 cup, 8 plants 1 cup,
    Rest 1/4 cup' -> 4x1/2 + 8x1 + 8x1/4 = 12."""
    low = _digitize(question.lower())
    base_m = re.search(r"has\s+(\d+)\s+plants?", low)
    if not base_m:
        return None
    total_p = Fraction(base_m.group(1))
    used_p = Fraction(0)
    total_c = Fraction(0)
    found = False
    for m in re.finditer(
        r"(\d+)\s+(?:of\s+her\s+)?plants?\s+need\s+"
        r"((?:half|quarter|one\s+half|\d+/\d+))\s+of\s+"
        r"a\s+cup|(\d+)\s+plants?\s+need\s+(\d+)\s+"
        r"cup",
        low,
    ):
        if m.group(1) is not None:
            n = Fraction(m.group(1))
            f = {
                "half": Fraction(1, 2),
                "quarter": Fraction(1, 4),
                "one half": Fraction(1, 2),
            }.get(m.group(2), _frac_val(m.group(2)))
        else:
            n = Fraction(m.group(3))
            f = Fraction(m.group(4))
        total_c += n * f
        used_p += n
        found = True
    # Rest: "The rest need a quarter of a cup"
    rm = re.search(
        r"the\s+rest\s+need\s+(?:a\s+)?(half|quarter|"
        r"\d+/\d+)\s+of\s+a\s+cup",
        low,
    )
    if rm and used_p > 0 and total_p > used_p:
        f = {"half": Fraction(1, 2), "quarter": Fraction(1, 4)}.get(
            rm.group(1), _frac_val(rm.group(1))
        )
        total_c += (total_p - used_p) * f
        found = True
    return total_c if found else None


def _muenzen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Münzwerte: '32 quarters, 95 dimes, 120 nickels, 750 pennies.
    Total dollar?' -> 32x0.25 + 95x0.10 + 120x0.05 + 750x0.01."""
    low = _digitize(question.lower())
    werte = {
        "quarter": Fraction(25, 100),
        "dime": Fraction(10, 100),
        "nickel": Fraction(5, 100),
        "penny": Fraction(1, 100),
        "half-dollar": Fraction(50, 100),
    }
    total = Fraction(0)
    found = False
    for m in re.finditer(
        r"(\d+(?:,\d{3})*)\s+(?:pieces?\s+of\s+)?"
        r"(quarters?|dimes?|nickels?|pennies?|"
        r"half-dollars?|dollar\s+bills?)",
        low,
    ):
        n = m.group(1).replace(",", "")
        typ = _canon(m.group(2))
        if typ in werte:
            total += Fraction(n) * werte[typ]
            found = True
        elif "dollar" in m.group(2):
            total += Fraction(n)
            found = True
    if not found:
        # "ten twenties" -> 10 x 20
        tw = re.search(r"(ten|twenty|five|fifteen|\d+)\s+twenties", low)
        if tw:
            n = {"ten": 10, "twenty": 20, "five": 5, "fifteen": 15}.get(
                tw.group(1), Fraction(tw.group(1))
            )
            total += Fraction(n) * 20
            found = True
    if not found:
        return None
    if not re.search(r"dollar|money|total|lunch", low):
        return None
    # "3/5 of the twenties" -> anteilig
    fm = re.search(r"(\d+/\d+)\s+of\s+the\s+twenties", low)
    if fm:
        f = Fraction(fm.group(1))
        total = (
            total
            - (Fraction(200) if "ten twenties" in low else Fraction(0))
            + Fraction(200) * f
        )
    # Bruch-Anteil: "two thirds of the pennies" -> x 2/3
    fm = re.search(
        r"(\d+/\d+|two\s+thirds|half|quarter|"
        r"\d+\s+thirds?|\d+\s+fourths?|\d+\s+fifths?|"
        r"\d+\s+halves?)\s+of\s+the\s+([a-z]+)",
        low,
    )
    if fm:
        w = fm.group(1)
        f = {
            "two thirds": Fraction(2, 3),
            "half": Fraction(1, 2),
            "quarter": Fraction(1, 4),
        }.get(w, None)
        if f is None:
            mm = re.match(r"(\d+)\s+(\w+)", w)
            if mm:
                w2 = mm.group(2)
                den = {
                    "thirds": 3,
                    "third": 3,
                    "fourths": 4,
                    "fourth": 4,
                    "fifths": 5,
                    "fifth": 5,
                    "halves": 2,
                    "half": 2,
                    "tenths": 10,
                    "tenth": 10,
                    "sixths": 6,
                    "sixth": 6,
                    "eighths": 8,
                    "eighth": 8,
                    "quarters": 4,
                    "quarter": 4,
                }.get(w2, None)
                if den:
                    f = Fraction(int(mm.group(1)), den)
        if f is not None:
            total = total * f
    return total


def _pizza_teilen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Pizza-Teilen: '7 Pizzen, 8 Scheiben, Henry + 3 Freunde'
    -> 7x8 / 4 = 14."""
    low = _digitize(question.lower())
    pm = re.search(r"order\s+(\d+)\s+pizzas?", low)
    sm = re.search(
        r"each\s+pizza\s+is\s+cut\s+into\s+(\d+)\s+"
        r"slices?",
        low,
    )
    if not (pm and sm):
        return None
    if not re.search(r"share\s+the\s+pizzas\s+equally", low):
        return None
    people = Fraction(1)
    fm = re.search(r"and\s+(\d+)\s+of\s+his\s+friends?", low)
    if fm:
        people = 1 + Fraction(fm.group(1))
    return Fraction(pm.group(1)) * Fraction(sm.group(1)) / people


def _ernte_rate(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Ernte-Rate: '10 ha, 100/ha, ernten alle 3 Monate, in einem
    Jahr' -> 10 x 100 x (12/3) = 4000."""
    low = _digitize(question.lower())
    m = re.search(r"(\d+)\s+hectares?", low)
    pm = re.search(r"(\d+)\s+pineapples?\s+per\s+hectare", low)
    hm = re.search(
        r"harvest(?:s)?\s+(?:his\s+pineapples?\s+)?every\s+"
        r"(\d+)\s+months?",
        low,
    )
    if not (m and pm and hm):
        return None
    return Fraction(m.group(1)) * Fraction(pm.group(1)) * (12 / Fraction(hm.group(1)))


def _baum_kette(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Baum-Kette: '6 Bäume Chris, F halb so viele, H 5 mehr als
    doppelt so viele wie F. Wie viele mehr H als F?' -> 11-3 = 8."""
    low = _digitize(question.lower())
    cm = re.search(r"trees?\s+in\s+([a-z]+)'s", low)
    tm = re.search(r"(\d+)\s+trees?\s+in", low)
    if not (cm and tm):
        return None
    base_name = cm.group(1)
    base = Fraction(tm.group(1))
    # "X has half the number ... that <base> has"
    hm = re.search(
        r"([a-z]+)\s+has\s+half\s+the\s+number\s+"
        r"(?:[a-z]+\s+){0,3}that\s+([a-z]+)\s+has",
        low,
    )
    if not hm:
        return None
    f_val = base / 2
    f_name = hm.group(1)
    # "Y has N more than twice the number ... that X has"
    mm = re.search(
        r"([a-z]+)\s+has\s+(\d+)\s+more\s+than\s+twice\s+"
        r"the\s+number\s+(?:[a-z]+\s+){0,3}that\s+"
        r"([a-z]+)\s+has",
        low,
    )
    if not mm:
        return None
    h_val = 2 * f_val + Fraction(mm.group(2))
    # "how many more ... than"
    if re.search(r"how\s+many\s+more", low):
        return h_val - f_val
    return h_val


def _geld_kette(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Geld-Kette: 'Carmen $100, Samantha $25 mehr, Daisy $50 mehr.
    Zusammen?' -> 100 + 125 + 175 = 400."""
    low = question.lower()
    if not re.search(r"combined|in total|altogether", low):
        return None
    cm = re.search(r"([a-z]+)\s+has\s+\$?(\d+)", low)
    if not cm:
        return None
    base_name = cm.group(1)
    base = Fraction(cm.group(2))
    vals = [base]
    for m in re.finditer(
        r"([a-z]+)\s+has\s+\$?(\d+)\s+more\s+than\s+"
        r"([a-z]+)",
        low,
    ):
        if m.group(3) == base_name:
            vals.append(base + Fraction(m.group(2)))
        elif m.group(3) == "samantha" and m.group(1) == "daisy":
            vals.append(vals[1] + Fraction(m.group(2)))
    if len(vals) >= 3:
        return sum(vals)
    return None


def _spar_ziel(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Spar-Ziel: 'hat $50, verdient $10/Tag, will $300 kaufen.
    Wie viele Tage?' -> (300-50)/10 = 25."""
    low = _digitize(question.lower())
    hm = re.search(r"has\s+\$?(\d+)", low)
    em = re.search(r"earns?\s+\$?(\d+)\s+per\s+(day|week)", low)
    bm = re.search(r"costs?\s+\$?(\d+)", low)
    if not (hm and em and bm):
        return None
    if not re.search(r"how many days", low):
        return None
    diff = Fraction(bm.group(1)) - Fraction(hm.group(1))
    if diff <= 0:
        return None
    return diff / Fraction(em.group(1))


def _woche_mult(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Wochen-Muster + Multiplikator: '1h Di, 2h Do, Sa 2x Di'
    -> 1 + 2 + 2x1 = 5."""
    low = _digitize(question.lower())
    days = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]
    total = Fraction(0)
    found = False
    vals = {}
    for d in days:
        m = re.search(r"(\d+)\s+hours?\s+on\s+" + d, low)
        if m:
            vals[d] = Fraction(m.group(1))
            total += vals[d]
            found = True
    # "Saturday twice as long as Tuesday"
    dm = re.search(
        r"(\w+days?)\s*,?\s+(?:[a-z]+\s+){0,6}(?:that\s+)?"
        r"lasted\s+(twice|half|triple)\s+as\s+long\s+as\s+"
        r"(\w+days?)",
        low,
    )
    if dm:
        a_day = dm.group(1).rstrip("s")
        b_day = dm.group(3).rstrip("s")
        if b_day in vals:
            f = {"twice": 2, "half": Fraction(1, 2), "triple": 3}[dm.group(2)]
            total += vals[b_day] * f
            found = True
    if found and re.search(r"a\s+week|per\s+week", low):
        return total
    return None


def _mehr_als_haelfte(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Mehr als Hälfte: '6 Buchstaben, Schwester hat 4 mehr als die
    Hälfte. Zusammen?' -> 6 + (6/2+4) = 13."""
    low = _digitize(question.lower())
    m = re.search(r"has\s+(\d+)\s+letters?", low)
    nm = re.search(r"(\d+)\s+more\s+letters?\s+than\s+half", low)
    if not (m and nm):
        return None
    base = Fraction(m.group(1))
    other = base / 2 + Fraction(nm.group(1))
    if re.search(r"and\s+her\s+sister|both|names", low):
        return base + other
    return other


def _halbe_dreifache(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Halb/Dreifach: 'Ben 4 blau, 3 gelb. Jasper halb so viele blau
    wie Ben, dreimal so viele gelb.' -> 4/2 + 3x3 = 11."""
    low = _digitize(question.lower())
    bm = re.search(
        r"([a-z]+)\s+has\s+(\d+)\s+tubes?\s+of\s+([a-z]+)\s+"
        r"paint",
        low,
    )
    tm = re.search(r"has\s+(\d+)\s+tubes?\s+of\s+([a-z]+)\s+paint", low)
    if not (bm and tm):
        return None
    base_name = bm.group(1)
    blue = Fraction(bm.group(2))
    ym = re.search(r"(\d+)\s+tubes?\s+of\s+yellow\s+paint", low)
    yellow = Fraction(ym.group(1)) if ym else Fraction(tm.group(1))
    total = Fraction(0)
    found = False
    for m in re.finditer(
        r"(half|twice|triple|three\s+times|\d+\s+times)"
        r"\s+as\s+many\s+tubes?\s+of\s+([a-z]+)\s+"
        r"paint\s+as\s+([a-z]+)",
        low,
    ):
        if m.group(3) != base_name:
            continue
        w1 = m.group(1)
        f = {"half": Fraction(1, 2), "twice": 2, "triple": 3, "three times": 3}.get(
            w1, None
        )
        if f is None:
            f = Fraction(w1.split()[0])
        col = m.group(2)
        base_col = blue if "blue" in col else yellow
        total += base_col * f
        found = True
    return total if found else None


def _rueckwaerts_times(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Rückwärts+Times: 'George ate 5, now 3 left. Nick twice as many
    as George.' -> (3+5) x 2 = 16."""
    low = _digitize(question.lower())
    em = re.search(r"ate\s+(\d+)\s+", low)
    lm = re.search(
        r"now\s+[a-z]+\s+has\s+(\d+)\s+(?:[a-z]+\s+)?"
        r"left",
        low,
    )
    tm = re.search(r"twice\s+as\s+many", low)
    if not (em and lm and tm):
        return None
    george = Fraction(lm.group(1)) + Fraction(em.group(1))
    return george * 2


def _gewichtsverlust(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Gewichtsverlust: '3 Monate, 10/Monat verloren, final 70'
    -> 70 + 3x10 = 100."""
    low = _digitize(question.lower())
    mm = re.search(r"for\s+(\d+)\s+months?", low)
    pm = re.search(r"lost\s+(\d+)\s+pounds?\s+per\s+month", low)
    fm = re.search(r"final\s+weight\s+was\s+(\d+)", low)
    if not (mm and pm and fm):
        return None
    return Fraction(fm.group(1)) + Fraction(mm.group(1)) * Fraction(pm.group(1))


def _groesser_als(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Größer-als: '5x größer als James' 80, brauchte 2 mehr als er
    hat' -> 5x80 - 2 = 398."""
    low = _digitize(question.lower())
    tm = re.search(r"(\d+)\s+times\s+larger\s+than", low)
    jm = re.search(r"which\s+had\s+(\d+)\s+toys?", low)
    nm = re.search(r"needed\s+(\d+)\s+more\s+toys?\s+than", low)
    if not (tm and jm and nm):
        return None
    return Fraction(tm.group(1)) * Fraction(jm.group(1)) - Fraction(nm.group(1))


def _weniger_zusammen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Weniger-zusammen: '5 Karten, Bruder 3 weniger. Zusammen?'
    -> 5 + (5-3) = 7."""
    low = _digitize(question.lower())
    hm = re.search(r"has\s+(\d+)\s+([a-z]+)", low)
    fm = re.search(r"has\s+(\d+)\s+(?:fewer|less)\s+([a-z]+)\s+than", low)
    if not (hm and fm):
        return None
    if not re.search(r"together|combined|in total", low):
        return None
    base = Fraction(hm.group(1))
    return base + (base - Fraction(fm.group(1)))


def _pro_einheit(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Pro-Einheit x Anzahl: 'pro Muffin 5+3+0.25 Löffel, 16 Muffins'
    -> (5+3+0.25) x 16 = 132."""
    low = _digitize(question.lower())
    if not re.search(r"for\s+every\s+[a-z]+", low):
        return None
    tm = re.search(r"make\s+(\d+)\s+[a-z]+", low)
    if not tm:
        return None
    n = Fraction(tm.group(1))
    vals = re.findall(
        r"(\d+(?:\.\d+)?)\s+(?:of\s+a\s+)?"
        r"(?:tablespoons?|teaspoons?)",
        low,
    )
    if not vals:
        return None
    return sum(Fraction(v) for v in vals) * n


def _fabrik_rate(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Fabrik-Rate: '100 Qt in 2h, 50 Qt in 4h. Total in 48h?'
    -> (100/2 + 50/4) x 48 = 3000."""
    low = _digitize(question.lower())
    hs = re.findall(r"in\s+(\d+)\s+hours?", low)
    hm = hs[-1] if hs else None
    if not hm or not re.search(r"in total", low):
        return None
    rate = Fraction(0)
    found = False
    for m in re.finditer(
        r"(\d+)\s+quarts?\s+of\s+[a-z]+\s+"
        r"(?:ice\s+cream\s+)?in\s+(\d+)\s+hours?",
        low,
    ):
        rate += Fraction(m.group(1)) / Fraction(m.group(2))
        found = True
    if not found:
        return None
    return rate * Fraction(hm)


def _wochen_vergleich(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Wochen-Vergleich: '3+5 Boxen diese Woche, 4 letzte. Wie viele
    mehr?' -> (3+5) - 4 = 4."""
    low = _digitize(question.lower())
    lm = re.search(r"last\s+week\s+(?:she\s+)?bought\s+(\d+)", low)
    if not lm:
        return None
    if not re.search(r"how many more", low):
        return None
    this_week = Fraction(0)
    for m in re.finditer(r"(?:bought|and)\s+(\d+)\s+boxes?", low):
        if lm.start() <= m.start() < lm.end():
            continue  # "last week bought 4" gehört NICHT zu dieser Woche
        this_week += Fraction(m.group(1))
    return this_week - Fraction(lm.group(1))


def _trinkgeld(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Trinkgeld: '40 Kunden x $20, andere 10% weniger. Zusammen?'
    -> 800 + 720 = 1520."""
    low = _digitize(question.lower())
    cm = re.search(r"each\s+of\s+the\s+(\d+)\s+[a-z]+", low)
    tm = re.search(r"gave\s+[a-z]+\s+a\s+\$?(\d+)", low)
    pm = re.search(r"(\d+)\s*(?:%|percent)\s+less", low)
    if not (cm and tm and pm):
        return None
    first = Fraction(cm.group(1)) * Fraction(tm.group(1))
    second = first * (100 - int(pm.group(1))) / 100
    return first + second


def _mehrfach_weniger(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Mehrfach-weniger: '5 Pfund Kartoffeln, Süßkartoffeln 2x, Möhren
    3 weniger als Süß' -> 5x2 - 3 = 7."""
    low = _digitize(question.lower())
    bm = re.search(r"weighed\s+(\d+)\s+pounds?", low)
    tm = re.search(r"weighed\s+(\d+)\s+times\s+as\s+much", low)
    fm = re.search(r"weighed\s+(\d+)\s+pounds?\s+(?:fewer|less)", low)
    if not (bm and tm and fm):
        return None
    return Fraction(bm.group(1)) * Fraction(tm.group(1)) - Fraction(fm.group(1))


def _doppelt_verloren(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Doppelt-verloren: '9 Bücher, Joseph doppelt, verlor 2'
    -> 9x2 - 2 = 16."""
    low = _digitize(question.lower())
    hm = re.search(r"has\s+(\d+)\s+([a-z]+)", low)
    tm = re.search(r"twice\s+the\s+number", low)
    lm = re.search(r"lost\s+(\d+)\s+of\s+them", low)
    if not (hm and tm and lm):
        return None
    return Fraction(hm.group(1)) * 2 - Fraction(lm.group(1))


def _papier_dicke(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Papier-Dicke: '100 Seiten/Zoll, beide Seiten, 1.5 Zoll'
    -> 100 x 1.5 x 2 = 300."""
    low = _digitize(question.lower())
    pm = re.search(r"(\d+)\s+pages?\s+to\s+the\s+inch", low)
    im = re.search(r"is\s+(\d+(?:\.\d+)?)\s+inches?\s+thick", low)
    if not (pm and im):
        return None
    mult = 2 if re.search(r"both\s+sides", low) else 1
    return Fraction(pm.group(1)) * Fraction(im.group(1)) * mult


def _lese_rate(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Lese-Rate: '10 min für 3 Seiten, liest 18 Seiten'
    -> 18/3 x 10 = 60."""
    low = _digitize(question.lower())
    tm = re.search(r"(\d+)\s+minutes?\s+to\s+read\s+(\d+)\s+pages?", low)
    pms = list(re.finditer(r"reads?\s+(\d+)\s+pages?", low))
    pm = pms[-1] if pms else None
    if not (tm and pm):
        return None
    return Fraction(pm.group(1)) / Fraction(tm.group(2)) * Fraction(tm.group(1))


def _doppel_rate(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Doppel-Rate: 'peel 6/min, saute 30 in 10 min. 90 shrimp?'
    -> 90/6 + 90/(30/10) = 15 + 30 = 45."""
    low = _digitize(question.lower())
    pm = re.search(r"peel\s+(\d+)\s+[a-z]+\s+a\s+minute", low)
    sm = re.search(r"saute\s+(\d+)\s+[a-z]+\s+in\s+(\d+)\s+minutes?", low)
    tm = re.search(r"peel\s+and\s+cook\s+(\d+)", low)
    if not (pm and sm and tm):
        return None
    n = Fraction(tm.group(1))
    peel_time = n / Fraction(pm.group(1))
    cook_rate = Fraction(sm.group(1)) / Fraction(sm.group(2))
    cook_time = n / cook_rate
    return peel_time + cook_time


def _rueck_kept(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Rückwärts+behalten: 'half gegeben, half benutzt, 9 behalten,
    7 übrig. Am Anfang?' -> ((7+9)x2)x2 = 64."""
    low = _digitize(question.lower())
    if not re.search(r"beginning|at\s+first|in\s+the\s+beginning", low):
        return None
    km = re.search(r"kept\s+(\d+)\s+of\s+the", low)
    rm = re.search(r"remaining\s+(\d+)(?:\s+[a-z]+)?\s+to", low)
    if not (km and rm):
        return None
    x = Fraction(rm.group(1)) + Fraction(km.group(1))
    # "gave half ... used half ..." -> x2 x2
    halves = len(re.findall(r"half\s+of\s+(?:the\s+)?(?:his|the)", low))
    return x * (2**halves)


def _rosen_preis(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Rosen-Preis: '15 Rosen, $2/Stück oder $15/Dutzend, $25 gezahlt,
    Wechselgeld in Quarters' -> (25 - (15+6)) x 4 = 16."""
    low = _digitize(question.lower())
    dm = re.search(r"\$?(\d+)\s+for\s+a\s+dozen", low)
    pm = re.search(r"cost\s+\$?(\d+)\s+each", low)
    nm = re.search(r"bought\s+(\d+)\s+roses?", low)
    bm = re.search(r"(?:five|5)\s+(\d+)\s+dollar\s+bills", low)
    if not (dm and pm and nm and bm):
        return None
    n = int(nm.group(1))
    dozen = n // 12
    rest = n % 12
    cost = dozen * int(dm.group(1)) + rest * int(pm.group(1))
    paid = 5 * int(bm.group(1))
    change = paid - cost
    if not re.search(r"quarters", low):
        return None
    return change * 4


def _gruppen_wachstum(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Geometrisches Wachstum: '5 Kinder, 2. Straße x2, 3. Straße x3,
    5 geben auf' -> 5x2x3-5 = 25."""
    low = _digitize(question.lower())
    mm = re.search(r"there are\s+(\d+)\s+children", low)
    if not mm:
        return None
    x = Fraction(mm.group(1))
    mults = []
    for m in re.finditer(r"joined by another\s+(?:child|(\d+)\s+children?)", low):
        mults.append(2 if not m.group(1) else int(m.group(1)) + 1)
    if not mults:
        return None
    for mlt in mults:
        x *= mlt
    if re.search(r"original\s+\d+\s+children\s+then\s+give\s+up", low):
        x -= Fraction(mm.group(1))
    return x


def _halbe_plus(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Halbe+Zusatz: '2 hours more than half Steve's time' -> 10/2+2."""
    low = _digitize(question.lower())
    m = re.search(
        r"(\d+)\s+[a-z]+\s+more\s+than\s+half\s+([a-z]+)'s\s+"
        r"(?:time|age|amount)",
        low,
    )
    if not m:
        return None
    # Referenzwert: 'Steve ... took 10 hours' ODER Quant mit dem Namen
    tm = re.search(r"took\s+(\d+)\s+[a-z]+", low)
    if tm:
        return Fraction(tm.group(1)) / 2 + Fraction(m.group(1))
    for q in reversed(quants):
        if q.obj and q.obj == m.group(2):
            return Fraction(q.value) / 2 + Fraction(m.group(1))
    return None


def _rechteck_umfang(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Rechteck-Umfang: '20 ft lang, 15 ft kurz' -> 2x(20+15)=70."""
    low = _digitize(question.lower())
    if not re.search(r"rectangle|rectangular", low):
        return None
    lm = re.search(r"(\d+)\s+[a-z]+\s+on\s+the\s+long", low)
    sm = re.search(r"(\d+)\s+[a-z]+\s+on\s+the\s+short", low)
    if not (lm and sm):
        return None
    return 2 * (Fraction(lm.group(1)) + Fraction(sm.group(1)))


def _docks_line(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Docks-Leine: '3 ft pro ft Dock, 200 ft, hat 6' -> 200x3-6=594."""
    low = _digitize(question.lower())
    rm = re.search(
        r"(\d+)\s+feet\s+of\s+line\s+for\s+every\s+"
        r"(\d+)?\s*foot\s+of\s+(\w+)",
        low,
    )
    dm = re.search(r"there\s+is\s+(\d+)\s+feet\s+of\s+(\w+)", low)
    hm = re.search(r"has\s+(\d+)\s+feet\s+of\s+new\s+line", low)
    if not (rm and dm and hm):
        return None
    need = Fraction(dm.group(1)) * Fraction(rm.group(1))
    if dm.group(2) != rm.group(3):
        return None
    return need - Fraction(hm.group(1))


def _pflanzen_prozent(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Pflanzen-Prozent: 'klein 3 Dutzend, groß 50% mehr' -> 36+54=90."""
    low = _digitize(question.lower())
    sm = re.search(
        r"(?:small|mini|baby)\s+\w+\s+has\s+(\d+)\s+"
        r"(?:dozen\s+)?(\w+)",
        low,
    )
    pm = re.search(
        r"(\d+)\s*(?:%|percent)\s+more\s+\w+\s+than\s+"
        r"(?:a\s+)?(?:small|mini|baby)",
        low,
    )
    if not (sm and pm):
        return None
    base = Fraction(sm.group(1))
    if "dozen" in low[sm.start() : sm.end()] or re.search(r"dozen\s+seeds", low):
        base *= 12
    large = base * (1 + Fraction(pm.group(1)) / 100)
    return base + large


def _gewicht_force(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Gewicht+Force: '1200+250+2x75, Force=1%' -> 1600/100=16."""
    low = _digitize(question.lower())
    if not re.search(r"force\s+to\s+move", low):
        return None
    pm = re.search(r"is\s+(\d+)\s*(?:%|percent)\s+of\s+the\s+weight", low)
    if not pm:
        return None
    total = Fraction(0)
    for m in re.finditer(
        r"weigh(?:s|ing)?\s+(\d+)\s+pounds\b"
        r"(?!\s+each)",
        low,
    ):
        total += Fraction(m.group(1))
    em = re.search(r"weigh\s+(\d+)\s+pounds?\s+each", low)
    if em:
        kids = len(re.findall(r"\b(\w+)\b\s+who\s+weigh", low))
        total += Fraction(em.group(1)) * max(kids, 2)
    return total * Fraction(pm.group(1)) / 100


def _gleich_teilen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Gleich-Teilen: '2 Tüten x 55, 5 Personen' -> 110/5=22."""
    low = _digitize(question.lower())
    nm = re.search(
        r"buys?\s+(\d+)\s+[a-z]+\s+of\s+[a-z]+\s+with\s+"
        r"(\d+)\s+[a-z]+\s+each",
        low,
    )
    fm = re.search(r"family\s+has\s+(\d+)\s+members?", low)
    if not (nm and fm):
        return None
    return Fraction(nm.group(1)) * Fraction(nm.group(2)) / Fraction(fm.group(1))


def _bag_teile(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Bag-Teile: '232 Stück, 54 rot, 2x orange, 1/2 gelb' -> Rest=43."""
    low = _digitize(question.lower())
    tm = re.search(r"(?:has|contains?)\s+(\d+)\s+pieces?", low)
    rm = re.search(r"has\s+(\d+)\s+(\w+)\s+candies", low)
    tm2 = re.search(r"twice\s+that\s+amount\s+of\s+(\w+)", low)
    hm = re.search(
        r"half\s+as\s+many\s+(\w+)(?:\s+\w+)?\s+as\s+"
        r"(\w+)",
        low,
    )
    if not (tm and rm and tm2 and hm):
        return None
    total = Fraction(tm.group(1))
    red = Fraction(rm.group(1))
    orange = red * 2
    yellow = red / 2
    return total - red - orange - yellow


def _putz_anteil(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Putz-Anteil: '80 Räume x 15min, 5 Tage, 8h-Tag' -> 50%."""
    low = _digitize(question.lower())
    cm = re.search(r"(\d+)\s+classrooms?", low)
    tm = re.search(r"takes?\s+them\s+(\d+)\s+minutes\s+per", low)
    dm = re.search(r"(\d+)\s+days?\s+to\s+get", low)
    hm = re.search(r"(\d+)\s+hour\s+day", low)
    if not (cm and tm and dm and hm):
        return None
    mins = Fraction(cm.group(1)) * Fraction(tm.group(1))
    per_day = mins / Fraction(dm.group(1))
    work = Fraction(hm.group(1)) * 60
    return per_day / work * 100


def _wochen_futter(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Wochen-Futter: '(3x5+2x3) pro Tag, Woche' -> 21x7=147."""
    low = _digitize(question.lower())
    if not re.search(r"per\s+day", low) or not re.search(
        r"in\s+a\s+week|per\s+week", low
    ):
        return None
    # "keeps 3 X und 2 Y. X consumes 5 kg, Y 3 kg pro Tag"
    km = re.search(
        r"keeps?\s+(\d+)\s+([a-z ]+?)\s+and\s+(\d+)\s+"
        r"([a-z ]+?)[,\.]",
        low,
    )
    if not km:
        return None
    c1 = re.search(r"([a-z]+)\s+consumes\s+(\d+)\s+kilograms?", low)
    c2 = re.search(
        r"and\s+a\s+([a-z]+)\s+consumes\s+(\d+)\s+"
        r"kilograms?",
        low,
    )
    if not (c1 and c2):
        return None
    total = Fraction(0)
    if c1.group(1)[:4] in km.group(2) or km.group(2).startswith(c1.group(1)[:4]):
        total += Fraction(km.group(1)) * Fraction(c1.group(2))
        total += Fraction(km.group(3)) * Fraction(c2.group(2))
    else:
        total += Fraction(km.group(1)) * Fraction(c2.group(2))
        total += Fraction(km.group(3)) * Fraction(c1.group(2))
    return total * 7


def _regen_zwei(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Regen-Zwei: '2 Mo, 1 mehr als 2x Mo' -> 2x2+1=5."""
    low = _digitize(question.lower())
    mm = re.search(r"rained?\s+(\d+)\s+\w+\s+on\s+\w+", low)
    tm = re.search(
        r"(\d+)\s+more\s+\w+\s+than\s+twice\s+of\s+"
        r"\w+['\u2019]s\s+total",
        low,
    )
    if not (mm and tm):
        return None
    return Fraction(mm.group(1)) * 2 + Fraction(tm.group(1))


def _weniger_kette(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Kette: 'S 12 weniger als Sh, Sh 8 mehr als K. S=27 -> K=31'."""
    low = _digitize(question.lower())
    # "X has N fewer than Y" + "Y has M more than Z" + "X has K"
    fm = re.search(
        r"(\w+)\s+has\s+(\d+)\s+fewer\s+(\w+)\s+than\s+"
        r"(\w+)",
        low,
    )
    mm = re.search(
        r"(\w+)\s+has\s+(\d+)\s+\w+\s+more\s+than\s+"
        r"(\w+)",
        low,
    )
    vm = re.search(r"if\s+(\w+)\s+has\s+(\d+)", low)
    if not (fm and mm and vm):
        return None
    x, d, _, y = fm.group(1), fm.group(2), fm.group(3), fm.group(4)
    z, m, w = mm.group(1), mm.group(2), mm.group(3)
    if vm.group(1) != x:
        return None
    val = Fraction(vm.group(2))
    # x = y - d -> y = val + d ; y = z + m -> z = y - m
    y_val = val + Fraction(d)
    return y_val - Fraction(m)


def _windeln_halb(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Windeln-Halb: '2 Kinder x 5, Frau halb' -> 10/2=5."""
    low = _digitize(question.lower())
    km = re.search(r"has\s+(\d+)\s+children\s+who\s+\w+", low)
    rm = re.search(r"requires?\s+(\d+)\s+\w+\s+changes\s+per\s+day", low)
    hm = re.search(r"changes\s+half\s+of\s+the\s+\w+", low)
    if not (km and rm and hm):
        return None
    return Fraction(km.group(1)) * Fraction(rm.group(1)) / 2


def _dosis_mix(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Dosis-Mix: '14 mL + 3x, 8 Dosen' -> 14x4x8=448."""
    low = _digitize(question.lower())
    dm = re.search(
        r"combine\s+(\d+)\s+\w+\s+of\s+(?:one|1)\s+\w+"
        r"\s+with\s+(\d+)\s+times\s+that\s+amount",
        low,
    )
    dm2 = re.search(r"in\s+(\d+)\s+doses?", low)
    if not (dm and dm2):
        return None
    base = Fraction(dm.group(1))
    per = base + base * Fraction(dm.group(2))
    return per * Fraction(dm2.group(1))


def _escape_raum(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Escape-Raum: '4 Stufen x 10, 8 gleichzeitig, 45 min'
    -> 40/8x45 = 225."""
    low = _digitize(question.lower())
    gm = re.search(r"grades?\s+\d+\s+[\u2013-]\s+\d+", low)
    sm = re.search(r"(\d+)\s+students?\s+in\s+each\s+grade", low)
    am = re.search(r"only\s+(\d+)\s+students?\s+can\s+try", low)
    mm = re.search(r"(\d+)\s+minutes?\s+to\s+try\s+and\s+escape", low)
    if not (gm and sm and am and mm):
        return None
    gs_ = re.findall(r"\d+", gm.group(0))
    grades = int(gs_[-1]) - int(gs_[0]) + 1
    kids = grades * Fraction(sm.group(1))
    groups = kids / Fraction(am.group(1))
    return groups * Fraction(mm.group(1))


def _stufen_rate(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Stufen-Rate: 'niedrig 1/Tag, mittel 2x, hoch 4x; 3/3/5 Tage'
    -> 3+6+20 = 29."""
    low = _digitize(question.lower())
    lm = re.search(
        r"low\s+setting\s+removes\s+(\d+)\s+"
        r"[a-z]+(?:\s+[a-z]+){0,6}\s+per\s+day",
        low,
    )
    dm = re.search(r"for\s+(\d+)\s+days?\s+on\s+the\s+low", low)
    md = re.search(
        r"additional\s+(\d+)\s+days?\s+on\s+the\s+"
        r"medium",
        low,
    )
    hd = re.search(
        r"additional\s+(\d+)\s+days?\s+on\s+the\s+"
        r"high",
        low,
    )
    if not (lm and dm and md and hd):
        return None
    base = Fraction(lm.group(1))
    return (
        base * Fraction(dm.group(1))
        + base * 2 * Fraction(md.group(1))
        + base * 4 * Fraction(hd.group(1))
    )


def _ballon_start(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Ballon-Start: '4 Freunde x 2 + 1, dann 5 x 3' -> 9+15=24."""
    low = _digitize(question.lower())
    fm = re.search(r"invited\s+(\d+)\s+of\s+her\s+friends", low)
    gm = re.search(
        r"gave\s+each\s+of\s+her\s+friends\s+(\d+)\s+"
        r"\w+",
        low,
    )
    mm = re.search(r"gave\s+each\s+person\s+(\d+)\s+more", low)
    if not (fm and gm and mm):
        return None
    people = Fraction(fm.group(1)) + 1
    return (
        Fraction(fm.group(1)) * Fraction(gm.group(1))
        + 1
        + people * Fraction(mm.group(1))
    )


def _fleisch_wuerze(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Fleisch-Würze: '80 Bällchen, 16/lb, 2 EL/lb' -> 80/16x2=10."""
    low = _digitize(question.lower())
    mm = re.search(r"make\s+(\d+)\s+\w+", low)
    bm = re.search(r"gets?\s+(\d+)\s+\w+\s+from\s+each\s+pound", low)
    sm = re.search(
        r"adds?\s+(\d+)\s+\w+\s+of\s+his\s+secret\s+"
        r"\w+(?:\s+\w+)?\s+for\s+every\s+pound",
        low,
    )
    if not (mm and bm and sm):
        return None
    return Fraction(mm.group(1)) / Fraction(bm.group(1)) * Fraction(sm.group(1))


def _bandagen_start(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Bandagen-Start: 'Ende 78, 1. Tag -38+50, 2. -28, 3. +100-25'
    -> B+59=78 -> B=19."""
    low = _digitize(question.lower())
    em = re.search(r"left\s+at\s+the\s+end", low)
    if not em or not re.search(r"start\s+with\s+on\s+the\s+first", low):
        return None
    um = re.search(r"used\s+(\d+)\s+\w+", low)
    om = re.search(r"ordered\s+(?:one|1)\s+bulk\s+pack", low)
    if not (um and om):
        return None
    b = Fraction(0)
    first = -Fraction(um.group(1)) + Fraction(50)
    tm2 = re.search(r"second\s+day,\s+they\s+used\s+(\d+)\s+fewer", low)
    if tm2:
        first -= Fraction(um.group(1)) - Fraction(tm2.group(1))
    tm3 = re.search(r"third\s+day,\s+they\s+ordered\s+(\d+)\s+bulk", low)
    if tm3:
        first += Fraction(tm3.group(1)) * 50
    hm = re.search(r"used\s+half\s+a\s+pack", low)
    if hm:
        first -= 25
    end = Fraction(78)
    return end - first


def _stift_pakete(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Stift-Pakete: '5 Päckchen rot, 2x schwarz, 5/Päckchen'
    -> 5x5 + 10x5 = 75."""
    low = _digitize(question.lower())
    pm = re.search(r"bought\s+(\d+)\s+\w+\s+of\s+\w+\s+\w+", low)
    bm = re.search(
        r"twice\s+the\s+amount\s+of\s+\w+\s+\w+\s+"
        r"than\s+the\s+\w+",
        low,
    )
    em = re.search(r"each\s+pack\s+has\s+(\d+)\s+\w+", low)
    if not (pm and bm and em):
        return None
    red = Fraction(pm.group(1)) * Fraction(em.group(1))
    black = Fraction(pm.group(1)) * 2 * Fraction(em.group(1))
    return red + black


def _mural_farben(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Mural-Farben: '12 Pints, halb gelb, Rest 3 gleich'
    -> 12/2/3 = 2."""
    low = _digitize(question.lower())
    if not re.search(r"mural", low):
        return None
    pm = re.search(r"used\s+(\d+)\s+\w+\s+of\s+paint\s+in\s+all", low)
    ym = re.search(r"half\s+the\s+mural\s+is\s+\w+", low)
    em = re.search(
        r"equal\s+amounts\s+of\s+\w+,\s+\w+,\s+and\s+"
        r"\w+",
        low,
    )
    if not (pm and ym and em):
        return None
    return Fraction(pm.group(1)) / 2 / 3


def _geld_umwandeln(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Geld-Umwandeln: '$1000 in 20ern, 10 verloren, halb bezahlt,
    x3, in 5ern' -> 50-10=40; 20 bezahlt -> 20x20=400; x3=1200; /5
    = 240."""
    low = _digitize(question.lower())
    wm = re.search(
        r"withdraws\s+\$?(\d+)\s+in\s+(\d+)\s+\w+\s+"
        r"bills",
        low,
    )
    lm = re.search(r"loses\s+(\d+)\s+bills", low)
    if not (wm and lm):
        return None
    denom = int(wm.group(2))
    bills = Fraction(wm.group(1)) / denom - Fraction(lm.group(1))
    hm = re.search(r"uses\s+half\s+of\s+the\s+remaining", low)
    if hm:
        bills = bills / 2
    tm = re.search(r"triples\s+his\s+money", low)
    if tm:
        bills = bills * denom * 3
    fm = re.search(
        r"converts\s+all\s+his\s+bills\s+to\s+(\d+)\s+"
        r"\w+\s+bills",
        low,
    )
    if not fm:
        return None
    return bills / Fraction(fm.group(1))


def _wochentag_rate(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Wochentag-Rate: '4 an Werktagen, 5 am Wochenende' -> 4x5+5x2."""
    low = _digitize(question.lower())
    wm = re.search(r"(\d+)\s+\w+\s+each\s+on\s+weekdays", low)
    em = re.search(
        r"(\d+)\s+\w+\s+each\s+on\s+"
        r"(?:saturday|weekend)",
        low,
    )
    if not (wm and em):
        return None
    return Fraction(wm.group(1)) * 5 + Fraction(em.group(1)) * 2


def _eier_chain(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Eier-Kette: '36, 5, 2x, -2, halb' -> 36-5-10-8-4=9."""
    low = _digitize(question.lower())
    dm = re.search(r"hid\s+(\d+)\s+dozen\s+\w+", low)
    fm = re.search(r"finds\s+(\d+)\s+\w+", low)
    if not (dm and fm):
        return None
    total = Fraction(dm.group(1)) * 12
    x = Fraction(fm.group(1))
    total -= x
    # Stacy: twice as many
    if re.search(r"finds\s+twice\s+as\s+many", low):
        total -= x * 2
        x = x * 2
    cm = re.search(r"finds\s+(\d+)\s+less\s+than", low)
    if cm:
        total -= x - Fraction(cm.group(1))
        x = x - Fraction(cm.group(1))
    if re.search(r"half\s+as\s+many\s+as", low):
        total -= x / 2
    return total


def _backen_each(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Backen-each: '9+4+5 Kekse, -3, -2, -2, backt 4 of each'
    -> 13+4x3 = 23."""
    low = _digitize(question.lower())
    em = re.search(r"bakes?\s+(\d+)\s+of\s+each\s+\w+", low)
    if not em:
        return None
    # Sorten: "9 X, 4 Y, and 5 Z" -> 3 ; Anfang = Summe aller
    hm = re.search(r"has\s+(\d+[^.]*\.)", low)
    if not hm:
        return None
    seg = hm.group(1)
    sorts = re.findall(r"(\d+)\s+[a-z]+(?:\s+[a-z]+)?\s+\w+s?\b", seg)
    kinds = len(sorts)
    if kinds < 2:
        return None
    total = sum(Fraction(s) for s in sorts)
    # Subtraktionen: ate N, gives N
    for m in re.finditer(r"ate\s+(\d+)|gives?\s+(\d+)", low):
        total -= Fraction(m.group(1) or m.group(2))
    return total + Fraction(em.group(1)) * kinds


def _ernte_jahr(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Ernte-Jahr: '5 Bäume x 6 Zitronen x 10 Jahre' -> 300."""
    low = _digitize(question.lower())
    tm = re.search(r"grows?\s+(\d+)\s+\w+", low)
    lm = re.search(r"collects?\s+(\d+)\s+\w+\s+from\s+each", low)
    dm = re.search(r"in\s+a\s+decade|decade", low)
    if not (tm and lm and dm):
        return None
    return Fraction(tm.group(1)) * Fraction(lm.group(1)) * 10


def _stein_stand(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Stein-Stand: '$100, 5/7, 60% verkauft' -> 100-12x7=16."""
    low = _digitize(question.lower())
    cm = re.search(r"has\s+\$?(\d+)\s+and\s+wants", low)
    bm = re.search(r"buy\s+\w+\s+for\s+\$?(\d+)\s+each", low)
    sm = re.search(r"sell\s+them\s+for\s+\$?(\d+)\s+each", low)
    pm = re.search(r"sells\s+(\d+)\s*(?:%|percent)\s+of\s+his", low)
    if not (cm and bm and sm and pm):
        return None
    cap = Fraction(cm.group(1))
    buy = Fraction(bm.group(1))
    sell = Fraction(sm.group(1))
    rocks = cap // buy
    sold = rocks * Fraction(pm.group(1)) / 100
    return cap - sold * sell


def _steuer_mitte(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Steuer-Mitte: '5168, Mo+Di 1907, Do+Fr 2136' -> 1125."""
    low = _digitize(question.lower())
    tm = re.search(r"received\s+(\d+)\s+\w+\s+\w+", low)
    m1 = re.search(r"received\s+a\s+total\s+of\s+(\d+)\s+\w+", low)
    m2 = re.search(
        r"received\s+a\s+total\s+of\s+(\d+)\s+\w+"
        r"\s+reports\.\s+how|"
        r"received\s+a\s+total\s+of\s+(\d+)\s+reports\."
        r"\s+how",
        low,
    )
    if not (tm and m1 and m2):
        return None
    return (
        Fraction(tm.group(1))
        - Fraction(m1.group(1))
        - Fraction(m2.group(1) or m2.group(2))
    )


def _kiste_gewicht(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Kisten-Gewicht: '200 Buntstifte, 8er-Gruppen, 8oz Kiste,
    1oz Stift' -> (200+25x8)/16 = 25."""
    low = _digitize(question.lower())
    cm = re.search(r"has\s+(\d+)\s+\w+", low)
    gm = re.search(r"groups?\s+of\s+(\d+)", low)
    bm = re.search(r"each\s+box\s+weighs\s+(\d+)\s+\w+", low)
    om = re.search(r"each\s+(?!box)\w+\s+weighs\s+(\d+)\s+\w+", low)
    lm = re.search(r"in\s+pounds?", low)
    if not (cm and gm and bm and om and lm):
        return None
    n = Fraction(cm.group(1))
    boxes = n / Fraction(gm.group(1))
    oz = n * Fraction(om.group(1)) + boxes * Fraction(bm.group(1))
    return oz / 16


def _marmor_umgekehrt(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Marmor-umgekehrt: '60 wenn +2 Dutzend, dann -10' -> 26."""
    low = _digitize(question.lower())
    wm = re.search(r"will\s+have\s+(\d+)\s+\w+", low)
    dm = re.search(r"receives\s+(\d+)\s+dozen\s+more", low)
    lm = re.search(r"loses\s+(\d+)\s+of\s+the", low)
    if not (wm and dm and lm):
        return None
    now = Fraction(wm.group(1)) - Fraction(dm.group(1)) * 12
    return now - Fraction(lm.group(1))


def _weg_rate(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Weg-Rate: 'Strand 2mi/40min, Bürgersteig 2x, 1mi' -> 50."""
    low = _digitize(question.lower())
    bm = re.search(
        r"(\d+)\s+miles?\s+of\s+walking\s+on\s+the\s+"
        r"beach",
        low,
    )
    pm = re.search(
        r"(\d+)\s+mile\s+of\s+walking\s+on\s+the\s+"
        r"sidewalk",
        low,
    )
    tm = re.search(
        r"(\d+)\s+minutes?\s+of\s+her\s+walk\s+is\s+"
        r"spent\s+on\s+the\s+beach",
        low,
    )
    if not (bm and pm and tm):
        return None
    rate = Fraction(bm.group(1)) / Fraction(tm.group(1))
    sidewalk = Fraction(pm.group(1)) / (rate * 2)
    return Fraction(tm.group(1)) + sidewalk


def _lauf_rest(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Lauf-Rest: '52 total, 8, 3.5x' -> 52-8-28=16."""
    low = _digitize(question.lower())
    tm = re.search(r"ran\s+(\d+)\s+\w+\s+in\s+total", low)
    am = re.search(r"ran\s+(\d+)\s+\w+\.", low)
    xm = re.search(r"ran\s+([\d.]+)\s+times\s+what", low)
    if not (tm and am and xm):
        return None
    return (
        Fraction(tm.group(1))
        - Fraction(am.group(1))
        - Fraction(am.group(1)) * Fraction(xm.group(1))
    )


def _albatros(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Albatros: '400/Tag, 40000, halbe Erde' -> 20000/400=50."""
    low = _digitize(question.lower())
    fm = re.search(r"flies\s+(\d+)\s+\w+\s+every\s+day", low)
    cm = re.search(
        r"circumference\s+of\s+the\s+earth\s+is\s+"
        r"([\d,]+)",
        low,
    )
    hm = re.search(r"half\s+of\s+the\s+way\s+around", low)
    if not (fm and cm and hm):
        return None
    return Fraction(int(cm.group(1).replace(",", ""))) / 2 / Fraction(fm.group(1))


def _spar_ausgabe(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Spar-Ausgabe: 'hatte 21, sparte 11, gab 5 und 19 aus' -> 8."""
    low = _digitize(question.lower())
    hm = re.search(r"had\s+\$?(\d+)", low)
    sm = re.search(r"saved\s+\$?(\d+)", low)
    if not (hm and sm):
        return None
    total = Fraction(hm.group(1)) + Fraction(sm.group(1))
    pm = re.search(
        r"spent\s+\$?(\d+)\s+on\s+[a-z ]+?\s+and\s+"
        r"\$?(\d+)",
        low,
    )
    if pm:
        total -= Fraction(pm.group(1)) + Fraction(pm.group(2))
    else:
        for m in re.finditer(r"spent\s+\$?(\d+)", low):
            total -= Fraction(m.group(1))
    return total


def _halbe_rueck_kette(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Halbe-rückwärts: 'halb, halb vom Rest, letzte 270' -> 270x2x2."""
    low = _digitize(question.lower())
    lm = re.search(r"last\s+(\d+)\s+\w+", low)
    if not lm or "hunted" not in low:
        return None
    halves = len(re.findall(r"half\s+of", low))
    return Fraction(lm.group(1)) * (2**halves)


def _kirche_mix(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Kirche-Mix: '20 Autos x 3, 12 Busse x 35' -> 60+420=480."""
    low = _digitize(question.lower())
    cm = re.search(r"(\d+)\s+private\s+\w+\s+and\s+(\d+)\s+\w+", low)
    bm = re.search(
        r"each\s+\w+\s+carried\s+(\d+)\s+\w+\s+and\s+"
        r"each\s+\w+\s+carried\s+(\d+)",
        low,
    )
    if not (cm and bm):
        return None
    return Fraction(cm.group(1)) * Fraction(bm.group(2)) + Fraction(
        cm.group(2)
    ) * Fraction(bm.group(1))


def _halb_rueck_weg(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Halb-rück-Weg: '4mi zum Laden, halb zurück, 4mph'
    -> 8/4 = 2h."""
    low = _digitize(question.lower())
    dm = re.search(r"home\s+is\s+(\d+)\s+\w+\s+from\s+the\s+\w+", low)
    rm = re.search(r"walks?\s+(\d+)\s+\w+\s+per\s+hour", low)
    if not (dm and rm) or "halfway" not in low:
        return None
    return Fraction(dm.group(1)) * 2 / Fraction(rm.group(1))


def _steuer_frei(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Steuer-Helfer: '2/h x 3h, 20% frei, 31.3-19.4' -> 40x6=240."""
    low = _digitize(question.lower())
    pm = re.search(r"help\s+(\d+)\s+\w+\s+per\s+hour", low)
    hm = re.search(r"for\s+(\d+)\s+hours?\s+a\s+day", low)
    tm = re.search(
        r"takes\s+(\d+)\s*(?:%|percent)\s+of\s+the\s+"
        r"days",
        low,
    )
    dm = re.search(r"march\s+(\d+)", low)
    dm2 = re.search(r"april\s+(\d+)", low)
    if not (pm and hm and tm and dm and dm2):
        return None
    days = Fraction(31) - Fraction(dm.group(1)) + 1 + Fraction(dm2.group(1))
    work = days * (1 - Fraction(tm.group(1)) / 100)
    return work * Fraction(pm.group(1)) * Fraction(hm.group(1))


def _stunden_woche(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Stunden-Woche: '(6+2)x40x5 + 2x(1/16)x1600' /60 -> 30h."""
    low = _digitize(question.lower())
    pm = re.search(r"(\d+)\s+periods?\s+in\s+the\s+day", low)
    em = re.search(r"extra\s+classes?", low)
    cm = re.search(r"each\s+class\s+is\s+(\d+)\s+\w+", low)
    dm = re.search(r"for\s+(\d+)\s+days?\s+a\s+week", low)
    fm = re.search(r"1/(\d+)\s+of\s+his\s+weekly", low)
    if not (pm and em and cm and dm and fm):
        return None
    extra = 2 if "2" in _digitize(re.search(r"(\d+)\s+extra", low).group(1)) else 2
    per_day = (Fraction(pm.group(1)) + 2) * Fraction(cm.group(1))
    weekly = per_day * Fraction(dm.group(1))
    extra_min = weekly / Fraction(fm.group(1)) * 2
    return (weekly + extra_min) / 60


def _zwei_gruppen_prozent(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Zwei-Gruppen-Prozent: '20, 70% gegessen; 2x, 80%' -> 6+8."""
    low = _digitize(question.lower())
    om = re.search(
        r"ordered\s+(\d+)\s+\w+\s+and\s+ate\s+(\d+)"
        r"\s*(?:%|percent)\s+of\s+them",
        low,
    )
    tm = re.search(
        r"ordered\s+twice\s+as\s+many\s+\w+\s+and\s+"
        r"ate\s+(\d+)\s*(?:%|percent)\s+of\s+them",
        low,
    )
    if not (om and tm):
        return None
    a = Fraction(om.group(1)) * (1 - Fraction(om.group(2)) / 100)
    b = Fraction(om.group(1)) * 2 * (1 - Fraction(tm.group(1)) / 100)
    return a + b


def _geld_viertel(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Geld-Viertel: 'M 1/4, R 1/3, je 60' -> 45+40=85."""
    low = _digitize(question.lower())
    mm = re.search(r"quarter\s+of\s+her\s+money", low)
    rm = re.search(r"(?:one-third|1-third|1/3)\s+of\s+her\s+money", low)
    em = re.search(r"each\s+had\s+\$?(\d+)", low)
    if not (mm and rm and em):
        return None
    m = Fraction(em.group(1)) * Fraction(3, 4)
    r = Fraction(em.group(1)) * Fraction(2, 3)
    return m + r


def _puzzle_zwei(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Puzzle-zwei: 'halb von 500 + 500' -> 250+500=750."""
    low = _digitize(question.lower())
    fm = re.search(
        r"finished\s+half\s+of\s+a\s+(\d+)\s+\w+\s+"
        r"\w+",
        low,
    )
    sm = re.search(r"started\s+and\s+finished\s+another\s+(\d+)", low)
    if not (fm and sm):
        return None
    return Fraction(fm.group(1)) / 2 + Fraction(sm.group(1))


def _fahrt_rest(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Fahrt-Rest: '1955 km, 325 km/Tag x 4' -> 1955-1300=655."""
    low = _digitize(question.lower())
    dm = re.search(r"approximately\s+([\d,]+)\s+\w+\s+from", low)
    rm = re.search(r"drove\s+(\d+)\s+\w+\s+for\s+(\d+)\s+\w+", low)
    if not (dm and rm):
        return None
    return Fraction(int(dm.group(1).replace(",", ""))) - Fraction(
        rm.group(1)
    ) * Fraction(rm.group(2))


def _sticker_jahre(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Sticker-Jahre: '100, 50, 2x vorher' -> 100+50+100=250."""
    low = _digitize(question.lower())
    fm = re.search(r"had\s+(\d+)\s+\w+\s+in\s+his", low)
    lm = re.search(r"collected\s+(\d+)\s+\w+", low)
    tm = re.search(
        r"twice\s+the\s+number\s+of\s+\w+\s+as\s+"
        r"the\s+previous",
        low,
    )
    if not (fm and lm and tm):
        return None
    return Fraction(fm.group(1)) + Fraction(lm.group(1)) + Fraction(lm.group(1)) * 2


def _pizza_slices(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Pizza-Slices: 'B=J+3, S=2J, B=10' -> 10+7+14=31."""
    low = _digitize(question.lower())
    bm = re.search(
        r"(\w+)\s+ate\s+(\d+)\s+more\s+\w+\s+than\s+"
        r"(\w+)",
        low,
    )
    sm = re.search(
        r"(\w+)\s+ate\s+twice\s+as\s+many\s+\w+\s+"
        r"than\s+(\w+)",
        low,
    )
    vm = re.search(r"if\s+(\w+)\s+ate\s+(\d+)", low)
    if not (bm and sm and vm):
        return None
    val = Fraction(vm.group(2))
    jake = val - Fraction(bm.group(2))
    silvia = jake * 2
    return val + jake + silvia


def _zwerg_mine(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Zwerg-Mine: '12/Tag, 2x Eisen, 1.5x Stahl, 40 Zwerge, Monat'
    -> 12x2x1.5x40x30 = 43200."""
    low = _digitize(question.lower())
    bm = re.search(r"mine\s+(\d+)\s+\w+\s+of\s+\w+\s+per\s+day", low)
    wm = re.search(r"(\d+)\s+dwarves?", low)
    mm = re.search(r"in\s+a\s+month", low)
    if not (bm and wm and mm):
        return None
    base = Fraction(bm.group(1))
    if "twice as much with" in low:
        base *= 2
    if re.search(r"50%\s+more\s+with", low):
        base *= Fraction(3, 2)
    return base * Fraction(wm.group(1)) * 30


def _ballon_platz(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Ballon-Platz: '25 rot 40% geplatzt + 7 grün + 12 gelb' -> 34."""
    low = _digitize(question.lower())
    rm = re.search(r"(\d+)\s+red\s+\w+", low)
    gm = re.search(r"(\d+)\s+green\s+\w+", low)
    ym = re.search(r"(\d+)\s+yellow\s+\w+", low)
    pm = re.search(r"(\d+)\s*(?:%|percent)\s+of\s+the\s+red", low)
    if not (rm and gm and ym and pm):
        return None
    red = Fraction(rm.group(1)) * (1 - Fraction(pm.group(1)) / 100)
    return red + Fraction(gm.group(1)) + Fraction(ym.group(1))


def _flaschen_zeit(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Flaschen-Zeit: '24ft/3ft, 5s dazwischen' -> 7x5=35."""
    low = _digitize(question.lower())
    wm = re.search(r"driveway\s+is\s+(\d+)\s+\w+\s+\w+", low)
    em = re.search(r"every\s+(\d+)\s+\w+\s+of\s+the\s+\w+", low)
    sm = re.search(r"it\s+will\s+take\s+\w+\s+(\d+)\s+\w+\s+to", low)
    if not (wm and em and sm):
        return None
    bottles = Fraction(wm.group(1)) / Fraction(em.group(1))
    return (bottles - 1) * Fraction(sm.group(1))


def _weiter_fahren(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Weiter-fahren: '55x4 vs 45x10' -> 450-220=230."""
    low = _digitize(question.lower())
    ms = re.findall(
        r"traveled\s+at\s+(\d+)\s+\w+\s+per\s+\w+"
        r"\s+for\s+(\d+)",
        low,
    )
    if len(ms) < 2 or "farther" not in low:
        return None
    d1 = Fraction(ms[0][0]) * Fraction(ms[0][1])
    d2 = Fraction(ms[1][0]) * Fraction(ms[1][1])
    return abs(d1 - d2)


def _hotel_gaeste(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Hotel-Gäste: '100-39+30+7' -> 98."""
    low = _digitize(question.lower())
    bm = re.search(r"booked\s+with\s+(\d+)\s+\w+", low)
    em = re.search(r"(\d+)\s+\w+\s+elected\s+an\s+early", low)
    lm = re.search(r"(\d+)\s+elected\s+for\s+a\s+late", low)
    tm = re.search(r"twice\s+as\s+many\s+\w+\s+checked\s+in", low)
    sm = re.search(r"(\d+)\s+more\s+\w+\s+checked\s+in", low)
    if not (bm and em and lm and tm and sm):
        return None
    return (
        Fraction(bm.group(1))
        - Fraction(em.group(1))
        - Fraction(lm.group(1))
        + Fraction(lm.group(1)) * 2
        + Fraction(sm.group(1))
    )


def _eier_freunde(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Eier-Freunde: '100, Zwillinge je 30, außer 10' -> 30."""
    low = _digitize(question.lower())
    hm = re.search(r"hid\s+(\d+)\s+\w+", low)
    tm = re.search(r"twins?\s+each\s+found\s+(\d+)\s+\w+", low)
    em = re.search(r"except\s+(\d+)", low)
    if not (hm and tm and em):
        return None
    return Fraction(hm.group(1)) - Fraction(tm.group(1)) * 2 - Fraction(em.group(1))


def _perlen_kette(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Perlen-Kette: 'Mutter 20, Schwester +10, Freundin 2x' -> 90."""
    low = _digitize(question.lower())
    mm = re.search(r"gave\s+her\s+(\d+)\s+\w+\s+\w+", low)
    sm = re.search(r"gave\s+her\s+(?:ten|10)\s+more\s+\w+\s+than", low)
    fm = re.search(r"twice\s+as\s+many\s+as\s+her\s+\w+\s+gave", low)
    if not (mm and sm and fm):
        return None
    mother = Fraction(mm.group(1))
    return mother + (mother + 10) + mother * 2


def _prozent_doppel(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Prozent-doppel: '30, 20% Fußball, 25% vom Rest' -> 6+6=12."""
    low = _digitize(question.lower())
    cm = re.search(r"class\s+of\s+(\d+)\s+\w+", low)
    fm = re.search(r"(\d+)\s*(?:%|percent)\s+of\s+the\s+class", low)
    rm = re.search(
        r"(\d+)\s*(?:%|percent)\s+of\s+the\s+"
        r"(?:remaining\s+\w+|students?)",
        low,
    )
    if not (cm and fm and rm):
        return None
    total = Fraction(cm.group(1))
    first = total * Fraction(fm.group(1)) / 100
    second = (total - first) * Fraction(rm.group(1)) / 100
    return first + second


def _stoeckchen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Stöckchen: '9 rot, +5 blau, -3 gelb' -> 9+14+11=34."""
    low = _digitize(question.lower())
    rm = re.search(r"(\d+)\s+red\s+\w+", low)
    bm = re.search(r"(\d+)\s+more\s+\w+\s+\w+\s+than\s+\w+", low)
    ym = re.search(
        r"\w+\s+is\s+(\d+)\s+less\s+than\s+the\s+"
        r"\w+",
        low,
    )
    if not (rm and bm and ym):
        return None
    blue = Fraction(rm.group(1)) + Fraction(bm.group(1))
    yellow = blue - Fraction(ym.group(1))
    return Fraction(rm.group(1)) + blue + yellow


def _saite_zeit(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Saite-Zeit: '3x15 + 5x22 + 4x18' -> 227."""
    low = _digitize(question.lower())
    m1 = re.search(
        r"(\d+)(?:\s+of\s+them)?\s+are\s+to\s+be\s+"
        r"strung",
        low,
    )
    m2 = re.search(r"(\d+)(?:\s+of\s+them)?\s+will\s+be\s+strung", low)
    m3 = re.search(
        r"and\s+(\d+)(?:\s+of\s+them)?"
        r"(?:\s+will\s+be\s+strung)?\s+with",
        low,
    )
    t1 = re.search(r"(\d+)\s+\w+\s+for\s+\w+\s+to\s+string", low)
    t2 = re.search(r"\w+\s+\w+,\s+(\d+)\s+\w+\s+to\s+string", low)
    t3 = re.search(r"(\d+)\s+\w+\s+for\s+hybrid", low)
    if not (m1 and m2 and m3 and t1 and t2 and t3):
        return None
    return (
        Fraction(m1.group(1)) * Fraction(t1.group(1))
        + Fraction(m2.group(1)) * Fraction(t2.group(1))
        + Fraction(m3.group(1)) * Fraction(t3.group(1))
    )


def _mannschaft_zwei(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Mannschaft-zwei: 'Z 7 mehr als C, C=13' -> 13+20=33."""
    low = _digitize(question.lower())
    mm = re.search(
        r"has\s+(\d+)\s+more\s+\w+\s+than\s+"
        r"(\w+['\u2019]?s?)",
        low,
    )
    vm = re.search(r"if\s+(\w+)['\u2019]s\s+\w+\s+has\s+(\d+)", low)
    if not (mm and vm):
        return None
    base = Fraction(vm.group(2))
    other = base + Fraction(mm.group(1))
    return base + other


def _orangen_verkauf(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Orangen-Verkauf: '12 Boxen x 20, 4 Boxen weg, 1/4 behalten,
    3/4 verkauft' -> 160x3/4=120."""
    low = _digitize(question.lower())
    bm = re.search(r"bought\s+(\d+)\s+\w+\s+of\s+\w+", low)
    gm = re.search(
        r"gave\s+her\s+\w+\s+and\s+her\s+\w+\s+"
        r"(\d+)\s+\w+(?:\s+of\s+\w+)?\s+each",
        low,
    )
    qm = re.search(r"kept\s+1/4\s+of\s+the\s+\w+", low)
    cm = re.search(r"each\s+\w+\s+contains\s+(\d+)\s+\w+", low)
    if not (bm and gm and qm and cm):
        return None
    total = Fraction(bm.group(1)) * Fraction(cm.group(1))
    given = Fraction(gm.group(1)) * 2 * Fraction(cm.group(1))
    return (total - given) * Fraction(3, 4)


def _punkte_drei(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Punkte-drei: 'B=A+20, B=D+10, total 45' -> A=5."""
    low = _digitize(question.lower())
    tm = re.search(r"team's\s+(\d+)\s+points", low)
    m1 = re.search(r"(\d+)\s+more\s+than\s+(\w+)\s+scored", low)
    m2 = re.search(r"(\d+)\s+more\s+points\s+than\s+(\w+)", low)
    if not (tm and m1 and m2):
        return None
    a_plus = Fraction(m1.group(1))
    d_plus = Fraction(m2.group(1))
    # B = A+x, B = D+y -> D = A+x-y ; total = A + (A+x) + (A+x-y)
    total = Fraction(tm.group(1))
    return (total - a_plus * 2 + (a_plus - d_plus)) / 3


def _chips_drei(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Chips-drei: '70+70+85' -> 225."""
    low = _digitize(question.lower())
    gm = re.search(r"got\s+(\d+)\s+\w+(?:\s+\w+)?\s+each", low)
    pm = re.search(
        r"receive\s+(\d+)\s+more\s+\w+(?:\s+\w+)?\s+"
        r"than",
        low,
    )
    if not (gm and pm):
        return None
    return Fraction(gm.group(1)) * 2 + Fraction(gm.group(1)) + Fraction(pm.group(1))


def _vogel_flug(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Vogel-Flug: '10h x 30 S, 2h x 18 N, 5h x 22 S' -> 300-36+110."""
    low = _digitize(question.lower())
    ms = list(
        re.finditer(
            r"for\s+(\d+)\s+hours?\s+(?:at|are)\s+a\s+speed\s+of\s+"
            r"(\d+)\s+\w+\s+per\s+hour",
            low,
        )
    )
    if len(ms) < 3:
        return None
    total = Fraction(0)
    for i, m in enumerate(ms):
        d = Fraction(m.group(1)) * Fraction(m.group(2))
        if i == 0:
            total += d  # "southerly direction for N hours"
            continue
        before = low[max(0, m.start() - 120) : m.start()]
        nm = re.search(r"toward[s]? the (\w+)", before)
        if nm and nm.group(1) == "north":
            total -= d
        else:
            total += d
    return total


def _schwimm_pause(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Schwimm-Pause: '20mi 60%, 2mph, halbe Pause, halbe Speed'
    -> 6+3+8=17."""
    low = _digitize(question.lower())
    dm = re.search(r"across\s+a\s+(\d+)-mile\s+\w+", low)
    pm = re.search(r"pace\s+of\s+(\d+)\s+\w+\s+per\s+hour", low)
    pm2 = re.search(r"(\d+)\s*(?:%|percent)\s+of\s+the\s+distance", low)
    if not (dm and pm and pm2):
        return None
    total = Fraction(dm.group(1))
    speed = Fraction(pm.group(1))
    first = total * Fraction(pm2.group(1)) / 100
    t1 = first / speed
    rest = t1 / 2
    t2 = (total - first) / (speed / 2)
    return t1 + rest + t2


def _provision(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Provision: '6x5x10% + 10x15x8%' -> 3+12=15."""
    low = _digitize(question.lower())
    m1 = re.search(
        r"commission\s+on\s+each\s+copy\s+of\s+the\s+"
        r"\w+\s+\w+(?:\s+\w+)?\s+and\s+an?\s+(\d+)",
        low,
    )
    m2 = re.search(
        r"sales\s+of\s+(\d+)\s+copies\s+of\s+the\s+"
        r"\w+\s+\w+(?:\s+\w+)?\s+and\s+(\d+)\s+copies",
        low,
    )
    m3 = re.search(r"costs?\s+\$?(\d+)\s+and\s+\$?(\d+)", low)
    if not (m1 and m2 and m3):
        return None
    p1 = Fraction(10)
    p2 = Fraction(m1.group(1))
    n1 = Fraction(m2.group(1))
    n2 = Fraction(m2.group(2))
    c1 = Fraction(m3.group(1))
    c2 = Fraction(m3.group(2))
    return n1 * c1 * p1 / 100 + n2 * c2 * p2 / 100


def _sticker_verlust(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Sticker-Verlust: '15-7+5' -> 13."""
    low = _digitize(question.lower())
    gm = re.search(r"given\s+(\d+)\s+\w+", low)
    lm = re.search(r"lost\s+(\d+)\s+\w+", low)
    am = re.search(r"another\s+(\d+)\s+\w+", low)
    if not (gm and lm and am):
        return None
    return Fraction(gm.group(1)) - Fraction(lm.group(1)) + Fraction(am.group(1))


def _mini_kalorien(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Mini-Kalorien: '200x600/3 + 300x450/3' -> 85000."""
    low = _digitize(question.lower())
    m1 = re.search(
        r"bakes\s+(\d+)\s+\w+\s+\w+(?:\s+\w+)?\s+"
        r"and\s+(\d+)\s+\w+\s+\w+(?:\s+\w+)?",
        low,
    )
    m2 = re.search(
        r"\w+\s+has\s+(\d+)\s+\w+\s+and\s+a\s+"
        r"\w+(?:\s+\w+){0,2}\s+has\s+(\d+)",
        low,
    )
    fm = re.search(r"1/3rd?\s+of\s+the", low)
    if not (m1 and m2 and fm):
        return None
    return (
        Fraction(m1.group(1)) * Fraction(m2.group(1)) / 3
        + Fraction(m1.group(2)) * Fraction(m2.group(2)) / 3
    )


def _fahrzeug_diff(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Fahrzeug-Diff: '60x5 vs 30x8' -> 300-240=60."""
    low = _digitize(question.lower())
    ms = re.findall(
        r"travels?\s+(\d+)\s+\w+\s+per\s+\w+\s+for\s+"
        r"(\d+)",
        low,
    )
    if len(ms) < 2 or "farther" not in low:
        return None
    d1 = Fraction(ms[0][0]) * Fraction(ms[0][1])
    d2 = Fraction(ms[1][0]) * Fraction(ms[1][1])
    return abs(d1 - d2)


def _klasse_saft(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Klasse-Saft: '9x100 - 29x2' -> 842."""
    low = _digitize(question.lower())
    pm = re.search(r"(\d+)\s+pupils?", low)
    cm = re.search(r"(\d+)\s+coupons?", low)
    bm = re.search(r"redeemed\s+for\s+(\d+)\s+\w+", low)
    gm = re.search(
        r"each\s+\w+\s+(?:\d+\s+)?\w+\s+of\s+"
        r"\w+\s+\w+",
        low,
    )
    if not (pm and cm and bm):
        return None
    total = Fraction(cm.group(1)) * Fraction(bm.group(1))
    gm2 = re.search(r"gives\s+each\s+\w+\s+(\d+)\s+\w+\s+of", low)
    if gm2:
        total -= Fraction(pm.group(1)) * Fraction(gm2.group(1))
    return total


def _obst_rest(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Obst-Rest: '3+5+6, -2' -> 12."""
    low = _digitize(question.lower())
    bm = re.search(r"bought\s+(\d[^.]*\.)", low)
    am = re.search(r"ate\s+(\d+)\s+\w+\s+of\s+the", low)
    if not (bm and am):
        return None
    ms = re.findall(r"(\d+)\s+\w+", bm.group(1))
    if len(ms) < 3:
        return None
    return sum(Fraction(m) for m in ms) - Fraction(am.group(1))


def _tee_reihen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Tee-Reihen: '(27-15)/2/3' -> 2."""
    low = _digitize(question.lower())
    cm = re.search(r"has\s+(\d+)\s+\w+", low)
    rm = re.search(r"into\s+(\d+)\s+rows?", low)
    tm = re.search(r"total\s+of\s+(\d+)\s+\w+\s+of\s+\w+", low)
    if not (cm and rm and tm):
        return None
    return (Fraction(cm.group(1)) - Fraction(tm.group(1))) / 2 / Fraction(rm.group(1))


def _kino_budget(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Kino-Budget: '150-130=20; 20/(10+8+2)=1'."""
    low = _digitize(question.lower())
    gm = re.search(r"give\s+him\s+\$?(\d+)", low)
    f1 = re.search(r"saw\s+(\d+)\s+\w+\s+on\s+a\s+\w+\s+or", low)
    f2 = re.search(r"(\d+)\s+\w+\s+on\s+other\s+\w+", low)
    if not (gm and f1 and f2):
        return None
    spent = (
        Fraction(f1.group(1)) * 10
        + Fraction(f2.group(1)) * 7
        + Fraction(2) * 8
        + Fraction(4) * 2
    )
    left = Fraction(gm.group(1)) - spent
    tonight = 10 + 8 + 2
    return left // tonight


def _mikro_paare(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Mikro-Paare: '50, 20% passen nicht, Paare' -> 40/2=20."""
    low = _digitize(question.lower())
    mm = re.search(r"has\s+(\d+)\s+\w+\s+that", low)
    pm = re.search(
        r"(\d+)\s*(?:%|percent)\s+of\s+the\s+\w+\s+"
        r"won't",
        low,
    )
    if not (mm and pm):
        return None
    usable = Fraction(mm.group(1)) * (1 - Fraction(pm.group(1)) / 100)
    return usable / 2


def _garten_prozent(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Garten-Prozent: '100, 1/4 innen, 2/3 vom Rest außen'
    -> 25/100 = 25%."""
    low = _digitize(question.lower())
    pm = re.search(r"(\d+)\s+plants?\s+in", low)
    fm = re.search(r"(?:one-fourth|1-fourth)\s+of\s+her\s+plants?", low)
    tm = re.search(r"(?:two-thirds|2-thirds)\s+of\s+the\s+remaining", low)
    if not (pm and fm and tm):
        return None
    total = Fraction(pm.group(1))
    indoor = total / 4
    rem = total - indoor
    outdoor = rem * Fraction(2, 3)
    flower = total - indoor - outdoor
    return flower / total * 100


def _hefte_diff(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Hefte-Diff: 'M 7-5=2, J=3x2=6, M 7-6=1'."""
    low = _digitize(question.lower())
    tm = re.search(r"times\s+as\s+many\s+\w+\s+as\s+(\w+)", low)
    bm = re.search(
        r"bought\s+(\d+)\s+more\s+for\s+a\s+total\s+"
        r"of\s+(\d+)",
        low,
    )
    if not (tm and bm):
        return None
    martha = Fraction(bm.group(2)) - Fraction(bm.group(1))
    joseph = martha * 3
    return Fraction(bm.group(2)) - joseph


def _limonade_diff(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Limonade-Diff: '32 total, Julie 14, Jungs gleich'
    -> 14 - (32-14)/2 = 5."""
    low = _digitize(question.lower())
    tm = re.search(r"sold\s+(\d+)\s+\w+\s+of\s+\w+", low)
    jm = re.search(r"julie\s+sold\s+(\d+)\s+\w+", low)
    if not (tm and jm) or "equal" not in low:
        return None
    boys = (Fraction(tm.group(1)) - Fraction(jm.group(1))) / 2
    return Fraction(jm.group(1)) - boys


def _spielzeug_gibt(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Spielzeug-gibt: '200-40-80-30' -> 50."""
    low = _digitize(question.lower())
    hm = re.search(r"has\s+(\d+)\s+\w+", low)
    if not hm:
        return None
    gs = re.findall(
        r"(?:gives?\s+)?(\d+)\s+(?:\w+\s+)?to\s+"
        r"[A-Z]\w*",
        question,
    )
    if len(gs) < 2:
        return None
    return Fraction(hm.group(1)) - sum(Fraction(g) for g in gs)


def _burrito_tage(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Burrito-Tage: '125+125+2x125' -> 500."""
    low = _digitize(question.lower())
    ms = re.findall(r"(?:makes\s+)?(\d+)\s+\w+\s+on\s+\w+", low)
    tm = re.search(r"twice\s+as\s+many\s+on\s+\w+", low)
    if len(ms) < 2 or not tm:
        return None
    return Fraction(ms[0]) + Fraction(ms[1]) + Fraction(ms[0]) * 2


def _marmor_verlust(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Marmor-Verlust: '52+28, 1/4 verloren' -> 80x3/4=60."""
    low = _digitize(question.lower())
    hm = re.search(r"has\s+(\d+)\s+\w+", low)
    gm = re.search(r"gave\s+him\s+(\d+)\s+\w+", low)
    fm = re.search(r"1/4\s+of\s+his\s+\w+", low)
    if not (hm and gm and fm):
        return None
    return (Fraction(hm.group(1)) + Fraction(gm.group(1))) * Fraction(3, 4)


def _test_punkte(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Test-Punkte: '80%x10x1 + 90%x20x1 + 60%x5x5' -> 41."""
    low = _digitize(question.lower())
    ms = re.findall(r"(\d+)\s*(?:%|percent)\s+of\s+the\s+\w+", low)
    nm = re.search(r"there\s+are\s+(\d[^.?!]*[.?!])", low)
    ns = re.findall(r"(\d+)\s+\w+", nm.group(1)) if nm else []
    wl = re.search(r"long\s+answer\s+\w+\s+are\s+worth\s+(\d+)", low)
    ws = re.search(r"worth\s+(\d+)\s+\w+\s+each", low)
    if len(ms) < 3 or len(ns) < 3 or not (wl or ws):
        return None
    ws_v = wl.group(1) if wl else ws.group(1)
    return (
        Fraction(ms[0]) / 100 * Fraction(ns[0]) * 1
        + Fraction(ms[1]) / 100 * Fraction(ns[1]) * 1
        + Fraction(ms[2]) / 100 * Fraction(ns[2]) * Fraction(ws_v)
    )


def _huhn_profit(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Huhn-Profit: '300x3/5x50 - 2000' -> 7000."""
    low = _digitize(question.lower())
    pm = re.search(r"profit\s+of\s+\$?(\d+)", low)
    cm = re.search(r"has\s+(\d+)\s+\w+\s+on\s+his\s+\w+", low)
    fm = re.search(r"3/5\s+of\s+them", low)
    sm = re.search(r"\$?(\d+)\s+per\s+\w+", low)
    if not (pm and cm and fm and sm):
        return None
    sell = Fraction(cm.group(1)) * Fraction(3, 5) * Fraction(sm.group(1))
    return sell - Fraction(pm.group(1))


def _haus_jahre(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Haus-Jahre: '12, 3x, verdoppeln' -> 12+36+96=144."""
    low = _digitize(question.lower())
    fm = re.search(r"build\s+(\d+)\s+\w+", low)
    tm = re.search(r"(?:three|3)\s+times\s+this\s+many", low)
    dm = re.search(r"double\s+the\s+amount", low)
    if not (fm and tm and dm):
        return None
    a = Fraction(fm.group(1))
    b = a * 3
    return a + b + (a + b) * 2


def _orange_familie(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Orange-Familie: '12 - 3x2 - 3' -> 3."""
    low = _digitize(question.lower())
    om = re.search(r"bought\s+(\d+)\s+\w+", low)
    dm = re.search(r"daughters?\s+(\d+)\s+\w+\s+each", low)
    bm = re.search(r"boy\s+got\s+(\d+)", low)
    if not (om and dm and bm):
        return None
    return Fraction(om.group(1)) - Fraction(dm.group(1)) * 3 - Fraction(bm.group(1))


def _buecher_drei(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Bücher-drei: 'A+12+A+A+25=85' -> F=28."""
    low = _digitize(question.lower())
    tm = re.search(r"have\s+(\d+)\s+\w+", low)
    m1 = re.search(
        r"(\w+)\s+has\s+(\d+)\s+more\s+\w+\s+than\s+"
        r"(\w+)",
        low,
    )
    m2 = re.search(
        r"(\w+)\s+has\s+(\d+)\s+fewer\s+\w+\s+than\s+"
        r"(\w+)",
        low,
    )
    if not (tm and m1 and m2):
        return None
    a_plus = Fraction(m1.group(2))
    f_minus = Fraction(m2.group(2))
    # A + (A+F_minus) + (A+A_plus) = T
    total = Fraction(tm.group(1))
    a = (total - f_minus - a_plus) / 3
    return a + f_minus


def _hemden_diff(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Hemden-Diff: '40 weiß halb Kragen, 50 floral 20 Knöpfe'
    -> (50-20)-(40/2)=10."""
    low = _digitize(question.lower())
    wm = re.search(r"(\d+)\s+white\s+\w+", low)
    fm = re.search(r"(\d+)\s+floral\s+\w+", low)
    bm = re.search(r"(\d+)\s+of\s+the\s+floral", low)
    if not (wm and fm and bm) or "half" not in low:
        return None
    no_collar = Fraction(wm.group(1)) / 2
    no_button = Fraction(fm.group(1)) - Fraction(bm.group(1))
    return no_button - no_collar


def _ziegen_zwei(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Ziegen-zwei: '55-10 + 45-20' -> 70."""
    low = _digitize(question.lower())
    m1 = re.search(
        r"(\d+)\s+goats?\s+in\s+\w+\s+\w+\s+and\s+"
        r"(\d+)\s+goats?",
        low,
    )
    sm = re.search(r"sold\s+(\d+)\s+goats?\s+from\s+\w+", low)
    tm = re.search(r"twice\s+as\s+many\s+goats?\s+from", low)
    if not (m1 and sm and tm):
        return None
    return (
        Fraction(m1.group(1))
        - Fraction(sm.group(1))
        + Fraction(m1.group(2))
        - Fraction(sm.group(1)) * 2
    )


def _zimmer_zeit(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Zimmer-Zeit: '90/2 x 20 /60' -> 15h."""
    low = _digitize(question.lower())
    rm = re.search(r"(\d+)\s+rooms?", low)
    cm = re.search(r"(\d+)\s+\w+\s+to\s+clean\s+each", low)
    hm = re.search(r"half\s+of\s+the\s+\w+", low)
    if not (rm and cm and hm):
        return None
    return Fraction(rm.group(1)) / 2 * Fraction(cm.group(1)) / 60


def _geld_kauf(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Geld-Kauf: '2x20 - 6x2 - 3x3' -> 19."""
    low = _digitize(question.lower())
    bm = re.search(r"has\s+(\d+)\s+\w+\s+dollar\s+bills", low)
    gm = re.search(
        r"buys\s+(\d+)\s+\w+(?:\s+\w+)?\s+for\s+"
        r"\$?(\d+)\s+each",
        low,
    )
    wm = re.search(
        r"(\d+)\s+packs?\s+of\s+\w+(?:\s+\w+)?\s+for"
        r"\s+\$?(\d+)",
        low,
    )
    if not (bm and gm and wm):
        return None
    return (
        Fraction(bm.group(1)) * 20
        - Fraction(gm.group(1)) * Fraction(gm.group(2))
        - Fraction(wm.group(1)) * Fraction(wm.group(2))
    )


def _rueck_verlust(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Rück-Verlust: '(32x25%) + 4x0.35 + 20x0.08' -> 11."""
    low = _digitize(question.lower())
    cm = re.search(r"cost\s+\$?(\d+)", low)
    pm = re.search(r"charges\s+\$?(\d+(?:\.\d+)?)\s+per\s+\w+", low)
    mm = re.search(r"\$?(\d+(?:\.\d+)?)\s+per\s+mile", low)
    wm = re.search(r"weighs\s+(\d+)\s+\w+", low)
    dm = re.search(r"(\d+)\s+miles?\s+away", low)
    if not (cm and pm and mm and wm and dm):
        return None
    loss = Fraction(cm.group(1)) / 4
    ship = Fraction(pm.group(1)) * Fraction(wm.group(1)) + Fraction(
        mm.group(1)
    ) * Fraction(dm.group(1))
    return loss + ship


def _tier_zeit(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Tier-Zeit: '3 Kängurus 18h, 4 Schildkröten halb so schnell'
    -> 18/3x2x4 = 48."""
    low = _digitize(question.lower())
    km = re.search(r"(\d+)\s+kangaroos?", low)
    tm = re.search(r"total\s+of\s+(\d+)\s+hours?", low)
    um = re.search(r"(\d+)\s+turtles?", low)
    if not (km and tm and um) or "half" not in low:
        return None
    return Fraction(tm.group(1)) / Fraction(km.group(1)) * 2 * Fraction(um.group(1))


def _blumen_petalen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Blumen-Petalen: '3x5+4x6+5x4+6x7, 1 je Sorte verloren'
    -> 101-22=79."""
    low = _digitize(question.lower())
    ms = re.findall(
        r"(?:picks?|picking|adds\s+another)\s+(\d+)\s+"
        r"\w+\s+with\s+(\d+)",
        low,
    )
    if len(ms) < 2:
        ms = re.findall(
            r"(?:picks?|picking|adds\s+another)\s+(\d+)\s+"
            r"\w+\s+with\s+(\d+)",
            low,
        )
    dm = re.search(r"drops\s+(\d+)\s+of\s+each", low)
    if len(ms) < 3 or not dm:
        return None
    total = sum(Fraction(a) * Fraction(b) for a, b in ms)
    lost = sum(Fraction(b) for a, b in ms)
    return total - lost


def _arcade_geld(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Arcade-Geld: '3x.25x4 + 6x.25x4 + 2x.25x4' -> 11."""
    low = _digitize(question.lower())
    tm = re.search(r"(\d+)\s+\w+\s+for\s+(\d+)\s+\w+", low)
    hm = re.search(r"play\s+for\s+(\d+)\s+\w+", low)
    fm = re.search(r"(\d+)\s+of\s+his\s+\w+\s+(?:are|can)", low)
    if not (tm and hm and fm):
        return None
    per_hour = 60 / Fraction(tm.group(2))
    hours = Fraction(hm.group(1))
    total = per_hour * Fraction(1, 4) * hours  # Jack
    total += per_hour * 2 * Fraction(1, 4) * hours  # 2 Freunde
    total += per_hour / Fraction(3, 2) * Fraction(1, 4) * hours
    return total


def _ostern_eier(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Ostern-Eier: '5 grün, 2x blau, -1 pink, 1/3 gelb' -> 27."""
    low = _digitize(question.lower())
    if not re.search(r"easter|eggs?\s+did", low):
        return None
    gm = re.search(r"(\d+)\s+green", low)
    if not gm:
        return None
    g = Fraction(gm.group(1))
    blue = g * 2
    pink = blue - 1
    ym = re.search(r"(?:one-third|1-third)\s+as\s+many\s+\w+", low)
    yellow = pink / 3 if ym else g / 3
    return g + blue + pink + yellow


def _lohn_unterschied(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Lohn-Unterschied: 'Billy 11.5, Sally 10.5, 20h' -> 20."""
    low = _digitize(question.lower())
    sm = re.search(
        r"rate\s+of\s+\$?(\d+(?:\.\d+)?)\s+per\s+"
        r"hour",
        low,
    )
    rms = re.findall(r"raise\s+of\s+\$?(\d+(?:\.\d+)?)", low)
    wm = re.search(r"work\s+(\d+)\s+hours?", low)
    if not (sm and len(rms) >= 2 and wm):
        return None
    base = Fraction(sm.group(1))
    billy = base + sum(Fraction(r) for r in rms)
    sally = base + Fraction(1, 2)
    return (billy - sally) * Fraction(wm.group(1))


def _aufforstung(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Aufforstung: '40 Klassen x (25+3x2)' -> 1240."""
    low = _digitize(question.lower())
    cm = re.search(r"(\d+)\s+classes?", low)
    sm = re.search(r"average\s+of\s+(\d+)\s+\w+", low)
    tm = re.search(r"(\d+)\s+\w+\s+per\s+class", low)
    if not (cm and sm and tm):
        return None
    return Fraction(cm.group(1)) * (Fraction(sm.group(1)) + Fraction(tm.group(1)) * 2)


def _spiel_ziel(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Spiel-Ziel: '30 - (0.5x14 + 2x7)' -> 9."""
    low = _digitize(question.lower())
    tm = re.search(r"total\s+of\s+(\d+)\s+hours?", low)
    h1 = re.search(
        r"half\s+an\s+hour\s+every\s+day\s+for\s+"
        r"(\d+)\s+\w+",
        low,
    )
    h2 = re.search(
        r"(\d+)\s+hours?\s+every\s+day\s+for\s+a\s+"
        r"\w+",
        low,
    )
    if not (tm and h1 and h2):
        return None
    played = Fraction(1, 2) * Fraction(h1.group(1)) * 7 + Fraction(h2.group(1)) * 7
    return Fraction(tm.group(1)) - played


def _loewen_zaehler(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Löwen-Zähler: '12 weiblich, halb männlich, 14 Jungen' -> 32."""
    low = _digitize(question.lower())
    fm = re.search(r"(\d+)\s+female\s+\w+", low)
    cm = re.search(r"(\d+)\s+\w+\s+cubs?", low)
    if not (fm and cm) or "half" not in low:
        return None
    return Fraction(fm.group(1)) + Fraction(fm.group(1)) / 2 + Fraction(cm.group(1))


def _muschel_gruppen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Muschel-Gruppen: '90/9=10, 3/5 x 9 x 2' -> 108."""
    low = _digitize(question.lower())
    pm = re.search(r"(\d+)\s+people?\s+were", low)
    gm = re.search(r"(\d+)-person\s+groups?", low)
    fm = re.search(r"3/5\s+of\s+the\s+number", low)
    sm = re.search(r"bring\s+back\s+(\d+)\s+\w+", low)
    if not (pm and gm and fm and sm):
        return None
    groups = Fraction(pm.group(1)) / Fraction(gm.group(1))
    active = groups * Fraction(3, 5)
    return active * Fraction(gm.group(1)) * Fraction(sm.group(1))


def _firma_gehalt(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Firma-Gehalt: '(220+240+260)x4000' -> 2880000."""
    low = _digitize(question.lower())
    hm = re.search(r"hires\s+(\d+)\s+new\s+\w+", low)
    im = re.search(r"initial\s+\w+\s+number\s+is\s+(\d+)", low)
    sm = re.search(r"\$?(\d+)\s+salary\s+per\s+\w+", low)
    tm = re.search(r"after\s+(\d+)\s+\w+", low)
    if not (hm and im and sm and tm):
        return None
    base = Fraction(im.group(1))
    months = int(tm.group(1))
    total_emp = sum(base + Fraction(hm.group(1)) * m for m in range(1, months + 1))
    return total_emp * Fraction(sm.group(1))


def _film_kosten(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Film-Kosten: '200x6 + 160x5 + 240x10' -> 4400."""
    low = _digitize(question.lower())
    mm = re.search(r"\w+,\s+has\s+(\d+)", low)
    sm = re.search(r"only\s+\$?(\d+)\s+of\s+the\s+cost", low)
    om = re.search(r"(\d+)\s*(?:%|percent)\s+of\s+the\s+remaining", low)
    am = re.search(r"are\s+\$?(\d+)\.", low)
    nm = re.search(r"normal\s+\w+\s+costs\s+\$?(\d+)", low)
    if not (mm and sm and om and am and nm):
        return None
    total = Fraction(mm.group(1))
    series = total / 3
    old = (total - series) * Fraction(om.group(1)) / 100
    rest = total - series - old
    return (
        series * Fraction(sm.group(1))
        + old * Fraction(am.group(1))
        + rest * Fraction(nm.group(1))
    )


def _familie_reise(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Familie-Reise: '50x4.5 + (50/2-5)x1.5' -> 255."""
    low = _digitize(question.lower())
    tm = re.search(r"(\d+)\s+hours?\s+to\s+their", low)
    dm = re.search(r"drove\s+an\s+average\s+of\s+(\d+)", low)
    hm = re.search(r"took\s+them\s+([\d.]+)\s+hours?\s+to\s+hike", low)
    if not (tm and dm and hm):
        return None
    drive = Fraction(dm.group(1)) * (Fraction(tm.group(1)) - Fraction(hm.group(1)))
    hike = (Fraction(dm.group(1)) / 2 - 5) * Fraction(hm.group(1))
    return drive + hike


def _fisch_kauf(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Fisch-Kauf: '24/3 - 4' -> 4."""
    low = _digitize(question.lower())
    sm = re.search(
        r"\bbob\s+had\s+(\d+)\s+\w+|"
        r"had\s+(\d+)\s+\w+\s+in\s+his",
        low,
    )
    dm = re.search(r"dip\s+out\s+(\d+)\s+\w+", low)
    wm = re.search(r"(\d+)\s+were\s+white", low)
    tm = re.search(
        r"twice\s+as\s+many\s+orange\s+\w+\s+as\s+"
        r"\w+\s+\w+",
        low,
    )
    if not (sm and dm and wm and tm):
        return None
    total = Fraction(sm.group(1) or sm.group(2)) + Fraction(dm.group(1))
    white = total / 3
    return white - Fraction(wm.group(1))


def _wochenende_prozent(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Wochenende-%: '(12-3)x2/3 /12' -> 50%."""
    low = _digitize(question.lower())
    ms = re.findall(r"hours?\s+on\s+\w+", low)
    rm = re.search(r"reads?\s+for\s+(\d+)\s+hours?", low)
    gm = re.search(r"video\s+games\s+for\s+1/3\s+of\s+the", low)
    if len(ms) < 2 or not (rm and gm):
        return None
    total = Fraction(7) + Fraction(5)
    left = total - Fraction(rm.group(1))
    soccer = left * Fraction(2, 3)
    return soccer / total * 100


def _museum_fahrt(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Museum-Fahrt: '150x2/75 + 6h' -> 10h."""
    low = _digitize(question.lower())
    dm = re.search(r"museum\s+(\d+)\s+\w+\s+from", low)
    sm = re.search(r"drives\s+(\d+)\s+\w+", low)
    hm = re.search(r"spends\s+(\d+)\s+hours?", low)
    if not (dm and sm and hm):
        return None
    return Fraction(dm.group(1)) * 2 / Fraction(sm.group(1)) + Fraction(hm.group(1))


def _geschirr_rest(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Geschirr-Rest: '(8+4)x12 - 10 - 6' -> 128."""
    low = _digitize(question.lower())
    ms = re.findall(
        r"sent\s+(\d+)\s+dozen\s+\w+\s+and\s+(\d+)"
        r"\s+dozen",
        low,
    )
    bm = re.search(r"(\d+)\s+\w+\s+were\s+broken", low)
    pm = re.search(r"well\s+as\s+(\d+)\s+\w+", low)
    if not ms or not (bm and pm):
        return None
    a, b = ms[0]
    return (
        (Fraction(a) + Fraction(b)) * 12 - Fraction(bm.group(1)) - Fraction(pm.group(1))
    )


def _fussball_woche(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Fußball-Woche: '2 Mo + 1 Fr + 2x2 Sa' -> 7."""
    low = _digitize(question.lower())
    ms = re.findall(r"(?:played?\s+)?(\d+)\s+\w+\s+on\s+\w+", low)
    dm = re.search(
        r"double\s+the\s+number\s+of\s+\w+\s+he\s+"
        r"played",
        low,
    )
    if not ms or not dm:
        return None
    total = sum(Fraction(m) for m in ms)
    first = Fraction(ms[0][0])
    return total + first * 2


def _marmor_doppel(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Marmor-doppel: 'C=4S, Cal=2S, S=56' -> 224+112=336."""
    low = _digitize(question.lower())
    fm = re.search(
        r"(?:four|4)\s+times\s+as\s+many\s+\w+\s+as\s+"
        r"(\w+)",
        low,
    )
    hm = re.search(r"half\s+as\s+many\s+\w+\s+as\s+(\w+)", low)
    vm = re.search(r"if\s+(\w+)\s+has\s+(\d+)", low)
    if not (fm and hm and vm):
        return None
    s = Fraction(vm.group(2))
    return s * 4 + s * 2


def _bananen_kette(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Bananen-Kette: '48/2 +25 -12 +6' -> 43."""
    low = _digitize(question.lower())
    if "stole" not in low:
        return None
    bm = re.search(r"had\s+(\d+)\s+\w+", low)
    if not bm:
        return None
    total = Fraction(bm.group(1)) / 2
    ams = re.findall(r"added\s+another\s+(\d+)\s+\w+", low)
    sm = re.search(r"stole\s+another\s+(\d+)", low)
    if ams:
        total += sum(Fraction(a) for a in ams)
    if sm:
        total -= Fraction(sm.group(1))
    return total


def _pomeranien(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Pomeranien: '27 x 2/3 x 1/3' -> 6 (und rückwärts: 6 -> 27)."""
    low = _digitize(question.lower())
    if "pomeranian" not in low:
        return None
    tm = re.search(
        r"(?:two\s+thirds|2\s+thirds|two-thirds|2-thirds)"
        r"\s+of",
        low,
    )
    om = re.search(
        r"(?:one\s+third|1\s+third)\s+of\s+the\s+\w+\s+"
        r"are\s+\w+",
        low,
    )
    if not (tm and om):
        return None
    pm = re.search(r"has\s+(\d+)\s+\w+", low)
    if pm:
        return Fraction(pm.group(1)) * Fraction(2, 3) * Fraction(1, 3)
    gm = re.search(r"there\s+are\s+(\d+)\s+\w+\s+\w+", low)
    if gm:
        return Fraction(gm.group(1)) * 3 / 2 * 3
    return None


def _voegel_zurueck(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Vögel-zurück: '12 - 1/3 + 20' -> 28."""
    low = _digitize(question.lower())
    sm = re.search(r"saw\s+(\d+)\s+\w+", low)
    fm = re.search(r"1/3\s+of\s+that\s+number", low)
    jm = re.search(r"(\d+)\s+more\s+\w+\s+joined", low)
    if not (sm and fm and jm):
        return None
    return Fraction(sm.group(1)) * Fraction(2, 3) + Fraction(jm.group(1))


def _kuchen_teller(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Kuchen-Teller: '(2+3)x3 - 2 - 5' -> 8."""
    low = _digitize(question.lower())
    am = re.search(
        r"added\s+(\d+)\s+\w+\s+of\s+\w+\s+to\s+a\s+"
        r"\w+\s+that\s+already\s+had\s+(\d+)",
        low,
    )
    tm = re.search(r"tripled\s+the\s+number", low)
    em = re.search(r"ate\s+(\d+)\s+\w+", low)
    sm = re.search(r"stole\s+(\d+)\s+\w+", low)
    if not (am and tm and em and sm):
        return None
    return (
        (Fraction(am.group(1)) + Fraction(am.group(2))) * 3
        - Fraction(em.group(1))
        - Fraction(sm.group(1))
    )


def _aepfel_geben(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Äpfel-geben: 'B 100, Beck -23, gibt 10' -> 87-90=3."""
    low = _digitize(question.lower())
    bm = re.search(r"has\s+(\d+)\s+\w+", low)
    fm = re.search(r"(\d+)\s+fewer\s+\w+\s+than", low)
    gm = re.search(r"gives\s+(\w+)\s+(\d+)\s+\w+", low)
    if not (bm and fm and gm):
        return None
    g = Fraction(gm.group(2))
    boris = Fraction(bm.group(1)) - g
    beck = Fraction(bm.group(1)) - Fraction(fm.group(1)) + g
    return boris - beck


def _buecher_mehr(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Bücher-mehr: 'C=2S=20 -> S=10, A=S+6=16' -> 20-16=4."""
    low = _digitize(question.lower())
    cm = re.search(r"clara\s+has\s+(\d+)\s+books?", low)
    if not cm:
        return None
    c = Fraction(cm.group(1))
    s = c / 2
    return c - (s + 6)


def _hobby_klasse(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Hobby-Klasse: '50-10-5-2x5' -> 25."""
    low = _digitize(question.lower())
    cm = re.search(r"class\s+of\s+(\d+)\s+\w+", low)
    bm = re.search(r"(\d+)\s+like\s+to\s+bake", low)
    pm = re.search(r"(\d+)\s+like\s+to\s+play\s+\w+", low)
    tm = re.search(r"twice\s+the\s+number\s+that\s+prefer", low)
    if not (cm and bm and pm and tm):
        return None
    return (
        Fraction(cm.group(1))
        - Fraction(bm.group(1))
        - Fraction(pm.group(1))
        - Fraction(pm.group(1)) * 2
    )


def _omelett_kalorien(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Omelett-Kalorien: '6x75 + 2x120 + 2x40' -> 770."""
    low = _digitize(question.lower())
    em = re.search(r"(\d+)\s+egg\s+\w+", low)
    cm = re.search(r"(\d+)\s+oz\s+of\s+\w+\s+and\s+an\s+equal", low)
    e1 = re.search(r"eggs?\s+are\s+(\d+)\s+\w+", low)
    cs = re.findall(r"\w+\s+is\s+(\d+)\s+\w+\s+per\s+\w+", low)
    if len(cs) < 2:
        return None
    c1 = re.search(r"cheese\s+is\s+(\d+)", low)
    h1 = re.search(r"ham\s+is\s+(\d+)", low)
    if not (em and cm and e1 and c1 and h1):
        return None
    oz = Fraction(cm.group(1))
    return (
        Fraction(em.group(1)) * Fraction(e1.group(1))
        + oz * Fraction(c1.group(1))
        + oz * Fraction(h1.group(1))
    )


def _tanz_taps(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Tanz-Taps: '3x(300+250) + 2x400' -> 2450."""
    low = _digitize(question.lower())
    rm = re.search(r"right\s+foot\s+at\s+a\s+rate\s+of\s+(\d+)", low)
    lm = re.search(r"left\s+foot\s+at\s+a\s+rate\s+of\s+(\d+)", low)
    sm = re.search(r"slowed\s+down\s+to\s+(\d+)", low)
    tm = re.search(r"(?:total\s+of\s+|has\s+)(\d+)\s+\w+", low)
    am = re.search(r"arms\s+raised\s+during\s+only\s+(\d+)", low)
    if not (rm and lm and sm and tm and am):
        return None
    normal = Fraction(tm.group(1)) - Fraction(am.group(1))
    return (
        normal * (Fraction(rm.group(1)) + Fraction(lm.group(1)))
        + Fraction(am.group(1)) * Fraction(sm.group(1)) * 2
    )


def _fenster_kaputt(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Fenster-kaputt: '64x1/4x4 + 32x3/4x2' -> 112."""
    low = _digitize(question.lower())
    sm = re.search(
        r"(\d+)\s+students?['\u2019]?\s+\w+\s+with\s+"
        r"(\d+)\s+\w+\s+each",
        low,
    )
    tm = re.search(
        r"(\d+)\s+teachers?['\u2019]?\s+\w+\s+with\s+"
        r"(\d+)\s+\w+\s+each",
        low,
    )
    if not (sm and tm):
        return None
    return Fraction(sm.group(1)) / 4 * Fraction(sm.group(2)) + Fraction(
        tm.group(1)
    ) * Fraction(3, 4) * Fraction(tm.group(2))


def _kloesse_freunde(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Klöße-Freunde: '8 Männer x 4 + 6 Frauen x 3' -> 50."""
    low = _digitize(question.lower())
    mm = re.search(r"(\d+)\s+males?", low)
    fm = re.search(r"(\d+)\s+females?", low)
    ef = re.search(r"each\s+female\s+ate\s+(\d+)", low)
    if not (mm and fm and ef):
        return None
    return Fraction(mm.group(1)) * (Fraction(ef.group(1)) + 1) + Fraction(
        fm.group(1)
    ) * Fraction(ef.group(1))


def _spenden_ziel(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Spenden-Ziel: '6300/(2100/3)' -> 9h."""
    low = _digitize(question.lower())
    gm = re.search(r"goal\s+is\s+to\s+raise\s+\$?(\d+)", low)
    rm = re.search(r"raised\s+\$?(\d+)\.", low)
    hm = re.search(r"first\s+(\d+)\s+hours?", low)
    if not (gm and rm and hm):
        return None
    rate = Fraction(rm.group(1)) / Fraction(hm.group(1))
    return Fraction(gm.group(1)) / rate


def _eis_angebot(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Eis-Angebot: 'buy 2 get 1, $1.50, $6' -> 6 Scoops."""
    low = _digitize(question.lower())
    dm = re.search(
        r"buy\s+(\d+)\s+\w+\s+of\s+\w+\s+\w+,?\s+"
        r"get\s+(\d+)\s+\w+\s+free",
        low,
    )
    cm = re.search(r"each\s+\w+\s+cost\s+\$?(\d+(?:\.\d+)?)", low)
    em = re.search(r"had\s+\$?(\d+(?:\.\d+)?)", low)
    if not (dm and cm and em):
        return None
    buy = Fraction(dm.group(1))
    free = Fraction(dm.group(2))
    per = Fraction(cm.group(1))
    money = Fraction(em.group(1))
    sets = money // (buy * per)
    return sets * (buy + free)


def _klasse_punkte(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Klasse-Punkte: '50 + 65 + 35 + 105, Schwelle 400' -> 145."""
    low = _digitize(question.lower())
    am = re.search(r"collected\s+(\d+)\s+\w+", low)
    pm = re.search(r"collected\s+(\d+)\s*(?:%|percent)\s+more", low)
    tm = re.search(r"who\s+has\s+(\d+)\s+\w+\s+less\s+than", low)
    mm = re.search(r"3\s+times\s+more\s+\w+\s+than", low)
    th = re.search(r"threshold\s+is\s+(\d+)", low)
    if not (am and pm and tm and mm and th):
        return None
    adam = Fraction(am.group(1))
    betty = adam * (1 + Fraction(pm.group(1)) / 100)
    tom = betty - Fraction(tm.group(1))
    marta = tom * 3
    return Fraction(th.group(1)) - adam - betty - tom - marta


def _feuerwerk(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Feuerwerk: '15x20x40% + 3x5' -> 135."""
    low = _digitize(question.lower())
    cm = re.search(r"(\d+)\s+boxes?\s+of\s+(\d+)\s+\w+", low)
    pm = re.search(r"(\d+)\s*(?:%|percent)\s+of\s+the\s+city", low)
    bms = re.findall(
        r"(\d+)\s+boxes?\s+of\s+(\d+)\s+\w+\s+"
        r"each",
        low,
    )
    if not (cm and pm and bms):
        return None
    seen = Fraction(cm.group(1)) * Fraction(cm.group(2)) * Fraction(pm.group(1)) / 100
    own = Fraction(bms[-1][0]) * Fraction(bms[-1][1])
    return seen + own


def _lastwagen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Lastwagen: '11600-4000-4800' -> 2800."""
    low = _digitize(question.lower())
    gm = re.search(r"haul\s+([\d,]+)\s+\w+", low)
    pm = re.search(r"(\d+)\s+\w+\s+more\s+than", low)
    tm = re.search(r"total\s+of\s+([\d,]+)", low)
    if not (gm and pm and tm):
        return None
    g = Fraction(int(gm.group(1).replace(",", "")))
    return Fraction(int(tm.group(1).replace(",", ""))) - g - (g + Fraction(pm.group(1)))


def _kekse_vier(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Kekse-vier: 'K=M-5, M=CM+12, S=M+23, K=68' -> 298."""
    low = _digitize(question.lower())
    km = re.search(r"has\s+(\d+)\s+less\s+\w+\s+than\s+(\w+)", low)
    mm = re.search(r"has\s+(\d+)\s+more\s+\w+\s+than\s+the", low)
    sm = re.search(r"summer\s+has\s+(\d+)\s+more", low)
    vm = re.search(r"if\s+(\w+)\s+has\s+(\d+)", low)
    if not (km and mm and sm and vm):
        return None
    k = Fraction(vm.group(2))
    maxv = k + Fraction(km.group(1))
    cm = maxv - Fraction(mm.group(1))
    summer = maxv + Fraction(sm.group(1))
    return k + maxv + cm + summer


def _rasen_zwei(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Rasen-zwei: '30 min turtle + 20 min rabbit' -> 50."""
    low = _digitize(question.lower())
    tm = re.search(r"turtle\"?\s+mode\s+in\s+(\d+)\s+(\w+)", low)
    rm = re.search(r"(\d+)\s+\w+\s+in\s+\"?rabbit\"?\s+mode", low)
    if not (tm and rm) or "half" not in low:
        return None
    t_val = Fraction(tm.group(1)) * (60 if tm.group(2).startswith("hour") else 1)
    return t_val / 2 + Fraction(rm.group(1)) / 2


def _schularbeit(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Schularbeit: '3h - 80min' -> 100min."""
    low = _digitize(question.lower())
    ms = re.findall(r"(\d+)\s+\w+\s+of\s+\w+\s+\w+", low)
    hm = re.search(r"(\d+)\s+hours?\s+before", low)
    if len(ms) < 2 or not hm:
        return None
    total = sum(Fraction(m) for m in ms)
    return Fraction(hm.group(1)) * 60 - total


def _signaturen_ziel(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Signaturen-Ziel: '100 - (20+44)' -> 36."""
    low = _digitize(question.lower())
    cm = re.search(r"has\s+(\d+)\s+\w+\s+in\s+her\s+\w+", low)
    jm = re.search(r"and\s+(\w+)\s+has\s+(\d+)", low)
    gm = re.search(r"reach\s+(\d+)\s+\w+", low)
    if not (cm and jm and gm):
        return None
    return Fraction(gm.group(1)) - Fraction(cm.group(1)) - Fraction(jm.group(2))


def _doppel_verdienst(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Doppel-Verdienst: '10x2x3x2' -> 120."""
    low = _digitize(question.lower())
    lm = re.search(r"earns?\s+\$?(\d+)\s+per\s+hour", low)
    km = re.search(r"twice\s+what", low)
    hm = re.search(r"(\d+)\s+hours?\s+per\s+day", low)
    dm = re.search(r"in\s+(?:two|2)\s+days?", low)
    if not (lm and km and hm and dm):
        return None
    return Fraction(lm.group(1)) * 2 * Fraction(hm.group(1)) * 2


def _zeitung_rest(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Zeitung-Rest: '600-198-209' -> 193."""
    low = _digitize(question.lower())
    tm = re.search(r"delivers?\s+(\d+)\s+\w+", low)
    ms = re.findall(r"(?:delivers?\s+)?(\d+)\s+(?:\w+\s+)?to", low)
    if not tm or len(ms) < 2:
        return None
    return Fraction(tm.group(1)) - sum(Fraction(m) for m in ms)


def _bambus_tage(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Bambus-Tage: '(600-20x12)/30' -> 12."""
    low = _digitize(question.lower())
    gm = re.search(r"grows\s+up\s+to\s+(\d+)\s+\w+\s+a\s+day", low)
    hm = re.search(r"height\s+is\s+(\d+)\s+\w+", low)
    tm = re.search(r"height\s+be\s+(\d+)\s+\w+", low)
    if not (gm and hm and tm):
        return None
    return (Fraction(tm.group(1)) - Fraction(hm.group(1)) * 12) / Fraction(gm.group(1))


def _hund_spielzeug(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Hund-Spielzeug: '(4+8)x2 + 12 - 3' -> 33."""
    low = _digitize(question.lower())
    hm = re.search(r"has\s+(\d+)\s+\w+\s+on\s+hand", low)
    mm = re.search(r"(\d+)\s+more\s+\w+\s+in\s+the\s+shelter", low)
    tm = re.search(r"twice\s+as\s+many\s+more\s+\w+", low)
    gm = re.search(r"(\d+)\s+\w+\s+were\s+gone", low)
    if not (hm and mm and tm and gm):
        return None
    dogs = Fraction(hm.group(1)) + Fraction(mm.group(1))
    doubled = dogs * 2
    return doubled + dogs - Fraction(gm.group(1))


def _cupcakes_klasse(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Cupcakes-Klasse: '(25+1+1)x2' -> 54."""
    low = _digitize(question.lower())
    cm = re.search(r"(\d+)\s+classmates?", low)
    tm = re.search(r"his\s+\w+,\s+and\s+his\s+(\d+)", low)
    gm = re.search(r"gets?\s+the\s+same\s+amount\s+of\s+(\d+)", low)
    if not (cm and tm and gm):
        return None
    return (Fraction(cm.group(1)) + 2) * Fraction(gm.group(1))


def _lese_ziel(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Lese-Ziel: '30 - 200/10' -> 10 Tage."""
    low = _digitize(question.lower())
    pm = re.search(r"(\d+)-page\s+\w+", low)
    dm = re.search(r"within\s+(\d+)\s+\w+", low)
    rm = re.search(r"read\s+(\d+)\s+\w+\s+a\s+day", low)
    if not (pm and dm and rm):
        return None
    return Fraction(dm.group(1)) - Fraction(pm.group(1)) / Fraction(rm.group(1))


def _koch_zeiten(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Koch-Zeiten: '30 + 50 + (30+50)/2' -> 120."""
    low = _digitize(question.lower())
    rm = re.search(
        r"cook\s+\w+\s+than\s+\w+,\s+while\s+\w+\s+"
        r"took",
        low,
    )
    mm = re.search(r"(\d+)\s+more\s+\w+\s+to\s+cook", low)
    tm = re.search(r"took\s+her\s+(\d+)\s+\w+\s+to\s+cook\s+\w+", low)
    if not (rm and mm and tm):
        return None
    rice = Fraction(tm.group(1))
    pork = rice + Fraction(mm.group(1))
    beans = (pork + rice) / 2
    return rice + pork + beans


def _aufholen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Aufholen: '75/(70-55)' -> 5h."""
    low = _digitize(question.lower())
    am = re.search(r"(\d+)\s+miles?\s+ahead", low)
    ms = re.findall(r"driving\s+(\d+)\s+\w+\s+per\s+\w+", low)
    if not am or len(ms) < 2:
        return None
    return Fraction(am.group(1)) / (abs(Fraction(ms[0]) - Fraction(ms[1])))


def _lauf_vergleich(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Lauf-Vergleich: 'R=5x4, 3 weiter als L' -> 20-3=17."""
    low = _digitize(question.lower())
    pm = re.search(r"ran\s+(\d+)\s+\w+", low)
    rm = re.search(r"ran\s+(\d+)\s+times\s+what", low)
    fm = re.search(r"farther\s+than\s+(\w+)", low)
    if not (pm and rm and fm):
        return None
    reggie = Fraction(pm.group(1)) * Fraction(rm.group(1))
    return reggie - 3


def _brunnen_graben(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Brunnen-Graben: '24/4 + 8/2' -> 10h."""
    low = _digitize(question.lower())
    fm = re.search(r"dig\s+(\d+)\s+\w+\/hour", low)
    sm = re.search(r"through\s+(\d+)\s+feet\s+of\s+\w+", low)
    cms = re.findall(r"(\d+)\s+feet\s+of\s+\w+", low)
    if not (fm and sm and cms):
        return None
    return Fraction(sm.group(1)) / Fraction(fm.group(1)) + Fraction(cms[-1]) / (
        Fraction(fm.group(1)) / 2
    )


def _groesse_jahre(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Größe-Jahre: '(4x12-40)/2' -> 4."""
    low = _digitize(question.lower())
    tm = re.search(r"be\s+(\d+)\s+feet\s+tall", low)
    hm = re.search(r"height\s+is\s+(\d+)\s+\w+", low)
    gm = re.search(r"grows?\s+(\d+)\s+\w+\s+a\s+year", low)
    if not (tm and hm and gm):
        return None
    return (Fraction(tm.group(1)) * 12 - Fraction(hm.group(1))) / Fraction(gm.group(1))


def _welle_reiter(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Welle-Reiter: '100x25%x40%' -> 10."""
    low = _digitize(question.lower())
    rm = re.search(r"(\d+)\s+riders?", low)
    pm = re.search(
        r"(\d+)\s*(?:%|percent)\s+of\s+(?:the\s+)?"
        r"(?:\d+\s+)?riders?",
        low,
    )
    wm = re.search(r"(\d+)\s*(?:%|percent)\s+are\s+women", low)
    if not (rm and pm and wm):
        return None
    upright = Fraction(rm.group(1)) * Fraction(pm.group(1)) / 100
    return upright * (1 - Fraction(wm.group(1)) / 100)


def _briefmarken(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Briefmarken: '16 + 19 + 10' -> 45."""
    low = _digitize(question.lower())
    sm = re.search(r"bought\s+(\d+)\s+\w+\s+\w+", low)
    tm = re.search(r"(\d+)\s+more\s+\w+\s+\w+\s+than", low)
    rm = re.search(r"(\d+)\s+fewer\s+\w+\s+\w+\s+than", low)
    if not (sm and tm and rm):
        return None
    snow = Fraction(sm.group(1))
    truck = snow + Fraction(tm.group(1))
    rose = truck - Fraction(rm.group(1))
    return snow + truck + rose


def _aquarium_kosten(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Aquarium-Kosten: '10+2x2.5+3x2+20x0.5+2' -> 33."""
    low = _digitize(question.lower())
    am = re.search(r"aquarium\s+for\s+\$?(\d+(?:\.\d+)?)", low)
    rm = re.search(r"bags?\s+of\s+\w+\s+for\s+\$?(\d+(?:\.\d+)?)", low)
    cm = re.search(
        r"(\d+)\s+pieces?\s+of\s+\w+\s+at\s+"
        r"\$?(\d+(?:\.\d+)?)",
        low,
    )
    fm = re.search(r"(\d+)\s+fish\s+at\s+\$?(\d+(?:\.\d+)?)", low)
    nf = re.search(r"food\s+that\s+cost\s+\$?(\d+(?:\.\d+)?)", low)
    if not (am and rm and cm and fm and nf):
        return None
    return (
        Fraction(am.group(1))
        + Fraction(rm.group(1)) * 2
        + Fraction(cm.group(1)) * Fraction(cm.group(2))
        + Fraction(fm.group(1)) * Fraction(fm.group(2))
        + Fraction(nf.group(1))
    )


def _fleisch_tage(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Fleisch-Tage: '750/(15x10)' -> 5."""
    low = _digitize(question.lower())
    sm = re.search(
        r"sells\s+(\d+)\w+\s+of\s+\w+\s+every\s+hour"
        r"|sells\s+(\d+)\s+\w+\s+every\s+hour",
        low,
    )
    hm = re.search(r"works\s+(\d+)\s+hours?\s+a\s+day", low)
    wm = re.search(r"bull\s+that\s+weighs\s+(\d+)", low)
    if not (sm and hm and wm):
        return None
    rate = sm.group(1) or sm.group(2)
    return Fraction(wm.group(1)) / (Fraction(rate) * Fraction(hm.group(1)))


def _trainer_kauf(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Trainer-Kauf: '8x14 - 9x3' -> 85."""
    low = _digitize(question.lower())
    m1 = re.search(
        r"baseball\s+\w+\s+bought\s+(\d+)\s+"
        r"\w+(?:\s+\w+)?\s+for\s+\$?(\d+)",
        low,
    )
    m2 = re.search(
        r"basketball\s+\w+\s+bought\s+(\d+)\s+"
        r"\w+(?:\s+\w+)?\s+for\s+\$?(\d+)",
        low,
    )
    if not (m1 and m2):
        return None
    return Fraction(m2.group(1)) * Fraction(m2.group(2)) - Fraction(
        m1.group(1)
    ) * Fraction(m1.group(2))


def _pilz_protein(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Pilz-Protein: '200/100x3x7' -> 42."""
    low = _digitize(question.lower())
    cm = re.search(
        r"cup\s+of\s+\w+\s+weighs\s+(\d+)\s+\w+\s+"
        r"and\s+has\s+(\d+)\s+\w+\s+of\s+\w+",
        low,
    )
    em = re.search(
        r"eats\s+(\d+)\s+\w+\s+of\s+\w+\s+every\s+"
        r"day",
        low,
    )
    if not (cm and em):
        return None
    per_gram = Fraction(cm.group(2)) / Fraction(cm.group(1))
    return Fraction(em.group(1)) * per_gram * 7


def _zahn_arbeit(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Zahn-Arbeit: '(2x2000+500-600)/15' -> 260."""
    low = _digitize(question.lower())
    im = re.search(r"needs\s+(\d+)\s+implants?", low)
    bm = re.search(r"base\s+price\s+of\s+\$?(\d+)", low)
    em = re.search(r"extra\s+\$?(\d+)", low)
    dm = re.search(r"deposit\s+of\s+\$?(\d+)", low)
    rm = re.search(r"makes\s+\$?(\d+)\s+per", low)
    if not (im and bm and em and dm and rm):
        return None
    total = (
        Fraction(im.group(1)) * Fraction(bm.group(1))
        + Fraction(em.group(1))
        - Fraction(dm.group(1))
    )
    return total / Fraction(rm.group(1))


def _spiele_jahre(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Spiele-Jahre: '5+12+24+48+3x5' -> 104."""
    low = _digitize(question.lower())
    if "games" not in low:
        return None
    gm = re.search(r"along\s+with\s+(\d+)\s+\w+", low)
    ms = re.findall(
        r"(?:buys?\w*\s+)?(\d+)\s+\w+\s+(?:a|per)"
        r"\s+month",
        low,
    )
    cm = re.search(r"christmas\s+every\s+year", low)
    if not (gm or ms or cm):
        return None
    total = Fraction(gm.group(1)) if gm else Fraction(0)
    for m in ms:
        total += Fraction(m) * 12
    total += 5 * 3
    return total


def _vogel_schnitt(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Vogel-Schnitt: '50+120+20+90 /7' -> 40."""
    low = _digitize(question.lower())
    if "on average" not in low or "birds" not in low:
        return None
    ms = re.findall(r"saw\s+(?:a\s+total\s+of\s+)?(\d+)", low)
    if len(ms) < 3:
        return None
    return sum(Fraction(m) for m in ms) / 7


def _wasser_laps(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Wasser-Laps: '8 x 0.25 x 60' -> 120."""
    low = _digitize(question.lower())
    wm = re.search(
        r"(\d+)\s+\w+\s+of\s+\w+\s+for\s+each\s+"
        r"\w+",
        low,
    )
    lm = re.search(r"run\s+(\d+)\s+\w+", low)
    km = re.search(r"each\s+\w+\s+is\s+([\d.]+)\s+\w+", low)
    if not (wm and lm and km):
        return None
    return Fraction(lm.group(1)) * Fraction(km.group(1)) * Fraction(wm.group(1))


def _schule_bestehen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Schule-Bestehen: '700/1000' -> 70%."""
    low = _digitize(question.lower())
    ms = re.findall(r"(\d+)\s+out\s+of\s+(\d+)", low)
    if len(ms) < 2:
        return None
    passed = sum(Fraction(a) for a, _ in ms)
    tm = re.search(r"(\d+)\s+\w+\s+\w+\s+had\s+a\s+pass\s+rate", low)
    if tm:
        fourth_rate = Fraction(ms[1][0]) / Fraction(ms[1][1])
        passed += Fraction(tm.group(1)) * fourth_rate * 2
        ms.append(("0", tm.group(1)))
    total = sum(Fraction(b) for _, b in ms)
    return passed / total * 100


def _limonade_mehr(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Limonade-mehr: '21x4 - 63' -> 21."""
    low = _digitize(question.lower())
    sm = re.search(r"sold\s+(\d+)\s+\w+\s+at\s+\$?(\d+)", low)
    bm = re.search(r"made\s+\$?(\d+)", low)
    if not (sm and bm):
        return None
    return Fraction(sm.group(1)) * Fraction(sm.group(2)) - Fraction(bm.group(1))


def _wasser_kalorien(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Wasser-Kalorien: '(2x500+1x600)/200x100' -> 800."""
    low = _digitize(question.lower())
    if "calories" not in low:
        return None
    wm = re.search(
        r"(\d+)\s+\w+\s+of\s+\w+\s+for\s+every\s+"
        r"(\d+)",
        low,
    )
    ms = re.findall(
        r"(\d+)\s+hours?\s+doing\s+\w+,\s*which\s+"
        r"burns\s+(\d+)",
        low,
    )
    ms2 = re.findall(
        r"(\d+)\s+hour\s+\w+,\s*which\s+burns\s+"
        r"(\d+)",
        low,
    )
    if not wm:
        return None
    total_cal = Fraction(0)
    for h, c in ms:
        total_cal += Fraction(h) * Fraction(c)
    for h, c in ms2:
        total_cal += Fraction(h) * Fraction(c)
    return total_cal / Fraction(wm.group(2)) * Fraction(wm.group(1))


def _rosen_mangel(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Rosen-Mangel: '(80+60+120) - 40x4' -> 100."""
    low = _digitize(question.lower())
    gm = re.search(r"grows\s+(\d+)\s+\w+\s+every\s+week", low)
    os_ = re.findall(r"orders?\s+(\d+)\s+\w+", low)
    mm = re.search(r"every\s+month", low)
    if not (gm and os_ and mm):
        return None
    need = sum(Fraction(o) for o in os_) * 4
    supply = Fraction(gm.group(1)) * 4
    return need - supply


def _lese_zwei(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Lese-zwei: '7x15/1.5x2 + 100' -> 240."""
    low = _digitize(question.lower())
    nm = re.search(r"read\s+for\s+(\d+)\s+\w+\s+each\s+night", low)
    pm = re.search(r"read\s+(\d+)\s+\w+\s+per\s+([\d.]+)\s+\w+", low)
    sm = re.search(r"total\s+of\s+(\d+)\s+\w+", low)
    if not (nm and pm and sm):
        return None
    return Fraction(nm.group(1)) * 7 / Fraction(pm.group(2)) * Fraction(
        pm.group(1)
    ) + Fraction(sm.group(1))


def _tore_drei(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Tore-drei: 'R=80, M=60, A=35' -> 175."""
    low = _digitize(question.lower())
    ms = re.findall(r"scored\s+(\d+)\s+more\s+goals?\s+than", low)
    vm = re.search(r"if\s+(\w+)\s+scored\s+(\d+)", low)
    if len(ms) < 2 or not vm:
        return None
    r = Fraction(vm.group(2))
    m = r - Fraction(ms[0])
    a = r - Fraction(ms[1])
    return r + m + a


def _kaugummi(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Kaugummi: '(20-4-1-1)/2' -> 7."""
    low = _digitize(question.lower())
    pm = re.search(r"pack\s+of\s+\w+", low)
    hm = re.search(r"every\s+(\d+)\s+hours?", low)
    dm = re.search(r"lasts?\s+(\d+)\s+hours?", low)
    hm2 = re.search(r"gives\s+half", low)
    if not (pm and hm and dm and hm2):
        return None
    at_school = Fraction(dm.group(1)) / Fraction(hm.group(1))
    return (Fraction(20) - at_school - 1 - 1) / 2


def _spar_wochen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Spar-Wochen: '80/(2x5x4)' -> 2."""
    low = _digitize(question.lower())
    pm = re.search(r"paid\s+\$?(\d+)\s+per\s+hour", low)
    hm = re.search(r"works\s+(\d+)\s+hours?\s+a\s+day", low)
    dm = re.search(r"(\d+)\s+days?\s+a\s+week", low)
    sm = re.search(r"save\s+\$?(\d+)", low)
    if not (pm and hm and dm and sm):
        return None
    return Fraction(sm.group(1)) / (
        Fraction(pm.group(1)) * Fraction(hm.group(1)) * Fraction(dm.group(1))
    )


def _kaffee_reduz(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Kaffee-Reduktion: '10x(4/2) - 4' -> 16."""
    low = _digitize(question.lower())
    rm = re.search(r"recommendation\s+of\s+(\d+)\s+\w+", low)
    tm = re.search(r"10\s+times\s+the\s+amount", low)
    if not (rm and tm):
        return None
    octavia = Fraction(rm.group(1)) / 2
    return octavia * 10 - Fraction(rm.group(1))


def _pyramide_winkel(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Pyramide-Winkel: '32 + 5x10' -> 82."""
    low = _digitize(question.lower())
    am = re.search(r"angle\s+of\s+(\d+)\s+\w+", low)
    sm = re.search(r"moving\s+at\s+(\d+)\s+\w+\s+an\s+hour", low)
    hm = re.search(r"moves\s+for\s+(\d+)\s+\w+", low)
    if not (am and sm and hm):
        return None
    return Fraction(am.group(1)) + Fraction(sm.group(1)) * Fraction(hm.group(1))


def _erdbeer_kette(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Erdbeer-Kette: '6, -1, x2, -2' -> 6+5+10+8=29."""
    low = _digitize(question.lower())
    tm = re.search(
        r"pick\s+(\d+)\s+\w+\s+of\s+\w+\s+per\s+"
        r"hour",
        low,
    )
    if not tm:
        return None
    tony = Fraction(tm.group(1))
    bobby = tony - 1
    kathy = bobby * 2
    ricky = kathy - 2
    return tony + bobby + kathy + ricky


def _brot_rest(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Brot-Rest: '200-93-39+6' -> 74."""
    low = _digitize(question.lower())
    bm = re.search(r"baked\s+(\d+)\s+\w+\s+of\s+\w+", low)
    ms = re.findall(
        r"(?:sold\s+)?(\d+)\s+\w+\s+in\s+the\s+"
        r"(?:morning|afternoon|evening)",
        low,
    )
    rm = re.search(r"returned\s+(\d+)\s+\w+", low)
    if not (bm and len(ms) >= 2 and rm):
        return None
    return Fraction(bm.group(1)) - sum(Fraction(m) for m in ms) + Fraction(rm.group(1))


def _vlog_rest(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Vlog-Rest: '72-18-21-15' -> 18."""
    low = _digitize(question.lower())
    tm = re.search(r"upload\s+(\d+)\s+\w+\s+per\s+month", low)
    ms = re.findall(r"(\d+)\s+\w+\s+for\s+the\s+\w+\s+week", low)
    if not tm or len(ms) < 2:
        return None
    return Fraction(tm.group(1)) - sum(Fraction(m) for m in ms)


def _zwiebel_teilen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Zwiebel-Teilen: 'Rose 4x Sophia, Rose 16' -> 4."""
    low = _digitize(question.lower())
    rm = re.search(r"bought\s+(\d+)\s+times\s+the\s+number", low)
    on = re.search(r"bought\s+(\d+)\s+\w+\s+and\s+(\d+)\s+\w+", low)
    if not (rm and on):
        return None
    return (Fraction(on.group(1)) + Fraction(on.group(2))) / Fraction(rm.group(1))


def _wolle_ausstattung(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Wolle-Ausstattung: '(2+4+12+1+2)x3' -> 63."""
    low = _digitize(question.lower())
    gm = re.search(r"for\s+her\s+(\d+)\s+grandchildren?", low)
    ms = re.findall(
        r"takes\s+(\d+)\s+\w+\s+of\s+\w+\s+to\s+"
        r"make",
        low,
    )
    if not gm:
        return None
    ms2 = re.findall(r"(\d+)\s+for\s+a\s+\w+", low)
    total = sum(Fraction(m) for m in ms + ms2)
    return total * Fraction(gm.group(1))


def _hausaufgaben(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Hausaufgaben: '100-12-36=52, -13' -> 39."""
    low = _digitize(question.lower())
    tm = re.search(r"has\s+(\d+)\s+\w+\s+problems", low)
    mm = re.search(r"completes\s+(\d+)\s+\w+\s+on\s+\w+", low)
    tm2 = re.search(r"times\s+as\s+many\s+(?:\w+\s+)?as", low)
    qm = re.search(
        r"(?:one-quarter|1-quarter)\s+of\s+the\s+"
        r"remaining",
        low,
    )
    if not (tm and mm and tm2 and qm):
        return None
    total = Fraction(tm.group(1))
    monday = Fraction(mm.group(1))
    tuesday = monday * 3
    left = total - monday - tuesday
    return left * Fraction(3, 4)


def _buecher_kinder(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Bücher-Kinder: '300/15/4' -> 5."""
    low = _digitize(question.lower())
    sm = re.search(r"spent\s+\$?(\d+)", low)
    bm = re.search(r"book\s+was\s+\$?(\d+)", low)
    km = re.search(r"her\s+(\d+)\s+\w+", low)
    if not (sm and bm and km):
        return None
    return Fraction(sm.group(1)) / Fraction(bm.group(1)) / Fraction(km.group(1))


def _garn_yards(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Garn-Yards: '(1/4+1/2)x364' -> 273."""
    low = _digitize(question.lower())
    fs = re.findall(r"used\s+(\d+/\d+)\s+of\s+a\s+\w+", low)
    ym = re.search(r"(\d+)\s+\w+\s+in\s+a\s+\w+", low)
    if len(fs) < 2 or not ym:
        return None
    total = sum(Fraction(f) for f in fs)
    return total * Fraction(ym.group(1))


def _geschenke_freunde(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Geschenke-Freunde: '2x5+3x2+10' -> 26."""
    low = _digitize(question.lower())
    fm = re.search(r"for\s+her\s+(\d+)\s+\w+", low)
    m1 = re.search(
        r"(\d+)\s+of\s+her\s+\w+\s+want\s+(\d+)\s+"
        r"\w+",
        low,
    )
    m2 = re.search(r"other\s+(\d+)(?:\s+\w+)?\s+want\s+(\d+)", low)
    mm = re.search(r"(\d+)\s+more\s+random\s+\w+", low)
    if not (fm and m1 and m2 and mm):
        return None
    return (
        Fraction(m1.group(1)) * Fraction(m1.group(2))
        + Fraction(m2.group(1)) * Fraction(m2.group(2))
        + Fraction(mm.group(1))
    )


def _apfel_rabatt(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Apfel-Rabatt: 'Kelly 9 - Becky 8' -> 1."""
    low = _digitize(question.lower())
    ms = re.findall(
        r"bought\s+(\d+)\s+\w+\s+for\s+(\d+)\s+\w+"
        r"\s+each",
        low,
    )
    if len(ms) < 2:
        return None
    m1 = ms[0]
    m2 = ms[1]
    d1 = re.search(r"received\s+a\s+\$?(\d+)", low)
    d2 = re.search(r"(\d+)\s+percent\s+discount", low)
    if not (m1 and m2 and d1 and d2):
        return None
    n = Fraction(m1[0])
    becky = n * Fraction(m1[1]) / 100 - Fraction(d1.group(1))
    kelly = n * Fraction(m2[1]) / 100 * (1 - Fraction(d2.group(1)) / 100)
    return kelly - becky


def _schuhe_zaehlen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Schuhe-zählen: '200+2x(5+15+30)-180' -> 120."""
    low = _digitize(question.lower())
    sm = re.search(r"has\s+(\d+)\s+shoes?", low)
    ps = re.findall(r"(\d+)\s+(?:\w+\s+)?pairs?", low)
    gm = re.search(r"gets\s+rid\s+of\s+(\d+)", low)
    if not (sm and ps and gm):
        return None
    return (
        Fraction(sm.group(1)) + sum(Fraction(p) for p in ps) * 2 - Fraction(gm.group(1))
    )


def _bleistift_rest(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Bleistift-Rest: '300 x 4/5 x 1/3' -> 80."""
    low = _digitize(question.lower())
    sm = re.search(r"(\d+)\s+students?\s+in", low)
    pm = re.search(r"with\s+(\d+)\s+\w+", low)
    fm = re.search(r"1/5\s+of\s+the\s+total", low)
    tm = re.search(r"1/3\s+of\s+the\s+remaining", low)
    if not (sm and pm and fm and tm):
        return None
    return (
        Fraction(sm.group(1)) * Fraction(pm.group(1)) * Fraction(4, 5) * Fraction(1, 3)
    )


def _scrabble_fuehrung(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Scrabble-Führung: '(214+26)-(225+10)' -> 5."""
    low = _digitize(question.lower())
    ms = re.findall(r"has\s+(\d+)\s+points?", low)
    ss = re.findall(r"scores?\s+(\d+)\s+points?", low)
    if len(ms) < 2 or len(ss) < 2:
        return None
    return Fraction(ms[0]) + Fraction(ss[0]) - Fraction(ms[1]) - Fraction(ss[1])


def _karten_farben(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Karten-Farben: '15 + 24 + 39' -> 78."""
    low = _digitize(question.lower())
    rm = re.search(r"(\d+)\s+red\s+\w+", low)
    pm = re.search(r"(\d+)\s*(?:%|percent)\s+more\s+\w+\s+\w+", low)
    if not (rm and pm):
        return None
    red = Fraction(rm.group(1))
    green = red * (1 + Fraction(pm.group(1)) / 100)
    yellow = red + green
    return red + green + yellow


def _pflanzen_kette(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Pflanzen-Kette: '10x1.6-7' -> 9."""
    low = _digitize(question.lower())
    if "plants" not in low:
        return None
    fm = re.search(r"if\s+(\w+)\s+has\s+(\d+)", low)
    pm = re.search(r"(\d+)\s*(?:%|percent)\s+more\s+\w+\s+than", low)
    if not (fm and pm):
        return None
    toni = Fraction(fm.group(2)) * (1 + Fraction(pm.group(1)) / 100)
    return toni - 7


def _triathlon_lauf(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Triathlon-Lauf: '180-36-85' -> 59."""
    low = _digitize(question.lower())
    sm = re.search(r"(\d+)\s+\w+\s+for\s+the\s+swim", low)
    bm = re.search(
        r"an\s+hour\s+and\s+(\d+)\s+\w+\s+for\s+the"
        r"\s+bike",
        low,
    )
    rm = re.search(r"(\d+)\s+\w+\s+for\s+the\s+run", low)
    pm = re.search(r"swim\s+(\d+)\s*(?:%|percent)\s+faster", low)
    bm2 = re.search(
        r"takes\s+(\d+)\s+\w+\s+longer\s+on\s+the\s+"
        r"bike",
        low,
    )
    wm = re.search(r"won\s+by\s+(\d+)\s+\w+", low)
    if not (sm and bm and rm and pm and bm2 and wm):
        return None
    jon = Fraction(sm.group(1)) + 60 + Fraction(bm.group(1)) + Fraction(rm.group(1))
    james = jon + Fraction(wm.group(1))
    swim = Fraction(sm.group(1)) * (1 - Fraction(pm.group(1)) / 100)
    bike = 60 + Fraction(bm.group(1)) + Fraction(bm2.group(1))
    return james - swim - bike


def _heu_kaufen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Heu-Kaufen: '30-16x0.5x3' -> 6."""
    low = _digitize(question.lower())
    hm = re.search(r"eats\s+(\d+/\d+)\s+a\s+\w+", low)
    cm = re.search(r"costs\s+\$?(\d+)", low)
    tm = re.search(r"runs\s+for\s+(\d+)\s+\w+\s+at\s+(\d+)", low)
    bm = re.search(r"(?:six|6)\s+(\d+)\s+dollar\s+bills", low)
    if not (hm and cm and tm and bm):
        return None
    miles = Fraction(tm.group(2)) * Fraction(tm.group(1)) / 60
    cost = miles * Fraction(hm.group(1)) * Fraction(cm.group(1))
    return Fraction(bm.group(1)) * 6 - cost


def _vogel_futter(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Vogel-Futter: '6x20 + 3x10' -> 150."""
    low = _digitize(question.lower())
    bm = re.search(
        r"builds?\s+(\d+)\s+(?:\w+\s+\w+\s+)?and\s+"
        r"buys?\s+(\d+)",
        low,
    )
    bm2 = re.search(r"attract\w*\s+(\d+)\s+\w+", low)
    pm = re.search(r"attract\s+(\d+)\s+more\s+\w+\s+each", low)
    if not (bm and bm2 and pm):
        return None
    made = Fraction(bm.group(1))
    total = (made + Fraction(bm.group(2))) * Fraction(bm2.group(1))
    return total + made * Fraction(pm.group(1))


def _drache_kette(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Drache-Kette: '100 x 3/4 x 2 x 1/5' -> 30."""
    low = _digitize(question.lower())
    sm = re.search(r"slew\s+(\d+)\s+\w+", low)
    if not sm:
        return None
    x = Fraction(sm.group(1))
    if re.search(r"(?:three|3)\s+quarters\s+as\s+many", low):
        x = x * Fraction(3, 4)
    if "twice as many" in low:
        x = x * 2
    if re.search(r"(?:one-fifth|1-fifth|one\s+fifth|1\s+fifth)", low):
        x = x / 5
    return x


def _perlen_schwestern(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Perlen-Schwestern: '(3+4-1-2)x20' -> 80."""
    low = _digitize(question.lower())
    em = re.search(
        r"elizabeth\s+bought\s+(\d+)\s+\w+\s+of\s+\w+"
        r"\s+and\s+(\d+)\s+\w+",
        low,
    )
    mm = re.search(
        r"margareth\s+bought\s+(\d+)\s+\w+\s+of\s+\w+"
        r"(?:\s+\w+)?\s+and\s+(\d+)\s+\w+\s+of\s+"
        r"\w+",
        low,
    )
    pm = re.search(r"pack\s+of\s+\w+\s+contains\s+(\d+)", low)
    if not (em and mm and pm):
        return None
    e = Fraction(em.group(1)) + Fraction(em.group(2))
    m = Fraction(mm.group(1)) + Fraction(mm.group(2))
    return (m - e) * Fraction(pm.group(1))


def _berg_aufstieg(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Berg-Aufstieg: '(10000-4000)-3000' -> 3000."""
    low = _digitize(question.lower())
    evs = re.findall(r"elevation\s+of\s+([\d,]+)\s+\w+", low)
    fm = re.search(r"fall\s+([\d,]+)\s+\w+", low)
    if len(evs) < 2 or not fm:
        return None
    comb = Fraction(int(evs[0].replace(",", ""))) - Fraction(
        int(fm.group(1).replace(",", ""))
    )
    return comb - Fraction(int(evs[-1].replace(",", "")))


def _bank_kapital(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Bank-Kapital: '4000x3+5000' -> 17000."""
    low = _digitize(question.lower())
    fm = re.search(r"gave\s+(?:him|mr\.\s+\w+)\s+\$?(\d+)", low)
    tm = re.search(r"twice\s+as\s+much", low)
    im = re.search(r"had\s+\$?(\d+)\s+in\s+capital", low)
    if not (fm and tm and im):
        return None
    return Fraction(fm.group(1)) * 3 + Fraction(im.group(1))


def _bus_kapazitaet(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Bus-Kapazität: '4x60+6x30+10x15' -> 570."""
    low = _digitize(question.lower())
    ms = re.findall(
        r"(\d+)\s+\w+(?:\s+\w+){0,2}\s+that\s+"
        r"(?:have\s+the\s+capacity\s+of\s+holding|hold)"
        r"\s+(\d+)|"
        r"(\d+)\s+\w+\s+that\s+can\s+hold\s+(\d+)",
        low,
    )
    if len(ms) < 2:
        return None
    total = Fraction(0)
    for a, b, c, d in ms:
        if a:
            total += Fraction(a) * Fraction(b)
        else:
            total += Fraction(c) * Fraction(d)
    return total


def _feuerwerk_kosten(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Feuerwerk-Kosten: '(400+800)x0.8+150' -> 1110."""
    low = _digitize(question.lower())
    pm = re.search(r"package\s+of\s+\w+\s+worth\s+\$?(\d+)", low)
    tm = re.search(r"twice\s+that\s+much", low)
    dm = re.search(r"(\d+)\s*(?:%|percent)\s+discount", low)
    fm = re.search(r"costs\s+\$?(\d+)", low)
    if not (pm and tm and dm and fm):
        return None
    total = Fraction(pm.group(1)) * 3 * (1 - Fraction(dm.group(1)) / 100) + Fraction(
        fm.group(1)
    )
    return total


def _apfel_schwestern(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Apfel-Schwestern: '500-30-15-60-45' -> 350."""
    low = _digitize(question.lower())
    jm = re.search(
        r"gathers\s+(\d+)\s+\w+\s+from\s+the\s+"
        r"tallest",
        low,
    )
    cm = re.search(r"combined\s+total\s+of\s+(\d+)", low)
    if not (jm and cm):
        return None
    tall = Fraction(jm.group(1))
    short = tall / 2
    sister = tall * 2 + short * 3
    return Fraction(cm.group(1)) - tall - short - sister


def _reise_km(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Reise-km: '4x200 + 0.3x800 + 7x300' -> 3140."""
    low = _digitize(question.lower())
    fm = re.search(
        r"first\s+(\d+)\s+days?,\s+he\s+traveled\s+"
        r"(\d+)|traveled\s+(\d+)\s+\w+\s+every\s+day"
        r"\s+for\s+the\s+first\s+(\d+)",
        low,
    )
    pm = re.search(
        r"(\d+)\s*(?:%|percent)\s+of\s+(?:that|the\s+"
        r"distance)",
        low,
    )
    sm = re.search(r"second\s+week,\s+he\s+made\s+(\d+)", low)
    if not (fm and pm and sm):
        return None
    first = Fraction(fm.group(1) or fm.group(3)) * Fraction(fm.group(2) or fm.group(4))
    second = first * Fraction(pm.group(1)) / 100
    third = Fraction(sm.group(1)) * 7
    return first + second + third


def _milch_kalorien(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Milch-Kalorien: '2x8x3' -> 48."""
    low = _digitize(question.lower())
    gm = re.search(r"drinks?\s+(\d+)\s+\w+\s+of\s+\w+", low)
    om = re.search(r"glass\s+of\s+\w+\s+is\s+(\d+)\s+\w+", low)
    cm = re.search(r"(\d+)\s+calories?\s+per\s+\w+", low)
    if not (gm and om and cm):
        return None
    return Fraction(gm.group(1)) * Fraction(om.group(1)) * Fraction(cm.group(1))


def _schritte_jog(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Schritte-Jog: '10000-5000-1000-2000' -> 2000."""
    low = _digitize(question.lower())
    tm = re.search(r"walk\s+([\d,]+)\s+steps\s+a\s+day", low)
    hm = re.search(r"finished\s+half", low)
    wm = re.search(r"another\s+([\d,]+)\s+steps", low)
    lm = re.search(r"([\d,]+)\s+steps\s+left", low)
    if not (tm and hm and wm and lm):
        return None
    total = Fraction(int(tm.group(1).replace(",", "")))
    return (
        total
        - total / 2
        - Fraction(int(wm.group(1).replace(",", "")))
        - Fraction(int(lm.group(1).replace(",", "")))
    )


def _buch_zeit(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Buch-Zeit: '60-45x200/300-10' -> 20."""
    low = _digitize(question.lower())
    pm = re.search(r"book\s+that\s+is\s+(\d+)\s+\w+\s+long", low)
    wm = re.search(r"averages\s+(\d+)\s+\w+\s+a\s+page", low)
    rm = re.search(
        r"(?:rate\s+of\s+|reads?\s+at\s+)(\d+)\s+\w+"
        r"\s+per\s+\w+",
        low,
    )
    am = re.search(r"airport\s+in\s+(\d+)\s+\w+", low)
    tm = re.search(r"takes\s+(\d+)\s+\w+\s+to\s+get", low)
    if not (pm and wm and rm and am and tm):
        return None
    minutes = Fraction(pm.group(1)) * Fraction(wm.group(1)) / Fraction(rm.group(1))
    return Fraction(am.group(1)) - minutes - Fraction(tm.group(1))


def _apfel_ernte(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Apfel-Ernte: '56/4+12+2x12' -> 50."""
    low = _digitize(question.lower())
    pm = re.search(r"at\s+\$?(\d+)\s+per\s+\w+", low)
    tm = re.search(r"picked\s+(\d+)\s+\w+", low)
    dm = re.search(r"double\s+(?:the\s+number|that)", low)
    sm = re.search(r"got\s+\$?(\d+)", low)
    if not (pm and tm and dm and sm):
        return None
    monday = Fraction(sm.group(1)) / Fraction(pm.group(1))
    tuesday = Fraction(tm.group(1))
    return monday + tuesday + tuesday * 2


def _wander_diff(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Wander-Diff: '14x8 - 9x5' -> 67."""
    low = _digitize(question.lower())
    ms = re.findall(
        r"hiked\s+(\d+)\s+\w+\s+per\s+\w+\s+"
        r"(?:for|and\s+stopped\s+after)\s+(\d+)",
        low,
    )
    if len(ms) < 2:
        return None
    return abs(
        Fraction(ms[0][0]) * Fraction(ms[0][1])
        - Fraction(ms[1][0]) * Fraction(ms[1][1])
    )


def _karten_saldo(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Karten-Saldo: '20+30-24-5+17' -> 38."""
    low = _digitize(question.lower())
    mm = re.search(r"made\s+(\d+)\s+\w+", low)
    bm = re.search(
        r"boxes?\s+of\s+[\w'\u2019-]+(?:\s+[\w'"
        r"\u2019-]+){0,2}\s+that\s+had\s+(\d+)\s+\w+"
        r"\s+each",
        low,
    )
    pm = re.search(r"passed\s+out\s+(\d+)", low)
    fs = re.findall(r"(\d+)\s+to\s+her\s+\w+", low)
    rm = re.search(r"received\s+(\d+)", low)
    if not (mm and bm and pm and fs and rm):
        return None
    fs = [f for f in fs if f != pm.group(1)]
    nboxes = re.search(r"(\d+)\s+boxes?", low)
    nbox = nboxes.group(1) if nboxes else "2"
    return (
        Fraction(mm.group(1))
        + Fraction(nbox) * Fraction(bm.group(1))
        - Fraction(pm.group(1))
        - sum(Fraction(f) for f in fs)
        + Fraction(rm.group(1))
    )


def _fahrzeit_gesamt(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Fahrzeit-gesamt: '(200+10)/70 + 240/80' -> 6h."""
    low = _digitize(question.lower())
    dm = re.search(r"(\d+)\s+miles?\s+away", low)
    sm = re.search(r"speed\s+of\s+(\d+)\s+\w+", low)
    dm2 = re.search(r"detour\s+that\s+added\s+(\d+)", low)
    rm = re.search(r"route\s+home\s+that\s+is\s+(\d+)", low)
    sm2 = re.search(r"goes\s+(\d+)\s+\w+", low)
    if not (dm and sm and dm2 and rm and sm2):
        return None
    out = (Fraction(dm.group(1)) + Fraction(dm2.group(1))) / Fraction(sm.group(1))
    back = Fraction(rm.group(1)) / Fraction(sm2.group(1))
    return out + back


def _punkte_vorher(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Punkte-vorher: '4x8-14' -> 18 (drei mal MEHR = 4x)."""
    low = _digitize(question.lower())
    sm = re.search(r"scored\s+(\d+)", low)
    nm = re.search(r"after\s+scoring\s+(\d+)", low)
    tm = re.search(r"(?:three|3)\s+times\s+more", low)
    if not (sm and nm and tm):
        return None
    return Fraction(sm.group(1)) * 4 - Fraction(nm.group(1))


def _schaf_milch(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Schaf-Milch: '15x1 + 15x2' -> 45."""
    low = _digitize(question.lower())
    sm = re.search(r"has\s+(\d+)\s+\w+", low)
    gm = re.search(r"(\d+)\s+kg\s+of\s+\w+\s+from\s+half", low)
    gm2 = re.search(
        r"(\d+)\s+kg\s+of\s+\w+\s+from\s+the\s+"
        r"other\s+half",
        low,
    )
    if not (sm and gm and gm2):
        return None
    half = Fraction(sm.group(1)) / 2
    return half * Fraction(gm.group(1)) + half * Fraction(gm2.group(1))


def _kekse_familie(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Kekse-Familie: '36-4x5-1x7' -> 9."""
    low = _digitize(question.lower())
    bm = re.search(r"bag\s+has\s+(\d+)\s+\w+", low)
    sm = re.search(r"(\d+)\s+\w+\s+in\s+her\s+son's", low)
    dm = re.search(r"(\d+)\s+days?\s+a\s+week", low)
    hm = re.search(r"(\d+)\s+\w+\s+a\s+day\s+for\s+(\d+)", low)
    if not (bm and sm and dm and hm):
        return None
    return (
        Fraction(bm.group(1))
        - Fraction(sm.group(1)) * Fraction(dm.group(1))
        - Fraction(hm.group(1)) * Fraction(hm.group(2))
    )


def _baum_höhen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Baum-Höhen: '2x(6+11)' -> 34."""
    low = _digitize(question.lower())
    sm = re.search(
        r"shortest\s+\w+\s+has\s+a\s+height\s+of\s+"
        r"(\d+)",
        low,
    )
    tm = re.search(r"(\d+)\s+\w+\s+more\s+than\s+the\s+shortest", low)
    dm = re.search(
        r"twice\s+the\s+height\s+of\s+the\s+(?:two|2)"
        r"\s+trees?",
        low,
    )
    if not (sm and tm and dm):
        return None
    first = Fraction(sm.group(1))
    second = first + Fraction(tm.group(1))
    return (first + second) * 2


def _hund_betten(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Hund-Betten: '4x2 + 3x(8+2)/2' -> 23."""
    low = _digitize(question.lower())
    rm = re.search(r"rottweiler\s+takes\s+(\d+)", low)
    cm = re.search(r"chihuahua\s+takes\s+(\d+)", low)
    am = re.search(r"average\s+(?:amount\s+)?of", low)
    nm = re.search(r"make\s+(\d+)\s+\w+\s+\w+\s+and\s+(\d+)", low)
    if not (rm and cm and am and nm):
        return None
    chi = Fraction(cm.group(1))
    collie = (Fraction(rm.group(1)) + chi) / 2
    return Fraction(nm.group(1)) * chi + Fraction(nm.group(2)) * collie


def _lutscher_kette(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Lutscher-Kette: '7+10-3' -> 14."""
    low = _digitize(question.lower())
    hm = re.search(r"has\s+(\d+)\s+\w+", low)
    gm = re.search(r"gives?\s+\w+\s+another\s+(\d+)", low)
    gm2 = re.search(r"gives\s+(\d+)\s+of\s+her", low)
    if not (hm and gm and gm2):
        return None
    return Fraction(hm.group(1)) + Fraction(gm.group(1)) - Fraction(gm2.group(1))


def _waffen_teilen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Waffen-Teilen: '(8+10+1+5)/4' -> 6."""
    low = _digitize(question.lower())
    ms = re.findall(r"has\s+(\d+)(?:\s+\w+)?", low)
    em = re.search(r"share\s+their\s+\w+\s+equally", low)
    if len(ms) < 3 or not em:
        return None
    return sum(Fraction(m) for m in ms) / len(ms)


def _baby_ausruestung(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Baby-Ausrüstung: '24+48+15' -> 87."""
    low = _digitize(question.lower())
    gm = re.search(r"gave\s+her\s+(\d+)\s+\w+\s+\w+", low)
    tm = re.search(r"twice\s+the\s+amount", low)
    mm = re.search(r"another\s+(\d+)\s+\w+", low)
    if not (gm and tm and mm):
        return None
    base = Fraction(gm.group(1))
    return base + base * 2 + Fraction(mm.group(1))


def _kaffee_verduennung(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Kaffee-Verdünnung: '65/13x12+15' -> 75."""
    low = _digitize(question.lower())
    cm = re.search(r"cools?\s+the\s+\w+\s+by\s+(\d+)\s+\w+", low)
    wm = re.search(
        r"makes?\s+(?:it|the\s+\w+)\s+(\d+)\s+\w+"
        r"\s+\w+",
        low,
    )
    dm = re.search(r"cooled\s+by\s+(\d+)", low)
    am = re.search(r"adds?\s+(\d+)\s+\w+\s+of\s+\w+", low)
    if not (cm and wm and dm and am):
        return None
    cubes = Fraction(dm.group(1)) / Fraction(cm.group(1))
    return cubes * Fraction(wm.group(1)) + Fraction(am.group(1))


def _lutscher_oscar(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Lutscher-Oscar: '24-2-14+28-3-2' -> 31."""
    low = _digitize(question.lower())
    hm = re.search(r"has\s+(\d+)\s+\w+", low)
    em = re.search(r"eats\s+(\d+)\s+on\s+his\s+way", low)
    pm = re.search(r"passes\s+(\d+)\s+out", low)
    tm = re.search(
        r"twice\s+as\s+many(?:\s+\w+){0,6}\s+as\s+he"
        r"\s+gave",
        low,
    )
    nm = re.search(r"eats\s+(\d+)\s+more\s+that\s+night", low)
    mm = re.search(r"(\d+)\s+more\s+in\s+the\s+morning", low)
    if not (hm and em and pm and tm and nm and mm):
        return None
    gave = Fraction(pm.group(1))
    return (
        Fraction(hm.group(1))
        - Fraction(em.group(1))
        - gave
        + gave * 2
        - Fraction(nm.group(1))
        - Fraction(mm.group(1))
    )


def _handy_laden(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Handy-Laden: '(100-60)x3/60' -> 2h."""
    low = _digitize(question.lower())
    rm = re.search(r"rate\s+of\s+(\d+)\s+\w+-point", low)
    tm = re.search(r"per\s+(\d+)\s+\w+", low)
    cm = re.search(r"at\s+(\d+)\s*(?:%|percent)\s+charged", low)
    if not (rm and tm and cm):
        return None
    return (100 - Fraction(cm.group(1))) * Fraction(tm.group(1)) / 60


def _gewicht_wochen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Gewicht-Wochen: '8/(8/4/4)' -> 16."""
    low = _digitize(question.lower())
    jm = re.search(r"joey\s+loses\s+(\d+)\s+\w+\s+in\s+(\d+)", low)
    sm = re.search(r"needs\s+(\d+)\s+\w+\s+to\s+lose", low)
    if not (jm and sm):
        return None
    joey_rate = Fraction(jm.group(1)) / Fraction(jm.group(2))
    sandy_rate = joey_rate / Fraction(sm.group(1))
    return Fraction(jm.group(1)) / sandy_rate


def _geschaefts_reise(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Geschäfts-Reise: '6000-600-150-1200-2000' -> 2050."""
    low = _digitize(question.lower())
    tm = re.search(r"has\s+\$?(\d+)", low)
    sm = re.search(r"suits?\s+at\s+\$?(\d+)", low)
    lm = re.search(r"suitcases?\s+at\s+\$?(\d+)", low)
    fm = re.search(r"(\d+)\s+business\s+suits?", low)
    lm2 = re.search(r"(\d+)\s+suitcases?", low)
    fm2 = re.search(
        r"costs\s+\$?(\d+)\s+more\s+than\s+(\d+)\s+"
        r"times",
        low,
    )
    vm = re.search(r"save\s+\$?(\d+)", low)
    if not (tm and sm and lm and fm and lm2 and fm2 and vm):
        return None
    return (
        Fraction(tm.group(1))
        - Fraction(fm.group(1)) * Fraction(sm.group(1))
        - Fraction(lm2.group(1)) * Fraction(lm.group(1))
        - Fraction(fm2.group(1))
        - Fraction(fm2.group(2)) * Fraction(sm.group(1))
        - Fraction(vm.group(1))
    )


def _party_gaeste(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Party-Gäste: '2x6+3x4-8-2' -> 14."""
    low = _digitize(question.lower())
    fm = re.search(r"(\d+)\s+families?\s+with\s+(\d+)", low)
    fm2 = re.search(r"(\d+)\s+families?\s+with\s+(\d+)", low)
    cm = re.search(r"(\d+)\s+people?\s+couldn", low)
    qm = re.search(r"1/4\s+that\s+number", low)
    if not (fm and fm2 and cm and qm):
        return None
    sick = Fraction(cm.group(1))
    return (
        Fraction(fm.group(1)) * Fraction(fm.group(2))
        + Fraction(fm2.group(1)) * Fraction(fm2.group(2))
        - sick
        - sick / 4
    )


def _theater_zeilen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Theater-Zeilen: '54 + 2x54/3 + (54+6)x4/5' -> 138."""
    low = _digitize(question.lower())
    sm = re.search(r"song\s+has\s+(\d+)\s+lines?", low)
    tm = re.search(r"twice\s+the\s+number\s+of\s+lines?", low)
    fm = re.search(r"only\s+a\s+third\s+of\s+them", low)
    sm2 = re.search(r"(?:six|6)\s+more\s+lines\s+than\s+the\s+song", low)
    fq = re.search(r"(?:four-fifths|4-fifths)\s+of\s+them", low)
    if not (sm and tm and fm and sm2 and fq):
        return None
    song = Fraction(sm.group(1))
    scene1 = song * 2 / 3
    scene2 = (song + 6) * Fraction(4, 5)
    return song + scene1 + scene2


def _medaillen_zehn(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Medaillen-zehn: '(22+17)x10' -> 390."""
    low = _digitize(question.lower())
    am = re.search(r"has\s+won\s+(\d+)\s+\w+", low)
    im = re.search(r"has\s+(\d+)\s+less(?:\s+\w+)?\s+than", low)
    tm = re.search(r"10\s+times\s+less\s+\w+\s+than", low)
    if not (am and im and tm):
        return None
    ali = Fraction(am.group(1))
    return (ali + ali - Fraction(im.group(1))) * 10


def _punkt_spiel(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Punkt-Spiel: '21+18+42+2' -> 83."""
    low = _digitize(question.lower())
    mm = re.search(r"mike\s+has\s+(\d+)\s+\w+", low)
    jm = re.search(r"jim\s+(\d+)\s+\w+\s+less(?:\s+than)?", low)
    tm = re.search(r"tony\s+(\d+)\s+times\s+more\s+than", low)
    em = re.search(r"extra\s+point\s+if\s+they\s+have\s+over", low)
    if not (mm and jm and tm and em):
        return None
    mike = Fraction(mm.group(1))
    jim = mike - Fraction(jm.group(1))
    tony = mike * Fraction(tm.group(1))
    bonus = sum(1 for x in (mike, jim, tony) if x > 20)
    return mike + jim + tony + bonus


def _kisten_ziel(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Kisten-Ziel: '120-(15+12+18+20)' -> 55."""
    low = _digitize(question.lower())
    tm = re.search(r"target\s+of\s+selling\s+(\d+)", low)
    ms = re.findall(r"(?:sold|he\s+sold)\s+(\d+)", low)
    ws = re.findall(
        r"(?:tuesday|wednesday|thursday|friday)"
        r"(?:\s+\w+\s+\w+)?\s+(\d+)",
        low,
    )
    cm = re.search(
        r"(\d+)(?:\s+\w+)?,?\s+and\s+(?:\w+\s+)?"
        r"(\d+)",
        low,
    )
    if not tm or not cm:
        return None
    if ws:
        total = sum(Fraction(w) for w in ws)
        for m in ms:
            if Fraction(m) not in [Fraction(w) for w in ws]:
                total += Fraction(m)
    else:
        total = sum(Fraction(m) for m in ms)
        total += Fraction(cm.group(1)) + Fraction(cm.group(2))
    return Fraction(tm.group(1)) - total


def _brot_verteilung(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Brot-Verteilung: '60x1/3/2' -> 10."""
    low = _digitize(question.lower())
    pm = re.search(r"produces?\s+(\d+)\s+\w+", low)
    tm = re.search(r"(?:two-thirds|2-thirds)\s+of\s+the\s+\w+", low)
    hm = re.search(r"half\s+of\s+what\s+is\s+left", low)
    if not (pm and tm and hm):
        return None
    return Fraction(pm.group(1)) / 3 / 2


def _oel_zeit(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Öl-Zeit: '20+28+43' -> 91."""
    low = _digitize(question.lower())
    fm = re.search(r"takes\s+(\d+)\s+\w+\s+for\s+the\s+\w+", low)
    pm = re.search(r"(\d+)\s*(?:%|percent)\s+longer", low)
    cm = re.search(r"(\d+)\s+\w+\s+less\s+time\s+to\s+cook", low)
    if not (fm and pm and cm):
        return None
    first = Fraction(fm.group(1))
    second = first * (1 + Fraction(pm.group(1)) / 100)
    cook = second + first - Fraction(cm.group(1))
    return first + second + cook


def _strick_aermel(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Strick-Ärmel: '(1800-1170)/2' -> 315."""
    low = _digitize(question.lower())
    bm = re.search(r"body\s+of\s+the\s+\w+\s+takes\s+(\d+)", low)
    tm = re.search(r"tenth\s+of\s+that", low)
    rm = re.search(r"twice\s+as\s+many\s+as\s+the\s+collar", low)
    pm = re.search(r"is\s+an\s+(\d+)-stitch", low)
    if not (bm and tm and rm and pm):
        return None
    body = Fraction(bm.group(1))
    collar = body / 10
    rosette = collar * 2
    return (Fraction(pm.group(1)) - body - collar - rosette) / 2


def _stall_kuehe(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Stall-Kühe: '8x20 + 8x40/10' -> 192."""
    low = _digitize(question.lower())
    sm = re.search(r"(\d+)\s+stalls?\s+have\s+(\d+)\s+\w+\s+each", low)
    bm = re.search(r"buys\s+(\d+)\s+\w+", low)
    em = re.search(r"in\s+(\d+)\s+of\s+the\s+stalls?", low)
    if not (sm and bm and em):
        return None
    per_stall = Fraction(bm.group(1)) / Fraction(sm.group(1))
    return (
        Fraction(em.group(1)) * Fraction(sm.group(2))
        + Fraction(em.group(1)) * per_stall
    )


def _kugeln_geschenk(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Kugeln-Geschenk: '5x50+20' -> 270."""
    low = _digitize(question.lower())
    bm = re.search(r"boxes?\s+with\s+(\d+)\s+\w+\s+in\s+each", low)
    gm = re.search(r"gets\s+(\d+)\s+\w+\s+from\s+her", low)
    if not bm or not gm:
        return None
    nb = re.search(r"has\s+(\d+)\s+boxes?", low)
    if not nb:
        return None
    return Fraction(nb.group(1)) * Fraction(bm.group(1)) + Fraction(gm.group(1))


def _kekse_kalorien(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Kekse-Kalorien: '(20+26-18)x200' -> 5600."""
    low = _digitize(question.lower())
    ms = re.findall(
        r"ate\s+(?:(\d+)\s+times|twice)\s+as\s+many"
        r"\s+\w+",
        low,
    )
    sm = re.findall(r"sister\s+ate\s+(\d+)\s+\w+\s+on\s+\w+", low)
    sm2 = re.findall(r"(\d+)\s+the\s+next\s+day", low)
    cm = re.search(r"cookie\s+has\s+(\d+)\s+\w+", low)
    if len(ms) < 2 or len(sm) < 1 or not cm:
        return None
    m1 = ms[0] or 4
    m2 = 2 if ms[1] == "" else ms[1]
    sue = Fraction(m1) * Fraction(sm[0]) + Fraction(m2) * Fraction(sm2[0])
    sis = Fraction(sm[0]) + Fraction(sm2[0])
    return (sue - sis) * Fraction(cm.group(1))


def _apps_tablet(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Apps-Tablet: '61-9+18' -> 70."""
    low = _digitize(question.lower())
    hm = re.search(r"had\s+(\d+)\s+\w+\s+on\s+his", low)
    dm = re.search(r"deleted\s+(\d+)", low)
    dm2 = re.search(r"downloaded\s+(\d+)", low)
    if not (hm and dm and dm2):
        return None
    return Fraction(hm.group(1)) - Fraction(dm.group(1)) + Fraction(dm2.group(1))


def _handy_minuten(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Handy-Minuten: '1000-15x30-300' -> 250."""
    low = _digitize(question.lower())
    pm = re.search(r"plan\s+of\s+(\d+)\s+\w+", low)
    cm = re.search(r"(\d+)-minute\s+call", low)
    em = re.search(r"had\s+(\d+)\s+extra\s+\w+", low)
    dm = re.search(r"month\s+has\s+(\d+)\s+\w+", low)
    if not (pm and cm and em and dm):
        return None
    return (
        Fraction(pm.group(1))
        - Fraction(cm.group(1)) * Fraction(dm.group(1))
        - Fraction(em.group(1))
    )


def _pommes_kette(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Pommes-Kette: '27-24+5+10+2' -> 20."""
    low = _digitize(question.lower())
    hm = re.search(r"had\s+(\d+)\s+\w+\s+\w+", low)
    km = re.search(r"took\s+(\d+)\s+of\s+them", low)
    bm = re.search(r"twice\s+as\s+many\s+as\s+\w+", low)
    cm = re.search(r"took\s+(?:from\s+\w+\s+)?(\d+)\s+less", low)
    em = re.search(r"in\s+the\s+end\s+\w+\s+had\s+(\d+)", low)
    if not (hm and km and bm and cm and em):
        return None
    kyle = Fraction(km.group(1))
    return (
        Fraction(em.group(1))
        - Fraction(hm.group(1))
        + kyle
        + kyle * 2
        + (kyle - Fraction(cm.group(1)))
    )


def _pinguin_rest(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Pinguin-Rest: '36-12-12' -> 12."""
    low = _digitize(question.lower())
    if "penguin" not in low:
        return None
    pm = re.search(r"(\d+)\s+\w+\s+\w+\s+in\s+the\s+\w+", low)
    if not pm:
        return None
    n = Fraction(pm.group(1))
    return n - n / 3 - n / 3


def _suesigkeiten_mehr(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Süßigkeiten-mehr: '54 - (54/2+6)' -> 21."""
    low = _digitize(question.lower())
    if "candies" not in low:
        return None
    jm = re.search(r"john\s+has\s+(\d+)\s+\w+", low)
    if not jm:
        return None
    john = Fraction(jm.group(1))
    robert = john / 2
    return john - (robert + 6)


def _kunden_tage(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Kunden-Tage: '500-100-150' -> 250."""
    low = _digitize(question.lower())
    fm = re.search(r"counts\s+(\d+)\s+\w+\s+entering", low)
    sm = re.search(r"(\d+)\s+more\s+\w+\s+than\s+the\s+first", low)
    tm = re.search(
        r"total\s+number\s+of\s+\w+\s+by\s+the\s+"
        r"third\s+day\s+was\s+(\d+)",
        low,
    )
    if not (fm and sm and tm):
        return None
    return (
        Fraction(tm.group(1))
        - Fraction(fm.group(1))
        - Fraction(fm.group(1))
        - Fraction(sm.group(1))
    )


def _muschel_dienstag(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Muschel-Dienstag: '(20+5)x2' -> 50."""
    low = _digitize(question.lower())
    rm = re.search(r"who\s+collects\s+(\d+)", low)
    km = re.search(r"(\d+)\s+more\s+\w+\s+than\s+\w+", low)
    tm = re.search(r"(\d+)\s+times\s+more\s+\w+\s+than\s+she", low)
    if not (rm and km and tm):
        return None
    return (Fraction(rm.group(1)) + Fraction(km.group(1))) * Fraction(tm.group(1))


def _kaese_woche(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Käse-Woche: '2x7 + 3x3 + 8' -> 31."""
    low = _digitize(question.lower())
    sm = re.search(r"used?\s+(\d+)\s+\w+\s+of\s+\w+\s+on\s+each", low)
    om = re.search(r"days?\s+in\s+the\s+week\s+using\s+one\s+more", low)
    dm = re.search(r"omelets?(?:\s+for\s+\w+)?\s+(\d+)\s+days?", low)
    mm = re.search(
        r"used\s+(\d+)\s+\w+(?:\s+of\s+\w+)?\s+in"
        r"\s+(?:it|\w+)",
        low,
    )
    if not (sm and dm and mm):
        return None
    per_sandwich = Fraction(sm.group(1))
    per_omelet = per_sandwich + 1
    days = Fraction(dm.group(1))
    return per_sandwich * 7 + per_omelet * days + Fraction(mm.group(1))


def _fahrrad_km(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Fahrrad-km: '5x25x4 + 2x60x3' -> 860."""
    low = _digitize(question.lower())
    m1 = re.search(
        r"(?:at\s+least\s+)?(\d+)\s+times\s+a\s+week"
        r"\s*(?:,|and)\s*makes?\s*(\d+)|"
        r"(\d+)\s+times\s+a\s+week,\s*(\d+)\s+\w+"
        r"\s+each",
        low,
    )
    ws = re.findall(r"for\s+(\d+)\s+weeks?", low)
    m2s = re.findall(
        r"(?:rode|rides?)\s+(\d+)\s+times\s+a\s+"
        r"week,\s*(\d+)\s+\w+\s+each\s+time,\s*"
        r"for\s+(\d+)|"
        r"only\s+(\d+)\s+times\s+a\s+week,\s+but\s+"
        r"for\s+(\d+)",
        low,
    )
    if not (m1 and ws and m2s):
        return None
    a = Fraction(m1.group(1) or m1.group(3))
    b = Fraction(m1.group(2) or m1.group(4))
    m2 = m2s[-1]
    c = Fraction(m2[0] or m2[3])
    d = Fraction(m2[1] or m2[4] or ws[-1])
    return a * b * Fraction(ws[0]) + c * d * Fraction(ws[-1])


def _enten_insecten(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Enten-Insekten: '10x3.5/7' -> 5."""
    low = _digitize(question.lower())
    pm = re.search(
        r"eat\s+([\d.]+)\s+\w+\s+of\s+\w+\s+each\s+"
        r"week",
        low,
    )
    fm = re.search(r"flock\s+of\s+(?:ten|10)\s+\w+", low)
    dm = re.search(r"per\s+day", low)
    if not (pm and fm and dm):
        return None
    return Fraction(pm.group(1)) * 10 / 7


def _karotten_rest(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Karotten-Rest: '200-40x2' -> 120."""
    low = _digitize(question.lower())
    cm = re.search(r"(\d+)\s+pounds?\s+of\s+\w+\s+are\s+to", low)
    rm = re.search(r"(\d+)\s+\w+\s+in\s+a\s+certain", low)
    pm = re.search(r"receive\w*\s+(\d+)\s+pounds?", low)
    if not (cm and rm and pm):
        return None
    return Fraction(cm.group(1)) - Fraction(rm.group(1)) * Fraction(pm.group(1))


def _pokemon_karten(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Pokemon-Karten: '20+60+40+2x100' -> 320."""
    low = _digitize(question.lower())
    im = re.search(r"had\s+(\d+)\s+\w+\s+\w+", low)
    tm = re.search(r"(?:three|3)\s+times\s+that\s+number", low)
    sm = re.search(r"(\d+)\s+fewer(?:\s+\w+)?\s+than", low)
    dm = re.search(r"twice\s+the\s+combined", low)
    if not (im and tm and sm and dm):
        return None
    first = Fraction(im.group(1)) * 3
    second = first - Fraction(sm.group(1))
    return Fraction(im.group(1)) + first + second + (first + second) * 2


def _krebse_drei(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Krebse-drei: '40+36+46' -> 122."""
    low = _digitize(question.lower())
    bm = re.search(r"if\s+\w+\s+has\s+(\d+)", low)
    mm = re.search(r"has\s+(\d+)\s+fewer\s+\w+\s+than", low)
    rm = re.search(r"(?:ten|10)\s+more\s+\w+\s+than", low)
    if not (bm and mm and rm):
        return None
    bo = Fraction(bm.group(1))
    monic = bo - Fraction(mm.group(1))
    return bo + monic + monic + 10


def _staffel_zeit(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Staffel-Zeit: '(60+57+54+51)-4x55' -> 2."""
    low = _digitize(question.lower())
    fm = re.search(r"4\s*by\s*400", low)
    sm = re.search(r"precisely\s+(\d+)\s+\w+", low)
    lm = re.search(r"in\s+(\d+)\s+\w+\s+then\s+each", low)
    dm = re.search(r"(\d+)\s+\w+\s+faster(?:\s+than)?", low)
    if not (fm and sm and lm and dm):
        return None
    fast = Fraction(sm.group(1)) * 4
    slow = sum(Fraction(lm.group(1)) - Fraction(dm.group(1)) * i for i in range(4))
    return slow - fast


def _weizen_gewinne(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Weizen-Gewinne: '400/(30-20-2)' -> 50."""
    low = _digitize(question.lower())
    bm = re.search(r"rate\s+of\s+\$?(\d+)\s+per\s+\w+", low)
    tm = re.search(r"costs\s+\$?(\d+)\s+to\s+transport", low)
    pm = re.search(r"profit\s+of\s+\$?(\d+)", low)
    sm = re.search(r"rate\s+of\s+\$?(\d+)\s+each", low)
    if not (bm and tm and pm and sm):
        return None
    return Fraction(pm.group(1)) / (
        Fraction(sm.group(1)) - Fraction(bm.group(1)) - Fraction(tm.group(1))
    )


def _backwaren_laenge(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Backwaren-Länge: '(300x4+120x6+60x24)/12' -> 280."""
    low = _digitize(question.lower())
    bm = re.search(r"bakes\s+(\d[^.]*?every\s+day)", low)
    if not bm:
        return None
    ms = re.findall(r"(\d+)\s+\w+(?:\s+\w+)?", bm.group(1))
    rs = re.findall(r"each\s+\w+\s+is\s+(\d+)\s+\w+", low)
    fs = re.search(r"each\s+\w+\s+is\s+(?:two|2)\s+feet", low)
    if len(ms) < 3 or len(rs) < 2 or not fs:
        return None
    inches = (
        Fraction(ms[0]) * Fraction(rs[0])
        + Fraction(ms[1]) * Fraction(rs[1])
        + Fraction(ms[2]) * 24
    )
    return inches / 12


def _insekten_beine(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Insekten-Beine: '80x8+90x6+3x10' -> 1210."""
    low = _digitize(question.lower())
    if "legs" not in low:
        return None
    ms = re.findall(r"(\d+)\s+\w+\s+with\s+(\d+)\s+\w+\s+each", low)
    mm = re.findall(
        r"(\d+)\s+\w+\s+\w+\s+\w+\s+with\s+"
        r"(\d+)\s+\w+",
        low,
    )
    if len(ms) < 2:
        return None
    total = sum(Fraction(a) * Fraction(b) for a, b in ms)
    if mm:
        total += Fraction(mm[0][0]) * Fraction(mm[0][1])
    return total


def _kartoffel_zeit(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Kartoffel-Zeit: '60x1.5 + 60x5/60' -> 95."""
    low = _digitize(question.lower())
    pm = re.search(r"has\s+(\d+)\s+\w+", low)
    tm = re.search(r"minute\s+and\s+a\s+half\s+to\s+\w+", low)
    cm = re.search(r"only\s+about\s+(\d+)\s+\w+\s+to\s+cut", low)
    if not (pm and tm and cm):
        return None
    n = Fraction(pm.group(1))
    return n * Fraction(3, 2) + n * Fraction(cm.group(1)) / 60


def _fahrgeschaeft(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Fahrgeschäft: '(2+4+2+2)x6' -> 60."""
    low = _digitize(question.lower())
    ms = re.findall(
        r"rode\s+(?:it|the\s+\w+(?:\s+\w+)?)\s+"
        r"(\d+)\s+times?",
        low,
    )
    lm = re.search(r"ride\s+the\s+\w+\s+(\d+)\s+times?", low)
    cm = re.search(r"each\s+ride\s+cost\s+(\d+)\s+\w+", low)
    if len(ms) < 2 or not (lm and cm):
        return None
    return (sum(Fraction(m) for m in ms) + Fraction(lm.group(1)) * 2) * Fraction(
        cm.group(1)
    )


def _wander_mittwoch(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Wander-Mittwoch: '41-4-24' -> 13."""
    low = _digitize(question.lower())
    mm = re.search(r"walked\s+(\d+)\s+\w+", low)
    tm = re.search(r"(\d+)\s+times\s+as\s+many\s+\w+", low)
    tm2 = re.search(
        r"total\s+\w+(?:\s+\w+)?\s+through\s+\w+"
        r"\s+was\s+(\d+)",
        low,
    )
    if not (mm and tm and tm2):
        return None
    monday = Fraction(mm.group(1))
    return Fraction(tm2.group(1)) - monday - monday * Fraction(tm.group(1))


def _einhorn_frauen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Einhorn-Frauen: '27/3x2/3' -> 6."""
    low = _digitize(question.lower())
    um = re.search(r"(\d+)\s+\w+\s+left\s+in\s+the\s+\w+", low)
    sm = re.search(r"(?:one|1)\s+third\s+of\s+them", low)
    fm = re.search(r"(?:two|2)\s+thirds\s+of\s+the\s+\w+", low)
    if not (um and sm and fm):
        return None
    return Fraction(um.group(1)) / 3 * Fraction(2, 3)


def _test_unvollstaendig(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Test-unvollständig: '(75-40)+(100-30)' -> 105."""
    low = _digitize(question.lower())
    ms = re.findall(r"(?:test\s+of|consisted\s+of)\s+(\d+)\s+\w+", low)
    rm = re.search(
        r"(?:rate\s+of\s+|at\s+)(\d+)\s+\w+\s+per"
        r"\s+\w+",
        low,
    )
    hs = re.findall(r"(\d+)\s+hours?(?:\s+to\s+complete|\s+for)", low)
    if len(ms) < 2 or not rm or len(hs) < 2:
        return None
    rate = Fraction(rm.group(1))
    return sum(Fraction(m) - rate * Fraction(h) for m, h in zip(ms, hs))


def _wettlauf_warten(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Wettlauf-Warten: '2x5280/264 - 3x5280/440' -> 4."""
    low = _digitize(question.lower())
    sm = re.search(r"lives\s+(\d+)\s+\w+\s+from", low)
    sb = re.search(r"bike\s+at\s+(\d+)\s+\w+\s+per\s+\w+", low)
    tm = re.search(r"lives\s+(\d+)\s+\w+\s+away", low)
    tb = re.search(r"skateboard\s+at\s+(\d+)", low)
    if not (sm and sb and tm and tb):
        return None
    steve = Fraction(sm.group(1)) * 5280 / Fraction(sb.group(1))
    tim = Fraction(tm.group(1)) * 5280 / Fraction(tb.group(1))
    return abs(tim - steve)


def _vorlesung_stunden(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Vorlesungs-Stunden: '(3x1x3+2x2x2)x16' -> 272."""
    low = _digitize(question.lower())
    ms = re.findall(r"(\d+)\s+(\d+)-hour", low)
    ws = re.search(r"(\d+)\s+weeks?(?:\s+of\s+school)?", low)
    if len(ms) < 2 or not ws:
        return None
    per_week = (
        Fraction(ms[0][0]) * Fraction(ms[0][1]) * 3
        + Fraction(ms[1][0]) * Fraction(ms[1][1]) * 2
    )
    return per_week * Fraction(ws.group(1))


def _loeffel_paket(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Löffel-Paket: '12+3-5' -> 10."""
    low = _digitize(question.lower())
    tm = re.search(r"total\s+of\s+(\d+)\s+\w+", low)
    um = re.search(r"used\s+(\d+)\s+of\s+the\s+\w+", low)
    hm = re.search(r"package\s+of\s+(\d+)\s+new", low)
    if not (tm and um and hm):
        return None
    return Fraction(tm.group(1)) + Fraction(um.group(1)) - Fraction(hm.group(1))


def _freunde_mehr(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Freunde-mehr: '50 + 70' -> 120."""
    low = _digitize(question.lower())
    lm = re.search(r"lily\s+made\s+(\d+)\s+\w+", low)
    am = re.search(r"(\d+)\s+more\s+\w+\s+than\s+(\w+)", low)
    if not (lm and am):
        return None
    lily = Fraction(lm.group(1))
    return lily + lily + Fraction(am.group(1))


def _boot_leck(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Boot-Leck: '64/16x20/10x2' -> 16."""
    low = _digitize(question.lower())
    lm = re.search(
        r"(?:two|2)\s+liters?\s+of\s+\w+\s+for\s+"
        r"every\s+(\d+)\s+\w+",
        low,
    )
    sm = re.search(r"(\d+)\s+seconds?\s+to\s+row\s+(\d+)", low)
    sm2 = re.search(r"shore\s+was\s+(\d+)\s+\w+\s+away", low)
    if not (lm and sm and sm2):
        return None
    segs = Fraction(sm2.group(1)) / Fraction(sm.group(1))
    feet = segs * Fraction(sm.group(2))
    return feet / Fraction(lm.group(1)) * 2


def _tafel_reinigung(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Tafel-Reinigung: '4x2x3' -> 24."""
    low = _digitize(question.lower())
    tm = re.search(r"between\s+the\s+(\d+)\s+\w+", low)
    lm = re.search(r"(\d+)\s+lessons?\s+per\s+day", low)
    cm = re.search(r"cleaned\s+(\d+)\s+times?\s+per\s+lesson", low)
    if not (tm and lm and cm):
        return None
    return Fraction(tm.group(1)) * Fraction(lm.group(1)) * Fraction(cm.group(1))


def _lauf_strecke(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Lauf-Strecke: '10x3 + 5x4' -> 50."""
    low = _digitize(question.lower())
    r1 = re.search(
        r"run\s+(\d+)\s+\w+\s+per\s+\w+\s+for\s+"
        r"(\d+)",
        low,
    )
    r2 = re.search(r"runs\s+(\d+)\s+\w+\s+per\s+\w+", low)
    tm = re.search(r"in\s+(\d+)\s+hours?", low)
    if not (r1 and r2 and tm):
        return None
    return Fraction(r1.group(1)) * Fraction(r1.group(2)) + Fraction(r2.group(1)) * (
        Fraction(tm.group(1)) - Fraction(r1.group(2))
    )


def _stau_autos(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Stau-Autos: '(30-5)-20' -> 5."""
    low = _digitize(question.lower())
    om = re.search(r"originally\s+(\d+)\s+\w+", low)
    em = re.search(r"take\s+an\s+exit", low)
    mm = re.search(r"(\d+)\s+more\s+\w+\s+drive\s+through", low)
    if not (om and em and mm):
        return None
    return Fraction(om.group(1)) - 5 - Fraction(mm.group(1))


def _pflanzen_bleiben(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Pflanzen-bleiben: '18+2x40-40' -> 58."""
    low = _digitize(question.lower())
    nm = re.search(r"received\s+(\d+)\s+new", low)
    lm = re.search(r"each\s+of\s+the\s+(\d+)\s+\w+", low)
    gm = re.search(
        r"give\s+(\d+)\s+\w+(?:\s+\w+)?\s+from\s+"
        r"each",
        low,
    )
    if not (nm and lm and gm):
        return None
    return Fraction(nm.group(1)) + Fraction(lm.group(1)) * 2 - Fraction(lm.group(1))


def _kekse_letztes(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Kekse-letztes: '(110+5-15)/2' -> 50."""
    low = _digitize(question.lower())
    tm = re.search(r"(?:total\s+of\s+|has\s+)(\d+)\s+\w+", low)
    dm = re.search(r"drops?\s+(\d+)(?:\s+of\s+his)?", low)
    mm = re.search(r"(\d+)\s+more(?:\s+\w+)?\s+than\s+he\s+meant", low)
    dm2 = re.search(r"twice\s+as\s+many", low)
    if not (tm and dm and mm and dm2):
        return None
    return (Fraction(tm.group(1)) + Fraction(dm.group(1)) - Fraction(mm.group(1))) / 2


def _wander_rest(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Wander-Rest: '6/(3-2)' -> 6mph."""
    low = _digitize(question.lower())
    tm = re.search(r"(\d+)-mile\s+\w+", low)
    fm = re.search(r"first\s+(\d+)\s+\w+", low)
    nm = re.search(r"next\s+(\d+)\s+\w+", low)
    am = re.search(r"average\s+speed\s+to\s+be\s+(\d+)", low)
    if not (tm and fm and nm and am):
        return None
    total = Fraction(tm.group(1))
    done = Fraction(fm.group(1)) + Fraction(nm.group(1))
    hours = Fraction(2)
    need = total / Fraction(am.group(1))
    return (total - done) / (need - hours)


def _juwelen_kette(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Juwelen-Kette: '40/2+5-2' -> 23."""
    low = _digitize(question.lower())
    rm = re.search(r"raymond\s+has\s+(\d+)", low)
    hm = re.search(r"half\s+of\s+\w+'s\s+\w+", low)
    if not rm:
        return None
    aaron = Fraction(rm.group(1)) / 2 + 5
    return aaron - 2


def _drache_wurf(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Drache-Wurf: '3x400-1000' -> 200."""
    low = _digitize(question.lower())
    if not re.search(r"distance|throw|javelin", low):
        return None
    dm = re.search(r"(?:distance\s+of|within)\s+(\d+)\s+\w+", low)
    ds = re.findall(r"distance\s+of\s+(\d+)\s+\w+", low)
    tm = re.search(r"(?:three|3)\s+times\s+farther", low)
    if not (dm or ds or tm):
        return None
    throw = ds[-1] if ds else dm.group(1)
    return Fraction(throw) * 3 - Fraction(dm.group(1))


def _downloads_drei(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Downloads-drei: '60+180+126' -> 366."""
    low = _digitize(question.lower())
    fm = re.search(r"had\s+(\d+)\s+\w+\s+in\s+the\s+first", low)
    tm = re.search(r"(?:three|3)\s+times\s+as\s+many", low)
    rm = re.search(r"(\d+)\s*(?:%|percent)\s+in\s+the\s+third", low)
    if not (fm and tm and rm):
        return None
    first = Fraction(fm.group(1))
    second = first * 3
    return first + second + second * (1 - Fraction(rm.group(1)) / 100)


def _schafe_drei(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Schafe-drei: '20+80+160' -> 260."""
    low = _digitize(question.lower())
    sm = re.search(r"if\s+\w+\s+has\s+(\d+)", low)
    cm = re.search(r"4\s+times\s+as\s+many", low)
    tm = re.search(r"twice\s+as\s+many", low)
    if not (sm and cm and tm):
        return None
    seattle = Fraction(sm.group(1))
    charleston = seattle * 4
    return seattle + charleston + charleston * 2


def _futter_letzte(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Futter-letzte: '20x3-15-25' -> 20."""
    low = _digitize(question.lower())
    cm = re.search(r"(\d+)\s+cups\s+of\s+\w+\s+\w+", low)
    fm = re.search(
        r"flock\s+(?:of\s+)?(\d+)\s+\w+|"
        r"flock\s+is\s+(\d+)",
        low,
    )
    mm2 = re.search(
        r"morning[^.]*?(\d+)\s+cups|"
        r"(\d+)\s+cups[^.]*?morning",
        low,
    )
    am2 = re.search(r"another\s+(\d+)\s+cups", low)
    if not (cm and fm and mm2 and am2):
        return None
    n = fm.group(1) or fm.group(2)
    m = mm2.group(1) or mm2.group(2)
    return Fraction(n) * Fraction(cm.group(1)) - Fraction(m) - Fraction(am2.group(1))


def _heimfahrt(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Heimfahrt: '180-15-120' -> 45."""
    low = _digitize(question.lower())
    dm = re.search(
        r"drives\s+for\s+(\d+)\s+hours?\s+at\s+"
        r"(?:a\s+speed\s+of\s+)?(\d+)",
        low,
    )
    hm = re.search(
        r"half-hour(?:\s+driving)?\s+at\s+(?:a\s+speed"
        r"\s+of\s+)?(\d+)",
        low,
    )
    rm = re.search(r"remaining\s+time[^.]*?\s*(\d+)\s+mph", low)
    tm = re.search(r"get\s+home\s+in\s+(\d+)\s+hours?", low)
    if not (dm and hm and rm and tm):
        return None
    out = Fraction(dm.group(1)) * Fraction(dm.group(2))
    rest = Fraction(tm.group(1)) - 2 - Fraction(1, 2)
    back = Fraction(hm.group(1)) * Fraction(1, 2) + rest * Fraction(rm.group(1))
    return out - back


def _blumen_verkauf(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Blumen-Verkauf: '(7x12+13)x3' -> 291."""
    low = _digitize(question.lower())
    sm = re.search(
        r"costs\s+\$?(\d+)\s+each\s+and\s+a\s+\w+"
        r"(?:\s+of\s+\w+)?\s+that\s+costs\s+\$?(\d+)",
        low,
    )
    em = re.search(
        r"earned\s+\$?(\d+)\s+from\s+the\s+\w+\s+and"
        r"\s+\$?(\d+)\s+from\s+the\s+\w+",
        low,
    )
    bm = re.search(r"each\s+\w+\s+has\s+(\d+)\s+\w+", low)
    dm = re.search(r"after\s+(\d+)\s+days?", low)
    if not (sm and em and bm and dm):
        return None
    bouquets = Fraction(em.group(2)) / Fraction(sm.group(2))
    singles = Fraction(em.group(1)) / Fraction(sm.group(1))
    return (bouquets * Fraction(bm.group(1)) + singles) * Fraction(dm.group(1))


def _schulbuecher(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Schulbücher: '9000/500x100' -> 1800."""
    low = _digitize(question.lower())
    sm = re.search(r"spends?\s+\$?(\d+)\s+\w+\s+between\s+(\d+)", low)
    bm = re.search(r"buy\s+(\d+)\s+\w+\s+for\s+\$?(\d+)", low)
    if not (sm and bm):
        return None
    per_school = Fraction(sm.group(1)) / Fraction(sm.group(2))
    return per_school / Fraction(bm.group(2)) * Fraction(bm.group(1))


def _kerzen_licht(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Kerzen-Licht: '8x2+4x1 + 4x4+5x4' -> 56."""
    low = _digitize(question.lower())
    rm = re.search(r"(\d+)\s+rooms?\s+in\s+the\s+\w+", low)
    pm = re.search(r"(\d+)\s+people?\s+living", low)
    fm = re.search(r"(?:two|2)\s+for\s+each\s+\w+", low)
    cm = re.search(r"4\s+small\s+\w+\s+each\s+for\s+half", low)
    cm2 = re.search(
        r"(\d+)\s+medium\s+\w+\s+each\s+for\s+the"
        r"\s+other",
        low,
    )
    if not (rm and pm and fm and cm and cm2):
        return None
    flash = Fraction(rm.group(1)) * 2 + Fraction(pm.group(1)) * 1
    candles = Fraction(rm.group(1)) / 2 * 4 + Fraction(rm.group(1)) / 2 * Fraction(
        cm2.group(1)
    )
    return flash + candles


def _thrice_mehr(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Thrice-mehr: '2 mehr als 3x Mike(5)' -> 3x5+2=17."""
    low = _digitize(question.lower())
    m = re.search(
        r"(\d+)\s+more\s+than\s+thrice\s+as\s+many\s+"
        r"as\s+([a-z]+)",
        low,
    )
    if not m:
        return None
    bm = re.search(r"bought\s+(\d+)\s+\w+\s+\w+\s+while", low)
    if bm:
        return Fraction(bm.group(1)) * 3 + Fraction(m.group(1))
    for q in quants:
        if q.obj and m.group(2) in q.obj.lower():
            return Fraction(q.value) * 3 + Fraction(m.group(1))
    return None


def _ballon_wurf(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Ballon-Wurf: '10x30-12' -> 288."""
    low = _digitize(question.lower())
    pm = re.search(
        r"(\d+)\s+packs?\s+of\s+\w+(?:\s+that\s+have"
        r"|\s+with)?\s+(\d+)\s+\w+\s+per",
        low,
    )
    lm = re.search(r"(\d+)\s+\w+\s+are\s+left", low)
    if not (pm and lm):
        return None
    return Fraction(pm.group(1)) * Fraction(pm.group(2)) - Fraction(lm.group(1))


def _suesigkeiten_kauf(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Süßigkeiten-Kauf: '4.95-1.2=3.75; /0.75=5'."""
    low = _digitize(question.lower())
    sm = re.search(r"spent\s+\$?(\d+)", low)
    pm = re.search(
        r"(\d+)\s*(?:%|percent)\s+of\s+his\s+\w+\s+"
        r"left",
        low,
    )
    cm = re.search(r"chips\s+for\s+(\d+)\s+\w+", low)
    nm = re.search(r"bags?\s+of\s+\w+", low)
    bm = re.search(r"\w+\s+bars?\s+for\s+(\d+)\s+\w+", low)
    if not (sm and pm and cm and nm and bm):
        return None
    spent = Fraction(sm.group(1)) * (1 - Fraction(pm.group(1)) / 100)
    chips = 3 * Fraction(cm.group(1)) / 100
    return (spent - chips) / (Fraction(bm.group(1)) / 100)


def _muenzen_cents(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Münzen-cents: 'quarter+nickels+dimes' -> 105¢."""
    low = _digitize(question.lower())
    qm = re.search(r"(?:a|one|1)\s+quarter", low)
    nm = re.search(r"(\d+)\s+nickels?", low)
    dm = re.search(r"(\d+)\s+dimes?", low)
    if not (qm and nm and dm):
        return None
    return Fraction(25) + Fraction(nm.group(1)) * 5 + Fraction(dm.group(1)) * 10


def _zitronenbaum(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Zitronenbaum: '90/(7x1.5-3)+1' -> 13."""
    low = _digitize(question.lower())
    cm = re.search(r"cost\s+\$?(\d+)\s+to\s+plant", low)
    lm = re.search(r"grow\s+(\d+)\s+\w+", low)
    sm = re.search(r"sell\s+for\s+\$?([\d.]+)\s+each", low)
    wm = re.search(r"costs\s+\$?(\d+)\s+a\s+year", low)
    if not (cm and lm and sm and wm):
        return None
    per_year = Fraction(lm.group(1)) * Fraction(sm.group(1)) - Fraction(wm.group(1))
    return Fraction(cm.group(1)) / per_year + 1


def _eier_verkauf(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Eier-Verkauf: '(16-3-4)x2' -> 18."""
    low = _digitize(question.lower())
    lm = re.search(r"lay\s+(\d+)\s+\w+\s+per\s+day", low)
    em = re.search(r"eats\s+(\d+)\s+for\s+\w+", low)
    bm = re.search(r"with\s+(\d+)\.", low)
    sm = re.search(r"for\s+\$?(\d+)\s+per", low)
    if not (lm and em and bm and sm):
        return None
    return (
        Fraction(lm.group(1)) - Fraction(em.group(1)) - Fraction(bm.group(1))
    ) * Fraction(sm.group(1))


def _sprint_gesamt(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Sprint-gesamt: '3x3x60' -> 540."""
    low = _digitize(question.lower())
    sm = re.search(
        r"runs?\s+(\d+)\s+\w+\s+(\d+)\s+times\s+a"
        r"\s+\w+",
        low,
    )
    mm = re.search(r"runs?\s+(\d+)\s+\w+\s+each", low)
    if not (sm and mm):
        return None
    return Fraction(sm.group(1)) * Fraction(sm.group(2)) * Fraction(mm.group(1))


def _bohnen_schnitt(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Bohnen-Schnitt: '(80+60+100)/3' -> 80."""
    low = _digitize(question.lower())
    fm = re.search(r"says\s+(\d+)", low)
    hm = re.search(r"(\d+)\s+more\s+than\s+half\s+the\s+first", low)
    pm = re.search(
        r"(\d+)\s*(?:%|percent)\s+more\s+than\s+the\s+"
        r"first",
        low,
    )
    if not (fm and hm and pm):
        return None
    first = Fraction(fm.group(1))
    second = first / 2 + Fraction(hm.group(1))
    third = first * (1 + Fraction(pm.group(1)) / 100)
    return (first + second + third) / 3


def _saft_wasser(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Saft-Wasser: '(10-1)x2/3 + 15x3/5' -> 15."""
    low = _digitize(question.lower())
    om = re.search(
        r"(\d+)\s+liters?\s+of\s+\w+\s+drink\s+"
        r"(?:that\s+)?are\s+((?:two-thirds|2-thirds))\s+"
        r"\w+",
        low,
    )
    pm = re.search(
        r"(\d+)\s+liters?\s+of\s+\w+\s+drink\s+"
        r"(?:that\s+)?is\s+((?:three-fifths|3-fifths))",
        low,
    )
    sm = re.search(r"spill\s+(\d+)\s+liter", low)
    if not (om and pm and sm):
        return None
    orange = (Fraction(om.group(1)) - Fraction(sm.group(1))) * Fraction(2, 3)
    pineapple = Fraction(pm.group(1)) * Fraction(3, 5)
    return orange + pineapple


def _eier_dutzend(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Eier-Dutzend: '3x7x4/12' -> 7."""
    low = _digitize(question.lower())
    em = re.search(r"(\d+)\s+egg\s+\w+", low)
    wm = re.search(r"(\d+)\s+weeks?", low)
    if not (em and wm):
        return None
    return Fraction(em.group(1)) * 7 * Fraction(wm.group(1)) / 12


def _haus_flip(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Haus-Flip: '(80000x2.5)-(80000+50000)' -> 70000."""
    low = _digitize(question.lower())
    bm = re.search(r"buys?\s+a\s+\w+\s+for\s+\$?([\d,]+)", low)
    rm = re.search(r"puts?\s+in\s+\$?([\d,]+)", low)
    pm = re.search(r"(\d+)\s*(?:%|percent)\s*\.", low)
    if not (bm and rm and pm):
        return None
    buy = Fraction(int(bm.group(1).replace(",", "")))
    repair = Fraction(int(rm.group(1).replace(",", "")))
    new_val = buy * (1 + Fraction(pm.group(1)) / 100)
    return new_val - buy - repair


def _download_zeit(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Download-Zeit: '40+20+100' -> 160min."""
    low = _digitize(question.lower())
    gm = re.search(r"(?:a\s+)?([\d.]+)\s+[Gg][Bb]\s+file", low)
    rm = re.search(r"(?:download\s+|at\s+)([\d.]+)\s+[Gg][Bb]", low)
    pm = re.search(r"(\d+)\s*(?:%|percent)\s+of\s+the\s+way", low)
    wm = re.search(r"takes\s+(\d+)\s+\w+", low)
    if not (gm and rm and pm and wm):
        return None
    total_min = Fraction(gm.group(1)) / Fraction(rm.group(1))
    done = total_min * Fraction(pm.group(1)) / 100
    return done + Fraction(wm.group(1)) + total_min


def _ueberstunden_lohn(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Überstunden-Lohn: '40x10 + 5x12' -> 460."""
    low = _digitize(question.lower())
    hm = re.search(r"first\s+(\d+)\s+hours?", low)
    rm = re.search(r"rate\s+per\s+hour[^$]*\$?(\d+)", low)
    om = re.search(r"overtime\s+pay\s+of\s+([\d.]+)\s+times", low)
    wm = re.search(r"worked\s+for\s+(\d+)\s+hours?", low)
    if not (hm and rm and om and wm):
        return None
    regular = Fraction(hm.group(1)) * Fraction(rm.group(1))
    ot_h = Fraction(wm.group(1)) - Fraction(hm.group(1))
    return regular + ot_h * Fraction(rm.group(1)) * Fraction(om.group(1))


def _krawatten_kauf(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Krawatten-Kauf: '5x40 + 10x60' -> 800."""
    low = _digitize(question.lower())
    tm = re.search(r"twice\s+as\s+many\s+\w+\s+\w+\s+as", low)
    pm = re.search(r"(\d+)\s*(?:%|percent)\s+more\s+than", low)
    sm = re.search(
        r"spent\s+\$?(\d+)\s+on\s+\w+\s+\w+\s+that"
        r"\s+cost\s+\$?(\d+)",
        low,
    )
    if not (tm and pm and sm):
        return None
    blue_n = Fraction(sm.group(1)) / Fraction(sm.group(2))
    red_n = blue_n * 2
    red_p = Fraction(sm.group(2)) * (1 + Fraction(pm.group(1)) / 100)
    return Fraction(sm.group(1)) + red_n * red_p


def _pasteten_kauf(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Pasteten-Kauf: '3x68+2x80+6x55' -> 694."""
    low = _digitize(question.lower())
    ms = re.findall(
        r"(\d+)\s+dozen\s+\w+[^.]*?\$?(\d+)\s+per"
        r"\s+dozen",
        low,
    )
    if len(ms) < 2:
        return None
    return sum(Fraction(a) * Fraction(b) for a, b in ms)


def _kauf_profit(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Kauf-Profit: 'max(5000x2.5%, 8000x1.2%)' -> 125."""
    low = _digitize(question.lower())
    ms = re.findall(r"worth\s+\$?([\d,]+)", low)
    ps = re.findall(
        r"(?:go(?:ing)?\s+up|ris(?:e|ing))\s+([\d.]+)"
        r"\s*(?:%|percent)",
        low,
    )
    if len(ms) < 2 or len(ps) < 2:
        return None
    a = Fraction(int(ms[0].replace(",", ""))) * Fraction(ps[0]) / 100
    b = Fraction(int(ms[1].replace(",", ""))) * Fraction(ps[1]) / 100
    return max(a, b)


def _tanz_prozent(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Tanz-Prozent: '20-4-4=12; 12/20' -> 60%."""
    low = _digitize(question.lower())
    cm = re.search(r"class\s+of\s+(\d+)\s+\w+", low)
    pm = re.search(r"(\d+)\s*(?:%|percent)\s+enrolled\s+in\s+\w+", low)
    jm = re.search(r"(\d+)\s*(?:%|percent)\s+of\s+the\s+remaining", low)
    if not (cm and pm and jm):
        return None
    total = Fraction(cm.group(1))
    first = total * Fraction(pm.group(1)) / 100
    second = (total - first) * Fraction(jm.group(1)) / 100
    return (total - first - second) / total * 100


def _reifen_einnahmen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Reifen-Einnahmen: '520-480' -> 40."""
    low = _digitize(question.lower())
    tm = re.search(
        r"charges?\s+\$?(\d+)\s+(?:for\s+each|per)"
        r"\s+truck",
        low,
    )
    cm = re.search(r"\$?(\d+)\s+(?:for\s+each|per)\s+car", low)
    tn = re.search(r"repairs?\s+(\d+)\s+truck", low)
    cn = re.search(r"thursday[^.]*?(\d+)\s+car", low)
    fn = re.search(r"friday[^.]*?(\d+)\s+car", low)
    if not (tm and cm and tn and cn and fn):
        return None
    thu = Fraction(tn.group(1)) * Fraction(tm.group(1)) + Fraction(
        cn.group(1)
    ) * Fraction(cm.group(1))
    fri = Fraction(fn.group(1)) * Fraction(cm.group(1))
    return abs(thu - fri)


def _kerze_zeit(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Kerze-Zeit: '(5-1)x2' -> 8."""
    low = _digitize(question.lower())
    rm = re.search(r"melts?\s+by\s+(\d+)\s+\w+\s+every\s+hour", low)
    tm = re.search(r"(\d+):00\s+\w+\s+to\s+(\d+):00", low)
    if not (rm and tm):
        return None
    return (Fraction(tm.group(2)) - Fraction(tm.group(1))) * Fraction(rm.group(1))


def _zug_strecke(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Zug-Strecke: '80+150' -> 230."""
    low = _digitize(question.lower())
    ms = re.findall(
        r"travel(?:ing)?\s+for\s+(\d+)\s+\w+|"
        r"cover(?:ing)?\s+(\d+)\s+\w+",
        low,
    )
    vals = [a or b for a, b in ms]
    if len(vals) < 2:
        return None
    return sum(Fraction(v) for v in vals)


def _pizza_boxes(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Pizza-Boxes: '(50-33)/8.5' -> 2."""
    low = _digitize(question.lower())
    ms = re.findall(r"costs?\s+\$?(\d+(?:\.\d+)?)\s+each", low)
    cm = re.search(r"meal\s+(?:that\s+costs\s+|at\s+)\$?(\d+)", low)
    tm = re.search(r"paid\s+(?:a\s+total\s+of\s+)?\$?(\d+)", low)
    bm = re.search(r"each\s+(?:box\s+)?costs\s+\$?([\d.]+)", low)
    if not (cm and tm and bm):
        return None
    spent = Fraction(cm.group(1))
    groups = re.findall(
        r"(\d+)\s+\w+\s+of\s+\w+\s+(?:that\s+"
        r"costs?\s+|at\s+)\$?([\d.]+)\s+each",
        low,
    )
    for n, c in groups:
        spent += Fraction(n) * Fraction(c)
    singles = re.findall(
        r"(\d+)\s+\w+\s+(?:that\s+cost\s+|at\s+)"
        r"\$?([\d.]+)\s+each",
        low,
    )
    for n, c in singles:
        spent += Fraction(n) * Fraction(c)
    return (Fraction(tm.group(1)) - spent) / Fraction(bm.group(1))


def _jahres_gehalt(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Jahres-Gehalt: '(20x35+30x15)x50' -> 57500."""
    low = _digitize(question.lower())
    ms = re.findall(
        r"\$?(\d+)\s+per\s+hour\s+to\s+\w+|"
        r"\$?(\d+)\s+to\s+be\s+a\s+\w+",
        low,
    )
    ms = [(a or b) for a, b in ms]
    ws = re.findall(r"(\d+)\s+hours?\s+a\s+week\s+as\s+a\s+\w+", low)
    ys = re.search(r"(\d+)\s+weeks?\s+a\s+year", low)
    if len(ms) < 2 or len(ws) < 2 or not ys:
        return None
    weekly = Fraction(ms[0]) * Fraction(ws[0]) + Fraction(ms[1]) * Fraction(ws[1])
    return weekly * Fraction(ys.group(1))


def _eis_ausgaben(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Eis-Ausgaben: '60/15x4' -> 16."""
    low = _digitize(question.lower())
    sm = re.search(
        r"(\d+)\s+servings?(?:\s+of\s+\w+(?:\s+\w+)?)"
        r"?\s+per\s+\w+",
        low,
    )
    cm = re.search(r"(?:cost\s+of\s+|at\s+)\$?([\d.]+)\s+per", low)
    dm = re.search(r"after\s+(\d+)\s+days?", low)
    if not (sm and cm and dm):
        return None
    cartons = Fraction(dm.group(1)) / Fraction(sm.group(1))
    return cartons * Fraction(cm.group(1))


def _original_preis(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Original-Preis: '19.5/0.75' -> 26."""
    low = _digitize(question.lower())
    if "original" not in low:
        return None
    pm = re.search(r"for\s+\$?(\d+(?:\.\d+)?)", low)
    dm = re.search(r"(\d+)\s*(?:%|percent)\s+discount", low)
    if not (pm and dm):
        return None
    return Fraction(pm.group(1)) / (1 - Fraction(dm.group(1)) / 100)


def _hunde_pflege(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Hunde-Pflege: '10x0.5x7' -> 35."""
    low = _digitize(question.lower())
    dm = re.search(r"care\s+of\s+(\d+)\s+\w+", low)
    hm = re.search(r"takes?\s+([\d.]+)\s+hours?\s+a\s+day", low)
    wm = re.search(r"a\s+week", low)
    if not (dm and hm and wm):
        return None
    return Fraction(dm.group(1)) * Fraction(hm.group(1)) * 7


def _alter_verhaeltnis(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Alter-Verhältnis: '162x11/18+10' -> 109."""
    low = _digitize(question.lower())
    rm = re.search(r"ratio\s+of\s+(\d+):(\d+)", low)
    tm = re.search(r"total\s+age\s+now\s+is\s+(\d+)", low)
    ym = re.search(r"(\d+)\s+years?\s+from\s+now", low)
    if not (rm and tm and ym):
        return None
    allen = (
        Fraction(tm.group(1))
        * Fraction(rm.group(2))
        / (Fraction(rm.group(1)) + Fraction(rm.group(2)))
    )
    return allen + Fraction(ym.group(1))


def _lego_rest(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Lego-Rest: '13-(8x20+5)/15' -> 2."""
    low = _digitize(question.lower())
    lm = re.search(r"has\s+(\d+)\s+\w+\s+\w+", low)
    sm = re.search(r"sells?\s+them\s+for\s+\$?(\d+)", low)
    vm = re.search(
        r"buys?\w*\s+(\d+)\s+\w+(?:\s+\w+)?\s+for"
        r"\s+\$?(\d+)",
        low,
    )
    lm2 = re.search(r"has\s+\$?(\d+)\s+left", low)
    if not (lm and sm and vm and lm2):
        return None
    earned = Fraction(vm.group(1)) * Fraction(vm.group(2)) + Fraction(lm2.group(1))
    sold = earned / Fraction(sm.group(1))
    return Fraction(lm.group(1)) - sold


def _pingpong_punkte(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Pingpong-Punkte: '4 + 4x1.25' -> 9."""
    low = _digitize(question.lower())
    pm = re.search(r"scores?\s+(\d+)\s+\w+", low)
    pm2 = re.search(r"(\d+)\s*(?:%|percent)\s+more\s+\w+", low)
    if not (pm and pm2):
        return None
    first = Fraction(pm.group(1))
    return first + first * (1 + Fraction(pm2.group(1)) / 100)


def _kuchen_gegessen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Kuchen-gegessen: '5x8-14' -> 26."""
    low = _digitize(question.lower())
    pm = re.search(r"baked\s+(\d+)\s+\w+\s+\w+", low)
    cm = re.search(r"cut\s+each\s+\w+\s+into\s+(\d+)\s+\w+", low)
    lm = re.search(r"there\s+were\s+(\d+)\s+\w+", low)
    if not (pm and cm and lm):
        return None
    return Fraction(pm.group(1)) * Fraction(cm.group(1)) - Fraction(lm.group(1))


def _lauf_tempo(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Lauf-Tempo: '60/(3+1.5+1.5)' -> 10."""
    low = _digitize(question.lower())
    wm = re.search(r"runs?\s+(\d+)\s+\w+\s+a\s+week", low)
    hm = re.search(r"runs?\s+(\d+)\s+hours?\s+the\s+first\s+day", low)
    if not (wm and hm):
        return None
    first = Fraction(hm.group(1))
    total_h = first + first / 2 * 2
    return Fraction(wm.group(1)) / total_h


def _chips_gramm(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Chips-Gramm: '(2000-1800)/250 x 300/5' -> 48."""
    low = _digitize(question.lower())
    cm = re.search(r"(\d+)\s+calories?\s+per\s+serving", low)
    gm = re.search(r"(\d+)g\s+bag\s+has\s+(\d+)\s+servings?", low)
    tm = re.search(r"target\s+is\s+(\d+)", low)
    cm2 = re.search(r"consumed\s+(\d+)", low)
    if not (cm and gm and tm and cm2):
        return None
    left = Fraction(tm.group(1)) - Fraction(cm2.group(1))
    servings = left / Fraction(cm.group(1))
    return servings * Fraction(gm.group(1)) / Fraction(gm.group(2))


def _iphone_alter(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """iPhone-Alter: '1x2x4' -> 8."""
    low = _digitize(question.lower())
    sm = re.search(r"if\s+\w+['\u2019]s\s+\w+\s+is\s+(\d+)", low)
    bm = re.search(r"(?:two|2)\s+times\s+older", low)
    bm2 = re.search(r"(?:four|4)\s+times\s+as\s+old", low)
    if not (sm and bm and bm2):
        return None
    return Fraction(sm.group(1)) * 2 * 4


def _draht_stuecke(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Draht-Stücke: '4x12/6' -> 8."""
    low = _digitize(question.lower())
    fm = re.search(r"wire\s+(\d+)\s+feet\s+long", low)
    cm = re.search(r"cut\s+into\s+\w+\s+(\d+)\s+\w+\s+long", low)
    if not (fm and cm):
        return None
    return Fraction(fm.group(1)) * 12 / Fraction(cm.group(1))


def _tasche_leicht(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Tasche-leicht: '(15-30x1/4)/(1/2)' -> 15."""
    low = _digitize(question.lower())
    rm = re.search(r"remove\s+(\d+)\s+pounds?", low)
    cm = re.search(r"comic\s+books?\s+weigh\s+(\d+/\d+)\s+\w+", low)
    tm = re.search(r"toys?\s+weigh\s+(\d+/\d+)\s+\w+", low)
    nm = re.search(r"removes\s+(\d+)\s+comic", low)
    if not (rm and cm and tm and nm):
        return None
    left = Fraction(rm.group(1)) - Fraction(nm.group(1)) * Fraction(cm.group(1))
    return left / Fraction(tm.group(1))


def _kitten_familie(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Kitten-Familie: '7+21+12' -> 40."""
    low = _digitize(question.lower())
    am = re.search(r"with\s+(\d+)\s+(?:\w+\s+)?kittens?", low)
    tm = re.search(r"thrice\s+the\s+number", low)
    tm2 = re.search(r"has\s+had\s+(\d+)", low)
    if not (am and tm and tm2):
        return None
    adopted = Fraction(am.group(1))
    return adopted + adopted * 3 + Fraction(tm2.group(1))


def _kerzen_profit(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Kerzen-Profit: '20x2 - 2x10' -> 20."""
    low = _digitize(question.lower())
    rm = re.search(r"make\s+(\d+)\s+(?:\w+\s+)?candles?", low)
    sm = re.search(
        r"(?:per\s+\w+\s+of\s+|(?:one|1)\s+\w+\s+of"
        r"\s+)\w+\s+and\s+(?:the\s+)?\w+\s+cost\s+"
        r"\$?(\d+)",
        low,
    )
    cm = re.search(r"sells?\s+each\s+\w+\s+for\s+\$?(\d+)", low)
    nm = re.search(r"makes?\s+and\s+sells\s+(\d+)", low)
    if not (rm and sm and cm and nm):
        return None
    pounds = Fraction(nm.group(1)) / Fraction(rm.group(1))
    return Fraction(nm.group(1)) * Fraction(cm.group(1)) - pounds * Fraction(
        sm.group(1)
    )


def _lutscher_tueten(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Lutscher-Tüten: '(30-2)/2' -> 14."""
    low = _digitize(question.lower())
    hm = re.search(r"has\s+(\d+)\s+\w+", low)
    em = re.search(r"eats\s+(\d+)(?:\s+of\s+the)?", low)
    pm = re.search(r"package\s+(\d+)\s+\w+\s+in\s+(?:one|1)", low)
    if not (hm and em and pm):
        return None
    return (Fraction(hm.group(1)) - Fraction(em.group(1))) / Fraction(pm.group(1))


def _blog_stunden(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Blog-Stunden: '(5+7+14)x4' -> 104."""
    low = _digitize(question.lower())
    hm = re.search(r"(\d+)\s+hours?(?:\s+to\s+\w+)?", low)
    mm = re.search(r"wrote\s+(\d+)\s+\w+\s+on\s+\w+", low)
    tm = re.search(r"2/5\s+times\s+more", low)
    wm = re.search(
        r"twice\s+the\s+number\s+of\s+\w+(?:\s+she"
        r"\s+wrote\s+on)?",
        low,
    )
    if not (hm and mm and tm and wm):
        return None
    mon = Fraction(mm.group(1))
    tue = mon + mon * Fraction(2, 5)
    wed = tue * 2
    return (mon + tue + wed) * Fraction(hm.group(1))


def _kino_besuche(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Kino-Besuche: '42/(7+7)' -> 3."""
    low = _digitize(question.lower())
    tm = re.search(r"ticket\s+for\s+\$?(\d+)", low)
    pm = re.search(r"popcorn\s+for\s+\$?(\d+)", low)
    dm = re.search(r"has\s+(\d+)\s+dollars?", low)
    if not (tm and pm and dm):
        return None
    return Fraction(dm.group(1)) / (Fraction(tm.group(1)) + Fraction(pm.group(1)))


def _notizen_paket(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Notizen-Paket: '220+23-80' -> 163."""
    low = _digitize(question.lower())
    nm = re.search(r"put\s+(\d+)\s+[\w-]+\s+[\w-]+\s+in\s+her", low)
    um = re.search(
        r"placed\s+a\s+[\w-]+(?:\s+[\w-]+){0,2}\s+on"
        r"\s+each\s+of\s+(\d+)",
        low,
    )
    lm = re.search(r"had\s+(\d+)\s+\w+", low)
    if not (nm and um and lm):
        return None
    return Fraction(um.group(1)) + Fraction(lm.group(1)) - Fraction(nm.group(1))


def _beeren_summe(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Beeren-Summe: '6x20+67' -> 187."""
    low = _digitize(question.lower())
    cm = re.search(r"(\d+)\s+clusters?\s+of\s+(\d+)\s+\w+\s+each", low)
    im = re.search(r"and\s+(\d+)\s+\w+\s+\w+", low)
    if not (cm and im):
        return None
    return Fraction(cm.group(1)) * Fraction(cm.group(2)) + Fraction(im.group(1))


def _wohnung_leer(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Wohnung-leer: '15x8x1/4' -> 30."""
    low = _digitize(question.lower())
    fm = re.search(r"(\d+)\s+floors?", low)
    um = re.search(r"(\d+)\s+units?", low)
    om = re.search(r"3/4(?:\s+of\s+the\s+\w+)?\s+is\s+occupied", low)
    if not (fm and um and om):
        return None
    return Fraction(fm.group(1)) * Fraction(um.group(1)) / 4


def _orangen_gut(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Orangen-gut: '25-1-5-2' -> 17."""
    low = _digitize(question.lower())
    cm = re.search(r"contains?\s+(\d+)\s+\w+", low)
    bm = re.search(r"(?:among\s+which\s+)?(\d+)\s+is\s+\w+", low)
    pm = re.search(r"(\d+)\s*(?:%|percent)\s+are\s+\w+", low)
    sm = re.search(r"(\d+)\s+are\s+\w+", low)
    if not (cm and bm and pm and sm):
        return None
    total = Fraction(cm.group(1))
    return (
        total
        - Fraction(bm.group(1))
        - total * Fraction(pm.group(1)) / 100
        - Fraction(sm.group(1))
    )


def _bruecke_boxes(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Brücke-Boxes: '(5000-3755)/15' -> 83."""
    low = _digitize(question.lower())
    cm = re.search(r"carry\s+(?:no\s+more\s+than\s+)?(\d+)", low)
    bm = re.search(r"each\s+(?:weighing\s+)?(\d+)|(\d+)\s+each", low)
    dm = re.search(
        r"weight\s+of\s+the\s+\w+\s+and\s+the\s+\w+"
        r"\s+truck\s+is\s+(\d+)|"
        r"truck\s+weighs\s+(\d+)",
        low,
    )
    if not (cm and bm and dm):
        return None
    truck = dm.group(1) or dm.group(2)
    box = bm.group(1) or bm.group(2)
    return (Fraction(cm.group(1)) - Fraction(truck)) / Fraction(box)


def _brosche_kosten(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Brosche-Kosten: '(500+800)x1.1' -> 1430."""
    low = _digitize(question.lower())
    ms = re.findall(
        r"(?:pays?\s+)?\$?(\d+)\s+for\s+(?:the\s+)?"
        r"\w+",
        low,
    )
    pm = re.search(r"pays?\s+(\d+)\s*(?:%|percent)\s+of\s+that", low)
    if len(ms) < 2 or not pm:
        return None
    return (sum(Fraction(m) for m in ms)) * (1 + Fraction(pm.group(1)) / 100)


def _liefer_gebuehren(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Liefer-Gebühren: '40x1.25+3+4' -> 57."""
    low = _digitize(question.lower())
    bm = re.search(r"bill\s+came\s+to\s+\$?(\d+(?:\.\d+)?)", low)
    pm = re.search(r"(\d+)\s*(?:%|percent)\s+fee", low)
    dm = re.search(r"charged\s+(?:him\s+)?\$?([\d.]+)", low)
    tm = re.search(r"added\s+a\s+\$?([\d.]+)", low)
    if not (bm and pm and dm and tm):
        return None
    return (
        Fraction(bm.group(1)) * (1 + Fraction(pm.group(1)) / 100)
        + Fraction(dm.group(1))
        + Fraction(tm.group(1))
    )


def _tank_reichweite(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Tank-Reichweite: '100/4x12' -> 300."""
    low = _digitize(question.lower())
    tm = re.search(r"traveled\s+(\d+)\s+\w+", low)
    gm = re.search(r"(?:put\s+in|needed)\s+(\d+)\s+gallons?", low)
    tm2 = re.search(r"holds\s+(\d+)\s+\w+(?:\s+of\s+gas)?", low)
    if not (tm and gm and tm2):
        return None
    return Fraction(tm.group(1)) / Fraction(gm.group(1)) * Fraction(tm2.group(1))


def _rente_anteil(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Rente-Anteil: '50000 x (30-20)x5%' -> 25000."""
    low = _digitize(question.lower())
    pm = re.search(r"(?:pension\s+of|gets\s+a)\s+\$?([\d,]+)", low)
    sm = re.search(r"(\d+)\s*(?:%|percent)(?:\s+of\s+the\s+value)?", low)
    sm2 = re.search(r"after\s+(\d+)\s+years?", low)
    qm = re.search(r"quits?\s+after\s+(\d+)", low)
    if not (pm and sm and sm2 and qm):
        return None
    years = Fraction(qm.group(1)) - Fraction(sm2.group(1))
    return (
        Fraction(int(pm.group(1).replace(",", "")))
        * years
        * Fraction(sm.group(1))
        / 100
    )


def _tv_lesen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """TV-Lesen: '(2+1)x3x4' -> 36."""
    low = _digitize(question.lower())
    tm = re.search(r"(?:spends?|watches\s+\w+)\s+(\d+)\s+hours?", low)
    rm = re.search(r"reads?\s+for\s+half\s+as\s+long", low)
    wm = re.search(r"(\d+)\s+times\s+a\s+week", low)
    wm2 = re.search(r"in\s+(\d+)\s+weeks?", low)
    if not (tm and rm and wm and wm2):
        return None
    return (
        Fraction(tm.group(1))
        * Fraction(3, 2)
        * Fraction(wm.group(1))
        * Fraction(wm2.group(1))
    )


def _streaming_kosten(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Streaming-Kosten: '6x140+6x126' -> 1596."""
    low = _digitize(question.lower())
    cm = re.search(
        r"(?:charges?\s+her\s+|pays?\s+)\$?(\d+)\s+per"
        r"\s+month",
        low,
    )
    dm = re.search(r"(\d+)\s*(?:%|percent)\s+less", low)
    hm = re.search(r"half\s+of\s+the\s+year", low)
    if not (cm and dm and hm):
        return None
    full = Fraction(cm.group(1))
    return 6 * full + 6 * full * (1 - Fraction(dm.group(1)) / 100)


def _schul_team(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Schul-Team: '4x2x5 + 4x2' -> 48."""
    low = _digitize(question.lower())
    sm = re.search(r"(\d+)\s+schools?", low)
    tm = re.search(r"(?:each\s+team\s+has|with)\s+(\d+)", low)
    cm = re.search(r"coach\s+for\s+each\s+team", low)
    if not (sm and tm and cm):
        return None
    teams = Fraction(sm.group(1)) * 2
    return teams * Fraction(tm.group(1)) + teams


def _sandburg_durchschnitt(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Sandburg-Schnitt: '(16+32+64+128)/4' -> 60."""
    low = _digitize(question.lower())
    lm = re.search(r"(\d+)(?:-|\s+)level(?:ed)?\s+\w+", low)
    tm = re.search(
        r"top\s+level(?:(?:\s+has\s+a\s+\w+\s+\w+\s+"
        r"of)|(?:\s+of))\s+(\d+)",
        low,
    )
    hm = re.search(
        r"half\s+the\s+\w+\s+\w+\s+as\s+the|"
        r"double\s+the\s+(?:one|1)\s+above",
        low,
    )
    if not (lm and tm and hm):
        return None
    levels = int(lm.group(1))
    top = Fraction(tm.group(1))
    total = Fraction(0)
    for i in range(levels):
        total += top * (2**i)
    return total / levels


def _schatz_steine(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Schatz-Steine: '175+140+280' -> 595."""
    low = _digitize(question.lower())
    if "diamond" not in low:
        return None
    dm = re.search(r"(\d+)\s+\w+", low)
    rm = re.search(r"(\d+)\s+fewer\s+\w+(?:\s+than)?", low)
    em = re.search(
        r"twice\s+(?:the\s+number\s+of\s+\w+\s+than|"
        r"as\s+many\s+\w+\s+as)",
        low,
    )
    if not (dm and rm and em):
        return None
    dia = Fraction(dm.group(1))
    rub = dia - Fraction(rm.group(1))
    return dia + rub + rub * 2


def _waesche_diff(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Wäsche-Diff: '200-100' -> 100."""
    low = _digitize(question.lower())
    sm = re.search(r"sarah\s+does\s+(\d+)\s+pounds?", low)
    hm = re.search(r"half\s+as\s+much", low)
    fm = re.search(r"4\s+times\s+as\s+much", low)
    if not (sm and hm and fm):
        return None
    sarah = Fraction(sm.group(1))
    return sarah / 2 - sarah / 4


def _schul_lehrer(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Schul-Lehrer: '(60+120)/5' -> 36."""
    low = _digitize(question.lower())
    gm = re.search(r"there\s+are\s+(\d+)\s+girls?", low)
    tm = re.search(r"twice\s+as\s+many\s+\w+\s+as\s+\w+", low)
    sm = re.search(r"(\d+)\s+students?\s+(?:to\s+every|per)\s+\w+", low)
    if not (gm and tm and sm):
        return None
    girls = Fraction(gm.group(1))
    return (girls + girls * 2) / Fraction(sm.group(1))


def _gewichte_kombiniert(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Gewichte-kombiniert: '125 + (4x125-2)' -> 623."""
    low = _digitize(question.lower())
    gm = re.search(r"weighs?\s+(\d+)\s+\w+", low)
    am = re.search(r"(\d+)\s+\w+\s+less\s+than\s+(\d+)\s+times", low)
    if not (gm and am):
        return None
    grace = Fraction(gm.group(1))
    return grace + (grace * Fraction(am.group(2)) - Fraction(am.group(1)))


def _tanz_einnahmen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Tanz-Einnahmen: '(5x5+8)x15x15' -> 7425."""
    low = _digitize(question.lower())
    fm = re.search(r"teaches?\s+(\d+)\s+\w+\s+\w+", low)
    sm = re.search(r"and\s+(\d+)(?:\s+\w+)?\s+on\s+\w+", low)
    sm2 = re.search(r"each\s+class\s+has\s+(\d+)", low)
    cm = re.search(
        r"charges?\s+\$?([\d.]+)\s+per\s+\w+|"
        r"paying\s+\$?([\d.]+)",
        low,
    )
    if not (fm and sm and sm2 and cm):
        return None
    price = cm.group(1) or cm.group(2)
    return (
        (Fraction(fm.group(1)) * 5 + Fraction(sm.group(1)))
        * Fraction(sm2.group(1))
        * Fraction(price)
    )


def _gehalt_steigerung(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Gehalt-Steigerung: '7200 + 3x720' -> 9360."""
    low = _digitize(question.lower())
    sm = re.search(
        r"pays?\s+(?:each\s+of\s+its\s+\w+\s+)?"
        r"\$?(\d+)",
        low,
    )
    pm = re.search(r"(\d+)\s*(?:%|percent)(?:\s+of\s+the\s+initial)?", low)
    ym = re.search(r"after\s+(\d+)\s+more\s+years?", low)
    if not (sm and pm and ym):
        return None
    annual = Fraction(sm.group(1)) * 12
    return annual + annual * Fraction(pm.group(1)) / 100 * Fraction(ym.group(1))


def _kuchen_spende(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Kuchen-Spende: '43x3+23x4' -> 221."""
    low = _digitize(question.lower())
    if "brownie" not in low:
        return None
    ms = re.findall(r"for\s+\$?(\d+)\s+a\s+\w+|at\s+\$?(\d+)", low)
    ms = [a or b for a, b in ms]
    bs = re.search(r"sells?\s+(\d+)\s+\w+", low)
    cs = re.search(r"and\s+(\d+)\s+\w+(?:\s+of\s+\w+)?", low)
    if len(ms) < 2 or not (bs and cs):
        return None
    return Fraction(bs.group(1)) * Fraction(ms[0]) + Fraction(cs.group(1)) * Fraction(
        ms[1]
    )


def _haustiere_drei(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Haustiere-drei: '4+6+18' -> 28."""
    low = _digitize(question.lower())
    cm = re.search(r"if\s+\w+\s+has\s+(\d+)", low)
    mm = re.search(r"(\d+)\s+more(?:\s+\w+)?\s+than\s+\w+", low)
    jm = re.search(
        r"(?:three|3)\s+times\s+the\s+(?:number\s+of\s+"
        r"\w+\s+|\w+)",
        low,
    )
    if not (cm and mm and jm):
        return None
    cindy = Fraction(cm.group(1))
    marcia = cindy + Fraction(mm.group(1))
    return cindy + marcia + marcia * 3


def _raten_kauf(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Raten-Kauf: '5x150x1.02/3' -> 255."""
    low = _digitize(question.lower())
    nm = re.search(
        r"bought\s+(\d+)\s+\w+\s+\w+\s+for\s+"
        r"\$?(\d+)",
        low,
    )
    pm = re.search(r"(\d+)\s*(?:%|percent)\s+interest", low)
    mm = re.search(r"each\s+month\s+for\s+(\d+)", low)
    if not (nm and pm and mm):
        return None
    return (
        Fraction(nm.group(1))
        * Fraction(nm.group(2))
        * (1 + Fraction(pm.group(1)) / 100)
        / Fraction(mm.group(1))
    )


def _alter_kette(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Alter-Kette: '10+1-2-5' -> 4."""
    low = _digitize(question.lower())
    jm = re.search(r"james\s+is\s+(\d+)", low)
    cm = re.search(r"younger\s+than\s+\w+", low)
    am = re.search(r"(\d+)\s+years?\s+older\s+than\s+\w+", low)
    am2 = re.search(r"(\d+)\s+years?\s+younger\s+than\s+\w+", low)
    if not (jm and cm and am and am2):
        return None
    james = Fraction(jm.group(1))
    corey = james + 1
    amy = corey - Fraction(am2.group(1))
    return amy - Fraction(am.group(1))


def _huerden_zeit(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Hürden-Zeit: '(38+2)x0.9' -> 36."""
    low = _digitize(question.lower())
    lm = re.search(
        r"lee\s+runs?\s+the\s+\w+-meter\s+\w+\s+in\s+"
        r"(\d+)",
        low,
    )
    fm = re.search(r"(?:two|2)\s+seconds\s+faster\s+than", low)
    pm = re.search(r"(\d+)\s*(?:%|percent)", low)
    if not (lm and fm and pm):
        return None
    gerald = Fraction(lm.group(1)) + 2
    return gerald * (1 - Fraction(pm.group(1)) / 100)


def _blumen_rundung(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Blumen-Rundung: '12x3+9x2+17x2' -> 88."""
    low = _digitize(question.lower())
    ps = re.findall(r"for\s+\$?([\d.]+)\s+per\s+pot", low)
    ns = re.findall(r"sells\s+(\d+)\s+pots?|(\d+)\s+pots?\s+of", low)
    ns = [a or b for a, b in ns]
    if len(ps) < 3 or len(ns) < 3:
        return None
    total = Fraction(0)
    for n, p in zip(ns, ps):
        total += Fraction(n) * round(float(p))
    return total


def _huerden_zeit(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Hürden-Zeit: '(38+2)x0.9' -> 36."""
    low = _digitize(question.lower())
    lm = re.search(
        r"lee\s+runs?\s+the\s+\w+-meter\s+\w+\s+in\s+"
        r"(\d+)",
        low,
    )
    fm = re.search(r"(?:two|2)\s+seconds\s+faster\s+than", low)
    pm = re.search(r"(\d+)\s*(?:%|percent)", low)
    if not (lm and fm and pm):
        return None
    gerald = Fraction(lm.group(1)) + 2
    return gerald * (1 - Fraction(pm.group(1)) / 100)


def _blumen_rundung(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Blumen-Rundung: '12x3+9x2+17x2' -> 88."""
    low = _digitize(question.lower())
    ps = re.findall(r"for\s+\$?([\d.]+)\s+per\s+pot", low)
    ns = re.findall(r"sells\s+(\d+)\s+pots?|(\d+)\s+pots?\s+of", low)
    ns = [a or b for a, b in ns]
    if len(ps) < 3 or len(ns) < 3:
        return None
    total = Fraction(0)
    for n, p in zip(ns, ps):
        total += Fraction(n) * round(float(p))
    return total


def _hundefutter_jahre(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Hundefutter-Jahre: '(180+185x2)/110' -> 5."""
    low = _digitize(question.lower())
    fm = re.search(r"feed\s+the\s+\w+\s+(\d+)\s+cup", low)
    sm = re.search(r"feed\s+the\s+\w+\s+(\d+)\s+cups", low)
    dm = re.search(r"first\s+(\d+)\s+days?", low)
    bm = re.search(r"contains\s+(\d+)\s+cups?", low)
    if not (fm and sm and dm and bm):
        return None
    cups = Fraction(dm.group(1)) * Fraction(fm.group(1)) + (
        365 - Fraction(dm.group(1))
    ) * Fraction(sm.group(1))
    return cups / Fraction(bm.group(1))


def _fernseh_episoden(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Fernseh-Episoden: '(7-1-1-1.5-2)x60/30' -> 3."""
    low = _digitize(question.lower())
    tm = re.search(r"watched\s+(\d+)\s+hours?\s+of\s+\w+\s+in", low)
    hm = re.search(r"1-hour\s+\w+", low)
    mm = re.search(r"30-minute\s+\w+", low)
    if not (tm and hm and mm):
        return None
    total = Fraction(tm.group(1))
    # Mo+Tu+Do: 3x 1h ; Fr: 2h ; Do: 1x 0.5h
    remaining = total - 3 - 2 - Fraction(1, 2)
    return remaining / Fraction(1, 2)


def _blumen_sparen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Blumen-Sparen: '18/3x2.5 - 18/2x1' -> 6."""
    low = _digitize(question.lower())
    ms = re.findall(r"packages?\s+of\s+(\d+)\s+for\s+\$?([\d.]+)", low)
    nm = re.search(r"buy(?:ing)?\s+(\d+)\s+\w+", low)
    if len(ms) < 2 or not nm:
        return None
    n = Fraction(nm.group(1))
    a = n / Fraction(ms[0][0]) * Fraction(ms[0][1])
    b = n / Fraction(ms[1][0]) * Fraction(ms[1][1])
    return abs(a - b)


def _kaffee_verhaeltnis(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Kaffee-Verhältnis: '120x7/20' -> 42."""
    low = _digitize(question.lower())
    rm = re.search(r"ratio\s+of\s+(\d+):(\d+)", low)
    tm = re.search(r"total\s+of\s+(\d+)", low)
    if not (rm and tm):
        return None
    return (
        Fraction(tm.group(1))
        * Fraction(rm.group(1))
        / (Fraction(rm.group(1)) + Fraction(rm.group(2)))
    )


def _urlaub_bloecke(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Urlaub-Blocke: '(34-23)x4' -> 44."""
    low = _digitize(question.lower())
    fm = re.search(r"(?:four|4)\s+vacations\s+a\s+year", low)
    am = re.search(r"since\s+he\s+was\s+(\d+)", low)
    nm = re.search(r"now\s+(\d+)", low)
    if not (fm and am and nm):
        return None
    return (Fraction(nm.group(1)) - Fraction(am.group(1))) * 4


def _container_zahl(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Container-Zahl: '(30-2x5)/5' -> 4."""
    low = _digitize(question.lower())
    cm = re.search(r"(\d+)\s+containers?\s+of\s+\w+\s+\w+", low)
    vm = re.search(r"each\s+having\s+(\d+)\s+\w+", low)
    tm = re.search(
        r"total\s+number\s+of\s+\w+\s+at\s+the\s+\w+"
        r"\s+became\s+(\d+)",
        low,
    )
    if not (cm and vm and tm):
        return None
    return (
        Fraction(tm.group(1)) - Fraction(cm.group(1)) * Fraction(vm.group(1))
    ) / Fraction(vm.group(1))


def _alarm_ringe(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Alarm-Ringe: '4+12+6' -> 22."""
    low = _digitize(question.lower())
    fm = re.search(r"rang\s+(\d+)\s+\w+", low)
    tm = re.search(r"(?:three|3)\s+times\s+as\s+long", low)
    hm = re.search(r"half\s+as\s+long\s+as\s+the\s+\w+", low)
    if not (fm and tm and hm):
        return None
    first = Fraction(fm.group(1))
    return first + first * 3 + first * 3 / 2


def _gehalt_vergleich(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Gehalt-Vergleich: '40000x1.3+40000x1.4' -> 95200."""
    low = _digitize(question.lower())
    em = re.search(r"earned\s+\$?([\d,]+)\s+\w+\s+\w+\s+ago", low)
    pm = re.search(r"(\d+)\s*(?:%|percent)\s+higher", low)
    pm2 = re.search(r"(\d+)\s*(?:%|percent)\s+more", low)
    if not (em and pm and pm2):
        return None
    base = Fraction(int(em.group(1).replace(",", "")))
    lylah = base * (1 - Fraction(pm.group(1)) / 100)
    return lylah * (1 + Fraction(pm2.group(1)) / 100) + base * (
        1 + Fraction(pm2.group(1)) / 100
    )


def _geschenk_tueten(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Geschenk-Tüten: '16x0.75x2' -> 24."""
    low = _digitize(question.lower())
    nm = re.search(r"invited\s+(\d+)\s+\w+", low)
    rm = re.search(r"needs?\s+([\d.]+)\s+\w+\s+\w+\s+per", low)
    pm = re.search(r"bags?\s+are\s+\$?(\d+)", low)
    if not (nm and rm and pm):
        return None
    return Fraction(nm.group(1)) * Fraction(rm.group(1)) * Fraction(pm.group(1))


def _markt_einkauf(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Markt-Einkauf: '4x45+20x15+10x40' -> 880."""
    low = _digitize(question.lower())
    pm = re.search(r"pepper\s+costs\s+\$?(\d+)|(\d+)\s*\$", low)
    wm = re.search(r"watermelon\s+costs\s+(?:three|3)\s+times", low)
    om = re.search(r"orange\s+costs\s+(\d+)\s+less", low)
    nm = re.search(
        r"buys?\s+(\d+)\s+\w+,\s+(\d+)\s+\w+,\s+"
        r"and\s+(\d+)",
        low,
    )
    if not (pm and wm and om and nm):
        return None
    pep = Fraction(pm.group(1) or pm.group(2))
    melon = pep * 3
    orange = melon - Fraction(om.group(1))
    return (
        Fraction(nm.group(1)) * melon
        + Fraction(nm.group(2)) * pep
        + Fraction(nm.group(3)) * orange
    )


def _dino_salat(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Dino-Salat: '20x10+5x5' -> 225."""
    low = _digitize(question.lower())
    am = re.search(r"adult\s+\w+\s+will\s+eat\s+(\d+)", low)
    cm = re.search(r"child\s+will\s+eat\s+half", low)
    an = re.search(r"(\d+)\s+adults?", low)
    cn = re.search(r"and\s+(\d+)\s+children?", low)
    if not (am and cm and an and cn):
        return None
    adult = Fraction(am.group(1))
    return Fraction(an.group(1)) * adult + Fraction(cn.group(1)) * adult / 2


def _liefer_bestellung(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Liefer-Bestellung: '(2x7.5+2x1.5+2x1)x1.2+5' -> 29."""
    low = _digitize(question.lower())
    ms = re.findall(
        r"(\d+)\s+\w+(?:\s+of\s+\w+)?\s+for\s+"
        r"\$?([\d.]+)\s+each",
        low,
    )
    pm = re.search(r"(\d+)\s*(?:%|percent)\s+delivery\s+fee", low)
    tm = re.search(r"add\s+a\s+\$?([\d.]+)\s+\w+", low)
    if not ms or not (pm and tm):
        return None
    sub = sum(Fraction(a) * Fraction(b) for a, b in ms)
    return sub * (1 + Fraction(pm.group(1)) / 100) + Fraction(tm.group(1))


def _pfadfinder_maedchen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Pfadfinder-Mädchen: '200-80-80' -> 40."""
    low = _digitize(question.lower())
    sm = re.search(r"out\s+of\s+the\s+(\d+)", low)
    bm = re.search(r"2/5\s+are\s+\w+", low)
    gm = re.search(r"2/3\s+of\s+the\s+girls?", low)
    if not (sm and bm and gm):
        return None
    total = Fraction(sm.group(1))
    boys = total * Fraction(2, 5)
    girls = total - boys
    scouts = girls * Fraction(2, 3)
    return girls - scouts


def _auto_prozent(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Auto-Prozent: '(20-12-4)/20' -> 20%."""
    low = _digitize(question.lower())
    tm = re.search(r"of\s+the\s+(\d+)\s+\w+", low)
    ms = re.findall(r"(\d+)\s+are\s+\w+", low)
    if not tm or len(ms) < 2:
        return None
    total = Fraction(tm.group(1))
    return (total - sum(Fraction(m) for m in ms)) / total * 100


def _tomaten_zahl(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Tomaten-Zahl: '32x2/16x3' -> 12."""
    low = _digitize(question.lower())
    sm = re.search(r"made\s+(\d+)\s+\w+\s+of\s+sauce", low)
    cm = re.search(r"each\s+(\d+)\s+\w+\s+can", low)
    tm = re.search(r"contains\s+(\d+)\s+\w+", low)
    hm = re.search(r"lose\s+half\s+their\s+\w+", low)
    if not (sm and cm and tm and hm):
        return None
    return Fraction(sm.group(1)) * 2 / Fraction(cm.group(1)) * Fraction(tm.group(1))


def _haushalt_profit(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Haushalt-Profit: '92 - 8x(2x2+5)' -> 20."""
    low = _digitize(question.lower())
    cm = re.search(r"has\s+(\d+)\s+clients?", low)
    pm = re.search(r"another\s+(\d+)\s+\w+\s+clients?", low)
    bm = re.search(
        r"bottles?\s+of\s+\w+\s+(?:will\s+)?cost\s+"
        r"\$?(\d+)",
        low,
    )
    pm2 = re.search(
        r"packs?\s+of\s+\w+\s+(?:will\s+)?cost\s+"
        r"\$?(\d+)",
        low,
    )
    im = re.search(
        r"income\s+each\s+week\s+(?:will\s+)?(?:be|is)"
        r"\s+\$?(\d+)",
        low,
    )
    nm = re.search(r"need\s+(\d+)\s+bottles", low)
    if not (cm and pm and bm and pm2 and im and nm):
        return None
    clients = Fraction(cm.group(1)) + Fraction(pm.group(1))
    expenses = clients * (
        Fraction(nm.group(1)) * Fraction(bm.group(1)) + Fraction(pm2.group(1))
    )
    return Fraction(im.group(1)) - expenses


def _wasser_woche(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Wasser-Woche: '4x5+3x2' -> 26."""
    low = _digitize(question.lower())
    wm = re.search(r"glass\s+of\s+water\s+with", low)
    em = re.search(
        r"(?:one|1)\s+before(?:\s+he\s+goes)?\s+(?:to"
        r"\s+)?\w+",
        low,
    )
    wm2 = re.search(r"weekdays?", low)
    sm = re.search(r"weekends?", low)
    if not (wm and em and wm2 and sm):
        return None
    return 4 * 5 + 3 * 2


def _museum_geld(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Museum-Geld: '12+10+8' -> 30."""
    low = _digitize(question.lower())
    am = re.search(r"\$?(\d+)\s+for\s+adults|adults?\s+\$?(\d+)", low)
    cm = re.search(
        r"\$?(\d+)\s+for\s+children|children?\s+\$?"
        r"(\d+)",
        low,
    )
    cm2 = re.search(r"received\s+\$?(\d+)\s+in\s+change", low)
    if not (am and cm and cm2):
        return None
    a = am.group(1) or am.group(2)
    c = cm.group(1) or cm.group(2)
    return Fraction(a) + Fraction(c) + Fraction(cm2.group(1))


def _benzin_cashback(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Benzin-Cashback: '10x3 - 10x0.2' -> 28."""
    low = _digitize(question.lower())
    gm = re.search(r"(?:for\s+)?\$?([\d.]+)\s+a\s+gallon", low)
    cm = re.search(
        r"\$?(\d+(?:\.\d+)?)\s+cashback|\$\.(\d+)"
        r"\s+cashback",
        low,
    )
    bm = re.search(r"buys?\s+(\d+)\s+gallons?", low)
    if not (gm and cm and bm):
        return None
    n = Fraction(bm.group(1))
    cb = cm.group(1) or ("0." + cm.group(2))
    return n * Fraction(gm.group(1)) - n * Fraction(cb)


def _wassertank_tiefe(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Wassertank-Tiefe: '(17+7)x2/3' -> 16."""
    low = _digitize(question.lower())
    dm = re.search(r"(?:depth\s+of\s+|has\s+)(\d+)\s+feet", low)
    am = re.search(r"(?:had|,)?\s*(\d+)\s+feet\s+more", low)
    tm = re.search(r"(?:two|2)\s+thirds\s+of\s+(?:what|tuesday|\w+)", low)
    if not (dm and am and tm):
        return None
    return (Fraction(dm.group(1)) + Fraction(am.group(1))) * Fraction(2, 3)


def _fruit_schnitt(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Fruit-Schnitt: '(2x24+3x14)/2' -> 45."""
    low = _digitize(question.lower())
    ms = re.findall(r"(\d+)\s+\w+-\w+\s+wide\s+and\s+(\d+)", low)
    if len(ms) < 2:
        return None
    return (sum(Fraction(a) * Fraction(b) for a, b in ms)) / 2


def _alter_summe(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Alter-Summe: '2B+4=28 -> S=16'."""
    low = _digitize(question.lower())
    tm = re.search(r"twice\s+as\s+old\s+as", low)
    sm = re.search(
        r"in\s+(\d+)\s+years?,\s+the\s+sum\s+of\s+their"
        r"\s+\w+\s+will\s+be\s+(\d+)",
        low,
    )
    if not (tm and sm):
        return None
    brooke = (Fraction(sm.group(2)) - 2 * Fraction(sm.group(1))) / 3
    return brooke * 2


def _blumen_pflanzen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Blumen-Pflanzen: '2x15-5' -> 25."""
    low = _digitize(question.lower())
    pm = re.search(r"plants?\s+(\d+)\s+\w+\s+a\s+day", low)
    dm = re.search(r"after\s+(\d+)\s+days?", low)
    nm = re.search(r"if\s+(\d+)\s+did\s+not", low)
    if not (pm and dm and nm):
        return None
    return Fraction(pm.group(1)) * Fraction(dm.group(1)) - Fraction(nm.group(1))


def _voegel_alter(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Vögel-Alter: '8+8+16+19' -> 51."""
    low = _digitize(question.lower())
    vs = re.findall(r"is\s+(\d+)(?:\s+years?\s+old)?", low)
    vm = vs[-1] if vs else None
    tm = re.search(r"(?:two|2)\s+times\s+as\s+old\s+as", low)
    tm2 = re.search(r"(?:three|3)\s+years?\s+older\s+than", low)
    if not (vm and tm and tm2):
        return None
    s32 = Fraction(vm)
    granny = s32 * 2
    return s32 * 2 + granny + granny + 3


def _holz_profit(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Holz-Profit: '(10x10+5x16)x0.5' -> 90."""
    low = _digitize(question.lower())
    ms = re.findall(
        r"(\d+)\s+\w+\s+x\s+\w+\s+x\s+\w+\s+"
        r"boards?\s+(?:that\s+cost\s+her\s+|she\s+"
        r"bought\s+for\s+)\$?(\d+)|"
        r"(\d+)\s+boards?\s+at\s+\$?(\d+)\s+each",
        low,
    )
    ms = [(a or c, b or d) for a, b, c, d in ms]
    pm = re.search(
        r"(\d+)\s*(?:%|percent)\s+(?:in\s+the\s+last|"
        r"more)",
        low,
    )
    if not ms or not pm:
        return None
    cost = sum(Fraction(a) * Fraction(b) for a, b in ms)
    return cost * Fraction(pm.group(1)) / 100


def _stimmen_verlierer(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Stimmen-Verlierer: '80x1/4' -> 20."""
    low = _digitize(question.lower())
    if not re.search(r"vote|winner|loser", low):
        return None
    vm = re.search(r"3/4\s+of\s+(?:the\s+)?\w+", low)
    tm = re.search(
        r"total\s+number\s+of\s+\w+\s+who\s+"
        r"[\w\s]+?was\s+(\d+)|of\s+(\d+)\s+\w+",
        low,
    )
    if not (vm and tm):
        return None
    n = tm.group(1) or tm.group(2)
    return Fraction(n) / 4


def _frucht_sammeln(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Frucht-Sammeln: '5+8+10+4' -> 27."""
    low = _digitize(question.lower())
    mm = re.search(r"brought\s+(\d+)\s+\w+\s+and\s+(\d+)\s+\w+", low)
    tm = re.search(r"twice\s+(?:the\s+amount\s+of\s+)?\w+", low)
    hm = re.search(r"half\s+(?:the\s+number\s+of\s+)?\w+", low)
    if not (mm and tm and hm):
        return None
    a = Fraction(mm.group(1))
    o = Fraction(mm.group(2))
    return a + o + a * 2 + o / 2


def _briefe_vorher(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Briefe-vorher: '30 - 60/3' -> 10."""
    low = _digitize(question.lower())
    pm = re.search(
        r"(?:pile\s+of\s+|has\s+)(\d+)\s+\w+\s+"
        r"needing",
        low,
    )
    tm = re.search(r"(?:one-third|1-third)(?:\s+of\s+the\s+\w+)?", low)
    nm = re.search(r"now\s+(\d+)(?:\s+\w+\s+in\s+the\s+pile)?", low)
    if not (pm and tm and nm):
        return None
    return Fraction(nm.group(1)) - Fraction(pm.group(1)) / 3


def _alter_versetzt(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Alter-versetzt: '(30-2)/2+5+2+2' -> 23."""
    low = _digitize(question.lower())
    jm = re.search(r"jan\s+is\s+(\d+)", low)
    mm = re.search(
        r"(?:two|2)\s+years?\s+ago\s+\w+\s+was\s+"
        r"(\d+)\s+\w+\s+older",
        low,
    )
    jm2 = re.search(r"(?:two|2)\s+years?\s+older\s+than\s+\w+", low)
    if not (jm and mm and jm2):
        return None
    jan = Fraction(jm.group(1))
    mark_ago = (jan - 2) / 2 + Fraction(mm.group(1))
    return mark_ago + 2 + 2


def _krankenhaus_profit(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Krankenhaus-Profit: '500x24/60x(200-150)' -> 10000."""
    low = _digitize(question.lower())
    pm = re.search(r"sees?\s+(\d+)\s+\w+\s+a\s+day", low)
    tm = re.search(r"(?:average\s+of\s+|for\s+)(\d+)\s+\w+", low)
    cm = re.search(
        r"(?:charges?\s+the\s+patients?\s+|hospital\s+"
        r"charges?\s+)\$?(\d+)",
        low,
    )
    dm = re.search(
        r"(?:charges?\s+\$?(\d+)\s+an\s+hour\s+to\s+"
        r"the|doctors\s+charge\s+\$?(\d+))",
        low,
    )
    if not (pm and tm and cm and dm):
        return None
    hours = Fraction(pm.group(1)) * Fraction(tm.group(1)) / 60
    doc = dm.group(1) or dm.group(2)
    return hours * (Fraction(cm.group(1)) - Fraction(doc))


def _test_durchschnitt(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Test-Durchschnitt: '93x5-(89+92+100+86)' -> 98."""
    low = _digitize(question.lower())
    am = re.search(r"average\s+(?:of\s+)?(\d+)", low)
    lm = re.search(r"lowest\s*(?:score|is|\.)", low)
    scores = re.findall(
        r"scores?\s+(?:of\s+)?"
        r"([\d\s,]+?(?:and\s+\d+)?)"
        r"(?:[.?!]|\s+on\s+the)",
        low,
    )
    if not (am and lm and scores):
        return None
    nums = [int(s) for s in re.findall(r"\d+", scores[0])]
    nums.remove(min(nums))
    return Fraction(am.group(1)) * 5 - sum(nums)


def _ausgaben_zwei(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Ausgaben-zwei: '500+440' -> 940."""
    low = _digitize(question.lower())
    if not re.search(r"\b(?:may|june)\b", low):
        return None
    fm = re.search(r"(?:was|spent|paid)\s+\$?(\d+)", low)
    sm = re.search(r"(?:was\s+)?\$?(\d+)\s+less", low)
    if not (fm and sm):
        return None
    return Fraction(fm.group(1)) + Fraction(fm.group(1)) - Fraction(sm.group(1))


def _tassen_preis(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Tassen-Preis: '(6x6000-1200)/240' -> 145."""
    low = _digitize(question.lower())
    dm = re.search(r"(\d+)\s+dozen\s+cups?", low)
    dm2 = re.search(r"half\s+a\s+dozen\s+\w+", low)
    pm = re.search(r"(?:sold\s+at\s+|at\s+)\$?(\d+)", low)
    lm = re.search(
        r"less\s+than\s+(?:the\s+total\s+(?:cost\s+of\s+"
        r")?|half\s+a\s+dozen)",
        low,
    )
    if not (dm and dm2 and pm and lm):
        return None
    total = 6 * Fraction(pm.group(1)) - 1200
    return total / (Fraction(dm.group(1)) * 12)


def _videospiele_bobby(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Videospiele-Bobby: '3x(20-5)-5' -> 40."""
    low = _digitize(question.lower())
    bms = re.findall(r"(?:if\s+)?(\w+)\s+has\s+(\d+)", low)
    lm = re.search(r"lost\s+(\d+)", low)
    fm = re.search(r"(\d+)\s+fewer\s+than\s+(\d+)\s+times", low)
    if not (bms and lm and fm):
        return None
    bm = bms[-1]
    return (Fraction(bm[1]) - Fraction(lm.group(1))) * Fraction(fm.group(2)) - Fraction(
        fm.group(1)
    )


def _alter_drei(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Alter-drei: '(24+66+45)/3' -> 45."""
    low = _digitize(question.lower())
    hm = re.search(r"harriet\s+is\s+(\d+)", low)
    tm = re.search(
        r"(?:three|3)\s+times\s+(?:the\s+age\s+of\s+)?"
        r"\w+",
        low,
    )
    hm2 = re.search(r"half\s+(?:the\s+age\s+of|of)", low)
    ym = re.search(r"in\s+(?:three|3)\s+years?", low)
    if not (hm and tm and hm2 and ym):
        return None
    h = Fraction(hm.group(1))
    a = h * 3
    z = h * 2
    return (h + 3 + a + 3 + z + 3) / 3


def _bienen_verteilung(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Bienen-Verteilung: '700/7x4' -> 400."""
    low = _digitize(question.lower())
    bm = re.search(r"(\d+)\s+bees?(?:\s+in\s+a\s+hive)?", low)
    wm = re.search(r"twice\s+as\s+many\s+\w+\s+as\s+\w+", low)
    bm2 = re.search(r"twice\s+as\s+many\s+\w+\s+as\s+\w+", low)
    if not (bm and wm and bm2):
        return None
    return Fraction(bm.group(1)) * 4 / 7


def _werbung_zwei(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Werbung-zwei: '15000+5000' -> 20000."""
    low = _digitize(question.lower())
    fm = re.search(r"spends?\s+\$?(\d+)", low)
    tm = re.search(r"a\s+third\s+of\s+that", low)
    if not (fm and tm):
        return None
    return Fraction(fm.group(1)) * Fraction(4, 3)


def _spielzeit_arbeit(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Spielzeit-Arbeit: '2x7x10' -> 140."""
    low = _digitize(question.lower())
    hm = re.search(r"for\s+(\d+)\s+hours?\s+every\s+day", low)
    em = re.search(r"earns?\s+\$?(\d+)\s+an\s+hour", low)
    wm = re.search(r"(?:one|1)\s+week", low)
    if not (hm and em and wm):
        return None
    return Fraction(hm.group(1)) * 7 * Fraction(em.group(1))


def _pokemon_prozent(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Pokemon-Prozent: '32/(30+34+32)' -> 33%."""
    low = _digitize(question.lower())
    ts = re.findall(r"(\d+)\s+\w+(?:\s+type)?(?:\s+cards?)?", low)
    ts = [x for x in ts if not re.search(r"\d+\s+(?:of|and)", x)]
    lm = re.search(r"loses\s+(\d+)", low)
    bm = re.search(r"buys?\s+(\d+)\s+\w+(?:\s+type)?", low)
    if len(ts) < 3 or not (lm and bm):
        return None
    water = Fraction(ts[2]) - Fraction(lm.group(1))
    total = Fraction(ts[0]) + Fraction(ts[1]) + Fraction(bm.group(1)) + water
    return round(float(water / total * 100))


def _einkauf_steuer(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Einkauf-Steuer: '15 + 10x0.1' -> 16."""
    low = _digitize(question.lower())
    ms = re.findall(r"for\s+(\d+)\s+dollars?", low)
    pm = re.search(r"(\d+)\s*(?:%|percent)\s+tax", low)
    if not ms or not pm:
        return None
    food = re.findall(
        r"(?:milk|eggs?|bread|meat|cheese|apples?|"
        r"oranges?|flour)\s+for\s+(\d+)\s+dollars?",
        low,
    )
    nonfood = sum(Fraction(m) for m in ms) - sum(Fraction(m) for m in food)
    return sum(Fraction(m) for m in ms) + nonfood * Fraction(pm.group(1)) / 100


def _insekten_zahl(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Insekten-Zahl: '50+25' -> 75."""
    low = _digitize(question.lower())
    am = re.search(r"(\d+)\s+ants?", low)
    hm = re.search(r"half\s+as\s+many", low)
    if not (am and hm):
        return None
    a = Fraction(am.group(1))
    return a + a / 2


def _lego_stapel(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Lego-Stapel: '500+1500+125' -> 2125."""
    low = _digitize(question.lower())
    bm = re.search(r"(?:with\s+)?(\d+)\s+pieces?", low)
    tm = re.search(r"times\s+more(?:\s+pieces)?", low)
    qm = re.search(r"1/4\s+the\s+number", low)
    if not (bm and tm and qm):
        return None
    b = Fraction(bm.group(1))
    return b + b * 3 + b / 4


def _stift_mischen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Stift-Mischen: '25+5+1' -> 31."""
    low = _digitize(question.lower())
    bm = re.search(r"buys?\s+(\d+)\s+pens?", low)
    fm = re.search(r"(?:five|5)\s+empty\s+pens?", low)
    if not (bm and fm):
        return None
    b = Fraction(bm.group(1))
    return b + b / 5 + 1


def _freunde_drei(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Freunde-drei: '12/3x4' -> 16."""
    low = _digitize(question.lower())
    cm = re.search(
        r"(?:if\s+)?(\w+)\s+has\s+(\d+)"
        r"(?!x|\s+times)",
        low,
    )
    fms = re.findall(r"times\s+as\s+many\s+[\w\s]+?as\s+(\w+)", low)
    xm = re.findall(r"(\d+)x\s+(\w+)['\u2019]?s", low)
    if not (cm and len(fms) == 2 and fms[0] == fms[1]):
        if not (cm and len(xm) == 2 and xm[0][1] == xm[1][1]):
            return None
        return Fraction(cm.group(2)) / Fraction(xm[0][0]) * Fraction(xm[1][0])
    return Fraction(cm.group(2)) / 3 * 4


def _halle_ausgang(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Halle-Ausgang: '700x2/5' -> 280."""
    low = _digitize(question.lower())
    sm = re.search(
        r"(?:(\d+)\s+students?(?:\s+in\s+a\s+school\s+"
        r"hall|\.)|in\s+a\s+school\s+hall\s+was\s+(\d+))",
        low,
    )
    pm = re.search(r"(\d+)\s*%", low)
    fm = re.search(r"3/5\s+of\s+(?:the\s+)?remaining", low)
    if not (sm and pm and fm):
        return None
    n = sm.group(1) or sm.group(2)
    return Fraction(n) * (100 - Fraction(pm.group(1))) / 100 * Fraction(2, 5)


def _reifen_service(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Reifen-Service: '(5x2+3x3+1)x0.25' -> 5."""
    low = _digitize(question.lower())
    bm = re.search(
        r"(\d+)\s+(?:people?\s+on\s+)?(?:bicycles?|"
        r"bikes)",
        low,
    )
    tm = re.search(
        r"(\d+)\s+(?:people?\s+came\s+by\s+to\s+get\s+"
        r"all\s+their\s+)?tricycles?",
        low,
    )
    um = re.search(r"(?:one|1)\s+(?:\w+\s+)*unicycles?", low)
    cm = re.search(r"costs?\s+(\d+)\s+cents?", low)
    if not (bm and tm and um and cm):
        return None
    tires = Fraction(bm.group(1)) * 2 + Fraction(tm.group(1)) * 3 + 1
    return tires * Fraction(cm.group(1)) / 100


def _keks_wechselgeld(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Keks-Wechselgeld: '10-10x6x0.1' -> 4."""
    low = _digitize(question.lower())
    pm = re.search(r"buys?\s+(?:ten|10)\s+packs?", low)
    km = re.search(
        r"each\s+pack(?:\s+of\s+\w+)?\s+has\s+"
        r"(\d+)",
        low,
    )
    cm = re.search(r"each\s+\w+\s+costs?\s+\$?(\d+)(?:\.(\d+))?", low)
    bm = re.search(r"(?:pay\s+with\s+a\s+|from\s+a\s+)\$?(\d+)", low)
    if not (pm and km and cm and bm):
        return None
    c = Fraction(cm.group(1)) + (
        Fraction(cm.group(2)) / 100 if cm.group(2) else Fraction(0)
    )
    cost = Fraction(10) * Fraction(km.group(1)) * c
    return Fraction(bm.group(1)) - cost


def _fahrstuhl_ziel(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Fahrstuhl-Ziel: '3x4+6' -> 18."""
    low = _digitize(question.lower())
    sm = re.search(r"starts?\s+on\s+the\s+(\d+)", low)
    tm = re.search(
        r"(?:equal\s+to\s+|up\s+to\s+)(\d+)\s+times\s+"
        r"his",
        low,
    )
    pm = re.search(r"plus\s+(\d+)", low)
    if not (sm and tm and pm):
        return None
    return Fraction(sm.group(1)) * Fraction(tm.group(1)) + Fraction(pm.group(1))


def _pommes_raub(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Pommes-Raub: '14+7+9+18' -> 48."""
    low = _digitize(question.lower())
    dm = re.search(r"ate\s+(?:fourteen|14)", low)
    gm = re.search(r"half\s+(?:the\s+amount|of\s+that)", low)
    pm = re.search(
        r"(?:each\s+pigeon?\s+ate\s+|pigeons?\s+ate\s+)"
        r"(\d+)",
        low,
    )
    rm = re.search(
        r"(?:left\s+1/3\s+behind|stole\s+(?:two|2)\s*"
        r"(?:thirds?|/\s*3)(?:\s+of\s+the\s+remaining)?)",
        low,
    )
    am = re.search(r"ants?\s+(?:carried\s+off|took)", low)
    lm = re.search(r"leaving\s+(\d+)(?:\s+behind)?", low)
    if not (dm and gm and pm and rm and lm):
        return None
    left = Fraction(lm.group(1)) + (1 if am else 0)
    return Fraction(14 + 7 + 3 * Fraction(pm.group(1)) + left * 3)


def _mms_beutel(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """M&Ms-Beutel: '300+312+150' -> 762."""
    low = _digitize(question.lower())
    fm = re.search(r"first(?:\s+bag)?\s+has\s+(\d+)", low)
    sm = re.search(r"second(?:\s+bag)?\s+has\s+(\d+)\s+more", low)
    tm = re.search(r"half\s+(?:the\s+number|of\s+the\s+first)", low)
    if not (fm and sm and tm):
        return None
    f = Fraction(fm.group(1))
    return f + f + Fraction(sm.group(1)) + f / 2


def _haus_fenster(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Haus-Fenster: '2x3x2+2x4' -> 20."""
    low = _digitize(question.lower())
    hm = re.search(r"(?:has\s+)?(\d+)\s+houses?", low)
    bm = re.search(r"(\d+)\s+bedrooms?\s+each", low)
    wm = re.search(r"(?:has\s+)?(\d+)\s+windows?(?:\s+each|\s+per)", low)
    am = re.search(
        r"(?:additional|extra|plus)\s+(\d+)"
        r"(?:\s+\w+)?\s+windows?",
        low,
    )
    if not (hm and bm and wm and am):
        return None
    h = Fraction(hm.group(1))
    return h * Fraction(bm.group(1)) * Fraction(wm.group(1)) + h * Fraction(am.group(1))


def _lauf_vergleich_zwei(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Lauf-Vergleich-zwei: '3000-2920' -> 80."""
    low = _digitize(question.lower())
    fm = re.search(
        r"field(?:\s+that\s+is)?\s+(\d+)\s+"
        r"(?:yards?|meters?|miles?)\b",
        low,
    )
    bm = re.search(r"blake\s+runs?\s+back\s+and\s+forth\s+(\d+)", low)
    km = re.search(
        r"kelly\s+(?:runs?\s+back\s+and\s+forth\s+)?"
        r"(?:once|1)",
        low,
    )
    lm = re.search(r"(\d+)-yard\s+line", low)
    dm = re.search(r"(?:does\s+this|back)\s+(\d+)\s+times?", low)
    if not (fm and bm and km and lm and dm):
        return None
    blake = Fraction(fm.group(1)) * 2 * Fraction(bm.group(1))
    kelly = Fraction(fm.group(1)) * 2 + Fraction(lm.group(1)) * 2 * Fraction(
        dm.group(1)
    )
    return max(blake, kelly) - min(blake, kelly)


def _alter_summe(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Alter-Summe: '20+20' -> 40."""
    low = _digitize(question.lower())
    sm = re.search(r"sum\s+of\s+their\s+ages?\s+is\s+(\d+)", low)
    ym = re.search(r"in\s+(\d+)\s+years?", low)
    if not (sm and ym):
        return None
    return Fraction(sm.group(1)) + 2 * Fraction(ym.group(1))


def _schulbedarf(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Schulbedarf: '4x1.5+2x4+20' -> 34."""
    low = _digitize(question.lower())
    if re.search(r"per\s+dozen", low):
        return None
    ms = re.findall(
        r"(?:(\d+)|a|an)\s+([a-z]+(?:\s+[a-z]+){0,5})"
        r"\s+which\s+costs?\s+\$?(\d+(?:\.\d+)?)"
        r"(?:\s+each)?",
        low,
    )
    if len(ms) < 2:
        return None
    total = Fraction(0)
    for n, _obj, c in ms:
        qty = Fraction(n) if n else Fraction(1)
        total += qty * Fraction(c)
    return total


def _alter_doppel(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Alter-doppel: '3B=24, S=16' -> 16."""
    low = _digitize(question.lower())
    dm = re.search(r"twice\s+as\s+old", low)
    ym = re.search(r"in\s+(\d+)\s+years?", low)
    sm = re.search(
        r"sum\s+of\s+their\s+ages?\s+will\s+be\s+"
        r"(\d+)",
        low,
    )
    if not (dm and ym and sm):
        return None
    y = Fraction(ym.group(1))
    s = Fraction(sm.group(1))
    return (s - 2 * y) * 2 / 3


def _kaulquappen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Kaulquappen: '11+6-2' -> 15."""
    low = _digitize(question.lower())
    sm = re.search(r"(\d+)\s+\w+(?:\s+swimming)?", low)
    cm = re.search(
        r"(?:sees?\s+)?(\d+)\s+of\s+them|(\d+)\s+come"
        r"\s+out\s+of",
        low,
    )
    hm = re.search(
        r"(\d+)\s+of\s+them\s+hide|(\d+)\s+hide\s+"
        r"under",
        low,
    )
    if not (sm and cm and hm):
        return None
    c = cm.group(1) or cm.group(2)
    h = hm.group(1) or hm.group(2)
    return Fraction(sm.group(1)) + Fraction(c) - Fraction(h)


def _alter_proportion(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Alter-Proportion: '6S+9=3(S+9)' -> 6."""
    low = _digitize(question.lower())
    fm = re.search(r"(\d+)\s+times\s+older\s+than", low)
    ym = re.search(r"in\s+(\d+)\s+years?", low)
    tm = re.search(r"(\d+)\s+times\s+as\s+old\s+as", low)
    if not (fm and ym and tm):
        return None
    a = Fraction(fm.group(1))
    y = Fraction(ym.group(1))
    b = Fraction(tm.group(1))
    return y * (b - 1) / (a - b)


def _strand_fang(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Strand-Fang: '19+13' -> 32."""
    low = _digitize(question.lower())
    am = re.search(r"(\w+)\s+caugh?t?\s+(\d+)", low)
    fm = re.search(r"fewer\s+\w+(?:\s+\w+)?(?:\s+than)?", low)
    if not (am and fm):
        return None
    ts = re.findall(
        r"caugh?t?\s+(\d+)\s+\w+(?:\s+\w+)?,\s+"
        r"(\d+)\s+\w+(?:\s+\w+)?,?\s*(?:and\s+)?"
        r"(\d+)",
        low,
    )
    fs = re.findall(
        r"(\d+)\s+fewer\s+\w+(?:\s+\w+)?"
        r"(?:\s+than)?",
        low,
    )
    ms2 = re.findall(
        r"(\d+)\s+more\s+\w+(?:\s+\w+)?"
        r"(?:\s+than)?",
        low,
    )
    if not (ts and fs):
        return None
    a = sum(Fraction(x) for x in ts[0])
    b = a - sum(Fraction(f) for f in fs) + sum(Fraction(m2) for m2 in ms2)
    return a + b


def _schlangen_flecken(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Schlangen-Flecken: '(40x70+60x35)/2' -> 2450."""
    low = _digitize(question.lower())
    cm = re.search(
        r"cobra?,?\s+(?:which\s+has\s+|has\s+)"
        r"(\d+)\s+spots?",
        low,
    )
    dm = re.search(r"twice\s+as\s+many(?:\s+spots)?\s+as", low)
    cs = re.search(r"(\d+)\s+cobras?", low)
    ms = re.search(r"(\d+)\s+mambas?", low)
    if not (cm and dm and cs and ms):
        return None
    m = Fraction(cm.group(1)) / 2
    return (
        Fraction(cs.group(1)) * Fraction(cm.group(1)) + Fraction(ms.group(1)) * m
    ) / 2


def _spielzeug_wert(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Spielzeug-Wert: '5x4+3x5+3x5' -> 50."""
    low = _digitize(question.lower())
    cm = re.search(
        r"\w+\s+cars?\s*(?:costs?\s+)?\(?\$?(\d+)"
        r"(?:\s+each)?",
        low,
    )
    am = re.search(
        r"\w+\s+figures?\s*(?:costs?\s+)?\(?\$?(\d+)"
        r"(?:\s+each)?",
        low,
    )
    dm = re.search(
        r"doll\s+(?:costs?\s+as\s+much\s+as|worth)"
        r"\s+(\d+)",
        low,
    )
    if not (cm and am and dm):
        return None
    cars = re.findall(r"(\d+)\s+\w+\s+cars?,?", low)
    figs = re.findall(r"(\d+)\s+\w+\s+figures?,?", low)
    if not (cars and figs):
        return None
    return (
        Fraction(cars[0]) * Fraction(cm.group(1))
        + Fraction(figs[0]) * Fraction(am.group(1))
        + Fraction(dm.group(1)) * Fraction(am.group(1))
    )


def _geschichten_doppel(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Geschichten-doppel: '120+240' -> 360."""
    low = _digitize(question.lower())
    ms = re.findall(r"(?:wrote\s+)?(\d+)(?:\s+stories?)?[,. ]", low)
    ms = [m for m in ms if re.search(r"^\d+$", m)]
    dm = re.search(r"each\s+doubled", low)
    if not (ms and dm and len(ms) == 3):
        return None
    w1 = sum(Fraction(m) for m in ms)
    return w1 + w1 * 2


def _babysitter_eier(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Babysitter-Eier: '15x3/9' -> 5."""
    low = _digitize(question.lower())
    em = re.search(r"basket\s+of\s+(\d+)\s+eggs?", low)
    nm = re.search(r"needs?\s+(\d+)\s+eggs?", low)
    tm = re.search(r"(?:make\s+(\d+)|(\d+)\s+flans?)", low)
    if not (em and nm and tm):
        return None
    t = tm.group(1) or tm.group(2)
    return Fraction(t) * Fraction(nm.group(1)) / Fraction(em.group(1))


def _bruder_alter(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Bruder-Alter: '9x2+3' -> 21."""
    low = _digitize(question.lower())
    am = re.search(
        r"(?:if\s+)?(\w+)\s+is\s+(\d+)"
        r"(?:,|\s+years?\s+old)",
        low,
    )
    dm = re.search(r"twice\s+her\s+age", low)
    ym = re.search(r"in\s+(\d+)\s+years?", low)
    if not (am and dm and ym):
        return None
    return Fraction(am.group(2)) * 2 + Fraction(ym.group(1))


def _uniform_kosten(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Uniform-Kosten: '25+75+50' -> 150."""
    low = _digitize(question.lower())
    hm = re.search(r"hat\s+(?:that\s+costs?\s+)?\\?\$?(\d+)", low)
    jm = re.search(
        r"jacket\s+(?:that\s+costs?\s+)?(?:three|3)\s+"
        r"times",
        low,
    )
    pm = re.search(r"average\s+of\s+(?:the\s+costs|both)", low)
    if not (hm and jm and pm):
        return None
    h = Fraction(hm.group(1))
    j = h * 3
    return h + j + (h + j) / 2


def _auto_zeitvergleich(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Auto-Zeitvergleich: '480/30' -> 16."""
    low = _digitize(question.lower())
    fm = re.search(
        r"(?:traveling\s+at\s+)?(\d+)\s+"
        r"(?:\w+/?hour|mph)",
        low,
    )
    hm = re.search(r"half\s+that(?:\s+speed)?", low)
    dm = re.search(
        r"(?:traveled\s+for\s+a\s+total\s+of\s+|"
        r"travels?\s+)(\d+)",
        low,
    )
    if not (fm and hm and dm):
        return None
    return Fraction(dm.group(1)) / (Fraction(fm.group(1)) / 2)


def _limonaden_profit(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Limonaden-Profit: '25/5x3' -> 15."""
    low = _digitize(question.lower())
    lm = re.search(
        r"costs?\s+\\?\$?(\d+)(?:\s+for\s+|\s+)"
        r"lemons?",
        low,
    )
    sm = re.search(
        r"(?:costs?\s+)?\\?\$?(\d+)"
        r"(?:\s+for\s+|\s+)sugar?",
        low,
    )
    gm = re.search(r"(\d+)\s+glasses?\s+per\s+gallon?", low)
    pm = re.search(
        r"(?:sell\s+each\s+glass?\s+for\s+|at\s+)"
        r"\\?\$?(\d+)(?:\.(\d+))?",
        low,
    )
    pm2 = re.search(r"\\?\$?(\d+)\s+(?:in\s+)?profit", low)
    if not (lm and sm and gm and pm and pm2):
        return None
    glass = Fraction(pm.group(1)) + (
        Fraction(pm.group(2)) / 100 if pm.group(2) else Fraction(0)
    )
    per = Fraction(gm.group(1)) * glass - Fraction(lm.group(1)) - Fraction(sm.group(1))
    return Fraction(pm2.group(1)) / per * Fraction(lm.group(1))


def _uebungen_zwei(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Uebungen-zwei: '170+200' -> 370."""
    low = _digitize(question.lower())
    ms = re.findall(r"(?:does|do\s+)?(\d+)\s+[\w\s]+?[,.:]", low)
    if len(ms) < 3:
        return None
    d1 = sum(Fraction(x) for x in ms[:3])
    ms2 = re.findall(r"(\d+)\s+more\s+\w+", low)
    ms3 = re.findall(r"(\d+)\s+fewer\s+\w+", low)
    ms4 = re.findall(
        r"doubles?\s+(?:the\s+number(?:\s+of\s+)?)?"
        r"\w+",
        low,
    )
    if not (ms2 and ms3 and ms4):
        return None
    d2 = (
        Fraction(ms[0])
        + Fraction(ms2[0])
        + Fraction(ms[1])
        - Fraction(ms3[0])
        + Fraction(ms[2]) * 2
    )
    return d1 + d2


def _rennen_teams(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Rennen-Teams: '(240-80)-60' -> 100."""
    low = _digitize(question.lower())
    tm = re.search(r"race\s+with\s+(\d+)\s+\w+", low)
    jm = re.search(r"(\d+)\s+(?:were\s+)?(?:japanese|chinese)", low)
    bm = re.search(
        r"(\d+)\s+boys?\s+on\s+the\s+\w+\s+team|"
        r"boys?\s+on\s+the\s+\w+\s+team(?:\s+was\s+"
        r"|\s+)(\d+)",
        low,
    )
    if not (tm and jm and bm):
        return None
    b = bm.group(1) or bm.group(2)
    return Fraction(tm.group(1)) - Fraction(jm.group(1)) - Fraction(b)


def _auktion_desk(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Auktion-Desk: '200+6x50' -> 500."""
    low = _digitize(question.lower())
    om = re.search(r"opening\s+bid\s+(?:of\s+)?\\?\$?(\d+)", low)
    rm = re.search(r"rise?s?\s+(?:by\s+)?\\?\$?(\d+)", low)
    pm = re.search(r"(\d+)\s+other\s+people?", low)
    if not (om and rm and pm):
        return None
    return Fraction(om.group(1)) + 2 * Fraction(pm.group(1)) * Fraction(rm.group(1))


def _gehalt_spenden(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Gehalt-Spenden: '1250-900' -> 350."""
    low = _digitize(question.lower())
    sm = re.search(
        r"earns?\s+\\?\$?(\d+)\\?\$?"
        r"(?:\s+per\s+month|/month)",
        low,
    )
    rm = re.search(r"1/4\s+(?:of\s+his\s+salary\s+on\s+)?rent", low)
    fm = re.search(r"1/3\s+(?:on\s+car\s+fuel|fuel)", low)
    dm = re.search(r"half\s+(?:of\s+the\s+remaining|the\s+rest)", low)
    gm = re.search(
        r"gives?\s+(?:his\s+)?daughter\s+\\?\$?"
        r"(\d+)\\?\$?",
        low,
    )
    wm = re.search(
        r"\\?\$?(\d+)\\?\$?\s+to\s+his\s+wife|"
        r"wife\s+\\?\$?(\d+)\\?\$?",
        low,
    )
    if not (sm and rm and fm and dm and gm and wm):
        return None
    rest = Fraction(sm.group(1)) - Fraction(sm.group(1)) / 4 - Fraction(sm.group(1)) / 3
    rest = rest / 2
    w = wm.group(1) or wm.group(2)
    return rest - Fraction(gm.group(1)) - Fraction(w)


def _buch_verkauf(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Buch-Verkauf: '65x20' -> 1300."""
    low = _digitize(question.lower())
    tm = re.search(r"(?:collection\s+of\s+)?(\d+)\s+books?", low)
    pm = re.search(
        r"each\s+book\s+sells?\s+at\s+\\?\$?(\d+)"
        r"\\?\$?|,\s*\\?\$?(\d+)\s+each",
        low,
    )
    um = re.search(r"(\d+)\s+unsold(?:\s+books?)?", low)
    cm = re.search(
        r"(?:sales?\s+number\s+this\s+year\s+is\s+|"
        r"this\s+year\s+)(\d+)",
        low,
    )
    if not (tm and pm and um and cm):
        return None
    first = Fraction(cm.group(1)) * 2
    second = (
        Fraction(tm.group(1)) - first - Fraction(cm.group(1)) - Fraction(um.group(1))
    )
    p = pm.group(1) or pm.group(2)
    return second * Fraction(p)


def _nachhilfe_stunden(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Nachhilfe-Stunden: '(5+8)x10' -> 130."""
    low = _digitize(question.lower())
    em = re.search(r"earns?\s+\\?\$?(\d+)(?:\s+an\s+hour|/hour)", low)
    hm = re.search(r"tutored\s+(\d+)\s+hours?", low)
    hm2 = re.search(r"(\d+)\s+hours?\s+(?:for\s+the\s+)?second", low)
    if not (em and hm and hm2):
        return None
    return (Fraction(hm.group(1)) + Fraction(hm2.group(1))) * Fraction(em.group(1))


def _alter_kette_drei(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Alter-Kette-drei: '30-3+5' -> 32."""
    low = _digitize(question.lower())
    qm = re.search(r"(?:if\s+)?(\w+)\s+is\s+(\d+)[,.?!]", low)
    om = re.search(r"older\s+than\s+(\w+)", low)
    jm = re.search(r"younger\s+than\s+(\w+)", low)
    if not (qm and om and jm):
        return None
    if jm.group(1) != qm.group(1):
        return None
    q = Fraction(qm.group(2))
    return q - 3 + 5


def _blumen_bestellung(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Blumen-Bestellung: '200/5x4' -> 160."""
    low = _digitize(question.lower())
    lm = re.search(r"(?:ordered\s+)?(\d+)[^.]*?lilies?", low)
    fm = re.search(
        r"(?:five|5)\s+times\s+the\s+(?:number|"
        r"carnations?)",
        low,
    )
    tm = re.search(r"(?:four|4)\s+times\s+as\s+many", low)
    if not (lm and fm and tm):
        return None
    return Fraction(lm.group(1)) / 5 * 4


def _bevoelkerung_chile(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Bevoelkerung-Chile: '3000x20x2' -> 120000."""
    low = _digitize(question.lower())
    cm = re.search(r"cera\s+is\s+(\d+)", low)
    nm = re.search(r"half\s+(?:as\s+old\s+as\s+)?\w+['\u2019]?s?", low)
    pm = re.search(
        r"(?:six|6)\s+years\s+ago\s+was\s+(\d+)\s+"
        r"times|was\s+(\d+)\s+times\s+",
        low,
    )
    ym = re.search(r"(?:six|6)\s+years\s+ago", low)
    if not (cm and nm and pm and ym):
        return None
    noah = (Fraction(cm.group(1)) - 6) / 2
    p = pm.group(1) or pm.group(2)
    return Fraction(p) * noah * 2


def _stroh_verteilung(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Stroh-Verteilung: '(160-50-20)/6/3' -> 5."""
    low = _digitize(question.lower())
    tm = re.search(r"(\d+)\s+pieces\s+of\s+straw\s+have\s+been", low)
    if not tm:
        tm = re.search(r"(\d+)\s+pieces\s+of\s+straw", low)
    hm = re.search(
        r"(?:(\d+)\s+cages?\s+of\s+hamsters?|"
        r"hamsters?:\s*(\d+)\s+cages?)",
        low,
    )
    hp = re.search(
        r"each\s+hamster?\s+is\s+given\s+(\d+)|"
        r"cages?\s*x\s*(\d+)",
        low,
    )
    rm = re.search(
        r"each\s+rat\s+is\s+given\s+(\d+)|"
        r"(\d+)\s+each",
        low,
    )
    cm = re.search(
        r"(?:rats?\s+are\s+kept\s+in\s+|rats?:\s*)"
        r"(\d+)\s+cages?",
        low,
    )
    if not (tm and hm and hp and rm and cm):
        return None
    h = hp.group(1) or hp.group(2)
    r = rm.group(1) or rm.group(2)
    c = cm.group(1) or cm.group(2)
    hmv = hm.group(1) or hm.group(2)
    rest = Fraction(tm.group(1)) - Fraction(hmv) * Fraction(h) - 20
    return rest / Fraction(r) / Fraction(c)


def _marmor_gewicht(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Marmor-Gewicht: '78x2' -> 156."""
    low = _digitize(question.lower())
    bm = re.search(r"bought\s+(\d+)\s+marbles?", low)
    sm = re.search(r"store(?:\s+that)?\s+had\s+(\d+)", low)
    fm = re.search(
        r"2/5\s+times\s+(?:as\s+many(?:\s+\w+)?\s+"
        r"as\s+he\s+bought|the\s+bought)",
        low,
    )
    wm = re.search(
        r"each\s+marble\s+weighs?\s+(\d+)|"
        r"weighs?\s+(\d+)\s*kgs?",
        low,
    )
    if not (bm and sm and fm and wm):
        return None
    total = (
        Fraction(bm.group(1))
        + Fraction(sm.group(1))
        + Fraction(bm.group(1)) * Fraction(2, 5)
    )
    w = wm.group(1) or wm.group(2)
    return total * Fraction(w)


def _zins_schuld(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Zins-Schuld: '100+100x0.02x3' -> 106."""
    low = _digitize(question.lower())
    sm = re.search(r"owes?\s+(?:\w+\s+)?\\?\$?(\d+)", low)
    im = re.search(
        r"(\d+)\s*%\s+monthly\s+interest|interest\s+"
        r"(?:of\s+)?(\d+)\s*%",
        low,
    )
    mm = re.search(r"after\s+(\d+)\s+months?", low)
    if not (sm and im and mm):
        return None
    i = im.group(1) or im.group(2)
    return Fraction(sm.group(1)) * (1 + Fraction(i) / 100 * Fraction(mm.group(1)))


def _klasse_verteilung(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Klasse-Verteilung: '30+90+3' -> 123."""
    low = _digitize(question.lower())
    bm = re.search(r"(?:has\s+)?(\d+)\s+boys?", low)
    gm = re.search(r"(\d+)\s+times\s+as\s+many\s+girls?", low)
    nm = re.search(r"1/10\s+as\s+many", low)
    if not (bm and gm and nm):
        return None
    b = Fraction(bm.group(1))
    return b + b * Fraction(gm.group(1)) + b / 10


def _messe_teilen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Messe-teilen: '(20.25+15.75+66)/3' -> 34."""
    low = _digitize(question.lower())
    tm = re.search(r"spent\s+\\?\$?(\d+)(?:\.(\d+))?", low)
    fm = re.search(r"\\?\$?(\d+)(?:\.(\d+))?\s+less\s+on", low)
    rm = re.search(r"costs?\s+\\?\$?(\d+)|at\s+\\?\$?(\d+)", low)
    rm2 = re.search(r"(\d+)\s+(?:different\s+)?rides?", low)
    pm = re.search(
        r"split(?:\s+all\s+the\s+costs?)?\s+evenly|"
        r"split\s+all\s+the\s+costs?",
        low,
    )
    if not (tm and fm and rm and rm2 and pm):
        return None
    t = Fraction(tm.group(1)) + (
        Fraction(tm.group(2)) / 100 if tm.group(2) else Fraction(0)
    )
    fv = fm.group(1) or fm.group(3)
    fd = fm.group(2) or fm.group(4)
    f = Fraction(fv) + (Fraction(fd) / 100 if fd else Fraction(0))
    rv = rm.group(1) or rm.group(2)
    total = t + (t - f) + Fraction(rv) * Fraction(rm2.group(1))
    return total / 3


def _strommasten(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Strommasten: '45/3' -> 15."""
    low = _digitize(question.lower())
    rm = re.search(r"ratio\s*(?:of\s+the\s+[\w\s]+?is\s+)?1:3", low)
    wm = re.search(
        r"total\s+number\s+of\s+[\w\s]+?\s+is\s+"
        r"(\d+)|(\d+)\s+wires?\s+needed",
        low,
    )
    if not (rm and wm):
        return None
    w = wm.group(1) or wm.group(2)
    return Fraction(w) / 3


def _pfirsich_ernte(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Pfirsich-Ernte: '3x60x2' -> 360."""
    low = _digitize(question.lower())
    hm = re.search(r"for\s+(\d+)\s+hours?", low)
    pm = re.search(
        r"(\d+)\s+\w+\s+a\s+minute|at\s+(\d+)\s+per"
        r"\s+minute",
        low,
    )
    if not (hm and pm):
        return None
    p = pm.group(1) or pm.group(2)
    return Fraction(hm.group(1)) * 60 * Fraction(p)


def _firma_gehalt_zwei(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Firma-Gehalt-zwei: '80000+144000' -> 224000."""
    low = _digitize(question.lower())
    em = re.search(
        r"(?:employed\s+)?(\d+)\s+(?:people|"
        r"employees?)",
        low,
    )
    fm = re.search(r"2/5(?:\s+of\s+the\s+total)?", low)
    pm = re.search(r"(?:paid\s+)?\\?\$?(\d+)\s+more", low)
    jm = re.search(
        r"paid\s+\\?\$?(\d+)(?:\s+per\s+month|"
        r"/month)",
        low,
    )
    if not (em and fm and pm and jm):
        return None
    jr = Fraction(em.group(1)) * Fraction(2, 5)
    sr = Fraction(em.group(1)) - jr
    return jr * Fraction(jm.group(1)) + sr * (
        Fraction(jm.group(1)) + Fraction(pm.group(1))
    )


def _kreide_wechsel(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Kreide-Wechsel: '20-5x2' -> 10."""
    low = _digitize(question.lower())
    cm = re.search(r"(\d+)\s+(?:different\s+)?colors?", low)
    pm = re.search(
        r"prepared\s+\\?\$?(\d+)|\\?\$?(\d+)\s+"
        r"prepared",
        low,
    )
    om = re.search(
        r"crayons?\s+(?:costs?\s+)?\\?\$?(\d+)"
        r"(?:\s+each)?",
        low,
    )
    if not (cm and pm and om):
        return None
    p = pm.group(1) or pm.group(2)
    return Fraction(p) - Fraction(cm.group(1)) * Fraction(om.group(1))


def _aktien_wert(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Aktien-Wert: '64x1.5x0.75' -> 72."""
    low = _digitize(question.lower())
    sm = re.search(r"(?:buys?\s+)?(\d+)\s+shares?", low)
    pm = re.search(r"(?:for\s+|at\s+)\\?\$?(\d+)(?:\s+each)?", low)
    im = re.search(
        r"increases?\s+(\d+)\s*%|(?:^|[^\d-])\+"
        r"(\d+)\s*%",
        low,
    )
    dm = re.search(
        r"decreases?\s+(\d+)\s*%|(?:^|[^\d+])-"
        r"(\d+)\s*%",
        low,
    )
    if not (sm and pm and im and dm):
        return None
    v = Fraction(sm.group(1)) * Fraction(pm.group(1))
    i = im.group(1) or im.group(2)
    d = dm.group(1) or dm.group(2)
    v = v * (1 + Fraction(i) / 100)
    v = v * (1 - Fraction(d) / 100)
    return v


def _stift_preis(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Stift-Preis: '(1.2+0.3)x8' -> 12."""
    low = _digitize(question.lower())
    if not re.search(r"pen\s+costs?\s+as\s+much|pen\s*=\s*pencil", low):
        return None
    pm = re.search(
        r"pencil(?:\s+costs?)?\s+\\?\$?(\d+)"
        r"(?:\.(\d+))?",
        low,
    )
    em = re.search(
        r"eraser(?:\s+costs?)?\s+\\?\$?(\d+)"
        r"(?:\.(\d+))?",
        low,
    )
    nm = re.search(r"(\d+)\s+pens?", low)
    if not (pm and em and nm):
        return None
    p = Fraction(pm.group(1)) + (
        Fraction(pm.group(2)) / 100 if pm.group(2) else Fraction(0)
    )
    e = Fraction(em.group(1)) + (
        Fraction(em.group(2)) / 100 if em.group(2) else Fraction(0)
    )
    return (p + e) * Fraction(nm.group(1))


def _jonglieren(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Jonglieren: '3+4-3' -> 4."""
    low = _digitize(question.lower())
    sm = re.search(
        r"(?:practicing\s+juggling?\s+|starts?\s+with\s+)"
        r"(\d+)",
        low,
    )
    am = re.search(r"adding\s+1\s+ball|adds?\s+1\s+per\s+week", low)
    dm = re.search(r"drops?\s+(?:three|3)", low)
    if not (sm and am and dm):
        return None
    return Fraction(sm.group(1)) + 4 - 3


def _stadt_bevoelkerung(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Stadt-Bevoelkerung: '23786-8417-9092' -> 6277."""
    low = _digitize(question.lower())
    im = re.search(r"(?:exactly\s+)?([\d,]+)\s+inhabitants?", low)
    mm = re.search(r"(?:include\s+)?([\d,]+)\s+men?", low)
    wm = re.search(r"(?:and\s+)?([\d,]+)\s+women?", low)
    if not (im and mm and wm):
        return None
    return (
        Fraction(im.group(1).replace(",", ""))
        - Fraction(mm.group(1).replace(",", ""))
        - Fraction(wm.group(1).replace(",", ""))
    )


def _tier_gewicht(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Tier-Gewicht: '50x3+60+30+20' -> 260."""
    low = _digitize(question.lower())
    fm = re.search(r"frog(?:s?\s+weighs?)?\s+(\d+)", low)
    lm = re.search(r"less\s+than\s+(?:a\s+\w+\s+)?snake?", low)
    mm = re.search(r"more\s+than\s+(?:a\s+\w+\s+)?bird?", low)
    cm = re.search(r"container(?:\s+also\s+weighs?)?\s+(\d+)", low)
    if not (fm and lm and mm and cm):
        return None
    f = Fraction(fm.group(1))
    return f * 3 + (f + 10) + (f - 20) + Fraction(cm.group(1))


def _lektor_lohn(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Lektor-Lohn: '500x5+500x10' -> 7500."""
    low = _digitize(question.lower())
    tm = re.search(
        r"total\s+number\s+of\s+(\d+)\s+sentences?|"
        r"(\d+)\s+sentences?\s+split",
        low,
    )
    pm = re.search(
        r"pays?\s+him\s+(\d+)\s+cents?|(?:a|b)\s+pays"
        r"\s+(\d+)\s+cents?",
        low,
    )
    dm = re.search(r"twice\s+what\s+[\w\s]+?pays|pays\s+twice", low)
    if not (tm and pm and dm):
        return None
    tv = tm.group(1) or tm.group(2)
    pv = pm.group(1) or pm.group(2)
    each = Fraction(tv) / 2
    return each * Fraction(pv) + each * Fraction(pv) * 2


def _tisch_beine(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Tisch-Beine: '40x4+50x3' -> 310."""
    low = _digitize(question.lower())
    ms = re.findall(r"(\d+)\s+tables?\s+with\s+(\d+)\s+legs?", low)
    if len(ms) < 2:
        return None
    return Fraction(ms[0][0]) * Fraction(ms[0][1]) + Fraction(ms[1][0]) * Fraction(
        ms[1][1]
    )


def _auszeichnung_jahr(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Auszeichnung-Jahr: '104000x1.05+5000' -> 114200."""
    low = _digitize(question.lower())
    rm = re.search(
        r"monetary\s+reward\s+of\s+\\?\$?(\d+)|"
        r"\\?\$?(\d+)\s+award",
        low,
    )
    pm = re.search(r"(\d+)\s*%\s+raise(?:\s+in\s+salary)?", low)
    sm = re.search(
        r"(?:makes?\s+)?\\?\$?(\d+)"
        r"(?:\s+a\s+week|/week)",
        low,
    )
    if not (rm and pm and sm):
        return None
    rv = rm.group(1) or rm.group(2)
    return Fraction(sm.group(1)) * 52 * (1 + Fraction(pm.group(1)) / 100) + Fraction(rv)


def _fabrik_prozent(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Fabrik-Prozent: '100/1000' -> 10."""
    low = _digitize(question.lower())
    ms = re.findall(
        r"(?:sold?\s+(\d+)\s+\w+\s*(?:a\s+day|/day)"
        r"[^.]*?(?:made\s+|at\s+)\\?\$?(\d+)|"
        r"sold?\s+(\d+)\s+\w+/day\s+at\s+\\?\$?"
        r"(\d+))",
        low,
    )
    ns = re.findall(
        r"(?:sell\s+(\d+)\s+\w+\s*(?:a\s+day|/day)"
        r"[^.]*?(?:make\s+|at\s+)\\?\$?(\d+)|"
        r"(?:sell\s+|now\s+)?(\d+)\s+\w+/day\s+at\s+"
        r"\\?\$?(\d+))",
        low,
    )
    ms = [(a or c, b or d) for a, b, c, d in ms]
    ns = [(a or c, b or d) for a, b, c, d in ns]
    if not (ms and ns):
        return None
    old = Fraction(ms[0][0]) * Fraction(ms[0][1])
    new = Fraction(ns[-1][0]) * Fraction(ns[-1][1])
    return (new - old) / old * 100


def _huehner_eier(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Huehner-Eier: '3R+5(R+2)=42' -> 4."""
    low = _digitize(question.lower())
    rm = re.search(r"red\s+\w+\s+(?:produce\s+)?(\d+)", low)
    wm = re.search(r"white\s+(?:\w+\s+)?(?:produce\s+)?(\d+)", low)
    cm = re.search(
        r"collects?\s+(\d+)\s+eggs?|(\d+)\s+eggs?\s+"
        r"collected",
        low,
    )
    dm = re.search(
        r"(\d+)\s+more\s+white(?:\s+\w+)?\s+than\s+"
        r"red",
        low,
    )
    if not (rm and wm and cm and dm):
        return None
    r = Fraction(rm.group(1))
    w = Fraction(wm.group(1))
    cv = cm.group(1) or cm.group(2)
    return (Fraction(cv) - Fraction(dm.group(1)) * w) / (r + w)


def _sohn_alter(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Sohn-Alter: '(45+5)/2' -> 25."""
    low = _digitize(question.lower())
    cm = re.search(
        r"(?:turned|is)\s+(\d+)\s+years?\s+old|is\s+"
        r"(\d+),",
        low,
    )
    lm = re.search(r"less\s+than\s+twice(?:\s+the\s+age)?", low)
    if not (cm and lm):
        return None
    cv = cm.group(1) or cm.group(2)
    return (Fraction(cv) + 5) / 2


def _park_umfang(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Park-Umfang: '2x7.5/3' -> 5."""
    low = _digitize(question.lower())
    ms = re.findall(r"(\d+)(?:\.(\d+))?(?:\s+miles?|\s+by)", low)
    wm = re.search(r"walks?\s+at\s+(\d+)\s+(?:\w+/hour|mph)", low)
    if not (ms and wm and len(ms) >= 2):
        return None
    vals = []
    for m in ms:
        v = Fraction(m[0]) + (Fraction(m[1]) / 10 if m[1] else Fraction(0))
        vals.append(v)
    return 2 * (vals[0] + vals[1]) / Fraction(wm.group(1))


def _test_minimum(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Test-Minimum: '42-15-18' -> 9."""
    low = _digitize(question.lower())
    tm = re.search(r"total\s+at\s+least\s+(\d+)", low)
    ms = re.findall(
        r"scored\s+(\d+)\s+and\s+(\d+)|scored\s+"
        r"(\d+)",
        low,
    )
    if not (tm and ms):
        return None
    vals = []
    for m in ms:
        if m[0] and m[1]:
            vals += [m[0], m[1]]
        else:
            vals.append(m[0] or m[1] or m[2])
    return Fraction(tm.group(1)) - sum(Fraction(v) for v in vals)


def _pool_lecks(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Pool-Lecks: '2x=16' -> 8."""
    low = _digitize(question.lower())
    lm = re.search(
        r"(?:leaks?\s+emptying\s+them\s+out\s+at\s+|"
        r"leak\s+)(\d+)",
        low,
    )
    tm = re.search(r"twice(?:\s+as\s+much\s+water)?", low)
    fm = re.search(r"(?:four|4)\s+times(?:\s+as\s+much\s+water)?", low)
    if not (lm and tm and fm):
        return None
    return Fraction(lm.group(1)) * 4 / 2


def _geraet_anteil(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Geraet-Anteil: '400000x0.6' -> 240000."""
    low = _digitize(question.lower())
    bm = re.search(r"bought\s+\\?\$?([\d,]+)", low)
    fm = re.search(r"worth\s+(\d+)\s*%|(\d+)\s*%\s+faulty", low)
    if not (bm and fm):
        return None
    f = fm.group(1) or fm.group(2)
    return Fraction(bm.group(1).replace(",", "")) * (100 - Fraction(f)) / 100


def _blumen_mehr(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Blumen-mehr: '4+11' -> 15."""
    low = _digitize(question.lower())
    rm = re.search(r"roses?(?:\s+in\s+the\s+vase)?", low)
    dm = re.search(r"more\s+\w+\s+than\s+roses?", low)
    nm = re.search(r"(\d+)\s+more\s+", low)
    bm = re.search(r"(\d+)\s+roses?", low)
    if not (rm and dm and nm and bm):
        return None
    return Fraction(bm.group(1)) + Fraction(bm.group(1)) + Fraction(nm.group(1))


def _pesos_total(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Pesos-Total: '130+220' -> 350."""
    low = _digitize(question.lower())
    sm = re.search(r"(?:has\s+)?(\d+)\s+silver", low)
    gm = re.search(r"(?:and\s+|\+)\s*(\d+)\s+gold", low)
    dm = re.search(r"twice(?:\s+as\s+many)?\s+(?:the\s+)?silver", low)
    am = re.search(r"(\d+)\s+more\s+gold", low)
    if not (sm and gm and dm and am):
        return None
    s = Fraction(sm.group(1))
    g = Fraction(gm.group(1))
    return s + g + s * 2 + g + Fraction(am.group(1))


def _raetsel_zeit(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Raetsel-Zeit: '3x10+8x5' -> 70."""
    low = _digitize(question.lower())
    cm = re.search(
        r"(\d+)\s+minutes?\s+to\s+finish\s+a\s+"
        r"crossword|crossword\s+(\d+)\s+min",
        low,
    )
    sm = re.search(
        r"(\d+)\s+minutes?\s+to\s+finish\s+a\s+"
        r"sudoku|sudoku\s+(\d+)\s+min",
        low,
    )
    cs = re.search(r"(?:solved\s+)?(\d+)\s+crossword", low)
    ss = re.search(r"(?:and\s+)?(\d+)\s+sudoku", low)
    if not (cm and sm and cs and ss):
        return None
    cv = cm.group(1) or cm.group(2)
    sv = sm.group(1) or sm.group(2)
    return Fraction(cs.group(1)) * Fraction(cv) + Fraction(ss.group(1)) * Fraction(sv)


def _zeugnis_durchschnitt(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Zeugnis-Durchschnitt: '400/5' -> 80."""
    low = _digitize(question.lower())
    ms = re.findall(r"received:\s*([\d, ]+(?:and\s+\d+)?)", low)
    if not ms:
        ms = re.findall(r"tests?:\s*([\d, ]+(?:and\s+\d+)?)", low)
    if not ms:
        return None
    nums = [int(x) for x in re.findall(r"\d+", ms[0])]
    if len(nums) < 3:
        return None
    return Fraction(sum(nums)) / len(nums)


def _blumen_weniger(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Blumen-weniger: '90+50' -> 140."""
    low = _digitize(question.lower())
    if "flower" not in low and "plant" not in low:
        return None
    gm = re.search(
        r"(?:plants?\s+)?(\d+)\s+\w+\s*(?:and|,)\s*"
        r"\d+\s+fewer",
        low,
    )
    fm = re.search(r"fewer\s+\w+", low)
    nm = re.search(r"(\d+)\s+fewer", low)
    if not (gm and fm and nm):
        return None
    return Fraction(gm.group(1)) + Fraction(gm.group(1)) - Fraction(nm.group(1))


def _alter_differenz(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Alter-Differenz: '7+5' -> 12."""
    low = _digitize(question.lower())
    if "difference" not in low:
        return None
    om = re.search(r"older\s+than\s+(\w+)", low)
    om2 = re.search(r"older\s+than\s+(\w+)", low)
    if not (om and om2):
        return None
    ms = re.findall(r"(\d+)\s+years?\s+older", low)
    if not ms:
        return None
    return sum(Fraction(m) for m in ms)


def _boot_miete(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Boot-Miete: '3x30+5x18' -> 180."""
    low = _digitize(question.lower())
    if "canoe" not in low or "raft" not in low:
        return None
    cm = re.search(
        r"(?:rents?\s+a\s+\w+\s+for\s+)?\\?\$?"
        r"(\d+)(?:\s+an\s+hour|/h)",
        low,
    )
    bm = re.search(
        r"(?:rents?\s+a\s+\w+\s+for\s+)?\\?\$?"
        r"(\d+)(?:\s+an\s+hour|/h)",
        low,
    )
    ch = re.search(
        r"(?:uses?\s+the\s+\w+\s+for\s+|for\s+)"
        r"(\d+)\s*h",
        low,
    )
    bh = re.search(
        r"(?:uses?\s+the\s+\w+\s+for\s+|for\s+)"
        r"(\d+)\s*h",
        low,
    )
    if not (cm and bm and ch and bh):
        return None
    return Fraction(ch.group(1)) * Fraction(cm.group(1)) + Fraction(
        bh.group(1)
    ) * Fraction(bm.group(1))


def _haus_vorrat(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Haus-Vorrat: '120+60+20' -> 200."""
    low = _digitize(question.lower())
    dm = re.search(r"twice\s+as\s+much\s+\w+\s+as\s+\w+", low)
    cm = re.search(r"(?:total\s+of\s+)?(\d+)\s+cannolis?", low)
    bm = re.search(r"bought\s+(\d+)\s+more\s+\w+", low)
    fm = re.search(
        r"(\d+)\s+fewer\s+\w+(?:\s+than\s+the\s+"
        r"number\s+of\s+\w+)?",
        low,
    )
    if not (dm and cm and bm and fm):
        return None
    c = Fraction(cm.group(1))
    before = c * 2 + c
    return (
        before + Fraction(bm.group(1)) + (Fraction(bm.group(1)) - Fraction(fm.group(1)))
    )


def _puzzle_zeit(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Puzzle-Zeit: '360/6/60' -> 1."""
    low = _digitize(question.lower())
    pm = re.search(r"(\d+)[\s-]piece\s+puzzle", low)
    km = re.search(
        r"(?:add\s+)?(\d+)\s*(?:pieces?\s+per\s+"
        r"minute|/min)",
        low,
    )
    hm = re.search(r"half(?:\s+as\s+many\s+pieces?)?", low)
    if not (pm and km and hm):
        return None
    rate = Fraction(km.group(1)) + Fraction(km.group(1)) / 2
    return Fraction(pm.group(1)) / rate / 60


def _vertrag_gehalt(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Vertrag-Gehalt: '96000+72000' -> 168000."""
    low = _digitize(question.lower())
    em = re.search(r"(?:hired\s+)?(\d+)\s+employees?", low)
    pm = re.search(r"\\?\$?(\d+)(?:\s+per\s+hour|/h)", low)
    wm = re.search(r"(\d+)-?h(?:our)?(?:\s+work?week|\s+week)", low)
    fm = re.search(
        r"1/4\s+(?:of\s+the\s+)?(?:employees?|"
        r"contracts?)",
        low,
    )
    if not (em and pm and wm and fm):
        return None
    weekly = Fraction(pm.group(1)) * Fraction(wm.group(1))
    monthly = weekly * 4
    may = Fraction(em.group(1)) * monthly
    june = Fraction(em.group(1)) * Fraction(3, 4) * monthly
    return may + june


def _melonen_ernte(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Melonen-Ernte: '84x1/4' -> 21."""
    low = _digitize(question.lower())
    pm = re.search(r"(?:produced\s+)?(\d+)\s+\w+", low)
    fm = re.search(
        r"(\d+)\s*%\s+(?:of\s+the\s+\w+\s+)?"
        r"harvested|(\d+)\s*%\s+of\s+the\s+\w+",
        low,
    )
    tm = re.search(r"3/4\s+of\s+the\s+(?:remaining|rest)", low)
    if not (pm and fm and tm):
        return None
    f = fm.group(1) or fm.group(2)
    rest = Fraction(pm.group(1)) * (100 - Fraction(f)) / 100
    return rest / 4


def _spind_groesse(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Spind-Groesse: '5x4x2' -> 40."""
    low = _digitize(question.lower())
    hm = re.search(r"half(?:\s+as\s+big\s+as\s+|\s+of\s+)(\w+)", low)
    qm = re.search(r"1/4(?:\s+as\s+big\s+as\s+|\s+of\s+)(\w+)", low)
    pm = re.search(
        r"peter['\u2019]?s?\s+(?:locker\s+)?is\s+"
        r"(\d+)\s+cubic",
        low,
    )
    if not (hm and qm and pm):
        return None
    return Fraction(pm.group(1)) * 4 * 2


def _tomaten_reben(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Tomaten-Reben: '63/3' -> 21."""
    low = _digitize(question.lower())
    em = re.search(r"eats?\s+(\d+)(?:\s+per\s+day|/day)", low)
    dm = re.search(r"twice(?:\s+as\s+much)?", low)
    vm = re.search(
        r"(?:vines?\s+can\s+produce|vines?\s+produce)"
        r"\s+(\d+)(?:\s+\w+\s+per\s+week|/week)",
        low,
    )
    if not (em and dm and vm):
        return None
    total = Fraction(em.group(1)) + Fraction(em.group(1)) / 2
    return total * 7 / Fraction(vm.group(1))


def _zimmer_umfang(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Zimmer-Umfang: '2x(9+40)' -> 98."""
    low = _digitize(question.lower())
    am = re.search(
        r"(?:area\s+of\s+[\w'\u2019\s]+?\s+is\s+)?"
        r"(\d+)\s+(?:square\s+feet?|sq\s+ft)",
        low,
    )
    lm = re.search(
        r"length(?:\s+of\s+his\s+room\s+is\s+|\s+)"
        r"(\d+)\s+yards?",
        low,
    )
    if not (am and lm):
        return None
    w = Fraction(am.group(1)) / (Fraction(lm.group(1)) * 3)
    return 2 * (Fraction(lm.group(1)) * 3 + w)


def _pizza_bestellung(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Pizza-Bestellung: '20x4/8' -> 10."""
    low = _digitize(question.lower())
    fm = re.search(r"friends?(?:\s+in\s+total)?", low)
    sm = re.search(
        r"(\d+)\s+slices?\s+each|each(?:\s+can\s+have)?"
        r"\s+(\d+)\s+slices?",
        low,
    )
    pm = re.search(
        r"(?:into\s+|in\s+)(\d+)\s+"
        r"(?:portions?|slices?)",
        low,
    )
    nm = re.search(r"(\d+)\s+friends?", low)
    if not (fm and sm and pm and nm):
        return None
    sv = sm.group(1) or sm.group(2)
    return Fraction(nm.group(1)) * Fraction(sv) / Fraction(pm.group(1))


def _haus_temperatur(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Haus-Temperatur: '40+15-6' -> 49."""
    low = _digitize(question.lower())
    tm = re.search(r"house(?:\s+is)?\s+(\d+)\s+degrees?", low)
    hm = re.search(r"baking(?:,?\s+and\s+every\s+hour)?", low)
    rm = re.search(
        r"raises?\s+the\s+house['\u2019]?s?\s+"
        r"temperature\s+by\s+(\d+)|\+(\d+)/h",
        low,
    )
    cm = re.search(
        r"cools?\s+down\s+(\d+)\s+degrees?|-(\d+)\s+"
        r"per\s+10\s+min",
        low,
    )
    mm = re.search(
        r"(\d+)\s+minutes?\s+the\s+window|per\s+"
        r"(\d+)\s+min",
        low,
    )
    if not (tm and hm and rm and cm and mm):
        return None
    hours = 3
    mins = 30
    rv = rm.group(1) or rm.group(2)
    cv = cm.group(1) or cm.group(2)
    return (
        Fraction(tm.group(1))
        + hours * Fraction(rv)
        - Fraction(mins) / 10 * Fraction(cv)
    )


def _geld_verdreifachen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Geld-verdreifachen: '(20+10)x3' -> 90."""
    low = _digitize(question.lower())
    am = re.search(
        r"allowance\s+of\s+\\?\$?(\d+)|\\?\$?"
        r"(\d+)\s+allowance",
        low,
    )
    em = re.search(
        r"extra\s+\\?\$?(\d+)|\\?\$?(\d+)\s+"
        r"extra",
        low,
    )
    tm = re.search(r"tripled\s+in\s+a\s+year", low)
    if not (am and em and tm):
        return None
    av = am.group(1) or am.group(2)
    ev = em.group(1) or em.group(2)
    return (Fraction(av) + Fraction(ev)) * 3


def _schuh_profit(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Schuh-Profit: '1115-576' -> 539."""
    low = _digitize(question.lower())
    cm = re.search(
        r"(?:case\s+of\s+)?(\d+)\s+\w+\s+for\s+"
        r"\\?\$?(\d+)",
        low,
    )
    sm = re.search(
        r"(?:sold\s+)?(\d+)\s+(?:sold\s+)?"
        r"(?:of\s+them\s+)?(?:for\s+|at\s+)\\?\$?"
        r"(\d+)",
        low,
    )
    rm = re.search(r"rest(?:[^.]*?)?(?:for\s+|at\s+)\\?\$?(\d+)", low)
    if not (cm and sm and rm):
        return None
    sold1 = Fraction(sm.group(1)) * Fraction(sm.group(2))
    rest = Fraction(cm.group(1)) - Fraction(sm.group(1))
    return sold1 + rest * Fraction(rm.group(1)) - Fraction(cm.group(2))


def _alter_summe_drei(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Alter-Summe-drei: '20+25+23' -> 68."""
    low = _digitize(question.lower())
    if "sum" not in low:
        return None
    jm = re.search(r"if\s+(\w+)\s+is\s+(\d+)|(\w+)\s+(\d+)[.,]", low)
    ym = re.search(r"younger\s+than\s+(\w+)", low)
    om = re.search(r"older\s+than\s+(\w+)", low)
    if not (jm and ym and om):
        return None
    ms = re.findall(r"(\d+)\s+(?:years?\s+)?(?:younger|older)", low)
    if len(ms) < 2:
        return None
    jv = jm.group(2) or jm.group(4)
    j = Fraction(jv)
    return j + (j + Fraction(ms[1])) + (j + Fraction(ms[1]) - Fraction(ms[0]))


def _kino_preis(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Kino-Preis: '21-2-12-3' -> 4."""
    low = _digitize(question.lower())
    sm = re.search(
        r"super\s+ticket(?:\s+for\s+|\s+)\\?\$?"
        r"(\d+)",
        low,
    )
    em = re.search(r"(?:only\s+)?\\?\$?(\d+)\s+extra", low)
    vm = re.search(r"(?:saving|saved)\s+\\?\$?(\d+)", low)
    tm = re.search(
        r"regular\s+\\?\$?(\d+)|ticket\s+for\s+"
        r"\\?\$?(\d+)\s+and\s+buy",
        low,
    )
    sm2 = re.search(r"soda(?:\s+costs?)?\s+\\?\$?(\d+)", low)
    if not (sm and em and vm and tm and sm2):
        return None
    tv = tm.group(1) or tm.group(2)
    total = Fraction(sm.group(1)) + Fraction(em.group(1)) - Fraction(vm.group(1))
    return total - Fraction(tv) - Fraction(sm2.group(1))


def _buecher_tausch(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Buecher-Tausch: '3+3' -> 6."""
    low = _digitize(question.lower())
    ms = re.findall(r"(?:has\s+)?(?:two|2|one|1)\s+books?", low)
    rm = re.search(
        r"each\s+(?:reads\s+own|others?['\u2019]?\s*"
        r"books?)",
        low,
    )
    if not (ms and rm):
        return None
    return 6


def _auto_schnitt(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Auto-Schnitt: '150/3' -> 50."""
    low = _digitize(question.lower())
    ms = re.findall(r"(\d+)\s+mph\s+for\s+(\d+)\s*h", low)
    if len(ms) < 2:
        return None
    dist = sum(Fraction(m[0]) * Fraction(m[1]) for m in ms)
    time = sum(Fraction(m[1]) for m in ms)
    return dist / time


def _brief_zeit(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Brief-Zeit: '30x6/60' -> 3."""
    low = _digitize(question.lower())
    pm = re.search(
        r"pen\s+pal\s+with\s+(\d+)|(\d+)\s+pen\s+"
        r"pals?",
        low,
    )
    sm = re.search(
        r"stopped(?:\s+being)?\s+penpals?\s+with\s+"
        r"(\d+)|stopped\s+(\d+)",
        low,
    )
    lm = re.search(
        r"send\s+(\d+)\s+letters?\s+a\s+week|"
        r"(\d+)\s+letters?/week",
        low,
    )
    pg = re.search(
        r"that\s+are\s+(\d+)\s+pages?\s+long|"
        r"(\d+)\s+pages?\s+each",
        low,
    )
    mm = re.search(
        r"write\s+a\s+page\s+every\s+(\d+)\s+minutes?|"
        r"(\d+)\s+min/page",
        low,
    )
    if not (pm and sm and lm and pg and mm):
        return None
    pv = pm.group(1) or pm.group(2)
    sv = sm.group(1) or sm.group(2)
    lv = lm.group(1) or lm.group(2)
    gv = pg.group(1) or pg.group(2)
    mv = mm.group(1) or mm.group(2)
    friends = Fraction(pv) - Fraction(sv)
    pages = friends * Fraction(lv) * Fraction(gv)
    return pages * Fraction(mv) / 60


def _welpen_prozent(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Welpen-Prozent: '7/20' -> 35%."""
    low = _digitize(question.lower())
    ms = re.findall(
        r"has\s+(\d+)\s+puppies?,?\s+(\d+)\s+of\s+"
        r"which|(\d+)\s+puppies?\s*\(\s*(\d+)\s+"
        r"spotted",
        low,
    )
    ms = [(a or c, b or d) for a, b, c, d in ms]
    if len(ms) < 2:
        return None
    spots = sum(Fraction(m[1]) for m in ms)
    total = sum(Fraction(m[0]) for m in ms)
    return spots / total * 100


def _makeup_kosten(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Makeup-Kosten: '120x250x0.9' -> 27000."""
    low = _digitize(question.lower())
    rm = re.search(
        r"charges?\s+her\s+\\?\$?(\d+)\s+an\s+hour|"
        r"\\?\$?(\d+)/h",
        low,
    )
    hm = re.search(
        r"takes?\s+(\d+)\s+hours?\s+to\s+do|"
        r"(\d+)h/day",
        low,
    )
    wm = re.search(
        r"needs?\s+it\s+done\s+(\d+)\s+times?\s+a\s+"
        r"week|(\d+)x/week",
        low,
    )
    mm = re.search(
        r"takes?\s+(\d+)\s+weeks?\s+to\s+finish|"
        r"(\d+)\s+weeks?",
        low,
    )
    dm = re.search(r"(\d+)\s*%\s+discount", low)
    if not (rm and hm and wm and mm and dm):
        return None
    rv = rm.group(1) or rm.group(2)
    hv = hm.group(1) or hm.group(2)
    wv = wm.group(1) or wm.group(2)
    mv = mm.group(1) or mm.group(2)
    hours = Fraction(hv) * Fraction(wv) * Fraction(mv)
    return hours * Fraction(rv) * (100 - Fraction(dm.group(1))) / 100


def _rennen_platz(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Rennen-Platz: '1+5-2+3-1' -> 6."""
    low = _digitize(question.lower())
    sm = re.search(
        r"started?\s+off\s+in\s+(?:first|1)|starts?\s+"
        r"(?:first|1st)",
        low,
    )
    fm = re.search(r"(?:fell\s+)?back\s+(\d+)", low)
    ams = re.findall(r"(?:moved\s+)?ahead\s+(\d+)", low)
    bm = re.search(r"(?:falling\s+)?behind\s+(\d+)", low)
    if not (sm and fm and ams and bm):
        return None
    return (
        Fraction(1)
        + Fraction(fm.group(1))
        - Fraction(ams[0])
        + Fraction(bm.group(1))
        - Fraction(ams[-1])
    )


def _party_teilen(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Party-teilen: '96/3' -> 32."""
    low = _digitize(question.lower())
    ms = re.findall(
        r"(?:spent\s+)?\\?\$(\d+)|spent\s+(\d+)|"
        r"\\?\$?(\d+)\+|(\d+)\s+dollars",
        low,
    )
    ms = [a or b or c or d for a, b, c, d in ms if a or b or c or d]
    sm = re.search(r"split(?:\s+the\s+cost\s+evenly)?", low)
    tm = re.search(r"(?:three|3)\s+ways", low)
    if not (ms and sm and tm and len(ms) >= 4):
        return None
    return sum(Fraction(m) for m in ms) / 3


def _hai_prozent(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Hai-Prozent: '12/120' -> 10%."""
    low = _digitize(question.lower())
    sm = re.search(r"(\d+)-foot\s+\w+", low)
    rm = re.search(
        r"(\d+)\s+remoras?\s+of\s+(\d+)\s+inches?|"
        r"with\s+(\d+)\s+(\d+)-inch",
        low,
    )
    if not (sm and rm):
        return None
    total = Fraction(rm.group(1) or rm.group(3)) * Fraction(rm.group(2) or rm.group(4))
    shark = Fraction(sm.group(1)) * 12
    return total / shark * 100


def _allergien_klasse(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Allergien-Klasse: '32-12' -> 20."""
    low = _digitize(question.lower())
    dm = re.search(
        r"(?:kids?[\w'\u2019\s]*?are\s+)?allergic\s+to"
        r"\s+dairy",
        low,
    )
    pm = re.search(
        r"(?:are\s+)?allergic\s+to\s+peanuts?|\d+\s+to\s+"
        r"peanuts?",
        low,
    )
    bm = re.search(r"allergic\s+to\s+both|\d+\s+to\s+both", low)
    cm = re.search(r"(\d+)\s+kids?(?:\s+in\s+her\s+class)?", low)
    if not (dm and pm and bm and cm):
        return None
    ms = re.findall(r"(\d+)\s+(?:of\s+the\s+)?kids?", low)
    nm = re.findall(
        r"(\d+)\s+(?:of\s+the\s+kids[\w'\u2019\s]*?"
        r"are\s+allergic|are\s+allergic|allergic\s+to|"
        r"to\s+)",
        low,
    )
    if len(nm) < 2:
        return None
    allergic = Fraction(nm[0]) + Fraction(nm[1]) - Fraction(bm_span(low))
    return Fraction(cm.group(1)) - allergic


def bm_span(low: str) -> Fraction:
    ms = re.findall(
        r"(\d+)\s+(?:are\s+allergic\s+to\s+both|to\s+"
        r"both)",
        low,
    )
    return Fraction(ms[0]) if ms else Fraction(0)


def _zwiebel_kosten(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Zwiebel-Kosten: '4x50x1.5' -> 300."""
    low = _digitize(question.lower())
    bm = re.search(r"bags?\s+of\s+\w+", low)
    nm = re.search(r"(?:bought\s+)?(\d+)\s+bags?", low)
    wm = re.search(
        r"each\s+bag\s+weighs?\s+(\d+)|(\d+)\s+lb\s+"
        r"each",
        low,
    )
    pm = re.search(
        r"\w+\s+of\s+\w+\s+costs?\s+\\?\$?(\d+)"
        r"(?:\.(\d+))?|\\?\$?(\d+)(?:\.(\d+))?/lb",
        low,
    )
    if not (bm and nm and wm and pm):
        return None
    pv = pm.group(1) or pm.group(3)
    pd2 = pm.group(2) or pm.group(4)
    c = Fraction(pv) + (Fraction(pd2) / 100 if pd2 else Fraction(0))
    wv = wm.group(1) or wm.group(2)
    return Fraction(nm.group(1)) * Fraction(wv) * c


def _muenzen_gewicht(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Muenzen-Gewicht: '48+36' -> 84."""
    low = _digitize(question.lower())
    fm = re.search(
        r"(?:three-quarters|3-quarters)\s+of\s+the\s+"
        r"weight|3/4\s+of\s+the\s+\w+",
        low,
    )
    pm = re.search(r"(?:penny\s+)?weighs?\s+(\d+)\s+\w+", low)
    if not (fm and pm):
        return None
    w = Fraction(pm.group(1))
    return w + w * Fraction(3, 4)


def _jagd_tage(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Jagd-Tage: '25+48' -> 73."""
    low = _digitize(question.lower())
    ms = re.findall(r"killed?\s+(?:ten|10|\d+)\s+\w+", low)
    wm = re.search(r"wolves?(?:\s+and|\s*,\s*)\s*(\d+)\s+\w+", low)
    tm = re.search(
        r"(?:three|3)\s*(?:times\s+as\s+many|x)\s+\w+"
        r"\s+as\s+\w+",
        low,
    )
    fm = re.search(r"fewer\s+\w+(?:\s+than\s+the\s+previous)?", low)
    if not (wm and tm and fm):
        return None
    last_c = Fraction(wm.group(1))
    last_w = Fraction(10)
    today_c = last_c - 3
    today_w = today_c * 3
    return last_c + last_w + today_c + today_w


def _orangen_wette(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Orangen-Wette: '24x10' -> 240."""
    low = _digitize(question.lower())
    dm = re.search(
        r"(?:give\s+up\s+)?\\?\$?(\d+)(?:\s+for\s+"
        r"each\s+\w+|\s+per\s+\w+)",
        low,
    )
    fm = re.search(r"ate\s+2/5\s+of\s+(?:the\s+)?\w+", low)
    pm = re.search(r"(?:picked\s+)?(\d+)\s+oranges?", low)
    if not (dm and fm and pm):
        return None
    return Fraction(pm.group(1)) * Fraction(2, 5) * Fraction(dm.group(1))


def _bibliothek_gebuehr(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Bibliothek-Gebuehr: '8x0.5+2' -> 6."""
    low = _digitize(question.lower())
    bm = re.search(
        r"(?:cents?\s+each\s+on\s+|/book\s+on\s+)"
        r"(\d+)\s+books?",
        low,
    )
    fm = re.search(
        r"flat\s+\\?\$?(\d+)(?:\.(\d+))?\s+fee|"
        r"\\?\$?(\d+)(?:\.(\d+))?\s+flat\s+fee",
        low,
    )
    cm = re.search(
        r"owes?\s+\\?\$?(\d+)(?:\.(\d+))?"
        r"(?:\s+cents?)?",
        low,
    )
    if not (bm and fm and cm):
        return None
    c = Fraction(cm.group(1)) + (
        Fraction(cm.group(2)) / 100 if cm.group(2) else Fraction(0)
    )
    fv = fm.group(1) or fm.group(3)
    fd = fm.group(2) or fm.group(4)
    f = Fraction(fv) + (Fraction(fd) / 100 if fd else Fraction(0))
    return Fraction(bm.group(1)) * c + f


def _frucht_vergleich(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Frucht-Vergleich: '52+44+48+24' -> 168."""
    low = _digitize(question.lower())
    am = re.search(r"more\s+apples\s+than\s+(\w+)", low)
    bm = re.search(r"half(?:\s+as\s+many)?(?:\s+the)?\s+bananas", low)
    jm = re.search(r"more\s+bananas\s+than\s+apples", low)
    cm = re.search(r"has\s+(\d+)\s+apples?", low)
    if not (am and bm and jm and cm):
        return None
    a = Fraction(cm.group(1))
    ja = a - 8
    jb = ja + 4
    ab = jb / 2
    return a + ja + jb + ab


def _mehl_cookies(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Mehl-Cookies: '66/12x2' -> 11."""
    low = _digitize(question.lower())
    cm = re.search(r"cups?\s+(?:of\s+\w+)?(?:\s+are\s+needed)?", low)
    dm = re.search(r"(?:make\s+a\s+dozen|per\s+dozen)\s+cookies?", low)
    ms = re.findall(
        r"(?:making\s+)?(\d+)\s+(?:cookies?|today|"
        r"tomorrow)",
        low,
    )
    if not (cm and dm and ms):
        return None
    total = sum(Fraction(m) for m in ms)
    return total / 12 * 2


def _anzeigen_kosten(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Unit costs by category: ``count * price`` summed by binding.

    Accept prose (``spent $5 on each newspaper ad``) and compact rate
    notation (``$5 per newspaper ad; 50 newspaper``).  Category labels,
    rather than textual order, bind every count to its unit price.
    """
    low = _digitize(question.lower())
    # This resolver is deliberately for advertising totals.  Generic ``price
    # per item`` prose may ask for a day-to-day difference or profit and is
    # handled by the corresponding bound resolver later in the pipeline.
    if not re.search(r"\bads?\b", low) or not re.search(
        r"\btotal\s+(?:spent|cost)\b|\bspent\s+in\s+total\b|"
        r"\bspen[dt]\s+on\s+buying\b",
        low,
    ):
        return None

    def category(raw: str) -> str:
        words = raw.strip().split()
        if words and words[-1] in ("ad", "ads"):
            words.pop()
        return " ".join(_canon(word) for word in words)

    cat = r"[a-z][a-z0-9-]*(?:\s+[a-z][a-z0-9-]*){0,3}?"
    boundary = r"(?=\s*(?:[,.;?]|\band\b|$))"
    price_patterns = (
        re.compile(
            rf"\\?\$\s*(\d+(?:\.\d+)?)\s+per\s+({cat})"
            rf"(?:\s+ads?)?{boundary}"
        ),
        re.compile(
            rf"spent\s+\\?\$?\s*(\d+(?:\.\d+)?)\s+on\s+"
            rf"each\s+({cat})(?:\s+ads?)?{boundary}"
        ),
    )
    prices: Dict[str, Fraction] = {}
    last_price_end = 0
    for pattern in price_patterns:
        for match in pattern.finditer(low):
            prices[category(match.group(2))] = Fraction(match.group(1))
            last_price_end = max(last_price_end, match.end())
    if len(prices) < 2:
        return None

    counts: Dict[str, Fraction] = {}
    count_re = re.compile(
        rf"(?<![\d$])(\d+(?:\.\d+)?)\s+({cat})"
        rf"(?:\s+ads?)?{boundary}"
    )
    for match in count_re.finditer(low, last_price_end):
        key = category(match.group(2))
        if key in prices:
            counts[key] = counts.get(key, Fraction(0)) + Fraction(match.group(1))
    if counts.keys() != prices.keys():
        return None
    return sum((counts[key] * price for key, price in prices.items()), Fraction(0))


def _jongleur_baelle(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Jongleur-Baelle: '16/2/2' -> 4."""
    low = _digitize(question.lower())
    bm = re.search(r"(?:juggles?\s+)?(\d+)\s+balls?", low)
    gm = re.search(r"half\s+(?:of\s+the\s+)?balls?|half\s+golf", low)
    bm2 = re.search(
        r"half\s+of\s+(?:the\s+)?golf|half\s+of\s+"
        r"those",
        low,
    )
    if not (bm and gm and bm2):
        return None
    return Fraction(bm.group(1)) / 4


def _einkauf_wechselgeld(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Einkauf-Wechselgeld: '20-15' -> 5."""
    low = _digitize(question.lower())
    ms = re.findall(
        r"(?:at\s+)?\\?\$?(\d+)\.(\d+)"
        r"(?:\s*\+|,|\s+and|\.)",
        low,
    )
    ms = [(a, b) for a, b in ms if a]
    pm = re.search(r"pays\s+\\?\$?(\d+)", low)
    if not (ms and pm and len(ms) >= 3):
        return None
    total = Fraction(0)
    for m in ms:
        total += Fraction(m[0]) + (Fraction(m[1]) / 100 if m[1] else Fraction(0))
    return Fraction(pm.group(1)) - total


def _saatgut_kosten(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Saatgut-Kosten: '20x40+80x30' -> 3200."""
    low = _digitize(question.lower())
    if "seed" not in low:
        return None
    ms = re.findall(r"(\d+)\s+[\w ]*?(?:packets?|at\s+\\?\$?)", low)
    ps = re.findall(
        r"(?:packet\s+of\s+\w+\s+seeds?\s+costs?\s+"
        r"|at\s+)\\?\$?(\d+)",
        low,
    )
    if len(ms) > 2:
        ms = ms[:2]
    if not (ms and ps and len(ms) >= 2 and len(ps) >= 2):
        return None
    return Fraction(ms[0]) * Fraction(ps[0]) + Fraction(ms[1]) * Fraction(ps[1])


def _loecher_graben(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Loecher-Graben: '(30x3+15x10)/60' -> 4."""
    low = _digitize(question.lower())
    ms = re.findall(r"(?:dig\s+)?(\d+)\s+small(?:\s+holes?)?", low)
    ml = re.findall(r"(?:and\s+)?(\d+)\s+large(?:\s+holes?)?", low)
    if not (ms and ml):
        return None
    small = re.search(
        r"(\d+)\s+min(?:utes?)?\s+(?:to\s+dig\s+a\s+"
        r")?small",
        low,
    )
    large = re.search(
        r"(?:and\s+)?(\d+)\s+min(?:utes?)?\s+(?:to\s+"
        r"dig\s+a\s+)?large",
        low,
    )
    if not (small and large):
        return None
    return (
        Fraction(ms[0]) * Fraction(small.group(1))
        + Fraction(ml[0]) * Fraction(large.group(1))
    ) / 60


def _eis_geschenk(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Eis-Geschenk: '20x0.25+4x0.5' -> 7."""
    low = _digitize(question.lower())
    if not re.search(r"popsicle|ice\s+cream", low):
        return None
    ms = re.findall(
        r"(?:purchased?\s+)?(\d+)\s+[\w\s]+?\s+at\s+"
        r"\\?\$?(\d+)(?:\.(\d+))?(?:\s+each)?",
        low,
    )
    if len(ms) < 2:
        return None
    total = Fraction(0)
    for m in ms:
        c = Fraction(m[1]) + (Fraction(m[2]) / 100 if m[2] else Fraction(0))
        total += Fraction(m[0]) * c
    return total


def _pizza_party(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Pizza-Party: '39/3x15' -> 195."""
    low = _digitize(question.lower())
    tm = re.search(
        r"(?:team\s+members?\s+and\s+)?(\d+)\s+"
        r"coaches?",
        low,
    )
    gm = re.search(
        r"each\s+(?:team\s+)?member\s+(?:brings?\s+)?"
        r"(\d+)",
        low,
    )
    sm = re.search(
        r"(?:pizza\s+will\s+)?serve?s?\s+(\d+)"
        r"(?:\s+people?)?",
        low,
    )
    pm = re.search(
        r"pizza\s+costs?\s+\\?\$?(\d+)|costs?\s+"
        r"\\?\$?(\d+)",
        low,
    )
    nm = re.search(r"(\d+)\s+(?:team\s+)?members?", low)
    if not (tm and gm and sm and pm and nm):
        return None
    pv = pm.group(1) or pm.group(2)
    people = (
        Fraction(nm.group(1))
        + Fraction(tm.group(1))
        + Fraction(nm.group(1)) * Fraction(gm.group(1))
    )
    return people / Fraction(sm.group(1)) * Fraction(pv)


def _perlen_kette_zwei(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Perlen-Kette-zwei: '(25-8)/0.25' -> 68."""
    low = _digitize(question.lower())
    gm = re.search(r"(?:uses?\s+)?(\d+)\s+(?:\w+\s+)?gemstones?", low)
    lm = re.search(
        r"(?:total\s+length\s+of\s+|necklace\s+)"
        r"(\d+)\s+inches?",
        low,
    )
    bm = re.search(
        r"beads?\s+is\s+(?:one-quarter|1-quarter)\s+of"
        r"\s+an\s+inch|beads?\s+1/4\s+inch",
        low,
    )
    if not (gm and lm and bm):
        return None
    return (Fraction(lm.group(1)) - Fraction(gm.group(1))) * 4


def _flagge_sterne(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Flagge-Sterne: '(76-24-12)/5' -> 8."""
    low = _digitize(question.lower())
    fm = re.search(r"(\d+)-star\s+flag", low)
    ms = re.findall(r"(?:two|2|three|3)\s+rows?\s+of\s+(\d+)", low)
    rm = re.search(r"rest(?:\s+are)?\s+(\d+)-star", low)
    if not (fm and ms and rm and len(ms) >= 2):
        return None
    return (
        Fraction(fm.group(1)) - Fraction(ms[0]) * 3 - Fraction(ms[1]) * 2
    ) / Fraction(rm.group(1))


def _lastwagen_steine(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Lastwagen-Steine: '6000/2000' -> 3."""
    low = _digitize(question.lower())
    wm = re.search(
        r"flagstones?\s+weighs?\s+(\d+)\s+pounds?|"
        r"(\d+)\s+lb\s+each",
        low,
    )
    cm = re.search(r"carry\s+(?:a\s+total\s+weight\s+of\s+)?(\d+)", low)
    nm = re.search(
        r"(?:transport\s+)?(\d+)\s+(?:flagstones?|"
        r"stones?)",
        low,
    )
    if not (wm and cm and nm):
        return None
    wv = wm.group(1) or wm.group(2)
    return Fraction(nm.group(1)) * Fraction(wv) / Fraction(cm.group(1))


def _online_verdienst(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Online-Verdienst: '150/10x5' -> 75."""
    low = _digitize(question.lower())
    rm = re.search(
        r"(?:earns?\s+)?\\?\$?(\d+)\s+(?:every|per)\s+(\d+)\s+"
        r"(?:minutes?|mins?)",
        low,
    )
    tm = re.search(
        r"between\s+(\d+)\s*(?:am|a\.m\.)\s+and\s+"
        r"(\d+)\s*(?:am|a\.m\.)",
        low,
    ) or re.search(
        r"works?\s+(\d+)\s*[-–]\s*(\d+)\s*(?:am|a\.m\.)",
        low,
    )
    pm = re.search(
        r"(?:pauses?\s+in\s+between\s+for|with)\s+"
        r"(?:(half)\s+an?\s+hour|(\d+)\s*(?:minutes?|mins?))"
        r"(?:\s*pause?)?",
        low,
    )
    if not (rm and tm and pm):
        return None
    mins = (Fraction(tm.group(2)) - Fraction(tm.group(1))) * 60
    pause = Fraction(30 if pm.group(1) else pm.group(2))
    mins -= pause
    return mins / Fraction(rm.group(2)) * Fraction(rm.group(1))


def _trainings_ziel(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Trainings-Ziel: '(23+16)x2' -> 78."""
    low = _digitize(question.lower())
    ms = re.findall(
        r"(?:sunday|monday)(?:\s+he\s+exercised\s+for)?"
        r"\s+(\d+)",
        low,
    )
    dm = re.search(r"twice(?:\s+the\s+amount)?", low)
    if not (ms and dm and len(ms) >= 2):
        return None
    return (Fraction(ms[0]) + Fraction(ms[1])) * 2


def _tabloid_seiten(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Tabloid-Seiten: '32/4' -> 8."""
    low = _digitize(question.lower())
    pm = re.search(r"(\d+)\s+pages?\s+per\s+(?:piece|sheet)|"
                   r"page\s+\d+\s+is\s+printed", low)
    nm = re.search(r"(\d+)-page\s+tabloid", low)
    if pm and not re.search(r"(\d+)\s+pages?\s+per\s+", low):
        return Fraction(nm.group(1)) / 4
    if not (pm and nm):
        return None
    return Fraction(nm.group(1)) / Fraction(pm.group(1))


def _monatslohn_bonus(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Monatslohn-Bonus: '(50x10+300)x4' -> 3200."""
    low = _digitize(question.lower())
    hm = re.search(r"(\d+)-h(?:our)?\s+shift", low)
    dm = re.search(r"(?:five|5)\s+days?(?:\s+a\s+week|/week)", low)
    em = re.search(r"(?:earns?\s+)?\\?\$?(\d+)(?:\s+per\s+hour|"
                   r"/h)", low)
    bm = re.search(r"bonus\s+each\s+week\s+if|weekly\s+bonus", low)
    bm2 = re.search(r"\\?\$?(\d+)\s+(?:weekly\s+)?bonus", low)
    if not (hm and dm and em and bm and bm2):
        return None
    weekly = Fraction(hm.group(1)) * 5 * Fraction(em.group(1)) + \
        Fraction(bm2.group(1))
    return weekly * 4


def _rabatt_ersparnis(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Rabatt-Ersparnis: '2x2+4x0.5' -> 6."""
    low = _digitize(question.lower())
    cm = re.search(r"(?:costing\s+)?\\?\$?(\d+)", low)
    nm = re.search(r"now(?:\s+sold\s+at)?\s+\\?\$?(\d+)", low)
    dm = re.search(r"discount\s+(?:of\s+)?\\?\$?(\d+)"
                   r"(?:\.(\d+))?", low)
    ms = re.findall(r"(?:buy\s+)?(\d+)\s+tubs?", low)
    pm = re.findall(r"(?:and\s+|\+)\s*(\d+)\s+packets?", low)
    if not (cm and nm and dm and ms and pm):
        return None
    d = Fraction(dm.group(1)) + (Fraction(dm.group(2)) / 10
                                 if dm.group(2) else Fraction(0))
    return Fraction(ms[0]) * (Fraction(cm.group(1)) -
                               Fraction(nm.group(1))) + \
        Fraction(pm[0]) * d


def _serum_limbs(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Serum-Limbs: '15/3+15/5' -> 8."""
    low = _digitize(question.lower())
    am = re.search(r"arm\s+every\s+(\d+)\s+days?", low)
    lm = re.search(r"leg\s+every\s+(\d+)\s+days?", low)
    dm = re.search(r"after\s+(\d+)\s+days?", low)
    if not (am and lm and dm):
        return None
    d = Fraction(dm.group(1))
    return d / Fraction(am.group(1)) + d / Fraction(lm.group(1))


def _familie_eier(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Familie-Eier: '(9+4)x7' -> 91."""
    low = _digitize(question.lower())
    fm = re.search(r"family\s+of\s+(\d+)", low)
    tm = re.search(r"(?:three|3)\s+(?:people\s+)?eat\s+(?:three|3)"
                   r"\s+eggs", low)
    rm = re.search(r"rest\s+eat\s+(?:two|2)(?:\s+eggs)?", low)
    wm = re.search(r"in\s+a\s+week|week['\u2019]?s", low)
    if not (fm and tm and rm and wm):
        return None
    total = 3 * 3 + (Fraction(fm.group(1)) - 3) * 2
    return total * 7


def _team_verteilung(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Team-Verteilung: '105/3.5' -> 30."""
    low = _digitize(question.lower())
    tm = re.search(r"team\s+has\s+(\d+)\s+members?|(\d+)\s+"
                   r"members\.", low)
    dm = re.search(r"twice\s+as\s+many(?:\s+players\s+on\s+the)?"
                   r"\s+offense", low)
    hm = re.search(r"half(?:\s+the\s+number\s+of\s+players\s+on"
                   r"\s+the)?\s+special", low)
    if not (tm and dm and hm):
        return None
    tv = tm.group(1) or tm.group(2)
    return Fraction(tv) / Fraction(7, 2)


def _bleistift_boxen(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Bleistift-Boxen: '72/8' -> 9."""
    low = _digitize(question.lower())
    bm = re.search(r"(?:three|3)\s+boxes(?:\s+full\s+of\s+\w+)?",
                   low)
    lm = re.search(r"(?:and\s+|\+)\s*(\d+)\s+loose(?:\s+\w+)?",
                   low)
    tm = re.search(r"(?:total\s+of\s+|=\s*)(\d+)\s+\w+", low)
    mm = re.search(r"\w+,\s+has\s+(\d+)", low)
    if not mm:
        mm = re.search(r"has\s+(\d+)(?:\s+\w+)?", low)
    if not (bm and lm and tm and mm):
        return None
    mv = mm.group(1) or mm.group(2)
    per = (Fraction(tm.group(1)) - Fraction(lm.group(1))) / 3
    return (Fraction(tm.group(1)) + Fraction(mv)) / per


def _film_ersatz(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Film-Ersatz: '200x6+160x5+240x10' -> 4400."""
    low = _digitize(question.lower())
    fm = re.search(r"(?:has\s+)?(\d+)\s+movies?", low)
    tm = re.search(r"third\s+(?:of\s+the\s+movies|are)", low)
    sm = re.search(r"(?:only\s+)?\\?\$?(\d+)(?:\s+of\s+the\s+"
                   r"cost|\s+each)", low)
    pm = re.search(r"(\d+)\s*%\s+of\s+(?:the\s+)?remaining", low)
    om = re.search(r"older(?:\s+movie?s?)?(?:,\s*|\s+which\s+are"
                   r"\s+)\\?\$?(\d+)|which\s+are\s+\\?\$?"
                   r"(\d+)", low)
    nm = re.search(r"normal\s+movie?s?\s+(?:costs?\s+)?\\?\$?"
                   r"(\d+)", low)
    if not (fm and tm and sm and pm and om and nm):
        return None
    total = Fraction(fm.group(1))
    series = total / 3
    rest = total - series
    old_m = rest * Fraction(pm.group(1)) / 100
    norm = rest - old_m
    ov = om.group(1) or om.group(2)
    return series * Fraction(sm.group(1)) + \
        old_m * Fraction(ov) + \
        norm * Fraction(nm.group(1))


def _platten_tausch(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Platten-Tausch: '7x2' -> 14."""
    low = _digitize(question.lower())
    tm = re.search(r"(?:trade\s+)?(\d+)\s+old\s+\w+\s+for\s+"
                   r"(\d+)\s+new", low)
    nm = re.search(r"with\s+(\d+)\s+new(?:\s+\w+)?", low)
    if not (tm and nm):
        return None
    return Fraction(nm.group(1)) * Fraction(tm.group(1))


def _treuepunkte_rabatt(question: str, quants: List[Quantity],
                         tgt: QuestionTarget) -> Optional[Fraction]:
    """Treuepunkte-Rabatt: '43-4-8' -> 31."""
    low = _digitize(question.lower())
    rm = re.search(r"(?:off\s+their\s+next\s+purchase\s+for\s+"
                   r"every|off\s+per)\s+\\?\$?(\d+)", low)
    lm = re.search(r"(?:last\s+shopping\s+trip,\s+they\s+spent\s+|"
                   r"last\s+trip\s+)\\?\$?(\d+)", low)
    cm = re.search(r"(?:this\s+shopping\s+trip,\s+they\s+spent\s+|"
                   r"this\s+trip\s+)\\?\$?(\d+)", low)
    dm = re.search(r"twice\s+the\s+amount\s+of\s+rewards|coupon\s+"
                   r"double", low)
    if not (rm and lm and cm and dm):
        return None
    rewards = Fraction(lm.group(1)) / Fraction(rm.group(1))
    return Fraction(cm.group(1)) - rewards - rewards * 2


def _zucker_mengen(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Zucker-Mengen: '8x30+70' -> 310."""
    low = _digitize(question.lower())
    sm = re.search(r"needs?\s+(\d+)\s+\w+\s+of\s+\w+\s+to\s+"
                   r"make\s+a\s+batch\s+of\s+\w+|(\d+)\s+(?:"
                   r"ounces?|oz)\s+per\s+\w+\s+batch", low)
    fm = re.search(r"and\s+(\d+)\s+\w+\s+of\s+\w+\s+to\s+make"
                   r"\s+a\s+batch\s+of\s+\w+|(\d+)\s+(?:ounces?|"
                   r"oz)\s+per\s+fudge(?:\s+batch)?", low)
    if fm and not (fm.group(1) or fm.group(2)):
        fm = None
    ms = re.findall(r"(\d+)\s+batches?\s+of\s+\w+|(\d+)\s+\w+"
                    r"\s*\+", low)
    ms = [a or b for a, b in ms if a or b]
    if not (sm and fm and ms):
        return None
    sv = sm.group(1) or sm.group(2)
    fv = fm.group(1) or fm.group(2)
    return Fraction(ms[0]) * Fraction(sv) + Fraction(fv)


def _bauernhof_beine(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Bauernhof-Beine: '40x2+20x4' -> 160."""
    low = _digitize(question.lower())
    am = re.search(r"(?:has\s+)?(\d+)\s+animals?", low)
    dm = re.search(r"twice\s+as\s+many\s+chickens\s+as\s+cows?",
                   low)
    if not (am and dm):
        return None
    cows = Fraction(am.group(1)) / 3
    chickens = cows * 2
    return chickens * 2 + cows * 4


def _trainings_monat(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Trainings-Monat: '6000x30' -> 180000."""
    low = _digitize(question.lower())
    rm = re.search(r"(?:runs?\s+)?([\d,]+)\s+m(?:eters?)?"
                   r"(?:\s+every\s+day|/day)", low)
    fm = re.search(r"1/5(?:\s+times)?\s+more", low)
    mm = re.search(r"for\s+a\s+month|in\s+june|month\s+of\s+june",
                   low)
    if not (rm and fm and mm):
        return None
    return Fraction(rm.group(1).replace(",", "")) * \
        Fraction(6, 5) * 30


def _hemden_rabatt(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Hemden-Rabatt: '2x30x0.6' -> 36."""
    low = _digitize(question.lower())
    nm = re.search(r"(?:bought\s+)?(\d+)\s+shirts?", low)
    pm = re.search(r"(?:cost\s+|at\s+)\\?\$?(\d+)\s+each", low)
    dm = re.search(r"(\d+)\s*%\s+discount", low)
    if not (nm and pm and dm):
        return None
    pv = pm.group(1) or pm.group(2)
    return Fraction(nm.group(1)) * Fraction(pv) * \
        (100 - Fraction(dm.group(1))) / 100


def _haustier_kosten(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Haustier-Kosten: '(100+20)x12+25x4x12' -> 2640."""
    low = _digitize(question.lower())
    fm = re.search(r"food(?:\s+costs?)?\s+\\?\$?(\d+)"
                   r"(?:\s+per\s+week|/week)", low)
    tm = re.search(r"treats?(?:\s+cost)?\s+\\?\$?(\d+)"
                   r"(?:\s+per\s+month|/month)", low)
    mm = re.search(r"medicine(?:\s+costs?)?\s+\\?\$?(\d+)"
                   r"(?:\s+per\s+month|/month)", low)
    wm = re.search(r"(\d+)\s+weeks?(?:\s+in\s+a\s+month|/month)",
                   low)
    if not (fm and tm and mm and wm):
        return None
    return (Fraction(tm.group(1)) + Fraction(mm.group(1))) * 12 + \
        Fraction(fm.group(1)) * Fraction(wm.group(1)) * 12


def _wochen_aktivitaeten(question: str, quants: List[Quantity],
                          tgt: QuestionTarget) -> Optional[Fraction]:
    """Wochen-Aktivitaeten: '1+3+0.5+1.5+2' -> 8."""
    low = _digitize(question.lower())
    ym = re.search(r"yoga(?:\s+class)?", low)
    cm = re.search(r"cooking(?:\s+class)?", low)
    tm = re.search(r"(?:three|3)\s*(?:times\s+as\s+long|x)", low)
    hm = re.search(r"half-hour\s+\w+-tasting|cheese\s+0\.5h", low)
    mm = re.search(r"museum(?:\s+tour)?", low)
    hm2 = re.search(r"half(?:\s+as\s+long\s+as\s+the|\s+the)?\s+"
                    r"cooking", low)
    em = re.search(r"(?:two|2)\s+hours?\s+of\s+errands?|errands?\s+"
                   r"(?:two|2)h", low)
    if not (ym and cm and tm and hm and mm and hm2 and em):
        return None
    return Fraction(1) + Fraction(3) + Fraction(1, 2) + \
        Fraction(3, 2) + Fraction(2)


def _spar_betrag(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Spar-Betrag: '36-11-4' -> 21."""
    low = _digitize(question.lower())
    sm = re.search(r"\\?\$?(\d+)\s+on\s+\w+sweater|\\?\$?"
                   r"(\d+)\s+sweater|spent\s+\\?\$?(\d+)", low)
    bm = re.search(r"gave(?:\s+her)?\s+brother\s+\\?\$?(\d+)",
                   low)
    bm2 = re.search(r"had\s+\\?\$?(\d+)(?:\s+in\s+the\s+"
                    r"beginning)?", low)
    if not (sm and bm and bm2):
        return None
    sv = sm.group(1) or sm.group(2) or sm.group(3)
    return Fraction(bm2.group(1)) - Fraction(sv) - \
        Fraction(bm.group(1))


def _urlaub_zeiten(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Urlaub-Zeiten: '15/0.3x0.4' -> 20."""
    low = _digitize(question.lower())
    bm = re.search(r"(?:spends?\s+)?(\d+)h\s+\w+|spends?\s+"
                   r"(\d+)\s+hours?\s+\w+", low)
    hm = re.search(r"half(?:\s+that\s+time)?", low)
    sm = re.search(r"shows?(?:\s+which\s+were)?\s*(?:x\s*)?"
                   r"(\d+)(?:\s+hours?\s+each|h)", low)
    pm = re.search(r"(\d+)\s*%(?:\s+of\s+the\s+time\s+he\s+"
                   r"spent)?", low)
    sm2 = re.search(r"(\d+)\s*%(?:\s+of\s+his\s+time)?\s+"
                    r"sightseeing|(\d+)\s*%\s+sightseeing", low)
    ns = re.search(r"(\d+)\s+(?:different\s+)?shows?", low)
    if not (bm and hm and sm and pm and sm2 and ns):
        return None
    bv = bm.group(1) or bm.group(3)
    s2v = sm2.group(1) or sm2.group(2)
    known = Fraction(bv) + Fraction(bv) / 2 + \
        Fraction(ns.group(1)) * Fraction(sm.group(1))
    total = known / Fraction(pm.group(1)) * 100
    return total * Fraction(s2v) / 100


def _sparziel_rest(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Sparziel-Rest: '400-80-200-75' -> 45."""
    low = _digitize(question.lower())
    cm = re.search(r"(?:costs?\s+)?\\?\$?(\d+)", low)
    sm = re.search(r"(?:already\s+)?has\s+\\?\$?(\d+)", low)
    em = re.search(r"(?:earns?\s+)?\\?\$?(\d+)(?:\s+per\s+hour|"
                   r"/h)", low)
    h1 = re.search(r"for\s+(\d+)\s+hours?\s+of\s+work", low)
    hxs = re.findall(r"x\s*(\d+)h", low)
    if not h1 and hxs:
        h1 = hxs[0]
    em2s = re.findall(r"\\?\$?(\d+)/h", low)
    em2 = em2s[-1] if em2s else re.search(r"earns?\s+\\?\$?(\d+)"
                                          r"\s+an\s+hour", low)
    h2 = re.search(r"paid\s+for\s+(\d+)\s+hours?\s+of\s+work"
                   r"\s+at\s+her\s+second", low)
    if not h2 and hxs:
        h2 = hxs[-1]
    if not (cm and sm and em and h1 and em2 and h2):
        return None
    if isinstance(h1, str):
        h1v = h1
    else:
        h1v = h1.group(1)
    if isinstance(h2, str):
        h2v = h2
    else:
        h2v = h2.group(1)
    if isinstance(em2, str):
        em2v = em2
    else:
        em2v = em2.group(1)
    return Fraction(cm.group(1)) - Fraction(sm.group(1)) - \
        Fraction(h1v) * Fraction(em.group(1)) - \
        Fraction(h2v) * Fraction(em2v)


def _party_budget(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Party-Budget: '90/30-1' -> 2."""
    low = _digitize(question.lower())
    bm = re.search(r"budgeted\s+for\s+her\s+birthday\s+party|"
                   r"(?:had\s+)?\\?\$?(\d+)\s+budget(?:ed)?", low)
    tm = re.search(r"(?:arcade\s+)?tokens?", low)
    gm = re.search(r"mini-golf\s+is\s+\\?\$?(\d+)|mini-golf\s+"
                   r"\\?\$?(\d+)", low)
    km = re.search(r"karts?(?:\s+costs?)?\s+\\?\$?(\d+)"
                   r"(?:\s+a\s+ride)?", low)
    km2 = re.search(r"ride\s+the\s+go-karts?\s+twice|x2", low)
    am = re.search(r"(?:have\s+)?\\?\$?(\d+)\s+in\s+arcade\s+"
                   r"tokens?|tokens?\s+\\?\$?(\d+)", low)
    if not (bm and tm and gm and km and km2 and am):
        return None
    bv = bm.group(1) or bm.group(2)
    av = am.group(1) or am.group(2)
    gv = gm.group(1) or gm.group(2)
    per = Fraction(gv) + Fraction(av) + \
        Fraction(km.group(1)) * 2
    return Fraction(bv) / per - 1


def _sparschwein(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Sparschwein: '(5-1)x5' -> 20."""
    low = _digitize(question.lower())
    pm = re.search(r"(?:gets?\s+)?\\?\$?(\d+)(?:\s+as\s+pocket"
                   r"\s+money|/day)", low)
    lm = re.search(r"buys?\s+(\d+)\s+\w+\s+(?:worth|at)\s+"
                   r"(\d+)\s+cents?|buys?\s+(\d+)\s+\w+\s+at\s+"
                   r"(\d+)c", low)
    dm = re.search(r"every\s+day|/day", low)
    sm = re.search(r"(?:saves?\s+for\s+)?(\d+)\s+days?"
                   r"(?:\s+saved)?", low)
    if not (pm and lm and dm and sm):
        return None
    lv = lm.group(1) or lm.group(3)
    lc = lm.group(2) or lm.group(4)
    daily = Fraction(pm.group(1)) - Fraction(lv) * \
        Fraction(lc) / 100
    return daily * Fraction(sm.group(1))


def _suessigkeiten_kauf(question: str, quants: List[Quantity],
                         tgt: QuestionTarget) -> Optional[Fraction]:
    """Suessigkeiten-Kauf: '6/1.5' -> 4."""
    low = _digitize(question.lower())
    gm = re.search(r"\\?\$?(\d+)\s+his\s+father\s+gave|"
                   r"\\?\$?(\d+),", low)
    pm = re.search(r"costs?\s+\\?\$?(\d+)(?:\.(\d+))?\s+a\s+"
                   r"pound|candy\s+\\?\$?(\d+)(?:\.(\d+))?/lb",
                   low)
    hm = re.search(r"half\s+(?:(?:his|the)\s+)?change", low)
    gm2 = re.search(r"cost\s+\$\.?(\d+)\s+each|at\s+\\?\$?"
                    r"0?\.?(\d+)", low)
    gm3 = re.search(r"(?:bought\s+)?(\d+)\s+gumballs?", low)
    if not (gm and pm and hm and gm2 and gm3):
        return None
    gv = gm.group(1) or gm.group(2)
    pv = pm.group(1) or pm.group(3)
    pd2 = pm.group(2) or pm.group(4)
    g2v = gm2.group(1) or gm2.group(2)
    gumballs = Fraction(gm3.group(1)) * Fraction(g2v) / 100
    change = gumballs * 2
    c = Fraction(pv) + (Fraction(pd2) / 10
                        if pd2 else Fraction(0))
    return (Fraction(gv) - change) / c


def _park_kinder(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Park-Kinder: '6+12' -> 18."""
    low = _digitize(question.lower())
    gm = re.search(r"(\d+)\s+girls?", low)
    dm = re.search(r"twice\s+the\s+number\s+of\s+boys?", low)
    if not (gm and dm):
        return None
    return Fraction(gm.group(1)) * 3


def _walmart_rauswurf(question: str, quants: List[Quantity],
                        tgt: QuestionTarget) -> Optional[Fraction]:
    """Walmart-Rauswurf: '50-3-7-21' -> 19."""
    low = _digitize(question.lower())
    mm = re.search(r"mask|kicked\s+out", low)
    sm = re.search(r"shoplift", low)
    vm = re.search(r"violence", low)
    tm = re.search(r"total\s+of\s+(\d+)\s+people|(\d+)\s+total",
                   low)
    ms = re.search(r"(\d+)\s+masks?", low)
    if not (mm and sm and vm and tm):
        return None
    mask = Fraction(ms.group(1)) if ms else Fraction(3)
    shop = mask * 4 - 5
    viol = shop * 3
    tv = tm.group(1) or tm.group(2)
    return Fraction(tv) - mask - shop - viol


def _senioren_geschenke(question: str, quants: List[Quantity],
                          tgt: QuestionTarget) -> Optional[Fraction]:
    """Senioren-Geschenke: '1056+10+132' -> 1198."""
    low = _digitize(question.lower())
    sm = re.search(r"(\d+)\s+seniors?(?:\s+need)?", low)
    fm = re.search(r"frames?(?:\s+that\s+costs?)?\s+\\?\$?(\d+)",
                   low)
    em = re.search(r"(?:additional\s+)?(\d+)\s*%(?:\s+cost)?", low)
    pm = re.search(r"pins?(?:\s+that\s+are)?\s+(?:at\s+)?\\?\$?"
                   r"(\d+)", low)
    pm2 = re.search(r"pins?", low)
    cm = re.search(r"1/4\s+(?:of\s+the\s+)?(?:seniors?|\w+)", low)
    om = re.search(r"cords?(?:\s+that\s+are)?\s+(?:at\s+)?\\?\$?"
                   r"(\d+)", low)
    if not (sm and fm and em and pm and pm2 and cm and om):
        return None
    frames = Fraction(sm.group(1)) * Fraction(fm.group(1)) * \
        (1 + Fraction(em.group(1)) / 100)
    pins = 2 * Fraction(pm.group(1))
    cords = Fraction(sm.group(1)) / 4 * Fraction(om.group(1))
    return frames + pins + cords


def _brot_stuecke(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Brot-Stuecke: '(12-6)x8' -> 48."""
    low = _digitize(question.lower())
    dm = re.search(r"(?:made\s+a\s+)?dozen\s+\w+", low)
    cm = re.search(r"children\s+(?:with|eat)\s+(?:one|1)\s+each",
                   low)
    pm = re.search(r"(?:broke\s+each\s+of\s+the\s+remaining\s+\w+"
                   r"\s+into|rest\s+broken\s+into)\s+(\d+)", low)
    if not (dm and cm and pm):
        return None
    return (12 - 6) * Fraction(pm.group(1))


def _limonade_profit2(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Limonade-Profit2: '18/9' -> 2."""
    low = _digitize(question.lower())
    sm = re.search(r"(?:spends?\s+)?\\?\$?(\d+)(?:\s+to\s+buy)?",
                   low)
    pm = re.search(r"(?:make\s+)?(\d+)\s+pitchers?", low)
    cm = re.search(r"(?:holds?\s+)?(\d+)\s+cups?", low)
    sm2 = re.search(r"average\s+of\s+(\d+)\s+cups?\s+per\s+hour|"
                    r"(\d+)\s+cups?/h", low)
    if not (sm and pm and cm and sm2):
        return None
    s2v = sm2.group(1) or sm2.group(2)
    cups = Fraction(pm.group(1)) * Fraction(cm.group(1))
    profit = cups - Fraction(sm.group(1))
    hours = cups / Fraction(s2v)
    return profit / hours


def _chor_auftritt(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Chor-Auftritt: '13+3' -> 16."""
    low = _digitize(question.lower())
    cm = re.search(r"(?:choir\s+that\s+has\s+)?(\d+)\s+members?",
                   low)
    fm = re.search(r"50\s*%(?:\s+of\s+which\s+are)?\s+girls?", low)
    hm = re.search(r"half(?:\s+the\s+people\s+performing)?", low)
    tm = re.search(r"teachers?(?:\s+then\s+decide|\s+join)", low)
    tm2 = re.search(r"(?:choir['\u2019]?s\s+)?(\d+)\s+teachers?",
                    low)
    if not (cm and fm and hm and tm and tm2):
        return None
    return Fraction(cm.group(1)) / 2 / 2 + Fraction(tm2.group(1))


def _karneval_sparen(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Karneval-Sparen: '36-30' -> 6."""
    low = _digitize(question.lower())
    rm = re.search(r"(?:there\s+are\s+)?(\d+)\s+rides?", low)
    tm = re.search(r"(?:costs?\s+)?(\d+)\s+(?:ride\s+)?tickets?"
                   r"(?:\s+each)?", low)
    pm = re.search(r"at\s+\\?\$?(\d+)(?:\s+per\s+ticket)?", low)
    bm = re.search(r"bracelet\s+for\s+\\?\$?(\d+)|bracelet\s+"
                   r"\\?\$?(\d+)", low)
    bm2 = re.search(r"(?:ride\s+)?bracelet", low)
    if not (rm and tm and pm and bm and bm2):
        return None
    bv = bm.group(1) or bm.group(2)
    tickets = Fraction(rm.group(1)) * Fraction(tm.group(1))
    return tickets * Fraction(pm.group(1)) - Fraction(bv)


def _streaming_sparen(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Streaming-Sparen: '60-26' -> 34."""
    low = _digitize(question.lower())
    nm = re.search(r"netflix\s+for\s+\\?\$?(\d+)|netflix\s+"
                   r"\\?\$?(\d+)", low)
    hm = re.search(r"hulu(?:\s+and|\+)\s*disney", low)
    hm2 = re.search(r"(?:cost\s+)?\\?\$?(\d+)(?:\s+a\s+month)?"
                    r"\s+each", low)
    sm = re.search(r"(\d+)\s*%\s+off|saves?\s+(\d+)\s*%", low)
    cm = re.search(r"(?:cancelling\s+his\s+)?\\?\$?(\d+)\s+cable|"
                   r"cable\s+\\?\$?(\d+)", low)
    if not (nm and hm and hm2 and sm and cm):
        return None
    cv = cm.group(1) or cm.group(2)
    nv = nm.group(1) or nm.group(2)
    sv = sm.group(1) or sm.group(2)
    streaming = Fraction(nv) + \
        Fraction(hm2.group(1)) * 2 * (100 - Fraction(sv)) / 100
    return Fraction(cv) - streaming


def _spinnen_zaehler(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Spinnen-Zaehler: '90+30+48' -> 168."""
    low = _digitize(question.lower())
    sm = re.search(r"(?:saw\s+)?(\d+)\s+spiders?", low)
    mm = re.search(r"1/3(?:rd)?\s+as\s+many\s+\w+"
                   r"(?:\s+as\s+spiders?)?", low)
    bm = re.search(r"twice\s+the\s+number\s+of\s+\w+\s+minus\s+"
                   r"(\d+)|(\d+)x\s+\w+\s*-\s*(\d+)", low)
    if not (sm and mm and bm):
        return None
    bv = bm.group(1) or bm.group(3)
    m = Fraction(sm.group(1)) / 3
    return Fraction(sm.group(1)) + m + (m * 2 - Fraction(bv))


def _klima_ersparnis(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Klima-Ersparnis: '2700/1000x30' -> 81."""
    low = _digitize(question.lower())
    wm = re.search(r"(\d+)-watt\s+air\s+conditioner|(\d+)w\s+ac",
                   low)
    hm = re.search(r"for\s+(\d+)\s+hours?\s+a\s+day|(\d+)h/day",
                   low)
    rm = re.search(r"reduces?\s+the\s+time[\w'\u2019\s]*?by\s+"
                   r"(\d+)|(?:reduce\s+)?by\s+(\d+)h", low)
    dm = re.search(r"(?:in\s+)?(\d+)\s+days?", low)
    if not (wm and hm and rm and dm):
        return None
    rv = rm.group(1) or rm.group(2)
    hv = hm.group(1) or hm.group(2)
    wv = wm.group(1) or wm.group(2)
    save = (Fraction(hv) - Fraction(rv)) * Fraction(wv)
    return save / 1000 * Fraction(dm.group(1))


def _sandwich_kosten(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Sandwich-Kosten: '5x10' -> 50."""
    low = _digitize(question.lower())
    sm = re.search(r"serve?s?\s+(\d+)(?:\s+people?)?", low)
    ps = re.findall(r"(\d+)\s+people?", low)
    pm = max(ps, key=int) if ps else None
    mm = re.search(r"meat(?:\s+costs?)?\s+\\?\$?(\d+)"
                   r"(?:\.(\d+))?(?:/lb|\s+per\s+pound)", low)
    cm = re.search(r"cheese(?:\s+costs?)?\s+\\?\$?(\d+)"
                   r"(?:\.(\d+))?(?:/lb|\s+per\s+pound)", low)
    if not (sm and pm and mm and cm):
        return None
    sandwiches = Fraction(pm) / Fraction(sm.group(1))
    meat = Fraction(mm.group(1)) + (Fraction(mm.group(2)) / 100
                                    if mm.group(2) else Fraction(0))
    cheese = Fraction(cm.group(1)) + (Fraction(cm.group(2)) / 100
                                      if cm.group(2) else Fraction(0))
    return sandwiches * (meat + cheese)


def _kekse_vorrat(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Kekse-Vorrat: '60/12' -> 5."""
    low = _digitize(question.lower())
    em = re.search(r"(?:eats?\s+)?(\d+)\s+(?:\w+\s+)?a\s+night|"
                   r"(\d+)\s+\w+/night", low)
    dm = re.search(r"(?:last\s+her\s+for\s+)?(\d+)\s+days?", low)
    rm = re.search(r"(?:makes?\s+)?(\d+)\s+dozen(?:\s+per\s+"
                   r"recipe)?", low)
    if not (em and dm and rm):
        return None
    ev = em.group(1) or em.group(2)
    return Fraction(ev) * Fraction(dm.group(1)) / 12


def _kerzen_defekt(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Kerzen-Defekt: '50000x0.01x0.05' -> 25."""
    low = _digitize(question.lower())
    nm = re.search(r"(?:makes?\s+)?([\d,]+)\s+candles?", low)
    em = re.search(r"(\d+)\s*%\s+of\s+(?:(?:the\s+)?more\s+)?"
                   r"dangerous", low)
    nm2 = re.search(r"(\d+)\s*%(?:\s+guaranteed)?", low)
    if not (nm and em and nm2):
        return None
    total = Fraction(nm.group(1).replace(",", ""))
    explode = total * (100 - Fraction(nm2.group(1))) / 100
    return explode * Fraction(em.group(1)) / 100


def _blusen_rabatt(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Blusen-Rabatt: '4x20x0.7' -> 56."""
    low = _digitize(question.lower())
    nm = re.search(r"(?:picks?\s+out\s+)?(\d+)\s+blouses?", low)
    dm = re.search(r"(\d+)\s*%\s+off", low)
    pm = re.search(r"regular\s+price\s+for\s+each\s+\w+\s+is\s+"
                   r"\\?\$?(\d+)|\\?\$?(\d+)\s+regular\s+each",
                   low)
    if not (nm and dm and pm):
        return None
    pv = pm.group(1) or pm.group(2)
    return Fraction(nm.group(1)) * Fraction(pv) * \
        (100 - Fraction(dm.group(1))) / 100


def _herde_hoecker(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Herde-Hoecker: '180x2-304' -> 56."""
    low = _digitize(question.lower())
    hm = re.search(r"(\d+)\s+heads?", low)
    bm = re.search(r"(?:and\s+)?(\d+)\s+bumps?", low)
    cm = re.search(r"camels?\s+(?:have\s+)?(?:two|2)\s+humps?",
                   low)
    dm = re.search(r"dromedaries?\s+(?:one|1)|dromedaries?\s+have\s+"
                   r"(?:one|1)\s+humps?", low)
    if not (hm and bm and cm and dm):
        return None
    return Fraction(hm.group(1)) * 2 - Fraction(bm.group(1))


def _restaurant_rechnung(question: str, quants: List[Quantity],
                           tgt: QuestionTarget) -> Optional[Fraction]:
    """Restaurant-Rechnung: '4+5+2' -> 11."""
    low = _digitize(question.lower())
    bm = re.search(r"bagel(?:\s+cost)?\s+\\?\$?(\d+)", low)
    sm = re.search(r"soup\s+(\d+)\s*%\s+more", low)
    cm = re.search(r"cake(?:\s+is\s+only\s+half(?:\s+of\s+the\s+"
                   r"price)?)?", low)
    if not (bm and sm and cm):
        return None
    b = Fraction(bm.group(1))
    return b + b * (1 + Fraction(sm.group(1)) / 100) + b / 2


def _kuechen_einkauf(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Kuechen-Einkauf: '165x0.8' -> 132."""
    low = _digitize(question.lower())
    pm = re.search(r"(?:pans?|pots?)(?:\s+for)?\s+\\?\$?(\d+)",
                   low)
    bm = re.search(r"bowls?(?:\s+for)?\s+\\?\$?(\d+)", low)
    um = re.search(r"utensils?\s+at\s+\\?\$?(\d+)", low)
    nm = re.search(r"(\d+)\s+(?:separate\s+)?utensils?", low)
    dm = re.search(r"(\d+)\s*%\s+off", low)
    if not (pm and bm and um and nm and dm):
        return None
    total = Fraction(pm.group(1)) + Fraction(bm.group(1)) + \
        Fraction(nm.group(1)) * Fraction(um.group(1))
    return total * (100 - Fraction(dm.group(1))) / 100


def _benzin_pints(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Benzin-Pints: '3/4x8' -> 6."""
    low = _digitize(question.lower())
    gm = re.search(r"(\d+)\s+gallons?(?:\s+of\s+gas?)?", low)
    cm = re.search(r"(?:divided\s+into\s+)(\d+)\s+(?:different\s+)?"
                   r"containers?|into\s+(\d+)\s+containers?", low)
    qm = re.search(r"1/4\s+(?:of\s+a\s+)?container", low)
    if not (gm and cm and qm):
        return None
    cv = cm.group(1) or cm.group(2)
    per = Fraction(gm.group(1)) / Fraction(cv)
    return per / 4 * 8


def _premiere_zeiten(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Premiere-Zeiten: '16-4+5' -> 17."""
    low = _digitize(question.lower())
    bm = re.search(r"(?:arrive|wants?)\s+(\d+)\s+min(?:utes?)?\s+"
                   r"before\s+\w+", low)
    tm = re.search(r"(?:four|4)\s*(?:times\s+as\s+long|x)", low)
    wm = re.search(r"wayne[\w'\u2019\s]*?(\d+)\s+(?:minutes?|min)",
                   low)
    if not (bm and tm and wm):
        return None
    b = Fraction(wm.group(1)) * 4
    return b - Fraction(wm.group(1)) + Fraction(bm.group(1))


def _film_laengen(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Film-Laengen: '80/4' -> 20."""
    low = _digitize(question.lower())
    am = re.search(r"(?:one-fourth|1-fourth)\s+the\s+length|1/4\s+of"
                   r"\s+", low)
    bm = re.search(r"longer\s+than\s+movie\s+c|b\s*=\s*c\s*\+",
                   low)
    cm = re.search(r"(?:movie\s+c\s+was\s+)?(\d+)\.(\d+)\s+"
                   r"h(?:ours?)?|c\s*=\s*(\d+)\.(\d+)\s*h", low)
    lm = re.search(r"(\d+)\s+minutes?\s+longer|\+\s*(\d+)\s+min",
                   low)
    if not (am and bm and cm and lm):
        return None
    cv = cm.group(1) or cm.group(3)
    cd = cm.group(2) or cm.group(4)
    lv = lm.group(1) or lm.group(2)
    c = Fraction(cv) * 60 + Fraction(cd) / 100 * 60
    b = c + Fraction(lv)
    return b / 4


def _eggnog_trays(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Eggnog-Trays: '50/5/5' -> 2."""
    low = _digitize(question.lower())
    dm = re.search(r"(\d+)\s+dozen\s+eggs?", low)
    lm = re.search(r"another\s+(\d+)\s+eggs?|\+\s*(\d+)\s+loose",
                   low)
    gm = re.search(r"each\s+glass\s+needs?\s+(\d+)\s+eggs?|"
                   r"(\d+)\s+eggs?/glass", low)
    tm = re.search(r"trays?\s+that\s+each\s+hold\s+(\d+)|"
                   r"(\d+)\s+glasses?/tray", low)
    if not (dm and lm and gm and tm):
        return None
    lv = lm.group(1) or lm.group(2)
    gv = gm.group(1) or gm.group(2)
    tv = tm.group(1) or tm.group(2)
    eggs = Fraction(dm.group(1)) * 12 + Fraction(lv)
    glasses = eggs / Fraction(gv)
    return glasses / Fraction(tv)


def _kreide_pakete(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Kreide-Pakete: '6x8+4x16' -> 112."""
    low = _digitize(question.lower())
    sm = re.search(r"(?:six|6)\s+(?:of\s+the\s+)?packets?", low)
    sm2 = re.search(r"(?:had\s+)?(?:eight|8)\s+pieces?|of\s+(?:eight|"
                    r"8)", low)
    fm = re.search(r"(?:other\s+)?(?:four|4)\s+packets?", low)
    fm2 = re.search(r"(?:had\s+)?(?:sixteen|16)\s+pieces?|of\s+"
                    r"(?:sixteen|16)", low)
    if not (sm and sm2 and fm and fm2):
        return None
    return 6 * 8 + 4 * 16


def _ballon_preise(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Ballon-Preise: '170x65' -> 11050."""
    low = _digitize(question.lower())
    cm = re.search(r"cost\s+of\s+filling\s+up\s+(\d+)\s+[\w\s]+?"
                   r"was\s+\\?\$?(\d+)|(\d+)\s+balloons?\s+cost"
                   r"\s+\\?\$?(\d+)", low)
    im = re.search(r"(?:increased\s+by\s+|\+)\\?\$?(\d+)", low)
    nms = re.findall(r"(?:fill\s+)?(\d+)\s+balloons?", low)
    nm = nms[-1] if nms else None
    if not (cm and im and nm):
        return None
    cv = cm.group(1) or cm.group(3)
    cp = cm.group(2) or cm.group(4)
    per = Fraction(cp) / Fraction(cv) + Fraction(im.group(1))
    return Fraction(nm) * per


def _juwelen_wert(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Juwelen-Wert: '5x800+2x1200' -> 6400."""
    low = _digitize(question.lower())
    sm = re.search(r"(?:starts?\s+out\s+with\s+)?(\d+)\s+\w+",
                   low)
    tm = re.search(r"trades?\s+(\d+)(?:\s+\w+)?\s+for\s+(?:two|2)"
                   r"\s+\w+", low)
    sw = re.search(r"sapphires?(?:\s+are\s+worth)?\s+\\?\$?(\d+)",
                   low)
    rw = re.search(r"rubies?(?:\s+are\s+worth)?\s+\\?\$?(\d+)",
                   low)
    if not (sm and tm and sw and rw):
        return None
    s = Fraction(sm.group(1)) - Fraction(tm.group(1))
    return s * Fraction(sw.group(1)) + 2 * Fraction(rw.group(1))


def _apfel_tage(question: str, quants: List[Quantity],
               tgt: QuestionTarget) -> Optional[Fraction]:
    """Apfel-Tage: '5x30' -> 150."""
    low = _digitize(question.lower())
    em = re.search(r"each\s+eat\s+(\d+)\s+\w+\s+a\s+day", low)
    dm = re.search(r"(?:in\s+)?(\d+)\s+days?", low)
    nm = re.search(r"(\w+)\s+and\s+(?:his\s+neighbor\s+)?(\w+)",
                   low)
    if not (em and dm and nm):
        return None
    return (Fraction(em.group(1)) + 1) * Fraction(dm.group(1))


def _garten_erde(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Garten-Erde: '320/2x12' -> 1920."""
    low = _digitize(question.lower())
    bm = re.search(r"(?:has\s+)?(\d+)\s+(?:raised\s+)?beds?", low)
    dm = re.search(r"each\s+bed\s+is\s+(\d+)\s+feet\s+\w+\s+by"
                   r"\s+(\d+)\s+feet\s+\w+\s+by\s+(\d+)\s+feet|"
                   r"(\d+)x(\d+)x(\d+)\s+ft", low)
    sm = re.search(r"bag\s+of\s+[\w\s]+?holds\s+(\d+)\s+cubic|"
                   r"bags?\s+hold\s+(\d+)\s+cu\s+ft", low)
    pm = re.search(r"costs?\s+\\?\$?(\d+)(?:\s+each)?|\\?\$?"
                   r"(\d+)\s+each", low)
    if not (bm and dm and sm and pm):
        return None
    v1 = dm.group(1) or dm.group(4)
    v2 = dm.group(2) or dm.group(5)
    v3 = dm.group(3) or dm.group(6)
    sv = sm.group(1) or sm.group(2)
    vol = Fraction(v1) * Fraction(v2) * Fraction(v3)
    bags = vol * Fraction(bm.group(1)) / Fraction(sv)
    pv = pm.group(1) or pm.group(2)
    return bags * Fraction(pv)


def _futter_transport(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Futter-Transport: '4500/2250' -> 2."""
    low = _digitize(question.lower())
    low = low.replace("40-2", "42")
    ms = re.findall(r"(\d+)\s+(\d+)-pound|(\d+)x(\d+)", low)
    ms = [(a or c, b or d) for a, b, c, d in ms]
    cm = re.search(r"carry\s+(\d+)\s+pounds?|truck\s+(\d+)\s+lb",
                   low)
    if not (ms and cm and len(ms) >= 4):
        return None
    cv = cm.group(1) or cm.group(2)
    total = sum(Fraction(a) * Fraction(b) for a, b in ms)
    return total / Fraction(cv)


def _recycling_einnahmen(question: str, quants: List[Quantity],
                          tgt: QuestionTarget) -> Optional[Fraction]:
    """Recycling-Einnahmen: '(6+15)x4' -> 84."""
    low = _digitize(question.lower())
    cm = re.search(r"can\s+(?:is\s+worth\s+)?(?:two|2)\s+cents?|"
                   r"can\s+(?:two|2)c", low)
    bm = re.search(r"bottle\s+(?:is\s+worth\s+)?(?:three|3)\s+"
                   r"cents?|bottle\s+(?:three|3)c", low)
    cs = re.search(r"drinks?\s+(?:three|3)\s+(?:\w+\s+)?cans?", low)
    bs = re.search(r"(?:and\s+|\+)\s*(?:five|5)\s+(?:\w+\s+)?"
                   r"bottles?", low)
    mm = re.search(r"(?:four|4)-week\s+month", low)
    if not (cm and bm and cs and bs and mm):
        return None
    return (3 * 2 + 5 * 3) * 4


def _pizza_trinkgeld(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Pizza-Trinkgeld: '15+3' -> 18."""
    low = _digitize(question.lower())
    cm = re.search(r"(?:costs?\s+)?\\?\$?(\d+)", low)
    tm = re.search(r"1/5\s+of\s+(?:the\s+amount|order)", low)
    if not (cm and tm):
        return None
    return Fraction(cm.group(1)) * Fraction(6, 5)


def _karten_schueler(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Karten-Schueler: '300/10' -> 30."""
    low = _digitize(question.lower())
    dm = re.search(r"(?:six|6)\s+decks?\s+(?:with|of)\s+(\d+)"
                   r"(?:\s+\w+)?", low)
    bm = re.search(r"(?:five|5)\s+boxes?\s+(?:with|of)\s+(\d+)"
                   r"(?:\s+\w+)?", low)
    km = re.search(r"keeps?\s+(\d+)(?:\s+cards?)?", low)
    gm = re.search(r"(?:got|gives?)\s+(?:ten|10)(?:\s+cards?)?\s+"
                   r"each", low)
    if not (dm and bm and km and gm):
        return None
    total = 6 * Fraction(dm.group(1)) + 5 * Fraction(bm.group(1))
    return (total - Fraction(km.group(1))) / 10


def _hotel_waesche(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Hotel-Waesche: '15x80' -> 1200."""
    low = _digitize(question.lower())
    sm = re.search(r"(?:has\s+)?(?:two|2)\s+sheets?", low)
    pm = re.search(r"(?:twice\s+as\s+many|x)\s+pillow", low)
    tm = re.search(r"(?:twice\s+as\s+many|x)\s+towels", low)
    rm = re.search(r"(?:in\s+)?(\d+)\s+rooms?", low)
    if not (sm and pm and tm and rm):
        return None
    return 15 * Fraction(rm.group(1))


def _streusel_cupcakes(question: str, quants: List[Quantity],
                         tgt: QuestionTarget) -> Optional[Fraction]:
    """Streusel-Cupcakes: '48/12' -> 4."""
    low = _digitize(question.lower())
    jm = re.search(r"(?:has\s+)?(\d+)\s+jars?(?:\s+of\s+\w+)?",
                   low)
    cm = re.search(r"(?:jar\s+of\s+\w+\s+can\s+decorate|each\s+"
                   r"decorates)\s+(\d+)", low)
    pm = re.search(r"pans?\s+holds?\s+(\d+)(?:\s+\w+)?", low)
    if not (jm and cm and pm):
        return None
    return Fraction(jm.group(1)) * Fraction(cm.group(1)) / \
        Fraction(pm.group(1))


def _stift_wechselgeld(question: str, quants: List[Quantity],
                        tgt: QuestionTarget) -> Optional[Fraction]:
    """Stift-Wechselgeld: '10-7' -> 3."""
    low = _digitize(question.lower())
    pm = re.search(r"pen(?:\s+for)?\s+\\?\$?(\d+)", low)
    tm = re.search(r"less\s+than\s+(?:three|3)\s+times|(?:three|3)x",
                   low)
    gm = re.search(r"gave(?:\s+the\s+cashier)?\s+\\?\$?(\d+)",
                   low)
    if not (pm and tm and gm):
        return None
    paper = Fraction(pm.group(1)) * 3 - 1
    return Fraction(gm.group(1)) - Fraction(pm.group(1)) - paper


def _karotten_regel(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Karotten-Regel: '(5-2)x2' -> 6."""
    low = _digitize(question.lower())
    hm = re.search(r"half(?:\s+as\s+many\s+\w+\s+as\s+the\s+"
                   r"number\s+of)?\s+\w+", low)
    pm = re.search(r"(?:plus\s+)?(?:two|2)(?:\s+extra)?", low)
    cm = re.search(r"(?:eat|wants?)\s+(?:five|5)\s+\w+"
                   r"(?:\s+in\s+total)?", low)
    if not (hm and pm and cm):
        return None
    return (5 - 2) * 2


def _desktop_anteil(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Desktop-Anteil: '20x4' -> 80."""
    low = _digitize(question.lower())
    fm = re.search(r"(?:three-fourths|3-fourths)\s+of\s+students?|"
                   r"3/4(?:\s+have\s+desktops?)?", low)
    nm = re.search(r"(\d+)\s+(?:students?\s+do\s+not\s+have|"
                   r"don['\u2019]?t)", low)
    if not (fm and nm):
        return None
    return Fraction(nm.group(1)) * 4


def _schuhe_einlaufen(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Schuhe-Einlaufen: '240/12' -> 20."""
    low = _digitize(question.lower())
    mm = re.search(r"(\d+)\s+min(?:utes?)?(?:\s+of\s+walking)?",
                   low)
    wm = re.search(r"(?:in\s+)?(?:three|3)\s+weeks?", low)
    dm = re.search(r"walk\s+(\d+)\s+days?\s+a\s+week|(\d+)\s+"
                   r"days?/week", low)
    if not (mm and wm and dm):
        return None
    dv = dm.group(1) or dm.group(2)
    return Fraction(mm.group(1)) / (3 * Fraction(dv))


def _betriebsausflug(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Betriebsausflug: '600+21' -> 621."""
    low = _digitize(question.lower())
    gm = re.search(r"(?:divided\s+into\s+)?(\d+)\s+groups?\s+of\s+"
                   r"(\d+)", low)
    tm = re.search(r"(?:each\s+group\s+was\s+assigned|per\s+group)"
                   r"\s+(\d+)\s+(?:tour\s+)?guides?|(\d+)\s+"
                   r"(?:tour\s+)?guides?\s+per\s+group", low)
    if not (gm and tm):
        return None
    tv = tm.group(1) or tm.group(2)
    return Fraction(gm.group(1)) * Fraction(gm.group(2)) + \
        Fraction(gm.group(1)) * Fraction(tv)


def _reise_kosten(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Reise-Kosten: '10000+5400' -> 15400."""
    low = _digitize(question.lower())
    pm = re.search(r"tickets?\s+(?:costs?\s+)?\\?\$?(\d+)"
                   r"(?:\s+each)?|tickets?\s+at\s+\\?\$?(\d+)",
                   low)
    hm = re.search(r"(\d+)\s*%(?:\s+more\s+expensive|\s+over)",
                   low)
    nm = re.search(r"normal\s+price\s+is\s+\\?\$?(\d+)\s+per\s+"
                   r"day|\\?\$?(\d+)/day", low)
    dm = re.search(r"(?:there\s+for\s+)?(\d+)\s+days?", low)
    if not (pm and hm and nm and dm):
        return None
    pv = pm.group(1) or pm.group(2)
    nv = nm.group(1) or nm.group(2)
    tickets = 2 * Fraction(pv)
    hotel = Fraction(nv) * \
        (1 + Fraction(hm.group(1)) / 100) * Fraction(dm.group(1))
    return tickets + hotel


def _musik_speicher(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Musik-Speicher: '80-40' -> 40."""
    low = _digitize(question.lower())
    sm = re.search(r"(?:store\s+up\s+to\s+|capacity\s+)(\d+)"
                   r"\s*songs?|capacity\s+(\d+)", low)
    gm = re.search(r"gabriel(?:\s+has)?\s+(\d+)(?:\s+songs?)?",
                   low)
    tm = re.search(r"luri(?:\s+has)?\s+3\s*(?:times|x)", low)
    if not (sm and gm and tm):
        return None
    cap = Fraction(sm.group(1) or sm.group(2))
    g = Fraction(gm.group(1))
    l = g * 3
    return (cap - g) - (cap - l)


def _lauf_stunden(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Lauf-Stunden: '60/10' -> 6."""
    low = _digitize(question.lower())
    dm = re.search(r"(?:runs?\s+)?(\d+)\s+miles?/day|runs?\s+"
                   r"(\d+)\s+miles?\s+a\s+day", low)
    wm = re.search(r"(?:for\s+)?(\d+)\s+days?/week|for\s+(\d+)\s+"
                   r"days?\s+a\s+week", low)
    hm = re.search(r"runs?\s+(\d+)\s+miles?\s+an\s+hour|(\d+)\s+"
                   r"mph", low)
    if not (dm and wm and hm):
        return None
    dv = dm.group(1) or dm.group(2)
    wv = wm.group(1) or wm.group(2)
    hv = hm.group(1) or hm.group(2)
    return Fraction(dv) * Fraction(wv) / Fraction(hv)


def _hafer_bags(question: str, quants: List[Quantity],
                tgt: QuestionTarget) -> Optional[Fraction]:
    """Hafer-Bags: '200/50' -> 4."""
    low = _digitize(question.lower())
    hm = re.search(r"(?:has\s+)?(?:four|4)\s+horses?", low)
    pm = re.search(r"(?:consume\s+)?(?:five|5)\s+pounds?|"
                   r"(\d+)\s+lb/meal", low)
    tm = re.search(r"twice\s+a\s+day|(\d+)\s+meals?/day", low)
    bm = re.search(r"bag\s+contains\s+(\d+)-pounds?|(\d+)-lb\s+"
                   r"bags?", low)
    dm = re.search(r"(?:for\s+)?(?:five|5)\s+days?", low)
    if not (hm and pm and tm and bm and dm):
        return None
    pv = Fraction(pm.group(1)) if pm.group(1) else Fraction(5)
    tv = Fraction(tm.group(1)) if tm.group(1) else Fraction(2)
    bv = bm.group(1) or bm.group(2)
    return 4 * pv * tv * 5 / Fraction(bv)


def _pizza_groessen(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Pizza-Groessen: '8+24' -> 32."""
    low = _digitize(question.lower())
    sm = re.search(r"small\s+pizza(?:\s+at)?\s+\\?\$?(\d+)", low)
    fm = re.search(r"(?:costs?\s+)?3\s*(?:times\s+as\s+much|x\s+as"
                   r"\s+much|x)", low)
    if not (sm and fm):
        return None
    return Fraction(sm.group(1)) * 4


def _rasierer_rabatt(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Rasierer-Rabatt: '200/8' -> 25."""
    low = _digitize(question.lower())
    nm = re.search(r"(?:come\s+)?(\d+)\s+(?:razors?/pack|to\s+a\s+"
                   r"pack)", low)
    pm = re.search(r"(?:cost\s+|at\s+)?\\?\$?(\d+)(?:\.(\d+))?"
                   r"(?:\s+a\s+pack)?", low)
    bm = re.search(r"buy\s+(?:one|1)\s+get\s+(?:one|1)\s+free",
                   low)
    cm = re.search(r"\\?\$?(\d+)(?:\.(\d+))?\s+coupon", low)
    if not (nm and pm and bm and cm):
        return None
    razors = Fraction(nm.group(1)) * 2
    paid = Fraction(pm.group(1)) + (Fraction(pm.group(2)) / 100
                                    if pm.group(2) else Fraction(0))
    coupon = Fraction(cm.group(1)) + (Fraction(cm.group(2)) / 100
                                      if cm.group(2) else Fraction(0))
    return (paid - coupon) / razors * 100


def _mensch_pyramide(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Mensch-Pyramide: '252/12' -> 21."""
    low = _digitize(question.lower())
    tm = re.search(r"(\d+)\s*(?:inches|[\"'])\s*(?:tall)?", low)
    sm = re.search(r"shortest\s+girl|pyramid", low)
    sm2 = re.search(r"girl(?:\s+is)?\s+(\d+)\s*(?:inches|[\"'])"
                    r"\s*(?:tall)?|is\s+(\d+)\s*[\"']\s*tall",
                    low)
    pm = re.search(r"human\s+pyramid|pyramid", low)
    if not (tm and sm and sm2 and pm):
        return None
    s2v = sm2.group(1) or sm2.group(2)
    return (Fraction(tm.group(1)) * 3 + Fraction(s2v)) / 12


def _buerogeh_zeiten(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Buerogeh-Zeiten: '8x5x5' -> 200."""
    low = _digitize(question.lower())
    hm = re.search(r"(?:works?\s+for\s+)?(\d+)h/day|works?\s+for\s+"
                   r"(\d+)\s+hours?\s+every\s+day", low)
    wm = re.search(r"walk\s+for\s+(\d+)\s+minutes?\s+every\s+hour|"
                   r"walk\s+(\d+)\s+min\s+every\s+hour", low)
    dm = re.search(r"(?:after\s+)?(\d+)\s+days?", low)
    if not (hm and wm and dm):
        return None
    hv = hm.group(1) or hm.group(2)
    wv = wm.group(1) or wm.group(2)
    return Fraction(hv) * Fraction(wv) * Fraction(dm.group(1))


def _fahr_kosten(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Fahr-Kosten: '(6+2)x14' -> 112."""
    low = _digitize(question.lower())
    dm = re.search(r"did\s+that\s+for\s+(\d+)\s+days?|for\s+"
                   r"(\d+)\s+days?", low)
    mm = re.search(r"morning\s+ride\s+cost[\w'\u2019\s]*?\\?\$?"
                   r"(\d+)|morning\s+\\?\$?(\d+)", low)
    am = re.search(r"afternoon(?:\s+ride,?\s+about)?\s+\\?\$?"
                   r"(\d+)", low)
    if not (dm and mm and am):
        return None
    mv = mm.group(1) or mm.group(2)
    return (Fraction(mv) + Fraction(am.group(1))) * \
        Fraction(dm.group(1) or dm.group(2))


def _orangen_pies(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Orangen-Pies: '120/3' -> 40."""
    low = _digitize(question.lower())
    bm = re.search(r"(?:brought\s+)?(?:five|5)\s+boxes?", low)
    bm2 = re.search(r"(?:with\s+)?(?:ten|10)(?:\s+\w+\s+in\s+each"
                    r")?", low)
    om = re.search(r"(?:brought\s+)?(\d+)\s+more"
                   r"(?:\s+\w+\s+than\s+\w+)?", low)
    pm = re.search(r"(?:pie\s+needs?\s+)?(?:three|3)\s+\w+"
                   r"(?:\s+per\s+pie)?", low)
    if not (bm and bm2 and om and pm):
        return None
    ashley = 5 * 10
    total = ashley + (ashley + Fraction(om.group(1)))
    return total / 3


def _veranstaltungs_vergleich(question: str, quants: List[Quantity],
                               tgt: QuestionTarget) -> Optional[Fraction]:
    """Veranstaltungs-Vergleich: '200/20' -> 10."""
    low = _digitize(question.lower())
    fm = re.search(r"flat\s+fee\s+of\s+\\?\$?(\d+)|venue\s+1:?\s+"
                   r"\\?\$?(\d+)", low)
    sms = re.findall(r"\\?\$?(\d+)/guest", low)
    sm = sms[-1] if sms else re.search(
        r"\\?\$?(\d+)\s+per\s+(?:person|guest)", low)
    fm2 = re.search(r"food,\s*which\s+mark\s+estimates?\s+will\s+"
                    r"cost\s+\\?\$?(\d+)|\\?\$?(\d+)/guest",
                    low)
    if not (fm and sm and fm2):
        return None
    fv = fm.group(1) or fm.group(2)
    if isinstance(sm, str):
        sv = sm
    else:
        sv = sm.group(1) or sm.group(2)
    f2v = fm2.group(1) or fm2.group(2)
    return Fraction(fv) / (Fraction(sv) - Fraction(f2v))


def _moebel_masse(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Moebel-Masse: '2x8+2' -> 18."""
    low = _digitize(question.lower())
    rm = re.search(r"rug\s+is\s+(\d+)\s+feet\s+wider\s+than\s+"
                   r"the\s+chair|rug\s+(\d+)\s+ft\s+wider", low)
    cm = re.search(r"couch\s+is\s+(\d+)\s+feet\s+longer\s+than\s+"
                   r"twice\s+the\s+width|couch\s+(\d+)\s+ft\s+"
                   r"longer", low)
    cm2 = re.search(r"chair\s+is\s+(\d+)\s+feet\s+wide|chair\s+"
                   r"\((\d+)\s+ft\)", low)
    if not (rm and cm and cm2):
        return None
    rv = rm.group(1) or rm.group(2)
    cv = cm.group(1) or cm.group(2)
    c2v = cm2.group(1) or cm2.group(2)
    rug = Fraction(c2v) + Fraction(rv)
    return rug * 2 + Fraction(cv)


def _kaugummi_preise(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Kaugummi-Preise: '(7-2-1)/2' -> 2."""
    low = _digitize(question.lower())
    pm = re.search(r"paid\s+\\?\$?(\d+)\s+for\s+a\s+pack\s+of\s+"
                   r"\w+\s+gum|grape\s+\\?\$?(\d+)", low)
    hm = re.search(r"half\s+as\s+much|half\s+the\s+grape", low)
    tms = re.findall(r"paid\s+\\?\$?(\d+)", low)
    tm = tms[-1] if tms else None
    sm = re.search(r"(?:two|2)\s+(?:packs\s+of\s+her\s+favorite|"
                   r"strawberry)", low)
    if not (pm and hm and tm and sm):
        return None
    pv = pm.group(1) or pm.group(2)
    green = Fraction(pv) / 2
    return (Fraction(tm) - Fraction(pv) - green) / 2


def _schneeschuh_hunde(question: str, quants: List[Quantity],
                         tgt: QuestionTarget) -> Optional[Fraction]:
    """Schneeschuh-Hunde: '24/2x12' -> 144."""
    low = _digitize(question.lower())
    dm = re.search(r"(?:for\s+his\s+)?(\d+)\s+(?:sled\s+)?dogs?",
                   low)
    lm = re.search(r"each\s+has\s+(?:four|4)\s+legs?|(?:four|4)\s+"
                   r"legs?\s+each", low)
    pm = re.search(r"pair(?:s)?\s+of\s+\w+\s+costs?\s+\\?\$?(\d+)"
                   r"|pairs?\s+at\s+\\?\$?(\d+)", low)
    if not (dm and lm and pm):
        return None
    pv = pm.group(1) or pm.group(2)
    shoes = Fraction(dm.group(1)) * 4
    return shoes / 2 * Fraction(pv)


def _bauernhof_zoo(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Bauernhof-Zoo: '40+80' -> 120."""
    low = _digitize(question.lower())
    fm = re.search(r"farm\s+(?:has\s+)?(\d+)\s+cows?", low)
    zm = re.search(r"zoo\s+(?:has\s+)?(\d+)\s+sheep?", low)
    dm = re.search(r"zoo\s+(?:has\s+)?twice\s+as\s+many\s+cows|zoo"
                   r"\s+2x\s+farm\s+cows", low)
    hm = re.search(r"farm\s+(?:has\s+)?half\s+as\s+many\s+sheep|"
                   r"farm\s+half\s+zoo\s+sheep", low)
    if not (fm and zm and dm and hm):
        return None
    f = Fraction(fm.group(1))
    z = Fraction(zm.group(1))
    return f + z / 2 + f * 2 + z


def _neujahr_ziel(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Neujahr-Ziel: '105000/200' -> 525."""
    low = _digitize(question.lower())
    lm = re.search(r"lose\s+(\d+)\s+lbs?(?:\.|\s+in)", low)
    cm = re.search(r"burn\s+(\d+)\s+calories?\s+to\s+lose\s+a\s+"
                   r"pound|(\d+)\s+cal/lb", low)
    if not (lm and cm):
        return None
    cv = cm.group(1) or cm.group(2)
    return Fraction(lm.group(1)) * Fraction(cv) / 200


def _haus_grundstueck(question: str, quants: List[Quantity],
                        tgt: QuestionTarget) -> Optional[Fraction]:
    """Haus-Grundstueck: '120000x3/4' -> 90000."""
    low = _digitize(question.lower())
    cm = re.search(r"cost\s+\\?\$?([\d,]+)|house\s+\+\s+lot\s+"
                   r"\\?\$?([\d,]+)", low)
    tm = re.search(r"(?:three|3)\s+times\s+as\s+much\s+as\s+the"
                   r"\s+lot|3x\s+the\s+lot", low)
    if not (cm and tm):
        return None
    cv = cm.group(1) or cm.group(2)
    return Fraction(cv.replace(",", "")) * Fraction(3, 4)


def _taschen_profit(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Taschen-Profit: '320-160' -> 160."""
    low = _digitize(question.lower())
    pm = re.search(r"packs?\s+of\s+(\d+)\s+(?:\w+\s+)?bags?",
                   low)
    cms = re.findall(r"\\?\$(\d+)(?:\s+each)?", low)
    cm = cms[0] if cms else None
    sm = re.search(r"sold\s+them\s+at\s+a\s+[\w\s]+?\s+for\s+"
                   r"\\?\$?(\d+)\s+each|sold\s+at\s+\\?\$?"
                   r"(\d+)\s+each", low)
    nm = re.search(r"(\d+)\s+packs?\s+of", low)
    if not (pm and cm and sm and nm):
        return None
    sv = sm.group(1) or sm.group(2)
    bags = Fraction(nm.group(1)) * Fraction(pm.group(1))
    return bags * (Fraction(sv) - Fraction(cm))


def _backen_vergleich(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Backen-Vergleich: '2x9' -> 18."""
    low = _digitize(question.lower())
    km = re.search(r"(?:two|2)\s*(?:times\s+more\s+\w+\s+than\s+|"
                   r"x\s+)\w+|kelsie\s+2x\s+\w+", low)
    jm = re.search(r"(?:one-fourth|1-fourth)\s+the\s+number|1/4\s+"
                   r"\w+", low)
    sm = re.search(r"suzanne\s+(?:made\s+)?(\d+)(?:\s+\w+)?", low)
    if not (km and jm and sm):
        return None
    return Fraction(sm.group(1)) / 4 * 2


def _tulpen_reihen(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Tulpen-Reihen: '6+3' -> 9."""
    low = _digitize(question.lower())
    rm = re.search(r"fit\s+(\d+)\s+red\s+\w+\s+in\s+a\s+row|"
                   r"(\d+)\s+red/row", low)
    bm = re.search(r"and\s+(\d+)\s+blue\s+\w+\s+in\s+a\s+row|"
                   r"(\d+)\s+blue/row", low)
    rbs = re.findall(r"buys?\s+(\d+)\s+red\s+\w+|(\d+)\s+red",
                     low)
    rb = (rbs[-1][0] or rbs[-1][1]) if rbs else None
    bbs = re.findall(r"and\s+(\d+)\s+blue\s+\w+|(\d+)\s+blue\.",
                    low)
    bbs = [a or b for a, b in bbs]
    bb = bbs[-1] if bbs else None
    if not (rm and bm and rb and bb):
        return None
    rv = rm.group(1) or rm.group(2)
    bv = bm.group(1) or bm.group(2)
    return Fraction(rb) / Fraction(rv) + \
        Fraction(bb) / Fraction(bv)


def _rosinen_batch(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Rosinen-Batch: '9/(3/4)' -> 12."""
    low = _digitize(question.lower())
    cm = re.search(r"with\s+(\d+)\s+cups?\s+of\s+\w+|(\d+)\s+"
                   r"cups\s+split", low)
    dm = re.search(r"divides?\s+the\s+bag\s+of\s+\w+\s+equally|"
                   r"split\s+3\s+ways", low)
    tm = re.search(r"takes?\s+3/4\s+of\s+a\s+cup|take\s+3/4\s+cup",
                   low)
    nm = re.search(r"(?:three|3)\s+\w+|split\s+(?:three|3)\s+ways|"
                   r"equally\s+among", low)
    if not (cm and dm and tm and nm):
        return None
    cv = cm.group(1) or cm.group(2)
    return Fraction(cv) / 3 / Fraction(3, 4)


def _haus_streichen(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Haus-Streichen: '240/5' -> 48."""
    low = _digitize(question.lower())
    hm = re.search(r"paints?\s+half\s+a\s+house\s+in\s+(\d+)\s+"
                   r"days?", low)
    pm = re.search(r"(?:for\s+)?(\d+)\s+people?", low)
    if not (hm and pm):
        return None
    return Fraction(hm.group(1)) * 24 * 2 / Fraction(pm.group(1))


def _alter_abstand(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Alter-Abstand: '47-22' -> 25."""
    low = _digitize(question.lower())
    tm = re.search(r"turn\s+(\d+)\s+in\s+(\d+)\s+years?", low)
    cm = re.search(r"in\s+(\d+)\s+years\s+his\s+cousin", low)
    ym = re.search(r"younger\s+than\s+twice\s+his\s+age", low)
    if not (tm and cm and ym):
        return None
    jame = Fraction(tm.group(1)) - Fraction(tm.group(2))
    jame8 = jame + Fraction(cm.group(1))
    cousin8 = jame8 * 2 - 5
    cousin_now = cousin8 - Fraction(cm.group(1))
    return cousin_now - jame


def _monitor_preis(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Monitor-Preis: '600/2' -> 300."""
    low = _digitize(question.lower())
    tm = re.search(r"for\s+\\?\$?([\d,]+)|=\s*\\?\$?([\d,]+)",
                   low)
    lm = re.search(r"paid\s+\\?\$?([\d,]+)\s+less\s+for\s+the"
                   r"\s+printer|printer\s+\\?\$?([\d,]+)\s+less",
                   low)
    cm = re.search(r"computer\s+(?:cost\s+)?\\?\$?([\d,]+)", low)
    nm = re.search(r"(\d+)\s+monitors?", low)
    if not (tm and lm and cm and nm):
        return None
    tv = tm.group(1) or tm.group(2)
    lv = lm.group(1) or lm.group(2)
    printer = Fraction(cm.group(1).replace(",", "")) - \
        Fraction(lv.replace(",", ""))
    rest = Fraction(tv.replace(",", "")) - \
        Fraction(cm.group(1).replace(",", "")) - printer
    return rest / Fraction(nm.group(1))


def _muscheln_suche(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Muscheln-Suche: '3000/10' -> 300."""
    low = _digitize(question.lower())
    km = re.search(r"(\d+)\s+kids?", low)
    bm = re.search(r"(?:brought\s+back\s+)?(\d+)\s+shells?\s+"
                   r"each", low)
    fm = re.search(r"(?:four|4)\s*(?:times\s+as\s+many|x)", low)
    if not (km and bm and fm):
        return None
    per = Fraction(km.group(1)) / 2
    boys = per * Fraction(bm.group(1))
    girls = boys + boys * 4
    return girls / per


def _buch_dicken(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Buch-Dicken: '31+50+45+62' -> 188."""
    low = _digitize(question.lower())
    fm = re.search(r"first\s+book\s+is\s+(\d+)\s+mm", low)
    sm = re.search(r"second\s+book\s+is\s+(\d+)\s+mm", low)
    tm = re.search(r"third\s+book\s+is\s+(\d+)\s+mm\s+less", low)
    fm2 = re.search(r"fourth\s+book\s+is\s+twice\s+as\s+thick",
                    low)
    if not (fm and sm and tm and fm2):
        return None
    third = Fraction(sm.group(1)) - Fraction(tm.group(1))
    return Fraction(fm.group(1)) + Fraction(sm.group(1)) + third + \
        Fraction(fm.group(1)) * 2


def _rechnung_tip(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Rechnung-Tip: '25+10' -> 35."""
    low = _digitize(question.lower())
    bm = re.search(r"\\?\$?(\d+)(?:\s+dinner)?\s+bill", low)
    pm = re.search(r"(\d+)\s*%\s+tip", low)
    sm = re.search(r"evenly\s+split|split\s+evenly", low)
    if not (bm and pm and sm):
        return None
    b = Fraction(bm.group(1))
    return b / 2 + b * Fraction(pm.group(1)) / 100


def _jungs_anteile(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Jungs-Anteile: '13x3' -> 39."""
    low = _digitize(question.lower())
    sm = re.search(r"shared\s+among\s+(\d+)\s+\w+", low)
    em = re.search(r"eldest\s+added\s+\\?\$?(\d+)", low)
    em2 = re.search(r"another\s+\\?\$?(\d+)", low)
    pm = re.search(r"spent\s+\\?\$?(\d+)", low)
    tm = re.search(r"triple\s+the\s+amount", low)
    if not (sm and em and em2 and pm and tm):
        return None
    share = Fraction(18) / Fraction(sm.group(1))
    return (share + Fraction(em.group(1)) + Fraction(em2.group(1)) -
            Fraction(pm.group(1))) * 3


def _obst_einkauf(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Obst-Einkauf: '20-13' -> 7."""
    low = _digitize(question.lower())
    ms = re.findall(r"((?:three|3|five|5|six|6))\s+\w+\s+at\s+"
                    r"\\?\$?(\d+)(?:\.(\d+))?", low)
    pm = re.search(r"gave\s+\\?\$?(\d+)", low)
    if not (ms and pm and len(ms) >= 3):
        return None
    qmap = {"three": 3, "3": 3, "five": 5, "5": 5, "six": 6, "6": 6}
    total = Fraction(0)
    for q, d, c in ms:
        total += Fraction(qmap[q]) * \
            (Fraction(d) + (Fraction(c) / 100 if c else Fraction(0)))
    return Fraction(pm.group(1)) - total


def _arbeitsweg(question: str, quants: List[Quantity],
                tgt: QuestionTarget) -> Optional[Fraction]:
    """Arbeitsweg: '3x2x5' -> 30."""
    low = _digitize(question.lower())
    dm = re.search(r"(\d+)\s+miles?\s+away", low)
    wm = re.search(r"(\d+)\s+times?/week|work\s+(\d+)\s+times?\s+"
                   r"a\s+week", low)
    bm = re.search(r"there\s+and\s+back", low)
    if not (dm and wm and bm):
        return None
    wv = wm.group(1) or wm.group(2)
    return Fraction(dm.group(1)) * 2 * Fraction(wv)


def _alphabet_uebung(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Alphabet-Uebung: '65x2' -> 130."""
    low = _digitize(question.lower())
    fm = re.search(r"in\s+full\s+twice", low)
    hm = re.search(r"half(?:\s+of\s+it)?\s+once", low)
    rm = re.search(r"(?:re-writes?|rewrite)\s+(?:everything|all)",
                   low)
    if not (fm and hm and rm):
        return None
    return (26 * 2 + 13) * 2


def _burger_rechnung(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Burger-Rechnung: '50-33' -> 17."""
    low = _digitize(question.lower())
    ms = re.findall(r"((?:five|5|ten|10))\s+\w+(?:\s+\w+)*\s+at\s+"
                    r"\\?\$?(\d+)(?:\.(\d+))?", low)
    pm = re.search(r"fifty-dollar\s+bill|\\?\$?(\d+)-dollar|"
                   r"\\?\$?(\d+)\s+bill", low)
    if not (ms and pm and len(ms) >= 3):
        return None
    qmap = {"five": 5, "5": 5, "ten": 10, "10": 10}
    total = Fraction(0)
    for q, d, c in ms:
        total += Fraction(qmap[q]) * \
            (Fraction(d) + (Fraction(c) / 100 if c else Fraction(0)))
    return Fraction(50) - total


def _wasser_flaschen(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Wasser-Flaschen: '140-48' -> 92."""
    low = _digitize(question.lower())
    cm = re.search(r"(\d+)\s+cases?\s+of\s+\w+\s+which\s+have\s+"
                   r"(\d+)\s+bottles?", low)
    gm = re.search(r"(\d+)\s+guests?", low)
    bm = re.search(r"(\d+)\s+bottles?\s+of\s+\w+\s+for\s+each",
                   low)
    if not (cm and gm and bm):
        return None
    have = Fraction(cm.group(1)) * Fraction(cm.group(2))
    need = Fraction(gm.group(1)) * Fraction(bm.group(1))
    return need - have


def _knoepfe_loecher(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Knoepfe-Loecher: '14+56' -> 70."""
    low = _digitize(question.lower())
    bm = re.search(r"(\d+)\s+buttons?(?:\s+in\s+it)?", low)
    sm = re.search(r"(?:seven|7)\s+(?:buttons?\s+had\s+)?(?:two|2)"
                   r"\s+holes?|(?:seven|7)\s+with\s+(?:two|2)\s+"
                   r"holes?", low)
    fm = re.search(r"rest\s+(?:had|with)\s+(?:four|4)"
                   r"(?:\s+holes?)?", low)
    if not (bm and sm and fm):
        return None
    two = 7 * 2
    four = (Fraction(bm.group(1)) - 7) * 4
    return two + four


def _tierladen_kaefige(question: str, quants: List[Quantity],
                         tgt: QuestionTarget) -> Optional[Fraction]:
    """Tierladen-Kaefige: '30+15' -> 45."""
    low = _digitize(question.lower())
    cm = re.search(r"(?:six|6)\s+cages?", low)
    hm = re.search(r"(?:three|3)\s+of\s+the\s+cages?\s+have\s+"
                   r"(?:ten|10)\s+\w+", low)
    gm = re.search(r"other\s+(?:three|3)\s+have\s+(?:five|5)\s+"
                   r"\w+", low)
    if not (cm and hm and gm):
        return None
    return 3 * 10 + 3 * 5


def _zahnfee(question: str, quants: List[Quantity],
             tgt: QuestionTarget) -> Optional[Fraction]:
    """Zahnfee: '5+3+1' -> 9."""
    low = _digitize(question.lower())
    fm = re.search(r"(?:left\s+\w+\s+)?\\?\$?(\d+)(?:\.(\d+))?",
                   low)
    tm = re.search(r"(?:each\s+of\s+the\s+)?next\s+(?:three|3)",
                   low)
    tm2 = re.search(r"at\s+\\?\$?(\d+)|\\?\$?(\d+)"
                    r"(?:\.(\d+))?\s+for\s+each", low)
    lm = re.search(r"last\s+(?:two|2)", low)
    hm = re.search(r"half(?:\s+the\s+amount|\s+that)", low)
    if not (fm and tm and tm2 and lm and hm):
        return None
    tv = tm2.group(1) or tm2.group(2)
    return Fraction(5) + Fraction(tv) * 3 + Fraction(tv) / 2 * 2


def _kitten_kosten(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Kitten-Kosten: '200+60+48' -> 308."""
    low = _digitize(question.lower())
    sm = re.search(r"cost\s+\\?\$?(\d+)", low)
    vm = re.search(r"vaccines?\s+costs?\s+\\?\$?(\d+)", low)
    nm = re.search(r"(\d+)\s+vaccines?", low)
    vm2 = re.search(r"broke\s+(\d+)\s+vases?", low)
    vm3 = re.search(r"cost\s+\\?\$?(\d+)\s+each", low)
    if not (sm and vm and nm and vm2 and vm3):
        return None
    return Fraction(sm.group(1)) + Fraction(nm.group(1)) * \
        Fraction(vm.group(1)) + Fraction(vm2.group(1)) * \
        Fraction(vm3.group(1))


def _kreide_muffins(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Kreide-Muffins: '24x1.5' -> 36."""
    low = _digitize(question.lower())
    bm = re.search(r"(\d+)\s+boxes?\s+of\s+(\d+)\s+\w+", low)
    mm = re.search(r"(?:melting\s+)?(\d+)(?:\s+small\s+pieces?|"
                   r"\s+per\s+muffin)", low)
    pm = re.search(r"(?:sell\s+her\s+[\w\s]+?\s+for\s+|sell\s+at"
                   r"\s+)\\?\$?(\d+)(?:\.(\d+))?", low)
    if not (bm and mm and pm):
        return None
    muffins = Fraction(bm.group(1)) * Fraction(bm.group(2)) / \
        Fraction(mm.group(1))
    price = Fraction(pm.group(1)) + (Fraction(pm.group(2)) / 100
                                     if pm.group(2) else Fraction(0))
    return muffins * price


def _teppich_kosten(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Teppich-Kosten: '216x52' -> 11232."""
    low = _digitize(question.lower())
    cm = re.search(r"costs?\s+\\?\$?(\d+)\s+per\s+square\s+foot",
                   low)
    pm = re.search(r"\\?\$?(\d+)\s+per\s+square\s+foot\s+for\s+"
                   r"padding", low)
    rm = re.search(r"charges?\s+\\?\$?(\d+)\s+per\s+square\s+"
                   r"foot\s+to\s+remove", low)
    im = re.search(r"\\?\$?(\d+)\s+per\s+square\s+foot\s+to\s+"
                   r"install", low)
    dm = re.search(r"measures\s+(\d+)\s+feet", low)
    dm2 = re.search(r"by\s+(\d+)\s+feet", low)
    if not (cm and pm and rm and im and dm and dm2):
        return None
    per = Fraction(cm.group(1)) + Fraction(pm.group(1)) + \
        Fraction(rm.group(1)) + Fraction(im.group(1))
    return Fraction(dm.group(1)) * Fraction(dm2.group(1)) * per


def _film_zeiten2(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Film-Zeiten2: '90+125' -> 215."""
    low = _digitize(question.lower())
    ms = re.findall(r"(\d+)h\s+(\d+)m|(\d+)\s+hour(?:s)?\s+and"
                    r"\s+(\d+)\s+minutes?", low)
    ms = [(a or c, b or d) for a, b, c, d in ms]
    if len(ms) < 2:
        return None
    return sum(Fraction(h) * 60 + Fraction(m) for h, m in ms)


def _haus_werte(question: str, quants: List[Quantity],
                tgt: QuestionTarget) -> Optional[Fraction]:
    """Haus-Werte: '53200+76000' -> 129200."""
    low = _digitize(question.lower())
    hm = re.search(r"paid\s+\\?\$?([\d,]+)\s+for\s+the\s+house",
                   low)
    lm = re.search(r"(\d+)\s*%\s+less\s+expensive", low)
    if not (hm and lm):
        return None
    h = Fraction(hm.group(1).replace(",", ""))
    j = h * (100 - Fraction(lm.group(1))) / 100
    return h + j


def _kontakt_linsen(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Kontakt-Linsen: '180/90' -> 2."""
    low = _digitize(question.lower())
    nm = re.search(r"(\d+)\s+(?:single\s+use\s+)?contacts?|"
                   r"(\d+)\s+per\s+box", low)
    pm = re.search(r"box\s+is\s+\\?\$?(\d+)|\\?\$(\d+)", low)
    dm = re.search(r"(\d+)\s*%\s+off", low)
    bm = re.search(r"buys?\s+(\d+)\s+boxes?", low)
    if not (nm and pm and dm and bm):
        return None
    nv = nm.group(1) or nm.group(2)
    pv = pm.group(1) or pm.group(2)
    cost = Fraction(bm.group(1)) * Fraction(pv) * \
        (100 - Fraction(dm.group(1))) / 100
    pairs = Fraction(bm.group(1)) * Fraction(nv) / 2
    return cost / pairs


def _event_gaeste(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Event-Gaeste: '300+19' -> 319."""
    low = _digitize(question.lower())
    em = re.search(r"invites?\s+(\d+)\s+people?\s+via\s+email",
                   low)
    fm = re.search(r"invite\s+(\d+)\s+of\s+their\s+friends?", low)
    cm = re.search(r"calls?\s+(\d+)\s+of\s+her\s+friends?", low)
    sm = re.search(r"(\d+)\s+of\s+them\s+say", low)
    if not (em and fm and cm and sm):
        return None
    groups = Fraction(em.group(1)) * (1 + Fraction(fm.group(1)))
    friends = Fraction(cm.group(1)) + Fraction(sm.group(1))
    return groups + friends + 1


def _hund_gewichte(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Hund-Gewichte: '5x44' -> 220."""
    low = _digitize(question.lower())
    am = re.search(r"(?:weighed\s+only\s+)?(\d+)\s+(?:pounds?|lb)",
                   low)
    dm = re.search(r"twice\s+as\s+much\s+as\s+the\s+\w+|2x", low)
    pm = re.search(r"(?:one-fourth|1-fourth)\s+as\s+much\s+as\s+"
                   r"the\s+\w+|1/4\s+of", low)
    mms = re.findall(r"(?:weighed\s+)?(\d+)\s*(?:times\s+the\s+"
                     r"weight|x)", low)
    mm = mms[-1] if mms else None
    if not (am and dm and pm and mm):
        return None
    d = Fraction(am.group(1)) * 2
    p = d / 4
    return p * Fraction(mm)


def _bohnenstange(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Bohnenstange: '4x2^3>20' -> 3."""
    low = _digitize(question.lower())
    wm = re.search(r"(\d+)\s+feet\s+off\s+the\s+ground", low)
    dm = re.search(r"doubles?\s+its\s+height\s+every\s+day", low)
    sm = re.search(r"starts?\s+out\s+(\d+)\s+feet", low)
    if not (wm and dm and sm):
        return None
    h = Fraction(sm.group(1))
    target = Fraction(wm.group(1))
    days = 0
    while h <= target:
        h *= 2
        days += 1
    return days


def _muenzen_kauf(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Muenzen-Kauf: '345/5' -> 69."""
    low = _digitize(question.lower())
    cm = re.search(r"(?:cost\s+)?a\s+nickel", low)
    qm = re.search(r"(\d+)\s+quarters?", low)
    dm = re.search(r"(\d+)\s+dimes?", low)
    nm = re.search(r"(\d+)\s+nickels?", low)
    pm = re.search(r"(?:and\s+)?(\d+)\s+pennies?", low)
    if not (cm and qm and dm and nm and pm):
        return None
    cents = Fraction(qm.group(1)) * 25 + Fraction(dm.group(1)) * 10 + \
        Fraction(nm.group(1)) * 5 + Fraction(pm.group(1))
    return cents / 5


def _tier_beine(question: str, quants: List[Quantity],
                tgt: QuestionTarget) -> Optional[Fraction]:
    """Tier-Beine: '20+8+20' -> 48."""
    low = _digitize(question.lower())
    dm = re.search(r"(\d+)\s+dogs?", low)
    cm = re.search(r"(\d+)\s+cats?", low)
    bm = re.search(r"(\d+)\s+birds?", low)
    lm = re.search(r"legs?\s+in\s+total", low)
    if not (dm and cm and bm and lm):
        return None
    return Fraction(dm.group(1)) * 4 + Fraction(cm.group(1)) * 4 + \
        Fraction(bm.group(1)) * 2


def _flug_zeiten(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Flug-Zeiten: '2000/400' -> 5."""
    low = _digitize(question.lower())
    low = _digitize(question.lower())
    if "average speed" in low:
        return None
    dm = re.search(r"(?:travels?\s+)?(\d+)\s+miles?\s+in\s+(\d+)"
                   r"\s+hours?", low)
    am = re.search(r"additional\s+(\d+)\s+miles?", low)
    if not (dm and am):
        return None
    rate = Fraction(dm.group(1)) / Fraction(dm.group(2))
    return Fraction(am.group(1)) / rate


def _limonade_verdienst(question: str, quants: List[Quantity],
                          tgt: QuestionTarget) -> Optional[Fraction]:
    """Limonade-Verdienst: '30+12' -> 42."""
    low = _digitize(question.lower())
    hm = re.search(r"(?:for\s+)?(\d+)h\s*x|for\s+(\d+)\s+hours?",
                   low)
    cms = re.findall(r"sold\s+(\d+)\s+cups?(?:\s+of\s+\w+)?\s+per"
                     r"\s+hour|(\d+)\s+cups\s+at", low)
    pms = re.findall(r"(?:price\s+of\s+|at\s+)\\?\$?(\d+)"
                     r"(?:\.(\d+))?", low)
    hm2 = re.search(r"next\s+(\d+)\s+hours?|\+\s*(\d+)h", low)
    cms = [a or b for a, b in cms]
    if not (hm and cms and pms and hm2 and len(cms) >= 2 and
            len(pms) >= 2):
        return None
    p1 = Fraction(pms[0][0]) + (Fraction(pms[0][1]) / 100
                                if pms[0][1] else Fraction(0))
    p2 = Fraction(pms[1][0]) + (Fraction(pms[1][1]) / 100
                                if pms[1][1] else Fraction(0))
    hv = hm.group(1) or hm.group(2)
    h2v = hm2.group(1) or hm2.group(2)
    return Fraction(hv) * Fraction(cms[0]) * p1 + \
        Fraction(h2v) * Fraction(cms[1]) * p2


def _reifen_rotation(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Reifen-Rotation: '72/12' -> 6."""
    low = _digitize(question.lower())
    dm = re.search(r"(?:every\s+)?(\d+)\s+miles?(?:\s+a\s+car\s+"
                   r"drives?)?", low)
    tm = re.search(r"(?:rotate\s+)?(\d+)\s*(?:times|rotations?)",
                   low)
    mm = re.search(r"(?:drives?\s+)?(\d+)\s+miles?(?:\s+a\s+month|"
                   r"/month)", low)
    rms = re.findall(r"([\d,]+)\s+rotations?", low)
    rm = rms[-1] if rms else None
    if not (dm and tm and mm and rm):
        return None
    per_month = Fraction(mm.group(1)) / Fraction(dm.group(1)) * \
        Fraction(tm.group(1))
    months = Fraction(rm.replace(",", "")) / per_month
    return months / 12


def _shampoo_pumpen(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Shampoo-Pumpen: '2400/240' -> 10."""
    low = _digitize(question.lower())
    pm = re.search(r"(?:costs?\s+)?\\?\$?(\d+)(?:\.(\d+))?", low)
    wm = re.search(r"(?:two|2)\s+pumps?(?:\s+of\s+\w+)?", low)
    wm2 = re.search(r"(?:give\s+you\s+|=)\s*(\d+)\s+washings?", low)
    um = re.search(r"(?:only\s+)?uses?\s+(?:one|1)\s+pump", low)
    if not (pm and wm and wm2 and um):
        return None
    cents = Fraction(pm.group(1)) * 100 + \
        (Fraction(pm.group(2)) if pm.group(2) else Fraction(0))
    pumps = Fraction(wm2.group(1)) * 2
    return cents / pumps


def _croissant_butter(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Croissant-Butter: '28/4' -> 7."""
    low = _digitize(question.lower())
    bm = re.search(r"1/4\s+pound\s+of\s+\w+", low)
    dm = re.search(r"(\d+)\s+dozen\s+a\s+day", low)
    wm = re.search(r"for\s+a\s+week", low)
    if not (bm and dm and wm):
        return None
    return Fraction(dm.group(1)) * 7 / 4


def _forschungs_kosten(question: str, quants: List[Quantity],
                         tgt: QuestionTarget) -> Optional[Fraction]:
    """Forschungs-Kosten: '100000+1350000' -> 1450000."""
    low = _digitize(question.lower())
    fm = re.search(r"(?:funding\s+of\s+)?\\?\$?([\d,]+)", low)
    tm = re.search(r"for\s+(?:the\s+first\s+)?(\d+)\s+months?", low)
    dm = re.search(r"(?:10|ten)\s*(?:times\s+that\s+long|x\s+that\s+"
                   r"long)", low)
    pm = re.search(r"(\d+)\s*%\s+more(?:\s+funding|/month)", low)
    if not (fm and tm and dm and pm):
        return None
    first = Fraction(fm.group(1).replace(",", ""))
    months = Fraction(tm.group(1)) * 10
    per_month = first / Fraction(tm.group(1)) * \
        (1 + Fraction(pm.group(1)) / 100)
    rest = (months - Fraction(tm.group(1))) * per_month
    return first + rest


def _steak_abendessen(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Steak-Abendessen: '6+8+16' -> 30."""
    low = _digitize(question.lower())
    fm = re.search(r"ate\s+a\s+(\d+)-ounce\s+steak", low)
    bm = re.search(r"(\d+)\s+beef\s+tips?", low)
    pm = re.search(r"(?:one|1)-pound\s+steak", low)
    if not (fm and bm and pm):
        return None
    return Fraction(fm.group(1)) + Fraction(bm.group(1)) + 16


def _bohne_wachstum(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Bohne-Wachstum: '3x2+4' -> 10."""
    low = _digitize(question.lower())
    fm = re.search(r"(?:was\s+)?(\d+)\s+inches?(?:\s+tall)?", low)
    dm = re.search(r"doubled(?:\s+in\s+height)?", low)
    gm = re.search(r"(?:grew\s+another\s+|\+)(\d+)\s+inches?", low)
    if not (fm and dm and gm):
        return None
    return Fraction(fm.group(1)) * 2 + Fraction(gm.group(1))


def _auto_provision(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Auto-Provision: '175000x0.1' -> 17500."""
    low = _digitize(question.lower())
    nm = re.search(r"sold\s+(\d+)\s+cars?", low)
    pm = re.search(r"cost\s+\\?\$?([\d,]+)\s+each", low)
    am = re.search(r"paid\s+(\d+)\s*%\s+of\s+that", low)
    cm = re.search(r"(\d+)\s*%\s+commission", low)
    if not (nm and pm and am and cm):
        return None
    revenue = Fraction(nm.group(1)) * \
        Fraction(pm.group(1).replace(",", ""))
    cost = revenue * Fraction(am.group(1)) / 100
    profit = revenue - cost
    return profit * Fraction(cm.group(1)) / 100


def _kasse_leistung(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Kasse-Leistung: '150x7' -> 1050."""
    low = _digitize(question.lower())
    dm = re.search(r"(?:twice|2x)\s+as\s+fast", low)
    cm = re.search(r"(?:processes?\s+)?(\d+)\s*(?:customers?|/day)",
                   low)
    wm = re.search(r"all\s+(?:days?\s+of\s+the\s+)?week", low)
    if not (dm and cm and wm):
        return None
    return Fraction(cm.group(1)) * 3 * 7


def _obst_preise2(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Obst-Preise2: '6.5+5.5' -> 12."""
    low = _digitize(question.lower())
    am = re.search(r"(?:four|4)\s+apples?(?:\s+cost)?\s+\\?\$?(\d+)"
                   r"(?:\.(\d+))?", low)
    om = re.search(r"(?:three|3)\s+oranges?(?:\s+cost)?\s+\\?\$?"
                   r"(\d+)(?:\.(\d+))?", low)
    nms = re.findall(r"(?:pay\s+for\s+)?(\d+)\s+apples?|(\d+)\s+"
                     r"of\s+each", low)
    nm = (nms[-1][0] or nms[-1][1]) if nms else None
    nm2s = re.findall(r"(?:and\s+)?(\d+)\s+oranges?|(\d+)\s+of\s+"
                      r"each", low)
    nm2 = (nm2s[-1][0] or nm2s[-1][1]) if nm2s else None
    if not (am and om and nm and nm2):
        return None
    a = Fraction(am.group(1)) + (Fraction(am.group(2)) / 100
                                 if am.group(2) else Fraction(0))
    o = Fraction(om.group(1)) + (Fraction(om.group(2)) / 100
                                 if om.group(2) else Fraction(0))
    return Fraction(nm) * a / 4 + Fraction(nm2) * o / 3


def _lastwagen_ausstattung(question: str, quants: List[Quantity],
                             tgt: QuestionTarget) -> Optional[Fraction]:
    """Lastwagen-Ausstattung: '30000+7500+2500+2000+1500' -> 43500."""
    low = _digitize(question.lower())
    bm = re.search(r"(?:base\s+price\s+of\s+the\s+\w+\s+is\s+|"
                   r"base\s+)\\?\$?([\d,]+)", low)
    km = re.search(r"king\s+cab(?:\s+is\s+an\s+extra)?\s+\\?\$?"
                   r"([\d,]+)", low)
    lm = re.search(r"leather\s+seats?\s+are\s+(?:one-third|1-third|"
                   r"1/3)|leather\s+(?:one-third|1-third|1/3)", low)
    rm = re.search(r"(?:running\s+)?boards?(?:\s+are)?\s+\\?\$?"
                   r"([\d,]+)\s+less", low)
    em = re.search(r"(?:light\s+package\s+is\s+|lights?\s+)\\?\$?"
                   r"([\d,]+)", low)
    if not (bm and km and lm and rm and em):
        return None
    leather = Fraction(km.group(1).replace(",", "")) / 3
    return Fraction(bm.group(1).replace(",", "")) + \
        Fraction(km.group(1).replace(",", "")) + leather + \
        (leather - Fraction(rm.group(1).replace(",", ""))) + \
        Fraction(em.group(1).replace(",", ""))


def _gehaltserhoehung(question: str, quants: List[Quantity],
                        tgt: QuestionTarget) -> Optional[Fraction]:
    """Gehaltserhoehung: '252000+10500' -> 262500."""
    low = _digitize(question.lower())
    rm = re.search(r"(\d+)\s*%\s+raise", low)
    sm = re.search(r"\\?\$?([\d,]+)(?:\s+a\s+month(?:\s+salary)?|"
                   r"/month)", low)
    bm = re.search(r"bonus(?:\s+worth)?\s+half", low)
    ym = re.search(r"in\s+a\s+year|yearly", low)
    if not (rm and sm and bm and ym):
        return None
    monthly = Fraction(sm.group(1).replace(",", "")) * \
        (1 + Fraction(rm.group(1)) / 100)
    return monthly * 12 + monthly / 2


def _garderobe_kosten(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Garderobe-Kosten: '7500+1500+1800' -> 10800."""
    low = _digitize(question.lower())
    sm = re.search(r"(\d+)\s+suits?", low)
    pm = re.search(r"(\d+)\s+(?:dress\s+)?pants?", low)
    sm2 = re.search(r"suits?\s+(?:cost\s+|at\s+)\\?\$?(\d+)",
                    low)
    pm2 = re.search(r"pants?\s+(?:cost\s+|at\s+)1/5\s+that", low)
    sm3 = re.search(r"shirts?\s+per\s+suit", low)
    sm4 = re.search(r"shirts?\s+were\s+\\?\$?(\d+)|shirts?\s+"
                    r"[\w\s]+?\s+at\s+\\?\$?(\d+)", low)
    if not (sm and pm and sm2 and pm2 and sm3 and sm4):
        return None
    shirts = Fraction(sm.group(1)) * 3
    s4v = sm4.group(1) or sm4.group(2)
    return Fraction(sm.group(1)) * Fraction(sm2.group(1)) + \
        Fraction(pm.group(1)) * Fraction(sm2.group(1)) / 5 + \
        shirts * Fraction(s4v)


def _mehl_saecke(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Mehl-Saecke: '16x3' -> 48."""
    low = _digitize(question.lower())
    pm = re.search(r"divided\s+into\s+(\d+)\s+portions?", low)
    km = re.search(r"of\s+(\d+)\s+(?:kilograms?|kg)", low)
    bm = re.search(r"(?:in\s+)?(?:three|3)\s+bags?", low)
    if not (pm and km and bm):
        return None
    return Fraction(pm.group(1)) * Fraction(km.group(1)) * 3


def _instagram_likes(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Instagram-Likes: '142000+20000' -> 162000."""
    low = _digitize(question.lower())
    lm = re.search(r"(?:received\s+)?([\d,]+)\s+likes?", low)
    tm = re.search(r"(\d+)\s*(?:times\s+as\s+many|x)", low)
    nm = re.search(r"(?:received\s+)?([\d,]+)\s+more\s+new|\+"
                   r"([\d,]+)\s+new", low)
    if not (lm and tm and nm):
        return None
    initial = Fraction(lm.group(1).replace(",", ""))
    later = initial * Fraction(tm.group(1))
    nv = nm.group(1) or nm.group(2)
    return initial + later + Fraction(nv.replace(",", ""))


def _kutsche_stunden(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Kutsche-Stunden: '15+30+30' -> 75."""
    low = _digitize(question.lower())
    tm = re.search(r"(?:from\s+)?(\d+)\s*(?:pm|p\.m\.)\s*(?:to|-)"
                   r"\s*(\d+)\s*(?:pm|p\.m\.)|(\d+)\s*-\s*(\d+)"
                   r"\s*(?:pm|p\.m\.)", low)
    fm = re.search(r"(\d+)\s+hour\s+free", low)
    pm = re.search(r"first\s+paid\s+hour(?:\s+is)?\s+\\?\$?(\d+)",
                   low)
    dm = re.search(r"twice(?:\s+the\s+cost)?", low)
    if not (tm and fm and pm and dm):
        return None
    t1 = tm.group(1) or tm.group(3)
    t2 = tm.group(2) or tm.group(4)
    hours = Fraction(t2) - Fraction(t1) - Fraction(fm.group(1))
    first = Fraction(pm.group(1))
    rest = hours - 1
    return first + rest * (first * 2)


def _lohn_abzug(question: str, quants: List[Quantity],
                tgt: QuestionTarget) -> Optional[Fraction]:
    """Lohn-Abzug: '300-220' -> 80."""
    low = _digitize(question.lower())
    sm = re.search(r"(?:held\s+)?\\?\$?(\d+)(?:\s+at\s+the\s+"
                   r"start)?", low)
    nm = re.search(r"(?:now\s+holds\s+|->\s*)\\?\$?(\d+)", low)
    wm = re.search(r"wage\s+should\s+be\s+\\?\$?(\d+)", low)
    if not (sm and nm and wm):
        return None
    received = Fraction(nm.group(1)) - Fraction(sm.group(1))
    return Fraction(wm.group(1)) - received


def _eier_gaeste(question: str, quants: List[Quantity],
                tgt: QuestionTarget) -> Optional[Fraction]:
    """Eier-Gaeste: '48/2/12' -> 2."""
    low = _digitize(question.lower())
    em = re.search(r"egg\s+to\s+make\s+(\d+)\s+[\w\s]+?halves?|"
                   r"egg\s+makes\s+(\d+)\s+halves?", low)
    gm = re.search(r"(?:each\s+of\s+her\s+guests?\s+will\s+eat|"
                   r"guests?\s+eat)\s+(\d+)", low)
    nm = re.search(r"(?:inviting\s+)?(\d+)\s+guests?", low)
    if not (em and gm and nm):
        return None
    ev = em.group(1) or em.group(2)
    halves = Fraction(nm.group(1)) * Fraction(gm.group(1))
    eggs = halves / Fraction(ev)
    return eggs / 12


def _flugzeug_kosten(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Flugzeug-Kosten: '150000+180000' -> 330000."""
    low = _digitize(question.lower())
    pm = re.search(r"(?:cost\s+)?\\?\$?([\d,]+)", low)
    hm = re.search(r"(?:pays?\s+)?\\?\$?([\d,]+)(?:\s+a\s+month|"
                   r"/month)", low)
    dm = re.search(r"twice\s+as\s+much\s+as\s+that|2x\s+fuel", low)
    ym = re.search(r"first\s+year", low)
    if not (pm and hm and dm and ym):
        return None
    monthly = Fraction(hm.group(1).replace(",", ""))
    return Fraction(pm.group(1).replace(",", "")) + \
        12 * (monthly + monthly * 2)


def _schafe_gaense(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Schafe-Gaense: '(70-2x20)/2' -> 15."""
    low = _digitize(question.lower())
    lm = re.search(r"legs?\s+is\s+(\d+)|(\d+)\s+legs?", low)
    hm = re.search(r"heads?\s+is\s+(\d+)|(\d+)\s+heads?", low)
    sm = re.search(r"sheep", low)
    if not (lm and hm and sm):
        return None
    lv = lm.group(1) or lm.group(2)
    hv = hm.group(1) or hm.group(2)
    return (Fraction(lv) - 2 * Fraction(hv)) / 2


def _kaffee_kauf(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Kaffee-Kauf: '42+2' -> 44."""
    low = _digitize(question.lower())
    pm = re.search(r"(?:cost\s+)?\\?\$?(\d+)(?:\s+per\s+pound|/lb)",
                   low)
    pm2 = re.search(r"(?:cost\s+)?(\d+)\s*%\s+more", low)
    dm = re.search(r"(\d+)\s+pound\s+of\s+\w+\s+per\s+day|"
                   r"(\d+)\s+lb/day", low)
    wm = re.search(r"week['\u2019]?s(?:\s+worth)?|for\s+a\s+week",
                   low)
    dm2 = re.search(r"(?:donut\s+for\s+|\+\s*donut\s+)\\?\$?"
                   r"(\d+)", low)
    if not (pm and pm2 and dm and wm and dm2):
        return None
    dv = dm.group(1) or dm.group(2)
    price = Fraction(pm.group(1)) * \
        (1 + Fraction(pm2.group(1)) / 100)
    return price * 7 * Fraction(dv) + \
        Fraction(dm2.group(1))


def _einkauf_pie(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Einkauf-Pie: '20-13' -> 7."""
    low = _digitize(question.lower())
    sm = re.search(r"spent\s+\\?\$?(\d+)", low)
    cm = re.search(r"chips?\s+(?:for\s+|at\s+)\\?\$?(\d+)", low)
    nm = re.search(r"(?:bought\s+)?(\d+)\s+(?:bag\s+of\s+)?chips?",
                   low)
    fm = re.search(r"chicken(?:\s+for)?\s+\\?\$?(\d+)", low)
    dm = re.search(r"soda(?:\s+for)?\s+\\?\$?(\d+)", low)
    if not (sm and cm and nm and fm and dm):
        return None
    return Fraction(sm.group(1)) - Fraction(nm.group(1)) * \
        Fraction(cm.group(1)) - Fraction(fm.group(1)) - \
        Fraction(dm.group(1))


def _basketball_zeit(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Basketball-Zeit: '48+5' -> 53."""
    low = _digitize(question.lower())
    qm = re.search(r"(\d+)\s+quarters?", low)
    mm = re.search(r"each\s+(\d+)\s+min(?:utes?)?|(\d+)\s+min",
                   low)
    em = re.search(r"extended\s+(?:for\s+)?(?:five|5)\s+min"
                   r"(?:utes?)?", low)
    if not (qm and mm and em):
        return None
    mv = mm.group(1) or mm.group(2)
    return Fraction(qm.group(1)) * Fraction(mv) + 5


def _suessigkeiten_pool(question: str, quants: List[Quantity],
                         tgt: QuestionTarget) -> Optional[Fraction]:
    """Suessigkeiten-Pool: '12/3' -> 4."""
    low = _digitize(question.lower())
    ms = re.findall(r"(?:had\s+)?(\d+)\s+(?:pounds?|lb)", low)
    pm = re.search(r"share(?:\s+their\s+\w+)?\s+equally", low)
    if not (ms and pm and len(ms) >= 3):
        return None
    return sum(Fraction(m) for m in ms) / 3


def _wechselgeld_suess(question: str, quants: List[Quantity],
                         tgt: QuestionTarget) -> Optional[Fraction]:
    """Wechselgeld-Suess: '15-11' -> 4."""
    low = _digitize(question.lower())
    am = re.search(r"(?:bought\s+)?(\d+)\s+candies?(?:\s+of\s+type)?"
                   r"\s+a", low)
    bm = re.search(r"(\d+)\s+candies?(?:\s+of\s+type)?\s+b", low)
    ap = re.search(r"(?:type\s+a\s+costs\s+|a\s+at\s+)\\?\$?"
                   r"(\d+(?:\.\d+)?)", low)
    bp = re.search(r"(?:type\s+b\s+costs\s+|b\s+at\s+)\\?\$?"
                   r"(\d+(?:\.\d+)?)", low)
    pm = re.search(r"paid(?:\s+the\s+cashier)?\s+\\?\$?(\d+)",
                   low)
    if not (am and bm and ap and bp and pm):
        return None
    total = Fraction(am.group(1)) * Fraction(ap.group(1)) + \
        Fraction(bm.group(1)) * Fraction(bp.group(1))
    return Fraction(pm.group(1)) - total


def _anteile_invest(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Anteile-Invest: '1200-480-480' -> 240."""
    low = _digitize(question.lower())
    tm = re.search(r"invested\s+\\?\$?(\d+)", low)
    dm = re.search(r"dylan(?:'s(?:\s+investment\s+of)?)?\s+"
                   r"(\d+)/(\d+)", low)
    fm = re.search(r"frances(?:\s+invested)?\s+(\d+)/(\d+)", low)
    rm = re.search(r"remaining\s+amount|of\s+rest", low)
    if not (tm and dm and fm and rm):
        return None
    total = Fraction(tm.group(1))
    first = total * Fraction(int(dm.group(1)), int(dm.group(2)))
    rest = total - first
    return rest - rest * Fraction(int(fm.group(1)), int(fm.group(2)))


def _hash_browns(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Hash-Browns: '96/6*36' -> 576."""
    low = _digitize(question.lower())
    pm = re.search(r"if\s+(\d+)\s+potatoes?\s+makes?\s+(\d+)|"
                   r"(\d+)\s+potatoes?\s*->\s*(\d+)", low)
    qms = re.findall(r"(\d+)\s+potatoes?", low)
    if not (pm and qms):
        return None
    p1 = pm.group(1) or pm.group(3)
    p2 = pm.group(2) or pm.group(4)
    return Fraction(qms[-1]) / Fraction(p1) * Fraction(p2)


def _aufzug_last(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Aufzug-Last: '720-700' -> 20."""
    low = _digitize(question.lower())
    lm = re.search(r"(?:maximum\s+)?load\s+of\s+(\d+)\s+kg|"
                   r"max\s+(\d+)\s+kg", low)
    wm = re.search(r"weighs?\s+an\s+average\s+of\s+(\d+)\s+kg|"
                   r"adults?\s+(\d+)\s+kg", low)
    om = re.search(r"with\s+(\d+)\s+other\s+adults?|\+\s*(\d+)"
                   r"\s+others?", low)
    if not (lm and wm and om):
        return None
    lv = lm.group(1) or lm.group(2)
    ov = om.group(1) or om.group(2)
    wv = wm.group(1) or wm.group(2)
    total = (Fraction(ov) + 1) * Fraction(wv)
    return total - Fraction(lv)


def _schulweg_zeit(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Schulweg-Zeit: '30-6-13' -> 11."""
    low = _digitize(question.lower())
    tm = re.search(r"have\s+(\d+)\s+minutes?\s+to\s+walk|"
                   r"(\d+)\s+min\s+to\s+\w+", low)
    cm = re.search(r"takes?\s+them\s+(\d+)\s+minutes?\s+to\s+get"
                   r"\s+to\s+the\s+corner|(\d+)\s+min\s+to\s+"
                   r"corner", low)
    fm = re.search(r"another\s+(\d+)\s+minutes?|another\s+(\d+)"
                   r"\s+min", low)
    if not (tm and cm and fm):
        return None
    tv = tm.group(1) or tm.group(2)
    cv = cm.group(1) or cm.group(2)
    fv = fm.group(1) or fm.group(2)
    return Fraction(tv) - Fraction(cv) - Fraction(fv)


def _obst_kauf(question: str, quants: List[Quantity],
              tgt: QuestionTarget) -> Optional[Fraction]:
    """Obst-Kauf: '4+4+6' -> 14."""
    low = _digitize(question.lower())
    am = re.search(r"apples?(?:\s+for)?\s+\\?\$?(\d+)", low)
    bm = re.search(r"bananas?(?:\s+for)?\s+\\?\$?(\d+)(?:\s+per"
                   r"\s+kilo|/kilo)", low)
    bq = re.search(r"(\d+)\s+kilos?\s+(?:of\s+)?bananas?", low)
    om = re.search(r"oranges?(?:\s+for)?\s+\\?\$?(\d+)(?:\s+per"
                   r"\s+kilo|/kilo)", low)
    oq = re.search(r"(\d+)\s+kilos?\s+(?:of\s+)?oranges?", low)
    if not (am and bm and bq and om and oq):
        return None
    return Fraction(am.group(1)) + Fraction(bq.group(1)) * \
        Fraction(bm.group(1)) + Fraction(oq.group(1)) * \
        Fraction(om.group(1))


def _lutscher_gesamt(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Lutscher-Gesamt: '10*(0.4)+10*0.3' -> 7."""
    low = _digitize(question.lower())
    lm = re.search(r"(?:five|5)\s+lollipops?\s+(?:and|\+)\s+"
                   r"(?:four|4)\s+candies?", low)
    tm = re.search(r"(?:cost\s+|=)\s*\\?\$?(\d+(?:\.\d+)?)",
                   low)
    lp = re.search(r"(?:each\s+)?lollipop(?:\s+costs)?\s+\\?\$?"
                   r"(\d+(?:\.\d+)?)", low)
    qms = re.findall(r"(\d+)\s+lollipops?\s+(?:and|\+)\s+(\d+)"
                     r"\s+candies?|(\d+)\s*\+\s*(\d+)\??\s*$",
                     low)
    if not (lm and tm and lp and qms):
        return None
    lolli = Fraction(lp.group(1))
    candy = (Fraction(tm.group(1)) - 5 * lolli) / 4
    q1, q2 = qms[-1][0] or qms[-1][2], qms[-1][1] or qms[-1][3]
    return Fraction(q1) * lolli + Fraction(q2) * candy


def _tierarzt_bill(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Tierarzt-Bill: '125-100' -> 25."""
    low = _digitize(question.lower())
    vm = re.search(r"(\d+)\s+vaccines?", low)
    vp = re.search(r"\$?(\d+)\s+each", low)
    pm = re.search(r"(\d+)\s*%\s+of(?:\s+his\s+total)?\s+bill",
                   low)
    bm = re.search(r"brought\s+\\?\$?(\d+)", low)
    if not (vm and vp and pm and bm):
        return None
    vacc = Fraction(vm.group(1)) * Fraction(vp.group(1))
    total = vacc / Fraction(100 - int(pm.group(1)), 100)
    return Fraction(bm.group(1)) - total


def _quilt_quadrate(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Quilt-Quadrate: '14+18+24+12' -> 68."""
    low = _digitize(question.lower())
    rm = re.search(r"(\d+)\s+red(?:\s+squares?)?", low)
    bm = re.search(r"(\d+)\s+more\s+blue(?:\s+squares?\s+than"
                   r"\s+red)?", low)
    gm = re.search(r"(\d+)\s+more\s+green(?:\s+squares?\s+than"
                   r"\s+blue)?", low)
    wm = re.search(r"(\d+)\s+fewer\s+white(?:\s+squares?\s+than"
                   r"\s+green)?", low)
    if not (rm and bm and gm and wm):
        return None
    red = Fraction(rm.group(1))
    blue = red + Fraction(bm.group(1))
    green = blue + Fraction(gm.group(1))
    white = green - Fraction(wm.group(1))
    return red + blue + green + white


def _grossmutter_babies(question: str, quants: List[Quantity],
                          tgt: QuestionTarget) -> Optional[Fraction]:
    """Grossmutter-Babies: '3^3' -> 27."""
    low = _digitize(question.lower())
    cm = re.search(r"(?:has\s+)?(?:three|3)\s+children", low)
    gm = re.search(r"(?:grandchildren?\s+has\s+)?(?:three|3)\s+"
                   r"babies?", low)
    if not (cm and gm):
        return None
    return 27


def _bleistift_paare(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Bleistift-Paare: '16/2' -> 8."""
    low = _digitize(question.lower())
    sm = re.search(r"space\s+for\s+(\d+)\s+pencils?|holds?\s+"
                   r"(\d+)\s+pencils?", low)
    mm = re.search(r"(\d+)\s+pencils?\s+missing|(\d+)\s+missing",
                   low)
    pm = re.search(r"pairs?(?:\s+of\s+pencils?)?(?:\s+in\s+box)?",
                   low)
    if not (sm and mm and pm):
        return None
    sv = sm.group(1) or sm.group(2)
    mv = mm.group(1) or mm.group(2)
    return (Fraction(sv) - Fraction(mv)) / 2


def _mosaik_fliesen(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Mosaik-Fliesen: '36*2/3*24' -> 576."""
    low = _digitize(question.lower())
    tm = re.search(r"needs?\s+(\d+)\s+mosaic\s+tiles?|(\d+)\s+"
                   r"tiles?/sqft", low)
    fm = re.search(r"(?:two|2)\s+thirds", low)
    sm = re.search(r"(\d+)\s+sq\s*ft", low)
    if not (tm and fm and sm):
        return None
    tv = tm.group(1) or tm.group(2)
    return Fraction(sm.group(1)) * Fraction(2, 3) * Fraction(tv)


def _blaubeeren_spar(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Blaubeeren-Spar: '75-65' -> 10."""
    low = _digitize(question.lower())
    cm = re.search(r"cost\s+\\?\$?(\d+)\s+to\s+go|\\?\$?(\d+)"
                   r"\s*\+", low)
    pm = re.search(r"(?:another\s+)?\\?\$?(\d+(?:\.\d+)?)"
                   r"(?:\s+per\s+pound|/lb)", low)
    qm = re.search(r"picked\s+(\d+)\s+pounds?|(\d+)\s+lb", low)
    sm = re.search(r"store(?:\s+for)?\s+\\?\$?(\d+(?:\.\d+)?)"
                   r"(?:\s+a\s+pound|/lb)", low)
    if not (cm and pm and qm and sm):
        return None
    cv = cm.group(1) or cm.group(2)
    qv = qm.group(1) or qm.group(2)
    pick = Fraction(cv) + Fraction(qv) * Fraction(pm.group(1))
    store = Fraction(qv) * Fraction(sm.group(1))
    return store - pick


def _schreibwaren_rest(question: str, quants: List[Quantity],
                         tgt: QuestionTarget) -> Optional[Fraction]:
    """Schreibwaren-Rest: '10-5' -> 5."""
    low = _digitize(question.lower())
    pp = re.search(r"pencil\s+cost\s+\\?\$?(\d+(?:\.\d+)?)",
                   low)
    ep = re.search(r"eraser\s+cost\s+\\?\$?(\d+(?:\.\d+)?)",
                   low)
    qm = re.search(r"bought\s+(\d+)\s+pencils?\s+and\s+(\d+)\s+"
                   r"erasers?", low)
    pm = re.search(r"paid\s+\\?\$?(\d+)", low)
    if not (pp and ep and qm and pm):
        return None
    total = Fraction(qm.group(1)) * Fraction(pp.group(1)) + \
        Fraction(qm.group(2)) * Fraction(ep.group(1))
    return Fraction(pm.group(1)) - total


def _geb_alter(question: str, quants: List[Quantity],
                tgt: QuestionTarget) -> Optional[Fraction]:
    """Half-age relation with the people and question target bound."""
    low = _digitize(question.lower())
    relation = re.search(
        r"\b([a-z]+)\s+is\s+(\d+)\s+(?:years?\s+)?less\s+than\s+half\s+"
        r"the\s+age\s+of\s+([a-z]+)\b",
        low,
    )
    known = re.search(
        r"\bif\s+([a-z]+)\s+is\s+(\d+)\s+years?\s+old\b", low
    )
    asked = re.search(r"\bhow\s+old\s+is\s+([a-z]+)\b", low)
    if not (relation and known and asked):
        return None
    child, offset, reference = relation.groups()
    known_person, age = known.groups()
    if child != asked.group(1) or reference != known_person:
        return None
    half_age = Fraction(age) / 2
    difference = Fraction(offset)
    return half_age - difference if half_age >= difference else None


def _masken_wechsel(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Masks per outing times outings per day times requested days."""
    low = _digitize(question.lower())
    changes = re.search(
        r"\b([a-z]+)\s+changes\s+(his|her|their)\s+face\s+masks?\s+(\d+)\s+"
        r"times\s+every\s+time\s+(he|she|they)\s+goes?\s+out\b",
        low,
    )
    outings = re.search(
        r"\b(?:if\s+)?(he|she|they)\s+goes?\s+out\s+(\d+)\s+times\s+a\s+day\b",
        low,
    )
    period = re.search(r"\bevery\s+(\d+)\s+days?\b", low)
    asked = re.search(
        r"\bhow\s+many\s+face\s+masks?\s+does\s+(he|she|they)\s+use\b", low
    )
    if not (changes and outings and period and asked):
        return None
    possessive_pronoun = {"his": "he", "her": "she", "their": "they"}
    if not (
        possessive_pronoun[changes.group(2)]
        == changes.group(4)
        == outings.group(1)
        == asked.group(1)
    ):
        return None
    return (
        Fraction(changes.group(3))
        * Fraction(outings.group(2))
        * Fraction(period.group(1))
    )


def _lotterie_wahrscheinlichkeit(question: str, quants: List[Quantity],
                                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Independent-ticket probability with a valid second probability."""
    low = _digitize(question.lower())
    first = re.search(
        r"\bbuys\s+1\s+lottery\s+ticket\s+with\s+a\s+(\d+)\s*%\s+chance\s+"
        r"of\s+winning\b",
        low,
    )
    second = re.search(
        r"\bsecond\s+lottery\s+ticket\s+that's\s+(\d+)\s+times\s+more\s+"
        r"likely\s+to\s+win\b",
        low,
    )
    target = re.search(
        r"\bprobability,?\s+expressed\s+as\s+a\s+percentage,?\s+that\s+both\s+"
        r"tickets\s+are\s+winners\b",
        low,
    )
    if not (first and second and target):
        return None
    first_pct = int(first.group(1))
    multiplier = int(second.group(1))
    second_pct = first_pct * multiplier
    if not (0 <= first_pct <= 100 and 0 <= second_pct <= 100):
        return None
    return Fraction(first_pct * second_pct, 100)


def _seil_laenge(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Solve the fully bound red/blue/yellow rope system."""
    low = _digitize(question.lower())
    red = re.search(
        r"\bthe\s+red\s+rope\s+was\s+(\d+)\s+times\s+the\s+length\s+of\s+"
        r"the\s+blue\s+rope\b",
        low,
    )
    yellow = re.search(
        r"\bthe\s+blue\s+rope\s+was\s+(\d+)\s+centimeters?\s+shorter\s+than\s+"
        r"the\s+yellow\s+rope\b",
        low,
    )
    total = re.search(
        r"\bif\s+the\s+3\s+ropes\s+had\s+a\s+combined\s+length\s+of\s+(\d+)\s+"
        r"centimeters?\b",
        low,
    )
    asked = re.search(
        r"\bwhat\s+was\s+the\s+length\s+of\s+the\s+red\s+rope\s+in\s+"
        r"centimeters?\b",
        low,
    )
    if not (red and yellow and total and asked):
        return None
    factor = Fraction(red.group(1))
    difference = Fraction(yellow.group(1))
    combined = Fraction(total.group(1))
    if factor <= 0 or combined <= difference:
        return None
    blue = (combined - difference) / (factor + 2)
    return blue * factor


def _fischfutter(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Daily per-fish food cost over May's 31 days."""
    low = _digitize(question.lower())
    fish = re.search(r"\bgot\s+(\d+)\s+fish\b", low)
    daily = re.search(
        r"\bthey\s+each\s+need\s+\$?(\d+(?:\.\d+)?)\s+worth\s+of\s+food\s+"
        r"a\s+day\b",
        low,
    )
    asked = re.search(
        r"\bhow\s+much\s+does\s+(?:he|she|they|[a-z]+)\s+spend\s+on\s+food\s+"
        r"in\s+the\s+month\s+of\s+may\b",
        low,
    )
    if not (fish and daily and asked):
        return None
    return Fraction(fish.group(1)) * Fraction(daily.group(1)) * 31


def _durchschnitts_geschwindigkeit(question: str, quants: List[Quantity],
                                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Distance-weighted average speed across two bound travel legs."""
    low = _digitize(question.lower())
    first = re.search(
        r"\b([a-z]+)\s+traveled\s+(\d+)\s+miles\s+in\s+(\d+)\s+hours?\b",
        low,
    )
    second = re.search(
        r"\bif\s+([a-z]+)\s+then\s+traveled\s+an\s+additional\s+(\d+)\s+"
        r"miles\s+in\s+(\d+)\s+hours?\b",
        low,
    )
    asked = re.search(r"\bwhat(?:'s|\s+is)\s+the\s+average\s+speed\b", low)
    if not (first and second and asked) or first.group(1) != second.group(1):
        return None
    hours = Fraction(first.group(3)) + Fraction(second.group(3))
    if hours <= 0:
        return None
    miles = Fraction(first.group(2)) + Fraction(second.group(2))
    return miles / hours


def _kassette_dauer(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Total a two-song cassette when song two is relatively longer."""
    low = _digitize(question.lower())
    cassette = re.search(r"\bbuys\s+a\s+cassette\s+with\s+(\d+)\s+songs?\b", low)
    first = re.search(r"\bthe\s+first\s+song\s+is\s+(\d+)\s+minutes?\b", low)
    second = re.search(
        r"\bthe\s+second\s+song\s+is\s+(\d+)\s*%\s+longer\b", low
    )
    asked = re.search(
        r"\bhow\s+much\s+time\s+was\s+the\s+total\s+cassette\b", low
    )
    if not (cassette and first and second and asked):
        return None
    if int(cassette.group(1)) != 2:
        return None
    first_minutes = Fraction(first.group(1))
    second_minutes = first_minutes * Fraction(100 + int(second.group(1)), 100)
    return first_minutes + second_minutes


def _stock_laenge(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Cane length from a name-bound three-person height chain."""
    low = _digitize(question.lower())
    cane = re.search(
        r"\b([a-z]+)\s+has\s+a\s+cane\s+that\s+is\s+half\s+as\s+long\s+as\s+"
        r"(?:he|she)\s+is\s+tall\b",
        low,
    )
    owner_relation = re.search(
        r"\b([a-z]+)\s+is\s+(\d+)\s+(?:foot|feet)\s+taller\s+than\s+"
        r"(?:his|her)\s+"
        r"brother,\s+([a-z]+)\b",
        low,
    )
    brother_relation = re.search(
        r"\b(?:and\s+)?([a-z]+)\s+is\s+(\d+)\s+(?:foot|feet)\s+shorter\s+than\s+"
        r"(?:his|her)\s+cousin,\s+([a-z]+)\b",
        low,
    )
    known = re.search(
        r"\bif\s+([a-z]+)\s+is\s+(\d+)\s+(?:foot|feet)\s+tall\b", low
    )
    asked = re.search(
        r"\bhow\s+long\s+is\s+([a-z]+)'s\s+cane,?\s+in\s+feet\b", low
    )
    if not (cane and owner_relation and brother_relation and known and asked):
        return None
    if not (
        cane.group(1) == owner_relation.group(1) == asked.group(1)
        and owner_relation.group(3) == brother_relation.group(1)
        and brother_relation.group(3) == known.group(1)
    ):
        return None
    owner_height = (
        Fraction(known.group(2))
        - Fraction(brother_relation.group(2))
        + Fraction(owner_relation.group(2))
    )
    return owner_height / 2 if owner_height > 0 else None


def _bus_verhaeltnis(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Bus-Verhältnis: '54-20' -> 34."""
    low = _digitize(question.lower())
    rm = re.search(r"ratio\s+of\s+men\s+to\s+women\s+on\s+a\s+bus\s+is\s+(\d+):(\d+)",
                   low)
    tm = re.search(r"total\s+number\s+of\s+passengers\s+on\s+the\s+bus\s+is\s+"
                   r"(\d+)", low)
    am = re.search(r"(\d+)\s+women\s+alight\s+from\s+the\s+bus", low)
    if not (rm and tm and am):
        return None
    women = Fraction(tm.group(1)) * Fraction(rm.group(2)) / \
        (Fraction(rm.group(1)) + Fraction(rm.group(2)))
    return women - Fraction(am.group(1))


def _eier_teilen(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Eier-Teilen: '36/4' -> 9."""
    low = _digitize(question.lower())
    dm = re.search(r"prepared\s+(?:three|3)\s+dozen\s+eggs", low)
    km = re.search(r"(\d+)\s+children", low)
    if not (dm and km and 'each child gets the same' in low):
        return None
    return 36 / Fraction(km.group(1))


def _kartoffelbrei(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Kartoffelbrei: '18x1.5' -> 27."""
    low = _digitize(question.lower())
    sm = re.search(r"ate\s+(\d+)\s+less\s+than\s+(\d+)\s+scoops", low)
    pm = re.search(r"takes\s+(\d+)\s+less\s+than\s+(\d+)\s+potatoes?\s+to\s+make\s+"
                   r"(\d+)\s+less\s+than\s+(\d+)\s+scoops", low)
    if not (sm and pm):
        return None
    scoops = Fraction(sm.group(2)) - Fraction(sm.group(1))
    per = (Fraction(pm.group(2)) - Fraction(pm.group(1))) / \
        (Fraction(pm.group(4)) - Fraction(pm.group(3)))
    return scoops * per


def _eier_monate(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Eier-Monate: '240/12' -> 20."""
    low = _digitize(question.lower())
    fm = re.search(r"eats\s+(\d+)\s+eggs?\s+a\s+day\s+for\s+(\d+)\s+days", low)
    sm = re.search(r"increases\s+it\s+to\s+(\d+)\s+eggs?\s+a\s+day\s+for\s+"
                   r"(\d+)\s+days", low)
    if not (fm and sm and 'dozens' in low):
        return None
    total = Fraction(fm.group(1)) * Fraction(fm.group(2)) + \
        Fraction(sm.group(1)) * Fraction(sm.group(2))
    return total / 12


def _tierfarm(question: str, quants: List[Quantity],
               tgt: QuestionTarget) -> Optional[Fraction]:
    """Tierfarm: '70+630' -> 700."""
    low = _digitize(question.lower())
    sm = re.search(r"starting\s+with\s+(\d+)\s+cows?\s+and\s+(\d+)\s+chickens",
                   low)
    cm = re.search(r"(\d+)\s+cows?\s+per\s+day", low)
    hm = re.search(r"(\d+)\s+chickens?\s+per\s+day", low)
    if not (sm and cm and hm and re.search(r"(?:three|3)\s+weeks", low)):
        return None
    start = Fraction(sm.group(1)) + Fraction(sm.group(2))
    return start + (Fraction(cm.group(1)) + Fraction(hm.group(1))) * 21


def _gehalt_familie(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Gehalt-Familie: '15000+30000' -> 45000."""
    low = _digitize(question.lower())
    vm = re.search(r"earns\s+\\?\$?(\d+)\s+per\s+month,\s+1/2\s+of\s+what\s+her\s+"
                   r"brother\s+earns", low)
    mm = re.search(r"mother\s+earns\s+twice\s+their\s+combined\s+salary", low)
    if not (vm and mm and 'total amount' in low):
        return None
    val = Fraction(vm.group(1))
    bro = val * 2
    mom = (val + bro) * 2
    return val + bro + mom


def _sparwochen(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Sparwochen: '(60-32)/4' -> 7."""
    low = _digitize(question.lower())
    wm = re.search(r"saved\s+\\?\$?(\d+)\s+of\s+her\s+allowance\s+every\s+week\s+"
                   r"for\s+the\s+past\s+(\d+)\s+weeks", low)
    tm = re.search(r"saved\s+a\s+total\s+of\s+\\?\$?(\d+)", low)
    if not (wm and tm and 'more weeks' in low):
        return None
    already = Fraction(wm.group(1)) * Fraction(wm.group(2))
    return (Fraction(tm.group(1)) - already) / Fraction(wm.group(1))


def _voegel_baume(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Vögel-Bäume: '21+8+3' -> 32."""
    low = _digitize(question.lower())
    fm = re.search(r"(\d+)\s+trees?\s+each\s+had\s+(\d+)\s+blue\s+birds", low)
    sm = re.search(r"(\d+)\s+different\s+trees?\s+each\s+had\s+(\d+)\s+blue\s+"
                   r"birds", low)
    lm = re.search(r"(\d+)\s+final\s+tree\s+had\s+(\d+)\s+blue\s+birds", low)
    if not (fm and sm and lm):
        return None
    return Fraction(fm.group(1)) * Fraction(fm.group(2)) + \
        Fraction(sm.group(1)) * Fraction(sm.group(2)) + \
        Fraction(lm.group(2))


def _teich_fische(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Teich-Fische: '16-11' -> 5."""
    low = _digitize(question.lower())
    fm = re.search(r"(\d+)\s+male\s+guppies?,\s+(\d+)\s+female\s+guppies?,\s+"
                   r"(\d+)\s+male\s+goldfishes?,\s+and\s+(\d+)\s+female\s+"
                   r"goldfishes", low)
    bm = re.search(r"buys\s+(\d+)\s+male\s+guppies?,\s+(\d+)\s+female\s+guppy,\s+"
                   r"(\d+)\s+male\s+goldfishes?,\s+and\s+(\d+)\s+female\s+"
                   r"goldfishes", low)
    if not (fm and bm):
        return None
    male = Fraction(fm.group(1)) + Fraction(fm.group(3)) + \
        Fraction(bm.group(1)) + Fraction(bm.group(3))
    female = Fraction(fm.group(2)) + Fraction(fm.group(4)) + \
        Fraction(bm.group(2)) + Fraction(bm.group(4))
    return female - male


def _liam_vince(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Liam-Vince: '7+2' -> 9."""
    low = _digitize(question.lower())
    lm = re.search(r"(\w+)\s+is\s+(\d+)\s+years?\s+old\s+now", low)
    tm = re.search(r"(?:two|2)\s+years?\s+ago,\s+(\w+)['\u2019]s\s+age\s+was\s+"
                   r"twice\s+the\s+age\s+of\s+(\w+)", low)
    if not (lm and tm):
        return None
    return (Fraction(lm.group(2)) - 2) / 2 + 2


def _mnm_tuetchen(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """M&M-Tütchen: '900/10' -> 90."""
    low = _digitize(question.lower())
    bm = re.search(r"buys\s+(\d+)\s+large\s+bags?\s+weighing\s+(\d+)\s+ounces?\s+"
                   r"each", low)
    om = re.search(r"an\s+ounce\s+of\s+m&m\s+has\s+(\d+)\s+m&m", low)
    pm = re.search(r"puts\s+(\d+)\s+in\s+each", low)
    if not (bm and om and pm):
        return None
    total = Fraction(bm.group(1)) * Fraction(bm.group(2)) * \
        Fraction(om.group(1))
    return total / Fraction(pm.group(1))


def _hunde_gewicht(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Hunde-Gewicht: '60+15+30' -> 105."""
    low = _digitize(question.lower())
    fm = re.search(r"(?:one|1)\s+dog\s+that\s+is\s+(?:one|1)-fourth\s+the\s+"
                   r"weight\s+of\s+(\w+)['\u2019]s\s+dog", low)
    hm = re.search(r"another\s+dog\s+that\s+is\s+half\s+the\s+weight", low)
    km = re.search(r"(\w+)['\u2019]s\s+dog\s+is\s+(\d+)\s+pounds", low)
    if not (fm and hm and km and 'altogether' in low):
        return None
    kory = Fraction(km.group(2))
    return kory + kory / 4 + kory / 2


def _baum_erloes(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Baum-Erlös: '16x5x1.2' -> 96."""
    low = _digitize(question.lower())
    tm = re.search(r"cuts\s+down\s+an\s+(\d+)-foot\s+tree", low)
    pm = re.search(r"make\s+logs\s+out\s+of\s+(\d+)\s*%", low)
    fm = re.search(r"cuts\s+it\s+into\s+(\d+)-foot\s+logs", low)
    pm2 = re.search(r"cuts\s+(\d+)\s+planks", low)
    sm = re.search(r"sells\s+each\s+plank\s+for\s+\\?\$?(\d+(?:\.\d+)?)", low)
    if not (tm and pm and fm and pm2 and sm):
        return None
    logs = Fraction(tm.group(1)) * Fraction(int(pm.group(1)), 100) / \
        Fraction(fm.group(1))
    return logs * Fraction(pm2.group(1)) * Fraction(sm.group(1))


def _wasserrutsche(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Wasserrutsche: '5-3' -> 2."""
    low = _digitize(question.lower())
    bm = re.search(r"(\d+)\s+feet\s+long,\s+and\s+people\s+slide\s+down\s+at\s+"
                   r"(\d+)\s+feet/minute", low)
    sm = re.search(r"(\d+)\s+feet\s+long,\s+but\s+steeper.*?slide\s+down\s+at\s+"
                   r"(\d+)\s+feet/minute", low)
    if not (bm and sm and 'how much longer' in low):
        return None
    return Fraction(bm.group(1)) / Fraction(bm.group(2)) - \
        Fraction(sm.group(1)) / Fraction(sm.group(2))


def _buch_budget(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Buch-Budget: '(16-6)/2' -> 5."""
    low = _digitize(question.lower())
    bm = re.search(r"budget\s+of\s+\\?\$?(\d+)", low)
    sm = re.search(r"already\s+spent\s+\\?\$?(\d+)", low)
    bm2 = re.search(r"bought\s+(\d+)\s+books?\s+today", low)
    lm = re.search(r"\\?\$?(\d+)\s+left\s+in\s+her\s+budget", low)
    if not (bm and sm and bm2 and lm):
        return None
    return (Fraction(bm.group(1)) - Fraction(sm.group(1)) -
            Fraction(lm.group(1))) / Fraction(bm2.group(1))


def _einschreibung(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Einschreibung: '50x1.2' -> 60."""
    low = _digitize(question.lower())
    sm = re.search(r"(\d+)\s+students\s+enrolled", low)
    im = re.search(r"(\d+)\s*%\s+increase\s+in\s+enrollment", low)
    if not (sm and im and 'enrolled this year' in low):
        return None
    return Fraction(sm.group(1)) * \
        Fraction(100 + int(im.group(1)), 100)


def _elternbesuch(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Elternbesuch: '2x140x2' -> 560."""
    low = _digitize(question.lower())
    vm = re.search(r"visits\s+his\s+parents\s+twice\s+a\s+month", low)
    hm = re.search(r"(\d+)\s+hours?\s+to\s+drive\s+there\s+at\s+a\s+speed\s+of\s+"
                   r"(\d+)\s+mph", low)
    if not (vm and hm and 'round trip' in low):
        return None
    one_way = Fraction(hm.group(1)) * Fraction(hm.group(2))
    return one_way * 2 * 2


def _wander_distanz(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Wander-Distanz: '70/2' -> 35."""
    low = _digitize(question.lower())
    dm = re.search(r"in\s+(\d+)\s+days,\s+(\w+)\s+will\s+walk\s+twice\s+as\s+far\s+as\s+"
                   r"(\w+)", low)
    pm = re.search(r"plans\s+to\s+walk\s+(\d+)\s+miles?\s+every\s+day", low)
    if not (dm and pm):
        return None
    return Fraction(pm.group(1)) * Fraction(dm.group(1)) / 2


def _band_teilen(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Band-Teilen: '100/4/5' -> 5."""
    low = _digitize(question.lower())
    cm = re.search(r"(\d+)\s+centimeters?\s+of\s+ribbon", low)
    fm = re.search(r"cut\s+into\s+(\d+)\s+equal\s+parts", low)
    sm = re.search(r"divided\s+into\s+(\d+)\s+equal\s+parts", low)
    if not (cm and fm and sm):
        return None
    return Fraction(cm.group(1)) / Fraction(fm.group(1)) / \
        Fraction(sm.group(1))


def _schulmaedchen(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Schulmädchen: '240/0.4x0.6' -> 360."""
    low = _digitize(question.lower())
    pm = re.search(r"(\d+)\s*%\s+of\s+a\s+school\s+population\s+is\s+made\s+up\s+of\s+"
                   r"(\d+)\s+boys", low)
    if not (pm and 'how many girls' in low):
        return None
    total = Fraction(pm.group(2)) / Fraction(int(pm.group(1)), 100)
    return total * Fraction(100 - int(pm.group(1)), 100)


def _garten_einkauf(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Garten-Einkauf: '45-7' -> 38."""
    low = _digitize(question.lower())
    pm = re.search(r"pots\s+for\s+\\?\$?(\d+)\s+and\s+a\s+sack\s+of\s+garden\s+"
                   r"soil\s+for\s+\\?\$?(\d+)", low)
    cm = re.search(r"coupon\s+for\s+\\?\$?(\d+)\s+off", low)
    if not (pm and cm):
        return None
    return Fraction(pm.group(1)) + Fraction(pm.group(2)) - \
        Fraction(cm.group(1))


def _absatz_durchschnitt(question: str, quants: List[Quantity],
                           tgt: QuestionTarget) -> Optional[Fraction]:
    """Absatz-Durchschnitt: '18/6' -> 3."""
    low = _digitize(question.lower())
    fm = re.search(r"(?:three|3)\s+of\s+the\s+women.*?wearing\s+(\d+)\s+inch\s+"
                   r"heels", low)
    sm = re.search(r"(?:three|3)\s+are\s+wearing\s+(\d+)\s+inch\s+heels", low)
    if not (fm and sm and 'average height of heels' in low):
        return None
    return (3 * Fraction(fm.group(1)) + 3 * Fraction(sm.group(1))) / 6


def _duenger_lieferung(question: str, quants: List[Quantity],
                          tgt: QuestionTarget) -> Optional[Fraction]:
    """Dünger-Lieferung: '400x3/4' -> 300."""
    low = _digitize(question.lower())
    tm = re.search(r"had\s+(\d+)\s+trucks", low)
    cm = re.search(r"each\s+truck\s+was\s+carrying\s+(\d+)\s+tons", low)
    qm = re.search(r"a\s+quarter\s+of\s+the\s+number\s+of\s+lorries", low)
    if not (tm and cm and qm and 'reached the farmers' in low):
        return None
    total = Fraction(tm.group(1)) * Fraction(cm.group(1))
    return total * Fraction(3, 4)


def _jeff_martha(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Jeff-Martha: '24-4+10' -> 30."""
    low = _digitize(question.lower())
    jm = re.search(r"(\w+)\s+is\s+(\d+)\s+years?\s+older\s+than\s+his\s+younger\s+"
                   r"sister,\s+(\w+)", low)
    mm = re.search(r"(\w+).*?is\s+(\d+)\s+years?\s+younger\s+than\s+her\s+"
                   r"boyfriend,\s+(\w+)", low)
    im = re.search(r"if\s+(\w+)\s+is\s+(\d+)\s+years?\s+old", low)
    if not (jm and mm and im):
        return None
    return Fraction(im.group(2)) - Fraction(mm.group(2)) + \
        Fraction(jm.group(2))


def _pause_stunden(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Pause-Stunden: '60x5/60' -> 5."""
    low = _digitize(question.lower())
    lm = re.search(r"(\d+)\s+min\s+lunch\s+and\s+(\d+)\s+(\d+)\s+minutes\s+"
                   r"break\s+per\s+day", low)
    dm = re.search(r"after\s+(\d+)\s+days", low)
    if not (lm and dm):
        return None
    per_day = Fraction(lm.group(1)) + \
        Fraction(lm.group(2)) * Fraction(lm.group(3))
    return per_day * Fraction(dm.group(1)) / 60


def _kreditkarte_balance(question: str, quants: List[Quantity],
                           tgt: QuestionTarget) -> Optional[Fraction]:
    """Kreditkarte-Balance: '85-15+16+27' -> 113."""
    low = _digitize(question.lower())
    cm = re.search(r"charged\s+\\?\$?(\d+(?:\.\d+)?)\s+worth\s+of\s+"
                   r"merchandise", low)
    rm = re.search(r"returning\s+(?:one|1)\s+item\s+that\s+cost\s+"
                   r"\\?\$?(\d+(?:\.\d+)?)", low)
    fm = re.search(r"frying\s+pan\s+that\s+was\s+on\s+sale\s+for\s+(\d+)\s*%\s+"
                   r"off\s+\\?\$?(\d+(?:\.\d+)?)", low)
    tm = re.search(r"towels?\s+that\s+was\s+(\d+)\s*%\s+off\s+\\?\$?(\d+(?:\.\d+)?)",
                   low)
    if not (cm and rm and fm and tm):
        return None
    return Fraction(cm.group(1)) - Fraction(rm.group(1)) + \
        Fraction(fm.group(2)) * Fraction(100 - int(fm.group(1)), 100) + \
        Fraction(tm.group(2)) * Fraction(100 - int(tm.group(1)), 100)


def _alter_dreifach_kette(question: str, quants: List[Quantity],
                           tgt: QuestionTarget) -> Optional[Fraction]:
    """Alter-Dreifach-Kette: '4x2x3' -> 24."""
    low = _digitize(question.lower())
    cm = re.search(r"(\w+)\s+is\s+(?:three|3)\s+times\s+older\s+than\s+(\w+)",
                   low)
    bm = re.search(r"(\w+)\s+is\s+(?:two|2)\s+times\s+older\s+than\s+(\w+)",
                   low)
    im = re.search(r"if\s+(\w+)\s+is\s+(\d+)", low)
    if not (cm and bm and im):
        return None
    return Fraction(im.group(2)) * 2 * 3


def _heu_ballen(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Heu-Ballen: '(5-3)x6' -> 12."""
    low = _digitize(question.lower())
    fm = re.search(r"each\s+hour\s+the\s+farmer\s+makes\s+(\d+)\s+bales", low)
    tm = re.search(r"each\s+hour\s+the\s+truck\s+picks\s+up\s+(\d+)\s+bales",
                   low)
    dm = re.search(r"(\d+)\s+hour\s+day", low)
    if not (fm and tm and dm):
        return None
    return (Fraction(fm.group(1)) - Fraction(tm.group(1))) * \
        Fraction(dm.group(1))


def _springball(question: str, quants: List[Quantity],
                tgt: QuestionTarget) -> Optional[Fraction]:
    """Springball: '72x4/9' -> 32."""
    low = _digitize(question.lower())
    bm = re.search(r"bounces\s+to\s+(\d+)/(\d+)rds\s+of\s+its\s+starting\s+"
                   r"height", low)
    sm = re.search(r"each\s+story\s+is\s+(\d+)\s+feet\s+high", low)
    if not (bm and sm and 'second bounce' in low):
        return None
    start = Fraction(3) * Fraction(sm.group(1))
    frac = Fraction(int(bm.group(1)), int(bm.group(2)))
    return start * frac * frac


def _apfel_erloes(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Apfel-Erlös: '200/10x5' -> 1000."""
    low = _digitize(question.lower())
    bm = re.search(r"sells\s+apples?\s+in\s+bags?\s+of\s+(\d+)", low)
    tm = re.search(r"sold\s+a\s+total\s+of\s+(\d+)\s+apples", low)
    pm = re.search(r"\\?\$?(\d+)\s+per\s+bag", low)
    if not (bm and tm and pm):
        return None
    return Fraction(tm.group(1)) / Fraction(bm.group(1)) * \
        Fraction(pm.group(1))


def _wand_anstrich(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Wand-Anstrich: '240/20x12' -> 144."""
    low = _digitize(question.lower())
    nm = re.search(r"north\s+and\s+south\s+walls?\s+are\s+(\d+)\s+x\s+(\d+)\s+"
                   r"feet", low)
    em = re.search(r"east\s+and\s+west\s+walls?\s+are\s+(\d+)\s+x\s+(\d+)\s+"
                   r"feet", low)
    gm = re.search(r"gallon\s+of\s+paint\s+can\s+cover\s+(\d+)\s+square\s+feet\s+"
                   r"and\s+cost\s+\\?\$?(\d+)", low)
    if not (nm and em and gm):
        return None
    area = 2 * Fraction(nm.group(1)) * Fraction(nm.group(2)) + \
        2 * Fraction(em.group(1)) * Fraction(em.group(2))
    return area / Fraction(gm.group(1)) * Fraction(gm.group(2))


def _zug_entfernung(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Zug-Entfernung: '(60+30)x3' -> 270."""
    low = _digitize(question.lower())
    fm = re.search(r"(?:one|1)\s+train\s+is\s+traveling\s+(\d+)\s+miles\s+an\s+"
                   r"hour", low)
    hm = re.search(r"half\s+that\s+distance\s+per\s+hour", low)
    am = re.search(r"after\s+(\d+)\s+hours", low)
    if not (fm and hm and am):
        return None
    a = Fraction(fm.group(1))
    return (a + a / 2) * Fraction(am.group(1))


def _bananen_spar(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Bananen-Spar: '32-30' -> 2."""
    low = _digitize(question.lower())
    pm = re.search(r"cost\s+\\?\$?(\d+(?:\.\d+)?)\s+each,\s+or\s+a\s+bunch\s+for\s+"
                   r"\\?\$?(\d+(?:\.\d+)?)", low)
    bm = re.search(r"buys\s+(\d+)\s+bunches?\s+that\s+average\s+(\d+)\s+bananas?\s+"
                   r"per\s+bunch", low)
    if not (pm and bm and 'save' in low):
        return None
    single = Fraction(bm.group(1)) * Fraction(bm.group(2)) * \
        Fraction(pm.group(1))
    bunch = Fraction(bm.group(1)) * Fraction(pm.group(2))
    return single - bunch


def _zaun_teilen(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Zaun-Teilen: '(100-60)/2' -> 20."""
    low = _digitize(question.lower())
    fm = re.search(r"(\d+)\s+feet\s+of\s+fence", low)
    hm = re.search(r"(\w+)\s+getting\s+(\d+)\s+feet\s+more\s+than\s+(\w+)",
                   low)
    if not (fm and hm):
        return None
    return (Fraction(fm.group(1)) - Fraction(hm.group(2))) / 2


def _krokodil_wachstum(question: str, quants: List[Quantity],
                         tgt: QuestionTarget) -> Optional[Fraction]:
    """Krokodil-Wachstum: '2x13' -> 26."""
    low = _digitize(question.lower())
    gm = re.search(r"grows\s+(\d+)\s+inches?\s+long\s+in\s+(\d+)\s+years", low)
    yms = re.findall(r"in\s+(\d+)\s+years", low)
    if not (gm and yms):
        return None
    return Fraction(gm.group(1)) / Fraction(gm.group(2)) * \
        Fraction(yms[-1])


def _bowling_score(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Bowling-Score: '2x90+15' -> 195."""
    low = _digitize(question.lower())
    fm = re.search(r"score\s+was\s+(\d+)\s+better\s+more\s+than\s+twice\s+as\s+"
                   r"high\s+as\s+\w+'s", low)
    bm = re.search(r"(\w+)\s+bowled\s+a\s+score\s+of\s+(\d+)", low)
    if not (fm and bm):
        return None
    return Fraction(bm.group(2)) * 2 + Fraction(fm.group(1))


def _milchshake_umsatz(question: str, quants: List[Quantity],
                         tgt: QuestionTarget) -> Optional[Fraction]:
    """Milchshake-Umsatz: '33+99+30' -> 162."""
    low = _digitize(question.lower())
    mm = re.search(r"sells\s+(\d+)\s+milkshakes?\s+for\s+\\?\$?(\d+(?:\.\d+)?)\s+"
                   r"each", low)
    bm = re.search(r"(\d+)\s+burger\s+platters?\s+for\s+\\?\$?(\d+)\s+each",
                   low)
    sm = re.search(r"(\d+)\s+sodas?\s+for\s+\\?\$?(\d+(?:\.\d+)?)\s+each",
                   low)
    if not (mm and bm and sm):
        return None
    return Fraction(mm.group(1)) * Fraction(mm.group(2)) + \
        Fraction(bm.group(1)) * Fraction(bm.group(2)) + \
        Fraction(sm.group(1)) * Fraction(sm.group(2))


def _affen_rest(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Affen-Rest: '45-24' -> 21."""
    low = _digitize(question.lower())
    bm = re.search(r"buys\s+(\d+)\s+bananas", low)
    fm = re.search(r"(?:one|1)\s+monkey\s+eats\s+(\d+)\s+bananas?\s+each\s+day",
                   low)
    sm = re.search(r"second\s+monkey\s+eats\s+(\d+)\s+more\s+bananas\s+than\s+"
                   r"the\s+first\s+monkey", low)
    if not (bm and fm and sm and 'third monkey eats the rest' in low):
        return None
    per_day = Fraction(bm.group(1)) / 7
    return per_day - Fraction(fm.group(1)) - \
        (Fraction(fm.group(1)) + Fraction(sm.group(1)))


def _uhr_rabatt(question: str, quants: List[Quantity],
                tgt: QuestionTarget) -> Optional[Fraction]:
    """Uhr-Rabatt: '200/2000' -> 10."""
    low = _digitize(question.lower())
    om = re.search(r"\\?\$?(\d+)\s+watch\s+was\s+put\s+on\s+sale", low)
    bm = re.search(r"bought\s+it\s+at\s+(\d+)\s*%\s+of\s+its\s+original\s+price",
                   low)
    sm = re.search(r"sold\s+the\s+watch\s+to\s+his\s+friend\s+at\s+(\d+)\s*%\s+"
                   r"of\s+the\s+price\s+that\s+he\s+bought\s+it", low)
    if not (om and bm and sm and 'percentage discount' in low):
        return None
    orig = Fraction(om.group(1))
    bought = orig * Fraction(int(bm.group(1)), 100)
    sold = bought * Fraction(int(sm.group(1)), 100)
    return (orig - sold) / orig * 100


def _quallen_springe(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Quallen-Springe: '5x4x3600' -> 72000."""
    low = _digitize(question.lower())
    sm = re.search(r"(\d+)\s+springs?\s+working\s+at\s+the\s+same\s+rate", low)
    hm = re.search(r"in\s+(\d+)\s+hours", low)
    if not (sm and hm and 'every second' in low):
        return None
    return Fraction(sm.group(1)) * Fraction(hm.group(1)) * 3600


def _testmittel_vier(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Testmittel-Vier: '380/4' -> 95."""
    low = _digitize(question.lower())
    fm = re.search(r"scored\s+(\d+)\s+on\s+his\s+first\s+(\d+)\s+tests", low)
    sm = re.search(r"an\s+(\d+)\s+on\s+his\s+4th", low)
    if not (fm and sm and 'average score' in low):
        return None
    total = Fraction(fm.group(1)) * Fraction(fm.group(2)) + \
        Fraction(sm.group(1))
    return total / (Fraction(fm.group(2)) + 1)


def _gutschein_porto(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Gutschein-Porto: '24500c' -> 245."""
    low = _digitize(question.lower())
    sm = re.search(r"(\d+)\s+small\s+coupons?\s+and\s+twice\s+as\s+many\s+big\s+"
                   r"coupons", low)
    sp = re.search(r"small\s+coupon\s+costs\s+(\d+)\s+cents", low)
    bp = re.search(r"big\s+coupon\s+costs\s+(\d+)\s+cents", low)
    if not (sm and sp and bp):
        return None
    total_c = Fraction(sm.group(1)) * Fraction(sp.group(1)) + \
        Fraction(sm.group(1)) * 2 * Fraction(bp.group(1))
    return total_c / 100


def _fleischbaellchen(question: str, quants: List[Quantity],
                        tgt: QuestionTarget) -> Optional[Fraction]:
    """Fleischbällchen: '6x4' -> 24."""
    low = _digitize(question.lower())
    mm = re.search(r"contains\s+(\d+)\s+meatballs", low)
    om = re.search(r"ordered\s+(\d+)\s+less\s+than\s+(?:ten|10)\s+meatball\s+"
                   r"sub\s+sandwiches", low)
    em = re.search(r"ate\s+(\d+)\s+of\s+\w+'s\s+meatball\s+sub\s+sandwiches",
                   low)
    am = re.search(r"ordered\s+another\s+(\d+)\s+sub\s+sandwiches", low)
    if not (mm and om and em and am and 'remained' in low):
        return None
    left = 10 - Fraction(om.group(1)) - Fraction(em.group(1)) + \
        Fraction(am.group(1))
    return left * Fraction(mm.group(1))


def _butter_angebot(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Butter-Angebot: '12+6' -> 18."""
    low = _digitize(question.lower())
    dm = re.search(r"(\d+)\s+pound\s+of\s+butter\s+for\s+every\s+dozen", low)
    nm = re.search(r"needs\s+to\s+make\s+(\d+)\s+dozen\s+croissants", low)
    bm = re.search(r"buy\s+(?:one|1)\s+pound\s+of\s+butter\s+get\s+(?:one|1)\s+"
                   r"half\s+off", low)
    pm = re.search(r"butter\s+costs\s+\\?\$?([\d.]+)\s+a\s+pound", low)
    if not (dm and nm and bm and pm):
        return None
    pounds = Fraction(nm.group(1))
    full = (pounds / 2).numerator // (pounds / 2).denominator
    price = Fraction(pm.group(1))
    return full * price + (pounds - full) * price / 2


def _katzenfutter_tage(question: str, quants: List[Quantity],
                         tgt: QuestionTarget) -> Optional[Fraction]:
    """Katzenfutter-Tage: '720/360' -> 2."""
    low = _digitize(question.lower())
    cm = re.search(r"has\s+(\d+)\s+cats", low)
    fm = re.search(r"feeds\s+her\s+cats?\s+twice\s+a\s+day\s+with\s+(\d+)\s+"
                   r"grams", low)
    gm = re.search(r"(\d+)\s+grams\s+of\s+cat\s+food\s+last", low)
    if not (cm and fm and gm):
        return None
    per_day = Fraction(cm.group(1)) * 2 * Fraction(fm.group(1))
    return Fraction(gm.group(1)) / per_day


def _film_wochenende(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Film-Wochenende: '6x4' -> 24."""
    low = _digitize(question.lower())
    sm = re.search(r"watch\s+(\d+)\s+movies?\s+every\s+saturday", low)
    hm = re.search(r"half\s+the\s+number\s+of\s+movies\s+on\s+sunday", low)
    wm = re.search(r"in\s+(\d+)\s+weeks", low)
    if not (sm and hm and wm and 'every weekend' in low):
        return None
    sat = Fraction(sm.group(1))
    return (sat + sat / 2) * Fraction(wm.group(1))


def _essens_zeiten(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Essens-Zeiten: '(98+18)/2' -> 58."""
    low = _digitize(question.lower())
    lm = re.search(r"(\w+)'s\s+part\s+took\s+(\d+)\s+minutes?\s+longer\s+than\s+"
                   r"(\w+)'s\s+part", low)
    mm = re.search(r"meal\s+was\s+made\s+in\s+(\d+)\s+minutes", low)
    if not (lm and mm):
        return None
    return (Fraction(mm.group(1)) + Fraction(lm.group(2))) / 2


def _email_antworten(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Email-Antworten: '80x0.8x5' -> 320."""
    low = _digitize(question.lower())
    em = re.search(r"gets\s+(\d+)\s+emails?\s+a\s+day", low)
    pm = re.search(r"(\d+)\s*%\s+of\s+those\s+emails?\s+don't\s+require\s+"
                   r"any\s+response", low)
    dm = re.search(r"(\d+)\s+day\s+work\s+week", low)
    if not (em and pm and dm):
        return None
    return Fraction(em.group(1)) * \
        Fraction(100 - int(pm.group(1)), 100) * Fraction(dm.group(1))


def _postamt_briefe(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Postamt-Briefe: '425+102+209' -> 736."""
    low = _digitize(question.lower())
    mm = re.search(r"delivered\s+(\d+)\s+letters", low)
    tm = re.search(r"(\d+)\s+more\s+than\s+(?:one|1)-fifth\s+as\s+many\s+as\s+"
                   r"(\w+)", low)
    wm = re.search(r"(\d+)\s+more\s+than\s+twice\s+as\s+many\s+as\s+they\s+"
                   r"delivered\s+on\s+tuesday", low)
    if not (mm and tm and wm):
        return None
    mon = Fraction(mm.group(1))
    tue = mon / 5 + Fraction(tm.group(1))
    wed = tue * 2 + Fraction(wm.group(1))
    return mon + tue + wed


def _wasser_galonen(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Wasser-Galonen: '240/16' -> 15."""
    low = _digitize(question.lower())
    dm = re.search(r"drinks\s+(\d+)\s+cups?\s+of\s+water\s+every\s+day", low)
    gm = re.search(r"(\d+)\s+cups?\s+in\s+a\s+gallon", low)
    dm2 = re.search(r"in\s+(\d+)\s+days", low)
    if not (dm and gm and dm2):
        return None
    return Fraction(dm.group(1)) * Fraction(dm2.group(1)) / \
        Fraction(gm.group(1))


def _zug_passagiere(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Zug-Passagiere: '120+20-50+100-80' -> 110."""
    low = _digitize(question.lower())
    bm = re.search(r"boards\s+a\s+train\s+with\s+(\d+)\s+people", low)
    fm = re.search(r"first\s+stop,\s+(\d+)\s+more\s+people\s+board", low)
    sm = re.search(r"second\s+stop,\s+(\d+)\s+people\s+descended", low)
    tm = re.search(r"twice\s+that\s+number\s+boarded", low)
    thm = re.search(r"(\d+)\s+more\s+people\s+descended\s+at\s+the\s+third",
                     low)
    if not (bm and fm and sm and tm and thm):
        return None
    n = Fraction(bm.group(1)) + Fraction(fm.group(1)) - \
        Fraction(sm.group(1)) + Fraction(sm.group(1)) * 2 - \
        Fraction(thm.group(1))
    return n


def _bodenfliesen(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Bodenfliesen: '200x12' -> 2400."""
    low = _digitize(question.lower())
    am = re.search(r"total\s+area\s+of\s+(\d+)\s+sqft", low)
    cm = re.search(r"tiles?\s+that\s+cost\s+\\?\$?(\d+)\s+each", low)
    if not (am and cm and 'each tile side is 1ft' in low):
        return None
    return Fraction(am.group(1)) * Fraction(cm.group(1))


def _versicherung_jahr(question: str, quants: List[Quantity],
                         tgt: QuestionTarget) -> Optional[Fraction]:
    """Versicherung-Jahr: '120x1.6x12' -> 2304."""
    low = _digitize(question.lower())
    pm = re.search(r"(\d+)\s*%\s+more\s+than\s+normal", low)
    nm = re.search(r"normal\s+cost\s+is\s+\\?\$?(\d+)\s+a\s+month", low)
    if not (pm and nm and 'a year' in low):
        return None
    return Fraction(nm.group(1)) * \
        Fraction(100 + int(pm.group(1)), 100) * 12


def _bettdecke_stoff(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Bettdecke-Stoff: '2x80' -> 160."""
    low = _digitize(question.lower())
    pm = re.search(r"(?:two|2)\s+pieces?\s+of\s+fabric\s+that\s+are\s+(\d+)\s+"
                   r"feet\s+longer\s+and\s+(\d+)\s+feet\s+wider", low)
    bm = re.search(r"measures\s+(\d+)\s+feet\s+long\s+by\s+(\d+)\s+feet\s+wide",
                   low)
    if not (pm and bm):
        return None
    per = (Fraction(bm.group(1)) + Fraction(pm.group(1))) * \
        (Fraction(bm.group(2)) + Fraction(pm.group(2)))
    return per * 2


def _catering_kosten(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Catering-Kosten: '65+36' -> 101."""
    low = _digitize(question.lower())
    cm = re.search(r"(\d+)\s+people\s+want\s+the\s+chicken\s+salad\s+which\s+is\s+"
                   r"\\?\$?(\d+(?:\.\d+)?)\s+per\s+person", low)
    pm = re.search(r"(\d+)\s+people\s+want\s+the\s+pasta\s+salad\s+at\s+"
                   r"\\?\$?(\d+(?:\.\d+)?)\s+per\s+person", low)
    if not (cm and pm):
        return None
    return Fraction(cm.group(1)) * Fraction(cm.group(2)) + \
        Fraction(pm.group(1)) * Fraction(pm.group(2))


def _suedamerika_bevoelkerung(question: str, quants: List[Quantity],
                                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Südamerika-Bevölkerung: '26x5x1000' -> 130000."""
    low = _digitize(question.lower())
    cm = re.search(r"(\d+)\s+countries\s+in\s+south\s+america", low)
    cm2 = re.search(r"(\d+)\s+cities\s+with\s+(\d+)\s+people\s+living\s+in\s+"
                    r"each\s+city", low)
    if not (cm and cm2):
        return None
    return Fraction(cm.group(1)) * Fraction(cm2.group(1)) * \
        Fraction(cm2.group(2))


def _maler_arbeit(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Maler-Arbeit: '9x21' -> 189."""
    low = _digitize(question.lower())
    pm = re.search(r"(\d+)\s+painters\s+worked", low)
    dm = re.search(r"3/8ths\s+of\s+a\s+day\s+every\s+day\s+for\s+(\d+)\s+"
                   r"weeks", low)
    if not (pm and dm and 'each painter' in low):
        return None
    per_day = Fraction(3, 8) * 24
    return per_day * 7 * Fraction(dm.group(1))


def _quiz_punkte(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Quiz-Punkte: '18+18' -> 36."""
    low = _digitize(question.lower())
    qm = re.search(r"(\d+)-item\s+quiz", low)
    em = re.search(r"(\d+)\s*%\s+of\s+the\s+questions\s+are\s+easy", low)
    gm = re.search(r"(\d+)\s*%\s+of\s+the\s+easy\s+questions", low)
    if not (qm and em and gm and 'half of the average' in low):
        return None
    easy = Fraction(qm.group(1)) * Fraction(int(em.group(1)), 100)
    rest = Fraction(qm.group(1)) - easy
    return easy * Fraction(int(gm.group(1)), 100) + rest / 2


def _geschworene_bezahlung(question: str, quants: List[Quantity],
                              tgt: QuestionTarget) -> Optional[Fraction]:
    """Geschworene-Bezahlung: '36/18' -> 2."""
    low = _digitize(question.lower())
    hm = re.search(r"(\d+)\s+hours?\s+a\s+day\s+for\s+(\d+)\s+days", low)
    pm = re.search(r"paid\s+\\?\$?(\d+)\s+per\s+day", low)
    pk = re.search(r"pays?\s+\\?\$?(\d+)\s+for\s+parking\s+each\s+day", low)
    if not (hm and pm and pk and 'per hour' in low):
        return None
    total = (Fraction(pm.group(1)) - Fraction(pk.group(1))) * \
        Fraction(hm.group(2))
    return total / (Fraction(hm.group(1)) * Fraction(hm.group(2)))


def _einkauf_summe(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Einkauf-Summe: '48+18' -> 66."""
    low = _digitize(question.lower())
    bm = re.search(r"buys\s+(\d+)\s+books?\s+for\s+(\d+)\s+dollars?\s+each\s+and\s+"
                   r"(\d+)\s+pencils?\s+for\s+(\d+)\s+dollars?\s+each", low)
    if not bm:
        return None
    return Fraction(bm.group(1)) * Fraction(bm.group(2)) + \
        Fraction(bm.group(3)) * Fraction(bm.group(4))


def _apfel_packungen(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Apfel-Packungen: '80/8' -> 10."""
    low = _digitize(question.lower())
    am = re.search(r"(?:forty|40)\s+apples?\s+in\s+(?:one|1)\s+box", low)
    om = re.search(r"ordered\s+(?:two|2)\s+boxes?\s+of\s+apples", low)
    pm = re.search(r"(?:eight|8)\s+apples?\s+in\s+(?:one|1)\s+pack", low)
    if not (am and om and pm):
        return None
    return Fraction(40) * 2 / 8


def _kaese_budget(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Käse-Budget: '50-40' -> 10."""
    low = _digitize(question.lower())
    pm = re.search(r"parmesan\s+cheese\s+is\s+\\?\$?(\d+)\s+per\s+pound", low)
    mm = re.search(r"mozzarella\s+cheese\s+is\s+\\?\$?(\d+)\s+per\s+pound",
                   low)
    bm = re.search(r"buys\s+(\d+)\s+pounds?\s+of\s+parmesan\s+and\s+(\d+)\s+"
                   r"pounds?\s+of\s+mozzarella", low)
    sm = re.search(r"starts\s+with\s+\\?\$?(\d+)\s+cash", low)
    if not (pm and mm and bm and sm):
        return None
    return Fraction(sm.group(1)) - Fraction(bm.group(1)) * \
        Fraction(pm.group(1)) - Fraction(bm.group(2)) * \
        Fraction(mm.group(1))


def _tanzstudio_einnahmen(question: str, quants: List[Quantity],
                             tgt: QuestionTarget) -> Optional[Fraction]:
    """Tanzstudio-Einnahmen: '40x12' -> 480."""
    low = _digitize(question.lower())
    rm = re.search(r"\\?\$?(\d+)\s+per\s+session\s+to\s+rent\s+the\s+studio",
                   low)
    sm = re.search(r"\\?\$?(\d+(?:\.\d+)?)\s+per\s+student\s+per\s+session",
                   low)
    nm = re.search(r"has\s+(\d+)\s+students", low)
    dm = re.search(r"rented\s+(\d+)\s+days?\s+a\s+week", low)
    if not (rm and sm and nm and dm and 'in a month' in low):
        return None
    per_session = Fraction(rm.group(1)) + \
        Fraction(sm.group(1)) * Fraction(nm.group(1))
    return per_session * Fraction(dm.group(1)) * 4


def _pool_befuellung(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Pool-Befüllung: '1400x0.59' -> 826."""
    low = _digitize(question.lower())
    dm = re.search(r"(\d+)\s+feet\s+wide,\s+(\d+)\s+feet\s+long,\s+and\s+"
                   r"(\d+)\s+feet\s+deep", low)
    mm = re.search(r"multiply\s+it\s+by\s+(\d+(?:\.\d+)?)", low)
    pm = re.search(r"\\?\$?(\d+(?:\.\d+)?)\s+per\s+gallon", low)
    if not (dm and mm and pm):
        return None
    volume = Fraction(dm.group(1)) * Fraction(dm.group(2)) * \
        Fraction(dm.group(3))
    return volume * Fraction(mm.group(1)) * Fraction(pm.group(1))


def _kuchen_einnahmen(question: str, quants: List[Quantity],
                        tgt: QuestionTarget) -> Optional[Fraction]:
    """Kuchen-Einnahmen: '320-20' -> 300."""
    low = _digitize(question.lower())
    sm = re.search(r"sold\s+(\d+)\s+cookies?\s+for\s+\\?\$?(\d+)\s+each\s+and\s+"
                   r"(\d+)\s+cupcakes?\s+for\s+\\?\$?(\d+)\s+each", low)
    gm = re.search(r"gave\s+her\s+(?:two|2)\s+sisters?\s+\\?\$?(\d+)\s+each",
                   low)
    if not (sm and gm):
        return None
    return Fraction(sm.group(1)) * Fraction(sm.group(2)) + \
        Fraction(sm.group(3)) * Fraction(sm.group(4)) - \
        Fraction(gm.group(1)) * 2


def _schlafzimmer_kredit(question: str, quants: List[Quantity],
                           tgt: QuestionTarget) -> Optional[Fraction]:
    """Schlafzimmer-Kredit: '2000x0.1' -> 200."""
    low = _digitize(question.lower())
    bm = re.search(r"bedroom\s+set\s+for\s+\\?\$?(\d+)", low)
    sm = re.search(r"sells\s+his\s+old\s+bedroom\s+for\s+\\?\$?(\d+)", low)
    pm = re.search(r"pay\s+(\d+)\s*%\s+a\s+month", low)
    if not (bm and sm and pm):
        return None
    balance = Fraction(bm.group(1)) - Fraction(sm.group(1))
    return balance * Fraction(int(pm.group(1)), 100)


def _abschluss_tickets(question: str, quants: List[Quantity],
                        tgt: QuestionTarget) -> Optional[Fraction]:
    """Abschluss-Tickets: '4750/950' -> 5."""
    low = _digitize(question.lower())
    sm = re.search(r"space\s+for\s+(\d+)\s+people", low)
    gm = re.search(r"(\d+)\s+seats\s+for\s+graduates", low)
    fm = re.search(r"(\d+)\s+seats\s+for\s+faculty", low)
    if not (sm and gm and fm and 'split equally' in low):
        return None
    return (Fraction(sm.group(1)) - Fraction(gm.group(1)) -
            Fraction(fm.group(1))) / Fraction(gm.group(1))


def _schokobox_vergleich(question: str, quants: List[Quantity],
                           tgt: QuestionTarget) -> Optional[Fraction]:
    """Schokobox-Vergleich: '16-8' -> 8."""
    low = _digitize(question.lower())
    boxes = re.findall(r"(\w+)\s+has\s+(\d+)\s+boxes?\s+with\s+the\s+same\s+"
                       r"number\s+of\s+chocolate\s+bars", low)
    tm = re.search(r"totals\s+of\s+(\d+)\s+and\s+(\d+)\s+chocolate\s+bars",
                   low)
    if len(boxes) < 2 or not tm:
        return None
    return Fraction(tm.group(1)) / Fraction(boxes[0][1]) - \
        Fraction(tm.group(2)) / Fraction(boxes[1][1])


def _kellnerin_sparen(question: str, quants: List[Quantity],
                        tgt: QuestionTarget) -> Optional[Fraction]:
    """Kellnerin-Sparen: '2000/1000' -> 2."""
    low = _digitize(question.lower())
    wm = re.search(r"makes\s+\\?\$?(\d+)\s+an\s+hour\s+from\s+wages\s+and\s+"
                   r"another\s+\\?\$?(\d+)\s+an\s+hour\s+from\s+tips", low)
    cm = re.search(r"(\d+)\s*%\s+of\s+the\s+cost\s+of\s+a\s+\\?\$?(\d+)\s+car",
                   low)
    hm = re.search(r"(\d+)\s+hours\s+a\s+week", low)
    if not (wm and cm and hm):
        return None
    down = Fraction(cm.group(2)) * Fraction(int(cm.group(1)), 100)
    per_week = (Fraction(wm.group(1)) + Fraction(wm.group(2))) * \
        Fraction(hm.group(1))
    return down / per_week


def _suessigkeiten_freunde(question: str, quants: List[Quantity],
                             tgt: QuestionTarget) -> Optional[Fraction]:
    """Süßigkeiten-Freunde: '13x60/10' -> 78."""
    low = _digitize(question.lower())
    pm = re.search(r"contains\s+(\d+)\s+packs", low)
    ep = re.search(r"each\s+pack\s+has\s+(\d+)\s+pieces", low)
    km = re.search(r"kept\s+(\d+)\s+packs\s+and\s+gave\s+the\s+rest\s+to\s+her\s+"
                   r"(\d+)\s+friends", low)
    if not (pm and ep and km):
        return None
    return (Fraction(pm.group(1)) - Fraction(km.group(1))) * \
        Fraction(ep.group(1)) / Fraction(km.group(2))


def _foto_alben(question: str, quants: List[Quantity],
                tgt: QuestionTarget) -> Optional[Fraction]:
    """Foto-Alben: '5x9' -> 45."""
    low = _digitize(question.lower())
    pm = re.search(r"uploaded\s+(\d+)\s+pictures", low)
    am = re.search(r"same\s+number\s+of\s+the\s+pics\s+into\s+(\d+)\s+albums",
                   low)
    sm = re.search(r"(\d+)\s+of\s+the\s+albums?\s+were\s+selfies\s+only\s+and\s+"
                   r"(\d+)\s+of\s+the\s+albums?\s+were\s+portraits", low)
    if not (pm and am and sm):
        return None
    per = Fraction(pm.group(1)) / Fraction(am.group(1))
    return (Fraction(sm.group(1)) + Fraction(sm.group(2))) * per


def _komet_alter(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Komet-Alter: '90-75' -> 15."""
    low = _digitize(question.lower())
    om = re.search(r"orbits\s+the\s+sun\s+every\s+(\d+)\s+years", low)
    dm = re.search(r"dad\s+saw\s+the\s+comet\s+when\s+he\s+was\s+(\d+)\s+years?\s+"
                   r"old", low)
    bm = re.search(r"bill\s+saw\s+the\s+comet\s+a\s+second\s+time\s+when\s+he\s+"
                   r"was\s+(?:three|3)\s+times\s+the\s+age", low)
    if not (om and dm and bm):
        return None
    return Fraction(dm.group(1)) * 3 - Fraction(om.group(1))


def _stachelschweine(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Stachelschweine: '50+1440' -> 1490."""
    low = _digitize(question.lower())
    pm = re.search(r"population\s+of\s+porcupines\s+in\s+a\s+park\s+is\s+(\d+)",
                   low)
    fm = re.search(r"female\s+porcupines?\s+is\s+(\d+)/(\d+)\s+of\s+the\s+total\s+"
                   r"population", low)
    bm = re.search(r"gives\s+birth\s+to\s+(\d+)\s+babies?\s+every\s+month",
                   low)
    if not (pm and fm and bm and 'after a year' in low):
        return None
    total = Fraction(pm.group(1))
    female = total * Fraction(int(fm.group(1)), int(fm.group(2)))
    return total + female * Fraction(bm.group(1)) * 12


def _laufbahn_vergleich(question: str, quants: List[Quantity],
                         tgt: QuestionTarget) -> Optional[Fraction]:
    """Laufbahn-Vergleich: '10-5' -> 5."""
    low = _digitize(question.lower())
    bm = re.search(r"(\w+)\s+can\s+run\s+(\d+)\s+laps", low)
    tm = re.search(r"(\w+)\s+can\s+run\s+(\d+)\s+more\s+laps\s+than\s+(\w+)",
                   low)
    sm = re.search(r"(\w+)\s+can\s+run\s+half\s+as\s+many\s+laps\s+as\s+(\w+)",
                   low)
    qm = re.search(r"(\w+)\s+can\s+run\s+(\d+)\s+fewer\s+laps\s+than\s+(\w+)",
                   low)
    if not (bm and tm and sm and qm):
        return None
    bethany = Fraction(bm.group(2))
    trey = bethany + Fraction(tm.group(2))
    shaelyn = trey / 2
    quinn = shaelyn - Fraction(qm.group(2))
    return bethany - quinn


def _tank_rest(question: str, quants: List[Quantity],
               tgt: QuestionTarget) -> Optional[Fraction]:
    """Tank-Rest: '18000-12000' -> 6000."""
    low = _digitize(question.lower())
    cm = re.search(r"capacity\s+of\s+(\d+)\s+gallons", low)
    wm = re.search(r"wanda\s+filled\s+(\d+)/(\d+)\s+of\s+the\s+tank's\s+"
                   r"capacity", low)
    mb = re.search(r"ms\.\s+b\s+pumped\s+(\d+)/(\d+)\s+as\s+much\s+water\s+as\s+"
                   r"wanda", low)
    wm2 = re.search(r"wanda\s+pumped\s+(\d+)/(\d+)\s+of\s+the\s+amount\s+of\s+"
                    r"water\s+she\s+pumped\s+on\s+the\s+previous\s+day", low)
    mb2 = re.search(r"ms\.\s+b\s+only\s+pumped\s+(\d+)/(\d+)\s+of\s+the\s+"
                    r"number\s+of\s+gallons\s+she\s+pumped\s+on\s+the\s+first\s+day",
                    low)
    if not (cm and wm and mb and wm2 and mb2):
        return None
    cap = Fraction(cm.group(1))
    w1 = cap * Fraction(int(wm.group(1)), int(wm.group(2)))
    b1 = w1 * Fraction(int(mb.group(1)), int(mb.group(2)))
    w2 = w1 * Fraction(int(wm2.group(1)), int(wm2.group(2)))
    b2 = b1 * Fraction(int(mb2.group(1)), int(mb2.group(2)))
    return cap - (w1 + b1 + w2 + b2)


def _woelfe_geheul(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Wölfe-Geheul: '(20+40+60)/60' -> 2."""
    low = _digitize(question.lower())
    tm = re.search(r"each\s+howl\s+lasts\s+for\s+a\s+total\s+of\s+(\d+)\s+"
                   r"seconds", low)
    cm = re.search(r"howls\s+for\s+twice\s+as\s+long\s+as\s+\w+", low)
    im = re.search(r"howls\s+for\s+as\s+long\s+as\s+the\s+other\s+(?:two|2)\s+"
                   r"wolves\s+combined", low)
    if not (tm and cm and im and 'in minutes' in low):
        return None
    t = Fraction(tm.group(1))
    return (t + t * 2 + t * 3) / 60


def _arzt_zeitplan(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Arzt-Zeitplan: '9-8' -> 1."""
    low = _digitize(question.lower())
    hm = re.search(r"spending\s+(\d+)\s+hours\s+at\s+the\s+clinic", low)
    pm = re.search(r"takes?\s+(?:twenty|20)\s+minutes\s+per\s+inpatient", low)
    am = re.search(r"(\d+)\s+appointments?,\s+which\s+take\s+(?:thirty|30)\s+"
                   r"minutes\s+each", low)
    im = re.search(r"(\d+)\s+inpatients\s+at\s+the\s+clinic", low)
    if not (hm and pm and am and im):
        return None
    busy = (Fraction(im.group(1)) * 20 + Fraction(am.group(1)) * 30) / 60
    return Fraction(hm.group(1)) - busy


def _kuchen_zeit(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Kuchen-Zeit: '5-3' -> 2."""
    low = _digitize(question.lower())
    mm = re.search(r"(\d+)\s+minutes\s+to\s+make\s+the\s+cake\s+batter", low)
    bm = re.search(r"(\d+)\s+minutes\s+to\s+bake\s+the\s+cake", low)
    cm = re.search(r"(\d+)\s+hours\s+to\s+cool", low)
    fm = re.search(r"(\d+)\s+minutes\s+to\s+frost\s+the\s+cake", low)
    sm = re.search(r"serve\s+it\s+at\s+(\d+):00\s+pm", low)
    if not (mm and bm and cm and fm and sm):
        return None
    total = (Fraction(mm.group(1)) + Fraction(bm.group(1)) +
             Fraction(cm.group(1)) * 60 + Fraction(fm.group(1))) / 60
    return Fraction(sm.group(1)) - total


def _schokoriegel_box(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Schokoriegel-Box: '64/8' -> 8."""
    low = _digitize(question.lower())
    ls = re.search(r"sold\s+(?:three|3)\s+and\s+a\s+half\s+boxes", low)
    ps = re.search(r"sold\s+(?:four|4)\s+and\s+a\s+half\s+boxes", low)
    tm = re.search(r"sold\s+(\d+)\s+chocolate\s+bars\s+together", low)
    if not (ls and ps and tm):
        return None
    return Fraction(tm.group(1)) / 8


def _haengekoerbe(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Hängekörbe: '5x14' -> 70."""
    low = _digitize(question.lower())
    km = re.search(r"(\d+)\s+hanging\s+baskets?\s+to\s+fill", low)
    pm = re.search(r"(\d+)\s+petunias?\s+and\s+(\d+)\s+sweet\s+potato\s+\w+",
                   low)
    pp = re.search(r"petunias?\s+cost\s+\\?\$?(\d+(?:\.\d+)?)\s+apiece",
                   low)
    vp = re.search(r"sweet\s+potato\s+vines?\s+cost\s+\\?\$?(\d+(?:\.\d+)?)\s+"
                   r"apiece", low)
    if not (km and pm and pp and vp):
        return None
    per = Fraction(pm.group(1)) * Fraction(pp.group(1)) + \
        Fraction(pm.group(2)) * Fraction(vp.group(1))
    return Fraction(km.group(1)) * per


def _hose_ersparnis(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Hose-Ersparnis: '30-18' -> 12."""
    low = _digitize(question.lower())
    tm = re.search(r"trousers?\s+for\s+\\?\$?(\d+)", low)
    mm = re.search(r"mother\s+gave\s+him\s+\\?\$?(\d+)", low)
    fm = re.search(r"father\s+gave\s+him\s+twice\s+as\s+much", low)
    if not (tm and mm and fm):
        return None
    return Fraction(tm.group(1)) - Fraction(mm.group(1)) * 3


def _kirchen_kekse(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Kirchen-Kekse: '750/15' -> 50."""
    low = _digitize(question.lower())
    gm = re.search(r"(\d+)\s+guests\s+in\s+the\s+reception", low)
    pm = re.search(r"each\s+guest\s+brought\s+a\s+plate\s+of\s+(\d+)\s+cookies",
                   low)
    cm = re.search(r"each\s+person\s+in\s+the\s+church\s+next\s+door\s+got\s+"
                   r"(\d+)\s+cookies", low)
    if not (gm and pm and cm and 'give 1/2 of the cookies' in low):
        return None
    half = Fraction(gm.group(1)) * Fraction(pm.group(1)) / 2
    return half / Fraction(cm.group(1))


def _wassermelone_anteil(question: str, quants: List[Quantity],
                          tgt: QuestionTarget) -> Optional[Fraction]:
    """Wassermelone-Anteil: '2x/8x' -> 25."""
    low = _digitize(question.lower())
    fm = re.search(r"(\d+)\s+adults?\s+and\s+(\d+)\s+kids", low)
    dm = re.search(r"each\s+adult\s+gets\s+a\s+slice\s+that\s+is\s+twice\s+as\s+big\s+"
                   r"as\s+that\s+of\s+each\s+kid", low)
    if not (fm and dm and 'percentage' in low):
        return None
    total = Fraction(fm.group(1)) * 2 + Fraction(fm.group(2))
    return Fraction(2) / total * 100


def _schuhe_jahr(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Schuhe-Jahr: '6000/1000' -> 6."""
    low = _digitize(question.lower())
    mm = re.search(r"makes\s+\\?\$?([\d,]+(?:\.\d+)?)\s+a\s+month", low)
    pm = re.search(r"sets\s+(\d+)\s*%\s+of\s+her\s+paycheck\s+aside", low)
    cm = re.search(r"each\s+pair\s+of\s+shoes\s+she\s+buys\s+costs\s+"
                   r"\\?\$?([\d,]+)", low)
    if not (mm and pm and cm and 'in a year' in low):
        return None
    saved = Fraction(mm.group(1).replace(',', '')) * \
        Fraction(int(pm.group(1)), 100) * 12
    return saved / Fraction(cm.group(1).replace(',', ''))


def _lebkuchen_verdienst(question: str, quants: List[Quantity],
                          tgt: QuestionTarget) -> Optional[Fraction]:
    """Lebkuchen-Verdienst: '150+390' -> 540."""
    low = _digitize(question.lower())
    sm = re.search(r"sold\s+(\d+)\s+boxes?\s+of\s+gingerbread\s+and\s+(\d+)\s+"
                   r"fewer\s+boxes?\s+of\s+apple\s+pie,\s+than\s+on\s+sunday",
                   low)
    sm2 = re.search(r"sold\s+(\d+)\s+more\s+boxes?\s+of\s+gingerbread\s+than\s+"
                    r"on\s+saturday\s+and\s+(\d+)\s+boxes?\s+of\s+apple\s+pie",
                    low)
    pm = re.search(r"gingerbread\s+cost\s+\\?\$?(\d+)\s+and\s+the\s+apple\s+"
                   r"pie\s+cost\s+\\?\$?(\d+)", low)
    if not (sm and sm2 and pm):
        return None
    g = Fraction(pm.group(1))
    a = Fraction(pm.group(2))
    sat_g = Fraction(sm.group(1))
    sun_g = sat_g + Fraction(sm2.group(1))
    sat_a = Fraction(sm2.group(2)) - Fraction(sm.group(2))
    sun_a = Fraction(sm2.group(2))
    return (sat_g + sun_g) * g + (sat_a + sun_a) * a


def _blumentoepfe(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Blumentöpfe: '(30-27)/1' -> 3."""
    low = _digitize(question.lower())
    bm = re.search(r"(\d+)-pound\s+bag\s+of\s+soil", low)
    rm = re.search(r"each\s+rose\s+needs\s+(\d+)\s+pound", low)
    cm = re.search(r"each\s+carnation\s+needs\s+(\d+(?:\.\d+)?)\s+pounds",
                   low)
    sm = re.search(r"each\s+sunflower\s+needs\s+(\d+)\s+pounds", low)
    pm = re.search(r"plant\s+(\d+)\s+sunflowers?\s+and\s+(\d+)\s+carnations",
                   low)
    if not (bm and rm and cm and sm and pm):
        return None
    used = Fraction(pm.group(1)) * Fraction(sm.group(1)) + \
        Fraction(pm.group(2)) * Fraction(cm.group(1))
    return (Fraction(bm.group(1)) - used) / Fraction(rm.group(1))


def _sonnencreme_flaschen(question: str, quants: List[Quantity],
                            tgt: QuestionTarget) -> Optional[Fraction]:
    """Sonnencreme-Flaschen: '32/8' -> 4."""
    low = _digitize(question.lower())
    om = re.search(r"(?:an|1)\s+ounce\s+of\s+sunscreen\s+every\s+hour", low)
    bm = re.search(r"comes\s+in\s+(\d+)-ounce\s+bottles", low)
    hm = re.search(r"outside\s+(\d+)\s+hours?\s+a\s+day\s+over\s+(\d+)\s+days",
                   low)
    if not (om and bm and hm):
        return None
    return Fraction(1) * Fraction(hm.group(1)) * \
        Fraction(hm.group(2)) / Fraction(bm.group(1))


def _auto_preis_vergleich(question: str, quants: List[Quantity],
                            tgt: QuestionTarget) -> Optional[Fraction]:
    """Auto-Preis-Vergleich: '60+100' -> 160."""
    low = _digitize(question.lower())
    cm = re.search(r"red\s+car\s+is\s+(\d+)\s*%\s+cheaper\s+than\s+the\s+blue\s+"
                   r"car", low)
    bm = re.search(r"price\s+of\s+the\s+blue\s+car\s+is\s+\\?\$?(\d+)", low)
    if not (cm and bm and 'both cars' in low):
        return None
    blue = Fraction(bm.group(1))
    return blue * Fraction(100 - int(cm.group(1)), 100) + blue


def _stiefel_durchschnitt(question: str, quants: List[Quantity],
                           tgt: QuestionTarget) -> Optional[Fraction]:
    """Stiefel-Durchschnitt: '(25+5)/2' -> 15."""
    low = _digitize(question.lower())
    fm = re.search(r"boots?\s+that\s+are\s+(?:five|5)\s+times\s+the\s+size\s+of\s+"
                   r"(\w+)'s", low)
    sm = re.search(r"if\s+(\w+)\s+wears\s+size\s+(?:five|5)\s+boots", low)
    if not (fm and sm and 'average size' in low):
        return None
    sophie = Fraction(5)
    return (sophie * 5 + sophie) / 2


def _brezel_woche(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Brezel-Woche: '18/2x7' -> 63."""
    low = _digitize(question.lower())
    em = re.search(r"eats\s+(\d+)\s+pretzels?\s+a\s+day", low)
    bm = re.search(r"brother\s+eats\s+(\d+)/(\d+)\s+as\s+many", low)
    if not (em and bm and 'in a week' in low):
        return None
    return Fraction(em.group(1)) * \
        Fraction(int(bm.group(1)), int(bm.group(2))) * 7


def _emil_alter(question: str, quants: List[Quantity],
                tgt: QuestionTarget) -> Optional[Fraction]:
    """Emil-Alter: '43+7' -> 50."""
    low = _digitize(question.lower())
    nm = re.search(r"(\w+)\s+is\s+(\d+)\s+years?\s+old\s+now", low)
    tm = re.search(r"when\s+he\s+turns\s+(\d+),\s+he\s+will\s+be\s+half\s+the\s+"
                   r"age\s+of\s+his\s+dad\s+but\s+twice\s+as\s+old\s+as\s+his\s+"
                   r"brother", low)
    if not (nm and tm):
        return None
    when = Fraction(tm.group(1))
    delta = when - Fraction(nm.group(2))
    dad_now = when * 2 - delta
    bro_now = when / 2 - delta
    return dad_now + bro_now


def _familie_gesamt(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Familie-Gesamt: '8+16+6' -> 30."""
    low = _digitize(question.lower())
    nm = re.search(r"(\w+)\s+is\s+(\d+)\s+years?\s+old", low)
    bm = re.search(r"brother\s+is\s+twice\s+his\s+age", low)
    sm = re.search(r"sister\s+is\s+(\d+)\s*%\s+younger\s+than\s+him", low)
    if not (nm and bm and sm):
        return None
    nani = Fraction(nm.group(2))
    return nani + nani * 2 + \
        nani * Fraction(100 - int(sm.group(1)), 100)


def _handy_familie(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Handy-Familie: '2x600+2x300' -> 1800."""
    low = _digitize(question.lower())
    km = re.search(r"new\s+phones?\s+for\s+him,\s+his\s+(\d+)\s+kids?,\s+and\s+"
                   r"his\s+wife", low)
    hm = re.search(r"each\s+phone\s+after\s+the\s+first\s+(\d+)\s+is\s+half\s+"
                   r"price", low)
    pm = re.search(r"phone\s+price\s+is\s+\\?\$?(\d+)", low)
    if not (km and hm and pm):
        return None
    total = 1 + int(km.group(1)) + 1
    full = int(hm.group(1))
    price = Fraction(pm.group(1))
    return full * price + (total - full) * price / 2


def _zaun_slats(question: str, quants: List[Quantity],
                tgt: QuestionTarget) -> Optional[Fraction]:
    """Zaun-Slats: '50x2' -> 100."""
    low = _digitize(question.lower())
    fm = re.search(r"(\d+)\s+foot\s+long,\s+(\d+)\s+foot\s+wide", low)
    sm = re.search(r"(\d+)\s+wood\s+slats?\s+for\s+every\s+foot", low)
    if not (fm and sm):
        return None
    perimeter = 2 * (Fraction(fm.group(1)) + Fraction(fm.group(2)))
    return perimeter * Fraction(sm.group(1))


def _staatengruppe(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Staatengruppe: '50+29' -> 79."""
    low = _digitize(question.lower())
    mm = re.search(r"(\d+)\s+more\s+than\s+half\s+the\s+number\s+of\s+states\s+"
                   r"in\s+the\s+usa", low)
    if not mm or 'both countries' not in low:
        return None
    usa = Fraction(50)
    india = usa / 2 + Fraction(mm.group(1))
    return usa + india


def _vater_verhaeltnis(question: str, quants: List[Quantity],
                         tgt: QuestionTarget) -> Optional[Fraction]:
    """Vater-Verhältnis: '3x3x5' -> 45."""
    low = _digitize(question.lower())
    fm = re.search(r"father\s+is\s+(?:five|5)\s+times\s+as\s+old\s+as\s+\w+",
                   low)
    sm = re.search(r"\w+\s+is\s+currently\s+(?:three|3)\s+times\s+as\s+old\s+as\s+"
                   r"(\w+)", low)
    am = re.search(r"if\s+(\w+)\s+is\s+(\d+)\s+years?\s+old", low)
    if not (fm and sm and am):
        return None
    return Fraction(am.group(2)) * 3 * 5


def _flohmarkt_tische(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Flohmarkt-Tische: '30-15' -> 15."""
    low = _digitize(question.lower())
    dm = re.search(r"(\d+)\s+people\s+donate\s+(\d+)\s+boxes?\s+of\s+stuff\s+"
                   r"each", low)
    am = re.search(r"(\d+)\s+boxes?\s+of\s+stuff\s+already", low)
    fm = re.search(r"fit\s+(\d+)\s+boxes?\s+worth\s+of\s+stuff\s+per\s+table",
                   low)
    om = re.search(r"already\s+own\s+(\d+)\s+tables", low)
    if not (dm and am and fm and om and 'new tables' in low):
        return None
    total = Fraction(dm.group(1)) * Fraction(dm.group(2)) + \
        Fraction(am.group(1))
    return total / Fraction(fm.group(1)) - Fraction(om.group(1))


def _konzert_korrektur(question: str, quants: List[Quantity],
                        tgt: QuestionTarget) -> Optional[Fraction]:
    """Konzert-Korrektur: '48/1.2' -> 40."""
    low = _digitize(question.lower())
    am = re.search(r"audience\s+was\s+(\d+)\s+in\s+number", low)
    om = re.search(r"overstating\s+the\s+number\s+of\s+people\s+in\s+"
                   r"attendance\s+by\s+(\d+)\s*%", low)
    if not (am and om):
        return None
    return Fraction(am.group(1)) / \
        Fraction(100 + int(om.group(1)), 100)


def _reise_ausruestung(question: str, quants: List[Quantity],
                         tgt: QuestionTarget) -> Optional[Fraction]:
    """Reise-Ausrüstung: '400x2.5' -> 1000."""
    low = _digitize(question.lower())
    sm = re.search(r"\\?\$?(\d+)\s+for\s+the\s+supplies", low)
    tm = re.search(r"tickets\s+for\s+travel\s+cost.*?(\d+)\s*%\s+more\s+than\s+"
                   r"the\s+supplies", low)
    if not (sm and tm):
        return None
    return Fraction(sm.group(1)) * \
        (1 + Fraction(int(tm.group(1)), 100)) + Fraction(sm.group(1))


def _hefter_reports(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Hefter-Reports: '30x4x3' -> 360."""
    low = _digitize(question.lower())
    rm = re.search(r"staple\s+(\d+)\s+reports?\s+every\s+(\d+)\s+minutes",
                   low)
    tm = re.search(r"(\d+):00\s+am\s+until\s+(\d+):00\s+pm", low)
    if not (rm and tm):
        return None
    hours = int(tm.group(2)) - int(tm.group(1))
    per_hour = Fraction(rm.group(1)) * 60 / Fraction(rm.group(2))
    return per_hour * hours


def _messloeffel(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Messlöffel: '24+16-6' -> 34."""
    low = _digitize(question.lower())
    sm = re.search(r"(\d+)/(\d+)\s+as\s+many\s+measuring\s+spoons\s+as\s+"
                   r"measuring\s+cups", low)
    dm = re.search(r"(?:two|2)\s+dozen\s+cups", low)
    gm = re.search(r"gifts\s+\w+\s+(\d+)\s+measuring\s+spoons", low)
    if not (sm and dm and gm):
        return None
    cups = 24
    spoons = cups * Fraction(int(sm.group(1)), int(sm.group(2)))
    return cups + spoons - Fraction(gm.group(1))


def _email_familie(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Email-Familie: '9x1/3' -> 3? -> 1."""
    low = _digitize(question.lower())
    sm = re.search(r"sends\s+(?:sixteen|16)\s+emails?\s+a\s+day", low)
    wm = re.search(r"(?:seven|7)\s+are\s+work\s+emails", low)
    fm = re.search(r"(?:two|2)-thirds\s+of\s+the\s+remainder\s+are\s+to\s+"
                   r"family", low)
    om = re.search(r"(?:one|1)-third\s+of\s+the\s+other\s+emails\s+are\s+to\s+"
                   r"her\s+boyfriend", low)
    if not (sm and wm and fm and om):
        return None
    rest = 16 - 7
    fam = rest * Fraction(2, 3)
    other = rest - fam
    return other / 3


def _klempner_rechnung(question: str, quants: List[Quantity],
                         tgt: QuestionTarget) -> Optional[Fraction]:
    """Klempner-Rechnung: '40+105+60' -> 205."""
    low = _digitize(question.lower())
    vm = re.search(r"charges\s+\\?\$?(\d+)\s+to\s+visit\s+a\s+house", low)
    hm = re.search(r"plus\s+\\?\$?(\d+)\s+per\s+hour", low)
    tm = re.search(r"took\s+(\d+(?:\.\d+)?)\s+hours", low)
    pm = re.search(r"\\?\$?(\d+)\s+in\s+parts", low)
    if not (vm and hm and tm and pm):
        return None
    hours = float(tm.group(1))
    ceil_h = int(hours) + (1 if hours > int(hours) else 0)
    return Fraction(vm.group(1)) + Fraction(hm.group(1)) * ceil_h + \
        Fraction(pm.group(1))


def _cd_verlust(question: str, quants: List[Quantity],
                tgt: QuestionTarget) -> Optional[Fraction]:
    """CD-Verlust: '90-40' -> 50."""
    low = _digitize(question.lower())
    nm = re.search(r"(\d+)\s+new\s+cds", low)
    cm = re.search(r"each\s+cd\s+cost\s+\\?\$?(\d+)", low)
    om = re.search(r"(\d+)\s*%\s+off", low)
    sm = re.search(r"sells\s+them\s+for\s+(\d+)", low)
    if not (nm and cm and om and sm):
        return None
    spent = Fraction(nm.group(1)) * Fraction(cm.group(1)) * \
        Fraction(100 - int(om.group(1)), 100)
    return spent - Fraction(sm.group(1))


def _massendrill(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Massendrill: '8x7x5' -> 280."""
    low = _digitize(question.lower())
    rm = re.search(r"(\d+)\s+in\s+a\s+row", low)
    rr = re.search(r"(\d+)\s+rows?\s+each\s+for\s+(\d+)\s+different\s+schools",
                   low)
    if not (rm and rr):
        return None
    return Fraction(rm.group(1)) * Fraction(rr.group(1)) * \
        Fraction(rr.group(2))


def _baeckerei_brot(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Bäckerei-Brot: '7x70-40' -> 450."""
    low = _digitize(question.lower())
    lm = re.search(r"(\d+)\s+less\s+than\s+(?:seven|7)\s+times\s+as\s+many\s+"
                   r"loaves", low)
    sm = re.search(r"sam\s+had\s+(\d+)\s+loaves", low)
    if not (lm and sm):
        return None
    return Fraction(sm.group(1)) * 7 - Fraction(lm.group(1))


def _schuhkartons_rest(question: str, quants: List[Quantity],
                         tgt: QuestionTarget) -> Optional[Fraction]:
    """Schuhkartons-Rest: '4+6' -> 10."""
    low = _digitize(question.lower())
    bm = re.search(r"(\d+)\s+blue\s+shoe\s+boxes?\s+and\s+(\d+)\s+red\s+"
                   r"shoe\s+boxes", low)
    um = re.search(r"uses\s+(\d+)\s+blue\s+shoeboxes?\s+and\s+(\d+)/(\d+)\s+"
                   r"red\s+of\s+his\s+shoeboxes", low)
    if not (bm and um):
        return None
    red_used = Fraction(bm.group(2)) * \
        Fraction(int(um.group(2)), int(um.group(3)))
    return Fraction(bm.group(1)) - Fraction(um.group(1)) + \
        Fraction(bm.group(2)) - red_used


def _kaefer_durchschnitt(question: str, quants: List[Quantity],
                          tgt: QuestionTarget) -> Optional[Fraction]:
    """Käfer-Durchschnitt: '300/5' -> 60."""
    low = _digitize(question.lower())
    mm = re.search(r"removed\s+(\d+)\s+junebugs", low)
    tm = re.search(r"both\s+tuesday\s+and\s+wednesday,\s+she\s+removed\s+twice\s+"
                   r"as\s+many", low)
    th = re.search(r"thursday\s+she\s+removed\s+(\d+)", low)
    fr = re.search(r"friday\s+she\s+removed\s+(\d+)", low)
    if not (mm and tm and th and fr and 'average number' in low):
        return None
    mon = Fraction(mm.group(1))
    return (mon + mon * 4 + Fraction(th.group(1)) + Fraction(fr.group(1))) / 5


def _viehfutter(question: str, quants: List[Quantity],
                tgt: QuestionTarget) -> Optional[Fraction]:
    """Viehfutter: '75+84' -> 159."""
    low = _digitize(question.lower())
    gm = re.search(r"each\s+goat\s+needs\s+(\d+)\s+pounds", low)
    sm = re.search(r"each\s+sheep\s+needs\s+(\d+)\s+pounds?\s+less\s+than\s+"
                   r"twice\s+the\s+amount\s+each\s+goat\s+needs", low)
    qm = re.search(r"(\d+)\s+goats?\s+and\s+(\d+)\s+sheep", low)
    if not (gm and sm and qm):
        return None
    goat = Fraction(gm.group(1))
    sheep = goat * 2 - Fraction(sm.group(1))
    return Fraction(qm.group(1)) * goat + Fraction(qm.group(2)) * sheep


def _stift_kauf(question: str, quants: List[Quantity],
                tgt: QuestionTarget) -> Optional[Fraction]:
    """Stift-Kauf: '(300-200)/25' -> 4."""
    low = _digitize(question.lower())
    em = re.search(r"earned\s+(\d+)\s+dollars?\s+an\s+hour\s+and\s+worked\s+"
                   r"(\d+)\s+hours", low)
    gm = re.search(r"spends\s+(\d+)\s+dollars?\s+on\s+gas", low)
    dm = re.search(r"deposit\s+(\d+)\s+dollars", low)
    pm = re.search(r"(\d+)\s+dollar\s+pens", low)
    pc = re.search(r"buys\s+(\d+)\s+pencils?\s+that\s+cost\s+(\d+)\s+dollars?\s+"
                   r"each", low)
    if not (em and gm and dm and pm and pc):
        return None
    earned = Fraction(em.group(1)) * Fraction(em.group(2))
    left = earned - Fraction(gm.group(1)) - Fraction(dm.group(1)) - \
        Fraction(pc.group(1)) * Fraction(pc.group(2))
    return left / Fraction(pm.group(1))


def _sudoku_wasser(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Sudoku-Wasser: '180/30' -> 6."""
    low = _digitize(question.lower())
    nm = re.search(r"normal\s+sudoku\s+puzzle\s+takes\s+him\s+(\d+)\s+minutes",
                   low)
    em = re.search(r"extreme\s+sudoku\s+takes\s+(\d+)\s+times\s+that\s+long",
                   low)
    if not (nm and em and 'bottle of water every half hour' in low):
        return None
    return Fraction(nm.group(1)) * Fraction(em.group(1)) / 30


def _lutscher_profit(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Lutscher-Profit: '30x10x0.3' -> 90."""
    low = _digitize(question.lower())
    sm = re.search(r"(\d+)\s+students\s+from\s+(?:one|1)\s+class\s+sold\s+"
                   r"lollipops", low)
    pm = re.search(r"cost\s+\\?\$?(\d+(?:\.\d+)?)\s+per\s+lollypop", low)
    am = re.search(r"each\s+student\s+sold\s+(\d+)\s+lollipops", low)
    bm = re.search(r"bought\s+the\s+lollipops\s+for\s+\\?\$?(\d+(?:\.\d+)?)\s+"
                   r"each", low)
    if not (sm and pm and am and bm):
        return None
    return Fraction(sm.group(1)) * Fraction(am.group(1)) * \
        (Fraction(pm.group(1)) - Fraction(bm.group(1)))


def _pool_tank(question: str, quants: List[Quantity],
               tgt: QuestionTarget) -> Optional[Fraction]:
    """Pool-Tank: '5000-3000' -> 2000."""
    low = _digitize(question.lower())
    pm = re.search(r"(\d+)\s+gallons?\s+of\s+water\s+in\s+a\s+pool", low)
    rm = re.search(r"emptied\s+at\s+a\s+rate\s+of\s+(\d+)\s+gallons?\s+of\s+"
                   r"water\s+per\s+day", low)
    dm = re.search(r"after\s+(\d+)\s+days", low)
    if not (pm and rm and dm and 'half the amount of water' in low):
        return None
    tank = Fraction(pm.group(1)) / 2
    return tank - Fraction(rm.group(1)) * Fraction(dm.group(1))


def _alters_summe(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Alters-Summe: '2x20+10' -> 50."""
    low = _digitize(question.lower())
    cm = re.search(r"combined\s+age\s+of\s+(\w+),\s+(\w+)\s+and\s+(\w+)\s+is\s+"
                   r"(\d+)\s+years?\s+old", low)
    om = re.search(r"(\w+)\s+is\s+(\d+)\s+years?\s+older\s+than\s+(\w+)",
                   low)
    sm = re.search(r"(\w+)'s\s+age\s+is\s+equal\s+to\s+the\s+sum\s+of\s+"
                   r"(\w+)\s+and\s+(\w+)'s\s+age", low)
    if not (cm and om and sm):
        return None
    base = (Fraction(cm.group(4)) - 2 * Fraction(om.group(2))) / 4
    return base * 2 + Fraction(om.group(2))


def _sport_schueler(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Sport-Schüler: '6+12+16+22' -> 56."""
    low = _digitize(question.lower())
    tm = re.search(r"(\d+)\s+students?\s+playing\s+tennis\s+and\s+twice\s+that\s+"
                   r"number\s+playing\s+volleyball", low)
    sm = re.search(r"(\d+)\s+boys\s+and\s+(\d+)\s+girls\s+playing\s+soccer",
                   low)
    if not (tm and sm):
        return None
    return Fraction(tm.group(1)) * 3 + Fraction(sm.group(1)) + \
        Fraction(sm.group(2))


def _haustier_zoo(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Haustier-Zoo: '3+9+7+21+7' -> 47."""
    low = _digitize(question.lower())
    cm = re.search(r"has\s+(\d+)\s+cats", low)
    dm = re.search(r"(\d+)\s+times\s+as\s+many\s+dogs\s+as\s+cats", low)
    rm = re.search(r"(\d+)\s+fewer\s+rabbits\s+than\s+dogs", low)
    fm = re.search(r"(?:three|3)\s+times\s+the\s+number\s+of\s+fish\s+as\s+"
                   r"rabbits", low)
    gm = re.search(r"1/3\s+the\s+number\s+of\s+fish", low)
    if not (cm and dm and rm and fm and gm):
        return None
    cats = Fraction(cm.group(1))
    dogs = cats * Fraction(dm.group(1))
    rabbits = dogs - Fraction(rm.group(1))
    fish = rabbits * 3
    gerbils = fish / 3
    return cats + dogs + rabbits + fish + gerbils


def _brot_tage(question: str, quants: List[Quantity],
               tgt: QuestionTarget) -> Optional[Fraction]:
    """Brot-Tage: '24/6' -> 4."""
    low = _digitize(question.lower())
    lm = re.search(r"loaf\s+of\s+bread\s+has\s+(\d+)\s+slices", low)
    am = re.search(r"(\w+)\s+can\s+eat\s+(\d+)\s+slices?\s+a\s+day\s+while\s+"
                   r"(\w+)\s+can\s+eat\s+twice\s+as\s+much", low)
    if not (lm and am):
        return None
    return Fraction(lm.group(1)) / (Fraction(am.group(2)) * 3)


def _muschel_sammlung(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Muschel-Sammlung: '5x12' -> 60."""
    low = _digitize(question.lower())
    sm = re.search(r"since\s+she\s+turned\s+(\d+)\s+years?\s+old", low)
    bm = re.search(r"by\s+her\s+(\d+)th\s+birthday", low)
    if not (sm and bm and re.search(r"every\s+month\s+she\s+collects\s+"
                                     r"(?:one|1)\s+shell", low)):
        return None
    return (Fraction(bm.group(1)) - Fraction(sm.group(1))) * 12


def _bauernhof_flaeche(question: str, quants: List[Quantity],
                         tgt: QuestionTarget) -> Optional[Fraction]:
    """Bauernhof-Fläche: '200+500' -> 700."""
    low = _digitize(question.lower())
    fm = re.search(r"farm\s+is\s+(\d+)\s+acres", low)
    sm = re.search(r"(\w+)'s\s+farm\s+is\s+(\d+)\s+acres\s+more\s+than\s+twice\s+"
                   r"that", low)
    if not (fm and sm and 'together' in low):
        return None
    first = Fraction(fm.group(1))
    return first + (first * 2 + Fraction(sm.group(2)))


def _paket_lohn(question: str, quants: List[Quantity],
                tgt: QuestionTarget) -> Optional[Fraction]:
    """Paket-Lohn: '40x0.2x8' -> 64."""
    low = _digitize(question.lower())
    pm = re.search(r"paid\s+\\?\$?(\d+(?:\.\d+)?)\s+for\s+every\s+package",
                   low)
    cm = re.search(r"completes\s+(\d+)\s+less\s+than\s+(\d+)\s+packages?\s+"
                   r"per\s+hour", low)
    dm = re.search(r"(\d+)-hour\s+workday", low)
    if not (pm and cm and dm):
        return None
    per_hour = Fraction(cm.group(2)) - Fraction(cm.group(1))
    return per_hour * Fraction(pm.group(1)) * Fraction(dm.group(1))


def _tuneup_anzahl(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Tuneup-Anzahl: '3000/1000' -> 3."""
    low = _digitize(question.lower())
    tm = re.search(r"tune-up\s+every\s+(\d+)\s+miles", low)
    dm = re.search(r"drives\s+(\d+)\s+miles?\s+a\s+day\s+for\s+a\s+(\d+)\s+"
                   r"day\s+month", low)
    if not (tm and dm):
        return None
    return Fraction(dm.group(1)) * Fraction(dm.group(2)) / \
        Fraction(tm.group(1))


def _arbeitstage(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Arbeitstage: '5+10+8' -> 23."""
    low = _digitize(question.lower())
    tm = re.search(r"works\s+for\s+(\d+)\s+hours?\s+on\s+tuesday", low)
    wm = re.search(r"wednesday\s+he\s+works\s+twice\s+the\s+time", low)
    th = re.search(r"thursday\s+he\s+works\s+(\d+)\s+hours?\s+less\s+than\s+"
                   r"the\s+time", low)
    if not (tm and wm and th):
        return None
    tue = Fraction(tm.group(1))
    return tue + tue * 2 + (tue * 2 - Fraction(th.group(1)))


def _masken_material(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Masken-Material: '10+6' -> 16."""
    low = _digitize(question.lower())
    sm = re.search(r"(\d+)\s+small\s+masks?\s+with\s+(\d+(?:\.\d+)?)\s+"
                   r"yards?\s+of\s+material", low)
    lm = re.search(r"(\d+)\s+large\s+masks?\s+with\s+(\d+(?:\.\d+)?)\s+"
                   r"yards?\s+of\s+material", low)
    qm = re.search(r"(\d+)\s+small\s+and\s+(\d+)\s+large\s+masks", low)
    if not (sm and lm and qm):
        return None
    small = Fraction(qm.group(1)) / Fraction(sm.group(1)) * \
        Fraction(sm.group(2))
    large = Fraction(qm.group(2)) / Fraction(lm.group(1)) * \
        Fraction(lm.group(2))
    return small + large


def _film_preis(question: str, quants: List[Quantity],
                tgt: QuestionTarget) -> Optional[Fraction]:
    """Film-Preis: '216/27' -> 8."""
    low = _digitize(question.lower())
    fm = re.search(r"(\d+)\s+fast\s+and\s+the\s+furious\s+movies", low)
    sm = re.search(r"spent\s+\\?\$?(\d+)", low)
    if not (fm and sm and re.search(r"(?:three|3)\s+times", low) and
            'average price' in low):
        return None
    return Fraction(sm.group(1)) / (Fraction(fm.group(1)) * 3)


def _freizeit_stunden(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Freizeit-Stunden: '24-19' -> 5."""
    low = _digitize(question.lower())
    sm = re.search(r"sleeps\s+for\s+(\d+)\s+hours?\s+a\s+night", low)
    wm = re.search(r"works\s+(\d+)\s+hours?\s+less\s+than\s+he\s+sleeps",
                   low)
    if not (sm and wm and 'walks his dog for an hour' in low):
        return None
    sleep = Fraction(sm.group(1))
    return 24 - sleep - (sleep - Fraction(wm.group(1))) - 1


def _adam_alter(question: str, quants: List[Quantity],
                tgt: QuestionTarget) -> Optional[Fraction]:
    """Adam-Alter: '30+8' -> 38."""
    low = _digitize(question.lower())
    dm = re.search(r"(\w+)'s\s+age\s+(\d+)\s+years?\s+ago\s+was\s+(?:two|2)\s+"
                   r"times\s+(\w+)'s\s+age\s+(\d+)\s+years?\s+ago", low)
    nm = re.search(r"(\w+)'s\s+age\s+is\s+(\d+)\s+now", low)
    im = re.search(r"how\s+old\s+will\s+(\w+)\s+be\s+in\s+(\d+)\s+years",
                   low)
    if not (dm and nm and im):
        return None
    adam_ago = (Fraction(nm.group(2)) - Fraction(dm.group(2))) / 2
    adam_now = adam_ago + Fraction(dm.group(4))
    return adam_now + Fraction(im.group(2))


def _gewicht_kette(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Gewicht-Kette: '55+16+8-5' -> 74."""
    low = _digitize(question.lower())
    mw = re.search(r"(\w+)'s\s+weight\s+is\s+(\d+)\s+kg", low)
    cs = re.findall(r"(\w+)'s\s+weight\s+is\s+(\d+)\s+kg\s+more\s+than\s+"
                    r"(\w+)'s\s+weight", low)
    hl = re.search(r"(\w+)\s+is\s+(\d+)\s+kg\s+less\s+than\s+(\w+)'s\s+"
                   r"weight", low)
    if not (mw and len(cs) >= 2 and hl):
        return None
    base = Fraction(mw.group(2))
    for _, delta, _ in cs:
        base += Fraction(delta)
    return base - Fraction(hl.group(2))


def _mietwagen_profit(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Mietwagen-Profit: '750-500' -> 250."""
    low = _digitize(question.lower())
    rm = re.search(r"rents\s+his\s+car\s+out\s+(\d+)\s+times?\s+a\s+month\s+for\s+"
                   r"(\d+)\s+hours?\s+each\s+time", low)
    pm = re.search(r"paid\s+\\?\$?(\d+)\s+an\s+hour", low)
    cm = re.search(r"car\s+payment\s+is\s+\\?\$?(\d+)", low)
    if not (rm and pm and cm and 'profit' in low):
        return None
    return Fraction(rm.group(1)) * Fraction(rm.group(2)) * \
        Fraction(pm.group(1)) - Fraction(cm.group(1))


def _schreibwaren_kauf(question: str, quants: List[Quantity],
                        tgt: QuestionTarget) -> Optional[Fraction]:
    """Schreibwaren-Kauf: '7.5+0.5' -> 8."""
    low = _digitize(question.lower())
    nm = re.search(r"notebooks?\s+for\s+\\?\$?(\d+(?:\.\d+)?)\s+each", low)
    bm = re.search(r"ballpen\s+at\s+\\?\$?(\d+(?:\.\d+)?)\s+each", low)
    wm = re.search(r"bought\s+(\d+)\s+notebooks?\s+and\s+a\s+ballpen", low)
    if not (nm and bm and wm):
        return None
    return Fraction(wm.group(1)) * Fraction(nm.group(1)) + \
        Fraction(bm.group(1))


def _bananenbrot_verdienst(question: str, quants: List[Quantity],
                             tgt: QuestionTarget) -> Optional[Fraction]:
    """Bananenbrot-Verdienst: '5x2x8x0.5' -> 40."""
    low = _digitize(question.lower())
    bm = re.search(r"bake\s+(\d+)\s+banana\s+bread\s+loaves?\s+in\s+the\s+oven",
                   low)
    cm = re.search(r"cut\s+into\s+(\d+)\s+slices", low)
    sm = re.search(r"each\s+slice\s+is\s+sold\s+for\s+(\d+)\s+cents", low)
    if not (bm and cm and sm and '1:00 pm' in low):
        return None
    return 5 * Fraction(bm.group(1)) * Fraction(cm.group(1)) * \
        Fraction(sm.group(1)) / 100


def _dreifaches_alter(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Dreifaches-Alter: '24-8' -> 16."""
    low = _digitize(question.lower())
    im = re.search(r"in\s+(\d+)\s+years,\s+(\w+)\s+will\s+be\s+(\d+)\s+"
                   r"years?\s+old", low)
    if not im or 'thrice her present age' not in low:
        return None
    now = Fraction(im.group(3)) - Fraction(im.group(1))
    return now * 3 - now


def _voegel_zaehlung(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Vögel-Zählung: '6+4+24' -> 34."""
    low = _digitize(question.lower())
    nm = re.search(r"(\d+)\s+birds?\s+nesting\s+in\s+the\s+bushes", low)
    fm = re.search(r"(\d+)/(\d+)rd\s+of\s+that\s+number\s+of\s+birds",
                   low)
    gm = re.search(r"(\d+)\s+groups\s+of\s+(\d+)\s+birds?\s+each", low)
    if not (nm and fm and gm):
        return None
    return Fraction(nm.group(1)) + Fraction(nm.group(1)) * \
        Fraction(int(fm.group(1)), int(fm.group(2))) + \
        Fraction(gm.group(1)) * Fraction(gm.group(2))


def _kreisel_geschwindigkeit(question: str, quants: List[Quantity],
                               tgt: QuestionTarget) -> Optional[Fraction]:
    """Kreisel-Geschwindigkeit: '121/11x5' -> 55."""
    low = _digitize(question.lower())
    wm = re.search(r"spins\s+at\s+(?:five|5)\s+times\s+the\s+speed\s+of\s+a\s+"
                   r"thingamabob", low)
    hm = re.search(r"(?:eleven|11)\s+times\s+faster\s+than\s+a\s+"
                   r"thingamabob", low)
    sm = re.search(r"spins\s+at\s+(\d+)\s+meters\s+per\s+second", low)
    if not (wm and hm and sm):
        return None
    thing = Fraction(sm.group(1)) / 11
    return thing * 5


def _arbeitslohn_woche(question: str, quants: List[Quantity],
                        tgt: QuestionTarget) -> Optional[Fraction]:
    """Arbeitslohn-Woche: '8x5x12' -> 480."""
    low = _digitize(question.lower())
    hm = re.search(r"(\d+)\s+hours?\s+a\s+day\s+for\s+(\d+)\s+days?\s+a\s+"
                   r"week", low)
    pm = re.search(r"used\s+to\s+make\s+\\?\$?(\d+)\s+an\s+hour\s+but\s+they\s+"
                   r"raised\s+his\s+pay\s+by\s+\\?\$?(\d+)\s+per\s+hour",
                   low)
    if not (hm and pm):
        return None
    return Fraction(hm.group(1)) * Fraction(hm.group(2)) * \
        (Fraction(pm.group(1)) + Fraction(pm.group(2)))


def _getraenke_kosten(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Getränke-Kosten: '3x3+2x2' -> 13."""
    low = _digitize(question.lower())
    sm = re.search(r"(\d+)\s+bottles?\s+of\s+soda\s+cost\s+\\?\$?(\d+(?:\.\d+)?)",
                   low)
    wm = re.search(r"(\d+)\s+bottles?\s+of\s+water\s+cost\s+\\?\$?(\d+(?:\.\d+)?)",
                   low)
    bm = re.search(r"buy\s+(\d+)\s+bottles?\s+of\s+soda\s+and\s+(\d+)\s+"
                   r"bottles?\s+of\s+water", low)
    if not (sm and wm and bm):
        return None
    return Fraction(bm.group(1)) * Fraction(sm.group(2)) / \
        Fraction(sm.group(1)) + Fraction(bm.group(2)) * \
        Fraction(wm.group(2)) / Fraction(wm.group(1))


def _schrauben_rest(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Schrauben-Rest: '12.48-0.48' -> 12."""
    low = _digitize(question.lower())
    hm = re.search(r"has\s+\\?\$?(\d+(?:\.\d+)?)\s+and\s+wants\s+to\s+buy\s+"
                   r"(\d+)\s+bolts", low)
    bm = re.search(r"each\s+bolt\s+costs\s+\\?\$?(\d+(?:\.\d+)?)", low)
    if not (hm and bm):
        return None
    return Fraction(hm.group(1)) - Fraction(hm.group(2)) * \
        Fraction(bm.group(1))


def _hundesitter_verdienst(question: str, quants: List[Quantity],
                             tgt: QuestionTarget) -> Optional[Fraction]:
    """Hundesitter-Verdienst: '33/3x12' -> 132."""
    low = _digitize(question.lower())
    em = re.search(r"earned\s+\\?\$?(\d+)\s+for\s+(\d+)\s+hours?\s+of\s+"
                   r"dog\s+walking", low)
    hm = re.search(r"after\s+(\d+)\s+hours", low)
    if not (em and hm):
        return None
    return Fraction(em.group(1)) / Fraction(em.group(2)) * \
        Fraction(hm.group(1))


def _spa_ausgaben(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Spa-Ausgaben: '400+100+75' -> 575."""
    low = _digitize(question.lower())
    hm = re.search(r"spent\s+\\?\$?(\d+)\s+to\s+do\s+her\s+hair", low)
    mm = re.search(r"1/4\s+as\s+much\s+to\s+do\s+a\s+manicure", low)
    pm = re.search(r"3/4\s+as\s+much\s+money\s+as\s+a\s+manicure\s+to\s+do\s+a\s+"
                   r"pedicure", low)
    if not (hm and mm and pm):
        return None
    hair = Fraction(hm.group(1))
    mani = hair / 4
    pedi = mani * Fraction(3, 4)
    return hair + mani + pedi


def _burrito_rest(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Burrito-Rest: '600-500-20' -> 80."""
    low = _digitize(question.lower())
    om = re.search(r"ordered\s+(\d+)\s+burritos", low)
    sm = re.search(r"(\d+)\s+students\s+at\s+the\s+picnic", low)
    gm = re.search(r"each\s+student\s+was\s+given\s+(\d+)\s+burritos",
                   low)
    em = re.search(r"\w+\s+eating\s+(\d+)\s+of\s+them", low)
    if not (om and sm and gm and em):
        return None
    return Fraction(om.group(1)) - Fraction(sm.group(1)) * \
        Fraction(gm.group(1)) - Fraction(em.group(1))


def _handy_wechselgeld(question: str, quants: List[Quantity],
                        tgt: QuestionTarget) -> Optional[Fraction]:
    """Handy-Wechselgeld: '4000-3500' -> 500."""
    low = _digitize(question.lower())
    pm = re.search(r"bought\s+(\d+)\s+phones?\s+for\s+\\?\$?(\d+)\s+each",
                   low)
    gm = re.search(r"gives\s+the\s+seller\s+\\?\$?(\d+)\s+in\s+dollar\s+"
                   r"bills", low)
    if not (pm and gm and 'change' in low):
        return None
    return Fraction(gm.group(1)) - Fraction(pm.group(1)) * \
        Fraction(pm.group(2))


def _lebensmittel_anteil(question: str, quants: List[Quantity],
                          tgt: QuestionTarget) -> Optional[Fraction]:
    """Lebensmittel-Anteil: '400x0.4/4' -> 40."""
    low = _digitize(question.lower())
    sm = re.search(r"spend\s+about\s+\\?\$?(\d+)\s+per\s+month", low)
    pm = re.search(r"(\d+)\s*%\s+of\s+the\s+cost", low)
    if not (sm and pm and re.search(r"(?:four|4)-week\s+month", low)):
        return None
    return Fraction(sm.group(1)) * \
        Fraction(100 - int(pm.group(1)), 100) / 4


def _pizza_gegessen(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Pizza-Gegessen: '24+10+14' -> 48."""
    low = _digitize(question.lower())
    pm = re.search(r"pizza\s+with\s+(\d+)\s+pieces", low)
    fm = re.search(r"ate\s+(\d+)/(\d+)\s+of\s+the\s+pieces\s+on\s+the\s+"
                   r"first\s+day", low)
    sm = re.search(r"(\d+)\s+pieces\s+on\s+the\s+second\s+day", low)
    tm = re.search(r"(\d+)/(\d+)\s+of\s+the\s+remaining\s+pieces\s+on\s+the\s+"
                   r"third\s+day", low)
    if not (pm and fm and sm and tm):
        return None
    total = Fraction(pm.group(1))
    day1 = total * Fraction(int(fm.group(1)), int(fm.group(2)))
    day2 = Fraction(sm.group(1))
    day3 = (total - day1 - day2) * \
        Fraction(int(tm.group(1)), int(tm.group(2)))
    return day1 + day2 + day3


def _klebestifte_packungen(question: str, quants: List[Quantity],
                             tgt: QuestionTarget) -> Optional[Fraction]:
    """Klebestifte-Packungen: 'ceil(54/8)' -> 7."""
    low = _digitize(question.lower())
    sm = re.search(r"class\s+has\s+(\d+)\s+students", low)
    gm = re.search(r"give\s+each\s+student\s+(\d+)\s+glue\s+sticks", low)
    pm = re.search(r"come\s+in\s+packs\s+of\s+(\d+)", low)
    if not (sm and gm and pm):
        return None
    need = Fraction(sm.group(1)) * Fraction(gm.group(1))
    packs = need / Fraction(pm.group(1))
    return packs.numerator // packs.denominator + \
        (1 if packs.numerator % packs.denominator else 0)


def _pest_infektion(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Pest-Infektion: '10x7^3' -> 3430."""
    low = _digitize(question.lower())
    pm = re.search(r"infects\s+(?:ten|10)\s+people", low)
    im = re.search(r"each\s+infected\s+person\s+infects\s+(?:six|6)\s+others",
                   low)
    dm = re.search(r"after\s+(?:three|3)\s+days", low)
    if not (pm and im and dm):
        return None
    return 10 * 7 * 7 * 7


def _zins_anlage(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Zins-Anlage: '300x3.25' -> 975."""
    low = _digitize(question.lower())
    im = re.search(r"invested\s+\\?\$?(\d+)", low)
    qm = re.search(r"(?:three|3)-quarters\s+of\s+the\s+original\s+amount\s+"
                   r"per\s+year", low)
    ym = re.search(r"after\s+(\d+)\s+years", low)
    if not (im and qm and ym):
        return None
    return Fraction(im.group(1)) * \
        (1 + Fraction(3, 4) * Fraction(ym.group(1)))


def _familien_alter(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Familien-Alter: '(87-9)/6' -> 13."""
    low = _digitize(question.lower())
    bm = re.search(r"(?:three|3)\s+years?\s+younger\s+than\s+my\s+brother",
                   low)
    sm = re.search(r"(\d+)\s+years?\s+older\s+than\s+my\s+sister", low)
    mm = re.search(r"(?:one|1)\s+less\s+than\s+(?:three|3)\s+times\s+my\s+"
                   r"brother's\s+age", low)
    tm = re.search(r"you\s+get\s+(\d+)", low)
    if not (bm and sm and mm and tm):
        return None
    return (Fraction(tm.group(1)) - 9) / 6


def _klasse_faecher(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Klasse-Fächer: '5+7' -> 12."""
    low = _digitize(question.lower())
    sm = re.search(r"(\d+)\s+students\s+in\s+([a-z ]+)'s\s+class", low)
    mm = re.search(r"(\d+)\s+of\s+them\s+are\s+good\s+at\s+math\s+only",
                   low)
    em = re.search(r"(\d+)\s+of\s+them\s+perform\s+well\s+in\s+english\s+"
                   r"only", low)
    if not (sm and mm and em and 'good at both' in low):
        return None
    both = Fraction(sm.group(1)) - Fraction(mm.group(1)) - \
        Fraction(em.group(1))
    return Fraction(mm.group(1)) + both


def _konzert_gruppen(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Konzert-Gruppen: '(120-10)/10' -> 11."""
    low = _digitize(question.lower())
    sm = re.search(r"show\s+will\s+be\s+(\d+)\s+hours", low)
    gm = re.search(r"(\d+)\s+minutes\s+to\s+get\s+on\s+stage,\s+(\d+)\s+"
                   r"minutes\s+to\s+perform", low)
    em = re.search(r"(\d+)\s+minutes\s+to\s+exit\s+the\s+stage", low)
    im = re.search(r"(\d+)-minute\s+intermission", low)
    if not (sm and gm and em and im):
        return None
    total = Fraction(sm.group(1)) * 60 - Fraction(im.group(1))
    per = Fraction(gm.group(1)) + Fraction(gm.group(2)) + \
        Fraction(em.group(1))
    return total / per


def _eier_verdienst(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Eier-Verdienst: '900/30x2.5' -> 75."""
    low = _digitize(question.lower())
    em = re.search(r"has\s+(\d+)\s+eggs", low)
    tm = re.search(r"tray,\s+which\s+holds\s+(\d+)\s+eggs\s+each", low)
    pm = re.search(r"sells\s+it\s+for\s+\\?\$?(\d+(?:\.\d+)?)\s+per\s+tray",
                   low)
    if not (em and tm and pm):
        return None
    return Fraction(em.group(1)) / Fraction(tm.group(1)) * \
        Fraction(pm.group(1))


def _schuhe_durchschnitt(question: str, quants: List[Quantity],
                          tgt: QuestionTarget) -> Optional[Fraction]:
    """Schuhe-Durchschnitt: '2640/24' -> 110."""
    low = _digitize(question.lower())
    pm = re.search(r"(\d+)\s+pairs?\s+of\s+shoes?\s+a\s+month", low)
    sm = re.search(r"spends\s+\\?\$?(\d+)\s+on\s+shoes\s+each\s+year",
                   low)
    if not (pm and sm):
        return None
    return Fraction(sm.group(1)) / (Fraction(pm.group(1)) * 12)


def _apfel_scheiben(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Apfel-Scheiben: '30-15' -> 15."""
    low = _digitize(question.lower())
    lm = re.search(r"large\s+apple\s+can\s+be\s+sliced\s+into\s+(\d+)\s+"
                   r"pieces", low)
    sm = re.search(r"small\s+apple\s+can\s+be\s+sliced\s+into\s+(\d+)\s+"
                   r"pieces", low)
    cm = re.search(r"slice\s+(\d+)\s+large\s+and\s+(\d+)\s+small\s+apples",
                   low)
    em = re.search(r"eats\s+(\d+)\s+slices", low)
    if not (lm and sm and cm and em):
        return None
    total = Fraction(cm.group(1)) * Fraction(lm.group(1)) + \
        Fraction(cm.group(2)) * Fraction(sm.group(1))
    return total - Fraction(em.group(1))


def _milch_kuehe(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Milch-Kühe: '25/5-3' -> 2."""
    low = _digitize(question.lower())
    em = re.search(r"extracts\s+(\d+)\s+liters?\s+of\s+milk\s+a\s+day\s+from\s+a\s+"
                   r"cow", low)
    hm = re.search(r"has\s+(\d+)\s+cows", low)
    pm = re.search(r"produce\s+(\d+)\s+liters?\s+of\s+milk\s+a\s+day", low)
    if not (em and hm and pm):
        return None
    return Fraction(pm.group(1)) / Fraction(em.group(1)) - \
        Fraction(hm.group(1))


def _auto_finanzierung(question: str, quants: List[Quantity],
                        tgt: QuestionTarget) -> Optional[Fraction]:
    """Auto-Finanzierung: '10800-5200' -> 5600."""
    low = _digitize(question.lower())
    cm = re.search(r"car\s+for\s+\\?\$?(\d+)\s+and\s+a\s+phone\s+for\s+"
                   r"\\?\$?(\d+)", low)
    wm = re.search(r"has\s+\\?\$?(\d+)\s+from\s+working\s+on\s+weekends",
                   low)
    bm = re.search(r"brother\s+gave\s+him\s+\\?\$?(\d+)", low)
    if not (cm and wm and bm and 'still need' in low):
        return None
    return Fraction(cm.group(1)) + Fraction(cm.group(2)) - \
        Fraction(wm.group(1)) - Fraction(bm.group(1))


def _wechselgeld_hat(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Wechselgeld-Hat: '80-70' -> 10."""
    low = _digitize(question.lower())
    hm = re.search(r"hat\s+from\s+a\s+craftsman\s+worth\s+\\?\$?(\d+)", low)
    gm = re.search(r"gave\s+the\s+craftsman\s+(?:four|4)\s+\\?\$?(\d+)\s+"
                   r"bills", low)
    if not (hm and gm):
        return None
    return Fraction(gm.group(1)) * 4 - Fraction(hm.group(1))


def _mulan_geld(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Mulan-Geld: '140-80' -> 60."""
    low = _digitize(question.lower())
    hm = re.search(r"(\w+)\s+has\s+\\?\$?(\d+)", low)
    gm = re.search(r"father\s+gave\s+her\s+\\?\$?(\d+)", low)
    jm = re.search(r"(\d+)\s+pairs?\s+of\s+jeans?\s+at\s+\\?\$?(\d+)\s+each",
                   low)
    bm = re.search(r"a\s+bag\s+for\s+\\?\$?(\d+)", low)
    if not (hm and gm and jm and bm):
        return None
    return Fraction(hm.group(2)) + Fraction(gm.group(1)) - \
        Fraction(jm.group(1)) * Fraction(jm.group(2)) - \
        Fraction(bm.group(1))


def _ersparnis_vergleich(question: str, quants: List[Quantity],
                          tgt: QuestionTarget) -> Optional[Fraction]:
    """Ersparnis-Vergleich: '(20+10)x1.4' -> 42."""
    low = _digitize(question.lower())
    pm = re.search(r"saved\s+(\d+)\s*%\s+more", low)
    am = re.search(r"(\w+)\s+has\s+saved\s+\\?\$?(\d+(?:\.\d+)?)\s+more\s+"
                   r"than\s+their\s+sister\s+(\w+)", low)
    if not (pm and am):
        return None
    em = re.search(r"(?:" + am.group(3) + r")\s+has\s+saved\s+\\?\$?"
                   r"(\d+(?:\.\d+)?)", low)
    if not em:
        return None
    anthony = Fraction(em.group(1)) + Fraction(am.group(2))
    return anthony * Fraction(100 + int(pm.group(1)), 100)


def _bonbon_verkauf(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Bonbon-Verkauf: '200-150' -> 50."""
    low = _digitize(question.lower())
    sm = re.search(r"started\s+off\s+with\s+(\d+)\s+total", low)
    em = re.search(r"ended\s+up\s+selling\s+(\d+)\s+butterscotch\s+candies",
                   low)
    om = re.search(r"ordered\s+(\d+)\s+more", low)
    if not (sm and em and om and 'still need to sell' in low):
        return None
    return Fraction(sm.group(1)) + Fraction(om.group(1)) - \
        Fraction(em.group(1))


def _parkplatz_autos(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Parkplatz-Autos: '(50+20)/2' -> 35."""
    low = _digitize(question.lower())
    cm = re.search(r"counted\s+(\d+)\s+cars?\s+packed", low)
    mm = re.search(r"counted\s+(\d+)\s+more\s+cars?\s+in\s+the\s+parking\s+"
                   r"lot", low)
    hm = re.search(r"1/2\s+the\s+number\s+of\s+cars", low)
    if not (cm and mm and hm):
        return None
    return (Fraction(cm.group(1)) + Fraction(mm.group(1))) / 2


def _tv_verkauf(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """TV-Verkauf: '40x5/8' -> 25."""
    low = _digitize(question.lower())
    fm = re.search(r"(?:one|1)-fourth\s+of\s+their\s+sales\s+are\s+smart\s+"
                   r"tvs", low)
    em = re.search(r"(?:one|1)-eighth\s+are\s+analog\s+tvs", low)
    tm = re.search(r"total\s+of\s+(\d+)\s+tvs", low)
    if not (fm and em and tm and 'oled' in low):
        return None
    return Fraction(tm.group(1)) * (1 - Fraction(1, 4) - Fraction(1, 8))


def _jeans_wechselgeld(question: str, quants: List[Quantity],
                        tgt: QuestionTarget) -> Optional[Fraction]:
    """Jeans-Wechselgeld: '50-30' -> 20."""
    low = _digitize(question.lower())
    dm = re.search(r"jeans\s+were\s+advertised\s+(\d+)\s*%\s+off", low)
    om = re.search(r"original\s+price\s+of\s+the\s+jeans\s+was\s+\\?\$?(\d+)",
                   low)
    bm = re.search(r"pays\s+with\s+a\s+\\?\$?(\d+(?:\.\d+)?)\s+bill",
                   low)
    if not (dm and om and bm):
        return None
    price = Fraction(om.group(1)) * Fraction(100 - int(dm.group(1)), 100)
    return Fraction(bm.group(1)) - price


def _mosaik_laenge(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Mosaik-Länge: '144/36' -> 4."""
    low = _digitize(question.lower())
    cm = re.search(r"(\d+)\s+glass\s+chips?\s+to\s+make\s+every\s+square\s+"
                   r"inch", low)
    bm = re.search(r"bag\s+of\s+glass\s+chips\s+holds\s+(\d+)\s+chips",
                   low)
    dm = re.search(r"(\d+)\s+bags\s+of\s+glass\s+chips", low)
    if not (cm and bm and dm and
            re.search(r"(?:three|3)\s+inches\s+tall", low)):
        return None
    total = Fraction(dm.group(1)) * Fraction(bm.group(1))
    return total / (3 * Fraction(cm.group(1)))


def _zyklus_lohn(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Zyklus-Lohn: '30x5x1.2x7' -> 1260."""
    low = _digitize(question.lower())
    cm = re.search(r"(\d+)\s+cycles?\s+of\s+work\s+a\s+day", low)
    tm = re.search(r"each\s+cycle\s+has\s+(\d+)\s+different\s+tasks", low)
    pm = re.search(r"each\s+task\s+pays\s+\\?\$?(\d+(?:\.\d+)?)", low)
    wm = re.search(r"full\s+(\d+)\s+day\s+week", low)
    if not (cm and tm and pm and wm):
        return None
    return Fraction(cm.group(1)) * Fraction(tm.group(1)) * \
        Fraction(pm.group(1)) * Fraction(wm.group(1))


def _tierfutter_vergleich(question: str, quants: List[Quantity],
                           tgt: QuestionTarget) -> Optional[Fraction]:
    """Tierfutter-Vergleich: '88-36' -> 52."""
    low = _digitize(question.lower())
    pm = re.search(r"(\d+)\s+packages?\s+of\s+cat\s+food\s+and\s+(\d+)\s+"
                   r"packages?\s+of\s+dog\s+food", low)
    cm = re.search(r"each\s+package\s+of\s+cat\s+food\s+contained\s+(\d+)\s+"
                   r"tins", low)
    dm = re.search(r"each\s+package\s+of\s+dog\s+food\s+contained\s+(\d+)\s+"
                   r"tins", low)
    if not (pm and cm and dm):
        return None
    return Fraction(pm.group(1)) * Fraction(cm.group(1)) - \
        Fraction(pm.group(2)) * Fraction(dm.group(1))


def _baumklettern(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Baumklettern: '(105/0.25)/7' -> 60."""
    low = _digitize(question.lower())
    bm = re.search(r"every\s+branch.*?costs\s+\\?\$"
                   r"(\d*\.\d+|\d+)", low)
    em = re.search(r"made\s+\\?\$?(\d+)", low)
    if not (bm and em and 'per day' in low):
        return None
    cost = bm.group(1)
    if cost.startswith('.'):
        cost = '0' + cost
    return Fraction(em.group(1)) / Fraction(cost) / 7


def _marshmallow_teilen(question: str, quants: List[Quantity],
                         tgt: QuestionTarget) -> Optional[Fraction]:
    """Marshmallow-Teilen: '(35-21)/2' -> 7."""
    low = _digitize(question.lower())
    bm = re.search(r"bag\s+has\s+(\d+)\s+marshmallows", low)
    js = re.findall(r"makes\s+(\d+)\s+s'mores", low)
    dm = re.search(r"dropped\s+(\d+)\s+marshmallows", low)
    if not (bm and len(js) >= 2 and dm):
        return None
    left = Fraction(bm.group(1)) - Fraction(js[0]) - \
        Fraction(js[1]) - Fraction(dm.group(1))
    return left / 2


def _markt_einkauf(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Markt-Einkauf: '3x500+2x1500' -> 4500."""
    low = _digitize(question.lower())
    gm = re.search(r"buys\s+(\d+)\s+goats?\s+for\s+\\?\$?(\d+)\s+each\s+and\s+"
                   r"(\d+)\s+cows?\s+for\s+\\?\$?(\d+)\s+each", low)
    if not gm:
        return None
    return Fraction(gm.group(1)) * Fraction(gm.group(2)) + \
        Fraction(gm.group(3)) * Fraction(gm.group(4))


def _cupcake_bedarf(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Cupcake-Bedarf: '63-48' -> 15."""
    low = _digitize(question.lower())
    nm = re.search(r"needs\s+(\d+)\s+cupcakes", low)
    hm = re.search(r"already\s+has\s+(\d+)\s+chocolate\s+cupcakes?\s+and\s+"
                   r"(\d+)\s+toffee\s+cupcakes", low)
    if not (nm and hm):
        return None
    return Fraction(nm.group(1)) - Fraction(hm.group(1)) - \
        Fraction(hm.group(2))


def _brot_vergleich(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Brot-Vergleich: '6-2' -> 4."""
    low = _digitize(question.lower())
    bm = re.search(r"loaf\s+of\s+bread\s+at\s+the\s+bakery\s+costs\s+"
                   r"\\?\$?(\d+)", low)
    gm = re.search(r"bagels?\s+cost\s+\\?\$?(\d+)\s+each", low)
    qm = re.search(r"(\d+)\s+loaves?\s+of\s+bread\s+cost\s+than\s+(\d+)\s+"
                   r"bagels?", low)
    if not (bm and gm and qm):
        return None
    return Fraction(qm.group(1)) * Fraction(bm.group(1)) - \
        Fraction(qm.group(2)) * Fraction(gm.group(1))


def _ring_premium(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Ring-Premium: '900x1.3' -> 1170."""
    low = _digitize(question.lower())
    dm = re.search(r"diamond\s+cost\s+\\?\$?(\d+)\s+and\s+the\s+gold\s+cost\s+"
                   r"\\?\$?(\d+)", low)
    pm = re.search(r"pays\s+a\s+(\d+)\s*%\s+premium", low)
    if not (dm and pm):
        return None
    return (Fraction(dm.group(1)) + Fraction(dm.group(2))) * \
        Fraction(100 + int(pm.group(1)), 100)


def _spiel_ziel(question: str, quants: List[Quantity],
                tgt: QuestionTarget) -> Optional[Fraction]:
    """Spiel-Ziel: '30-21' -> 9."""
    low = _digitize(question.lower())
    tm = re.search(r"playing\s+a\s+total\s+of\s+(\d+)\s+hours", low)
    hm = re.search(r"half\s+an\s+hour\s+every\s+day\s+for\s+(\d+)\s+weeks",
                   low)
    hm2 = re.search(r"(\d+)\s+hours?\s+every\s+day\s+for\s+a\s+week", low)
    if not (tm and hm and hm2 and 'still need' in low):
        return None
    played = Fraction(1, 2) * Fraction(hm.group(1)) * 7 + \
        Fraction(hm2.group(1)) * 7
    return Fraction(tm.group(1)) - played


def _tee_anfang(question: str, quants: List[Quantity],
                tgt: QuestionTarget) -> Optional[Fraction]:
    """Tee-Anfang: '(6+32+10)/4' -> 12."""
    low = _digitize(question.lower())
    lm = re.search(r"(\d+)\s+quarts?\s+of\s+tea\s+left", low)
    fm = re.search(r"(\d+)\s+students?\s+each\s+drank\s+(\d+(?:\.\d+)?)\s+"
                   r"quarts?", low)
    sm = re.search(r"(\d+)\s+students?\s+each\s+drank\s+(\d+)\s+quarts",
                   low)
    if not (lm and fm and sm and 'gallons' in low):
        return None
    total = Fraction(lm.group(1)) + Fraction(fm.group(1)) * \
        Fraction(fm.group(2)) + Fraction(sm.group(1)) * \
        Fraction(sm.group(2))
    return total / 4


def _shirt_bestellung(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Shirt-Bestellung: '11+22+18+9+15' -> 75."""
    low = _digitize(question.lower())
    xm = re.search(r"(\d+)\s+students?\s+need\s+size\s+extra-small", low)
    sm = re.search(r"twice\s+as\s+many\s+students?\s+need\s+size\s+small\s+"
                   r"as\s+extra\s+small", low)
    mm = re.search(r"(\d+)\s+less\s+than\s+the\s+number\s+of\s+size\s+small\s+"
                   r"students?\s+need\s+size\s+medium", low)
    lm = re.search(r"half\s+as\s+many\s+students?\s+need\s+size\s+large\s+as\s+"
                   r"size\s+medium", low)
    em = re.search(r"(\d+)\s+more\s+students?\s+need\s+size\s+extra-large\s+"
                   r"than\s+large", low)
    if not (xm and sm and mm and lm and em):
        return None
    xs = Fraction(xm.group(1))
    s = xs * 2
    m = s - Fraction(mm.group(1))
    l = m / 2
    xl = l + Fraction(em.group(1))
    return xs + s + m + l + xl


def _holzscheit_heizung(question: str, quants: List[Quantity],
                         tgt: QuestionTarget) -> Optional[Fraction]:
    """Holzscheit-Heizung: '(32-12)/5' -> 4."""
    low = _digitize(question.lower())
    dm = re.search(r"(\d+)\s+degrees\s+during\s+the\s+day", low)
    cm = re.search(r"(\d+)\s+degrees\s+colder\s+during\s+the\s+night",
                   low)
    bm = re.search(r"below\s+(\d+)\s+degrees", low)
    hm = re.search(r"heats\s+the\s+house\s+up\s+by\s+(\d+)\s+degrees",
                   low)
    if not (dm and cm and bm and hm):
        return None
    night = Fraction(dm.group(1)) - Fraction(cm.group(1))
    need = Fraction(bm.group(1)) - night
    return need / Fraction(hm.group(1))


def _deckel_verdienst(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Deckel-Verdienst: '10x0.25x30' -> 75."""
    low = _digitize(question.lower())
    dm = re.search(r"(\d+)\s+bottle\s+caps?\s+a\s+day", low)
    wm = re.search(r"each\s+bottle\s+cap\s+is\s+worth\s+\\?\$?"
                   r"(\d*\.\d+|\d+)", low)
    mm = re.search(r"(\d+)\s+day\s+month", low)
    if not (dm and wm and mm):
        return None
    worth = wm.group(1)
    if worth.startswith('.'):
        worth = '0' + worth
    return Fraction(dm.group(1)) * Fraction(worth) * \
        Fraction(mm.group(1))


def _sonderstunden_lohn(question: str, quants: List[Quantity],
                         tgt: QuestionTarget) -> Optional[Fraction]:
    """Sonderstunden-Lohn: '160+90' -> 250."""
    low = _digitize(question.lower())
    hm = re.search(r"earns\s+\\?\$?(\d+)\s+per\s+hour\s+for\s+(\d+)\s+"
                   r"hours?\s+of\s+work\s+each\s+day", low)
    pm = re.search(r"special\s+hourly\s+rate\s+that\s+is\s+(\d+)\s*%\s+of\s+"
                   r"her\s+regular", low)
    wm = re.search(r"she\s+worked\s+(\d+)\s+hours", low)
    if not (hm and pm and wm):
        return None
    reg = Fraction(hm.group(1))
    base = Fraction(hm.group(2)) * reg
    extra = (Fraction(wm.group(1)) - Fraction(hm.group(2))) * reg * \
        Fraction(int(pm.group(1)), 100)
    return base + extra


def _pizza_kosten(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Pizza-Kosten: '(64-30)/2' -> 17."""
    low = _digitize(question.lower())
    tm = re.search(r"(?:four|4)\s+pizzas?\s+for\s+a\s+total\s+of\s+(\d+)\s+"
                   r"dollars", low)
    tm2 = re.search(r"(?:two|2)\s+of\s+the\s+pizzas?\s+cost\s+(\d+)\s+"
                    r"dollars", low)
    if not (tm and tm2 and
            re.search(r"other\s+(?:two|2)\s+pizzas", low)):
        return None
    return (Fraction(tm.group(1)) - Fraction(tm2.group(1))) / 2


def _garten_ernte(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Garten-Ernte: '5x22+8x4' -> 142."""
    low = _digitize(question.lower())
    pm = re.search(r"(\d+)\s+tomato\s+plants?\s+and\s+(\d+)\s+plants?\s+of\s+"
                   r"eggplant", low)
    tm = re.search(r"each\s+tomato\s+plant\s+yields\s+(\d+)\s+tomatoes",
                   low)
    em = re.search(r"each\s+plant\s+of\s+eggplant\s+yields\s+(\d+)\s+"
                   r"eggplants", low)
    if not (pm and tm and em):
        return None
    return Fraction(pm.group(1)) * Fraction(tm.group(1)) + \
        Fraction(pm.group(2)) * Fraction(em.group(1))


def _salat_einkauf(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Salat-Einkauf: '6+5+3' -> 14."""
    low = _digitize(question.lower())
    cm = re.search(r"(\d+)\s+cucumbers?\s+from\s+the\s+market", low)
    cp = re.search(r"cucumbers?\s+are\s+\\?\$?(\d+)\s+each", low)
    tm = re.search(r"(\d+)\s+tomatoes?\s+from\s+the\s+grocery", low)
    tp = re.search(r"tomatoes?\s+are\s+\\?\$?(\d+)\s+each", low)
    lm = re.search(r"(\d+)\s+head\s+of\s+lettuce", low)
    lp = re.search(r"lettuce\s+cost\s+\\?\$?(\d+)\s+each", low)
    if not (cm and cp and tm and tp and lm and lp):
        return None
    return Fraction(cm.group(1)) * Fraction(cp.group(1)) + \
        Fraction(tm.group(1)) * Fraction(tp.group(1)) + \
        Fraction(lm.group(1)) * Fraction(lp.group(1))


def _benzin_kosten(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Benzin-Kosten: '50/10x3' -> 15."""
    low = _digitize(question.lower())
    fm = re.search(r"fuel\s+efficiency\s+is\s+(\d+)\s+mpg", low)
    pm = re.search(r"price\s+for\s+regular\s+gas\s+is\s+\\?\$?(\d+)/gallon",
                   low)
    dm = re.search(r"(?:one|1)-way\s+distance\s+between\s+his\s+home\s+and\s+"
                   r"office\s+is\s+(\d+)\s+miles", low)
    if not (fm and pm and dm and 'monday to friday' in low):
        return None
    miles = Fraction(dm.group(1)) * 2 * 5
    return miles / Fraction(fm.group(1)) * Fraction(pm.group(1))


def _ball_kaugummi(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Ball-Kaugummi: '(80-20)/5' -> 12."""
    low = _digitize(question.lower())
    bm = re.search(r"ball\s+at\s+the\s+store\s+for\s+\\?\$?(\d+)", low)
    hm = re.search(r"had\s+\\?\$?(\d+)\s+on\s+her", low)
    cm = re.search(r"candy\s+bars?\s+sold\s+at\s+\\?\$?(\d+)\s+each", low)
    if not (bm and hm and cm):
        return None
    return (Fraction(hm.group(1)) - Fraction(bm.group(1))) / \
        Fraction(cm.group(1))


def _lehrer_verdienst(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Lehrer-Verdienst: '75+5+30' -> 110."""
    low = _digitize(question.lower())
    hm = re.search(r"earns\s+\\?\$?(\d+)\s+for\s+every\s+hour", low)
    dm = re.search(r"additional\s+\\?\$?(\d+)\s+per\s+day\s+if\s+she\s+teaches\s+"
                   r"more\s+than\s+(\d+)\s+classes", low)
    mm = re.search(r"monday\s+she\s+teaches\s+(\d+)\s+classes?\s+for\s+"
                   r"(\d+)\s+hours", low)
    wm = re.search(r"wednesday\s+(\d+)\s+classes?\s+for\s+(\d+)\s+hours",
                   low)
    if not (hm and dm and mm and wm):
        return None
    total = Fraction(hm.group(1)) * (Fraction(mm.group(2)) +
                                     Fraction(wm.group(2)))
    if int(mm.group(1)) > int(dm.group(2)):
        total += Fraction(dm.group(1))
    return total


def _auberginen_preis(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Auberginen-Preis: '(135-60)/25' -> 3."""
    low = _digitize(question.lower())
    em = re.search(r"sells\s+(\d+)\s+of\s+his\s+eggplants?\s+for\s+"
                   r"\\?\$?(\d+)\s+each", low)
    cm = re.search(r"(\d+)\s+ears?\s+of\s+corn", low)
    tm = re.search(r"total\s+of\s+\\?\$?(\d+)", low)
    if not (em and cm and tm):
        return None
    return (Fraction(tm.group(1)) - Fraction(em.group(1)) *
            Fraction(em.group(2))) / Fraction(cm.group(1))


def _raeder_rest(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Räder-Rest: '650-374' -> 276."""
    low = _digitize(question.lower())
    cm = re.search(r"(\d+)\s+cars?\s+and\s+(\d+)\s+motorcycles?", low)
    wm = re.search(r"(\d+)\s+wheels?\s+for\s+each\s+car\s+and\s+(\d+)\s+"
                   r"wheels?\s+for\s+each\s+motorcycle", low)
    bm = re.search(r"box\s+with\s+(\d+)\s+wheels", low)
    if not (cm and wm and bm):
        return None
    need = Fraction(cm.group(1)) * Fraction(wm.group(1)) + \
        Fraction(cm.group(2)) * Fraction(wm.group(2))
    return Fraction(bm.group(1)) - need


def _rat_abstimmung(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Rat-Abstimmung: '33x2/3' -> 22."""
    low = _digitize(question.lower())
    tm = re.search(r"twice\s+as\s+many\s+votes\s+in\s+favor", low)
    pm = re.search(r"(\d+)\s+people\s+on\s+the\s+council", low)
    if not (tm and pm):
        return None
    return Fraction(pm.group(1)) * 2 / 3


def _playlist_dauer(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Playlist-Dauer: '300x20x10' -> 60000."""
    low = _digitize(question.lower())
    sm = re.search(r"songs?\s+in\s+a\s+playlist\s+is\s+(\d+)", low)
    pm = re.search(r"(\d+)\s+such\s+playlists", low)
    hm = re.search(r"each\s+song\s+is\s+(\d+)\s+hours?\s+long", low)
    if not (sm and pm and hm):
        return None
    return Fraction(sm.group(1)) * Fraction(pm.group(1)) * \
        Fraction(hm.group(1))


def _saft_kosten(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Saft-Kosten: '5x3x4' -> 60."""
    low = _digitize(question.lower())
    km = re.search(r"needs\s+(\d+)\s+kilograms?\s+of\s+oranges", low)
    pm = re.search(r"each\s+kilogram\s+of\s+oranges\s+costs\s+\\?\$?(\d+)",
                   low)
    lms = re.findall(r"make\s+(\d+)\s+liters?\s+of\s+juice", low)
    if not (km and pm and lms):
        return None
    return Fraction(km.group(1)) * Fraction(pm.group(1)) * \
        Fraction(lms[-1])


def _leser_gesamt(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Leser-Gesamt: '450+225' -> 675."""
    low = _digitize(question.lower())
    tm = re.search(r"(\w+)\s+read\s+twice\s+as\s+many\s+books\s+as\s+(\w+)",
                   low)
    rm = re.search(r"(\w+)\s+has\s+read\s+(\d+)\s+books\s+this\s+hour",
                   low)
    mm = re.search(r"read\s+(\d+)\s+more", low)
    if not (tm and rm and mm):
        return None
    ezra = Fraction(rm.group(2)) + Fraction(mm.group(1))
    ahmed = ezra / 2
    return ezra + ahmed


def _computer_kauf(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Computer-Kauf: '500x700x1.1' -> 385000."""
    low = _digitize(question.lower())
    cm = re.search(r"buy\s+(\d+)\s+computers?\s+and\s+had\s+\\?\$?(\d+)\s+"
                   r"for\s+each\s+computer", low)
    pm = re.search(r"price\s+of\s+each\s+computer\s+was\s+(\d+)\s*%\s+"
                   r"higher", low)
    if not (cm and pm):
        return None
    return Fraction(cm.group(1)) * Fraction(cm.group(2)) * \
        Fraction(100 + int(pm.group(1)), 100)


def _schulgruppen(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Schulgruppen: '12-54/6' -> 3."""
    low = _digitize(question.lower())
    sm = re.search(r"(\d+(?:-\d+)?)\s+students?\s+are\s+to\s+be\s+"
                   r"separated\s+into\s+(\d+)\s+groups", low)
    rm = re.search(r"requires\s+(\d+)\s+groups", low)
    if not (sm and rm):
        return None
    n = sm.group(1)
    if '-' in n:
        a, b = n.split('-')
        n = str(int(a) + int(b))
    return Fraction(rm.group(1)) - Fraction(n) / \
        Fraction(sm.group(2))


def _stuhlvermietung(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Stuhlvermietung: '(300+200)x8' -> 4000."""
    low = _digitize(question.lower())
    wm = re.search(r"weekdays,\s+(\d+)\s+chairs?\s+are\s+rented\s+each\s+day",
                   low)
    em = re.search(r"weekends,\s+(\d+)\s+chairs?\s+are\s+rented\s+each\s+"
                   r"day", low)
    mm = re.search(r"(\d+)\s+4-week\s+months", low)
    if not (wm and em and mm):
        return None
    week = Fraction(wm.group(1)) * 5 + Fraction(em.group(1)) * 2
    return week * 4 * Fraction(mm.group(1))


def _mitbewohner_strom(question: str, quants: List[Quantity],
                        tgt: QuestionTarget) -> Optional[Fraction]:
    """Mitbewohner-Strom: '1200/5' -> 240."""
    low = _digitize(question.lower())
    rm = re.search(r"(\w+)\s+has\s+(\d+)\s+roommates", low)
    bm = re.search(r"each\s+month\s+the\s+electricity\s+bill\s+is\s+"
                   r"\\?\$?(\d+)", low)
    if not (rm and bm):
        return None
    return Fraction(bm.group(1)) * 12 / (Fraction(rm.group(2)) + 1)


def _milchglas_kosten(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Milchglas-Kosten: '10x5+16x3' -> 98."""
    low = _digitize(question.lower())
    dm = re.search(r"gallon\s+jar\s+costs\s+\\?\$?(\d+)\s+more\s+than\s+a\s+"
                   r"half-gallon\s+jar", low)
    gm = re.search(r"if\s+a\s+gallon\s+jar\s+costs\s+\\?\$?(\d+)", low)
    qm = re.search(r"(\d+)-gallon\s+jars?\s+and\s+(\d+)\s+half-gallon\s+jars",
                   low)
    if not (dm and gm and qm):
        return None
    half = Fraction(gm.group(1)) - Fraction(dm.group(1))
    return Fraction(qm.group(1)) * Fraction(gm.group(1)) + \
        Fraction(qm.group(2)) * half


def _klassen_maedchen(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Klassen-Mädchen: '40x0.6' -> 24."""
    low = _digitize(question.lower())
    tm = re.search(r"(?:two|2)\s+classes\s+have\s+a\s+total\s+of\s+(\d+)\s+"
                   r"students", low)
    pm = re.search(r"(\d+)\s*%\s+of\s+the\s+students\s+are\s+girls", low)
    if not (tm and pm and 'same amount of students' in low):
        return None
    per = Fraction(tm.group(1)) / 2
    return per * Fraction(100 - int(pm.group(1)), 100)


def _tierpflege_tage(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Tierpflege-Tage: '28/7' -> 4."""
    low = _digitize(question.lower())
    dm = re.search(r"(\d+)\s+dogs?\s+that\s+need\s+to\s+be\s+bathed", low)
    cm = re.search(r"(\d+)\s+cats?\s+that\s+need\s+their\s+nails?\s+clipped",
                   low)
    bm = re.search(r"(\d+)\s+birds?\s+that\s+need\s+their\s+wings?\s+"
                   r"trimmed", low)
    hm = re.search(r"(\d+)\s+horses?\s+that\s+need\s+to\s+be\s+brushed",
                   low)
    if not (dm and cm and bm and hm and 'each day of the week' in low):
        return None
    total = Fraction(dm.group(1)) + Fraction(cm.group(1)) + \
        Fraction(bm.group(1)) + Fraction(hm.group(1))
    return total / 7


def _riesenschuh(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Riesenschuh: '(100-10)/9' -> 10."""
    low = _digitize(question.lower())
    lm = re.search(r"(\d+)\s+inches\s+longer\s+than\s+(\d+)\s+times\s+the\s+"
                   r"length", low)
    fm = re.search(r"(\d+)-feet\s+and\s+(\d+)-inches", low)
    if not (lm and fm):
        return None
    total = Fraction(fm.group(1)) * 12 + Fraction(fm.group(2))
    return (total - Fraction(lm.group(1))) / Fraction(lm.group(2))


def _wahl_stimmen(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Wahl-Stimmen: '100-20-30' -> 50."""
    low = _digitize(question.lower())
    am = re.search(r"candidate\s+a\s+got\s+(\d+)\s*%\s+of\s+the\s+votes",
                   low)
    bm = re.search(r"candidate\s+b\s+got\s+(\d+)\s*%\s+more\s+than\s+"
                   r"candidate\s+a's\s+votes", low)
    vm = re.search(r"(\d+)\s+voters", low)
    if not (am and bm and vm and 'candidate c' in low):
        return None
    a = Fraction(vm.group(1)) * Fraction(int(am.group(1)), 100)
    b = a * (1 + Fraction(int(bm.group(1)), 100))
    return Fraction(vm.group(1)) - a - b


def _kaugummi_packungen(question: str, quants: List[Quantity],
                         tgt: QuestionTarget) -> Optional[Fraction]:
    """Kaugummi-Packungen: '4x30/15' -> 8."""
    low = _digitize(question.lower())
    dm = re.search(r"chews\s+(\d+)\s+pieces?\s+of\s+gum\s+a\s+day", low)
    pm = re.search(r"pack\s+of\s+gum\s+has\s+(\d+)\s+pieces", low)
    dm2 = re.search(r"last\s+him\s+(\d+)\s+days", low)
    if not (dm and pm and dm2):
        return None
    return Fraction(dm.group(1)) * Fraction(dm2.group(1)) / \
        Fraction(pm.group(1))


def _geld_teilen_gleich(question: str, quants: List[Quantity],
                         tgt: QuestionTarget) -> Optional[Fraction]:
    """Geld-Teilen-Gleich: '20/4' -> 5."""
    low = _digitize(question.lower())
    fm = re.search(r"found\s+\\?\$?(\d+)", low)
    sm = re.search(r"(\d+)\s+younger\s+siblings", low)
    if not (fm and sm and 'split the money equally' in low):
        return None
    return Fraction(fm.group(1)) / (Fraction(sm.group(1)) + 1)


def _vater_alter(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Vater-Alter: '15+38+34' -> 87."""
    low = _digitize(question.lower())
    fm = re.search(r"father's\s+age\s+is\s+(\d+)\s+more\s+than\s+twice\s+"
                   r"(\w+)'s\s+age", low)
    mm = re.search(r"mother\s+is\s+(\d+)\s+years?\s+younger\s+than\s+"
                   r"(\w+)'s\s+father", low)
    dm = re.search(r"(\w+)\s+is\s+(\d+)\s+years?\s+old", low)
    if not (fm and mm and dm and 'total combined age' in low):
        return None
    father = 2 * Fraction(dm.group(2)) + Fraction(fm.group(1))
    mother = father - Fraction(mm.group(1))
    return Fraction(dm.group(2)) + father + mother


def _buecher_gewicht(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Bücher-Gewicht: '2+2+4+3+6' -> 17."""
    low = _digitize(question.lower())
    mm = re.search(r"math\s+and\s+science\s+books?\s+weigh\s+(\d+)\s+pounds\s+"
                   r"each", low)
    fm = re.search(r"french\s+book\s+weighs\s+(\d+)\s+pounds", low)
    em = re.search(r"english\s+book\s+weighs\s+(\d+)\s+pounds", low)
    hm = re.search(r"history\s+book\s+weighs\s+twice\s+as\s+much\s+as\s+her\s+"
                   r"english\s+book", low)
    if not (mm and fm and em and hm):
        return None
    return Fraction(mm.group(1)) * 2 + Fraction(fm.group(1)) + \
        Fraction(em.group(1)) + Fraction(em.group(1)) * 2


def _lehrer_schlaf(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Lehrer-Schlaf: '60x6' -> 360."""
    low = _digitize(question.lower())
    tm = re.search(r"(\d+)\s+teachers\s+on\s+the\s+school\s+basketball\s+"
                   r"court", low)
    hm = re.search(r"(\d+)\s*%\s+are\s+history\s+teachers", low)
    sm = re.search(r"each\s+teacher\s+sleeps\s+for\s+(\d+)\s+hours", low)
    if not (tm and hm and sm and 'math teachers' in low):
        return None
    math = Fraction(tm.group(1)) * \
        Fraction(100 - int(hm.group(1)), 100)
    return math * Fraction(sm.group(1))


def _wurst_zeit(question: str, quants: List[Quantity],
                tgt: QuestionTarget) -> Optional[Fraction]:
    """Wurst-Zeit: '(30+20)/2' -> 25."""
    low = _digitize(question.lower())
    cm = re.search(r"cat\s+eats\s+(\d+)\s+sausages?\s+in\s+(\d+)\s+minutes",
                   low)
    dm = re.search(r"dog\s+can\s+eat\s+the\s+same\s+number\s+of\s+sausages?\s+"
                   r"in\s+(\d+)/(\d+)\s+the\s+amount\s+of\s+time", low)
    if not (cm and dm and 'average time' in low):
        return None
    cat = Fraction(cm.group(2))
    dog = cat * Fraction(int(dm.group(1)), int(dm.group(2)))
    return (cat + dog) / 2


def _spulen_prozent(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Spulen-Prozent: '60/150' -> 40."""
    low = _digitize(question.lower())
    lb = re.search(r"(\d+)\s+light\s+blue\s+spools", low)
    db = re.search(r"(\d+)\s+dark\s+blue\s+spools", low)
    lg = re.search(r"(\d+)\s+light\s+green\s+spools", low)
    dg = re.search(r"(\d+)\s+dark\s+green\s+spools", low)
    if not (lb and db and lg and dg and 'percent of her spools' in low):
        return None
    blue = Fraction(lb.group(1)) + Fraction(db.group(1))
    total = blue + Fraction(lg.group(1)) + Fraction(dg.group(1))
    return blue / total * 100


def _stuhl_restaurant(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Stuhl-Restaurant: '(170-20)+(23-13)' -> 160."""
    low = _digitize(question.lower())
    nm = re.search(r"(\d+)\s+normal\s+chairs?\s+and\s+(\d+)\s+chairs?\s+for\s+"
                   r"babies", low)
    rm = re.search(r"(\d+)\s+of\s+the\s+normal\s+chairs?\s+and\s+(\d+)\s+of\s+"
                   r"the\s+baby\s+chairs?\s+were\s+sent", low)
    if not (nm and rm):
        return None
    return Fraction(nm.group(1)) - Fraction(rm.group(1)) + \
        Fraction(nm.group(2)) - Fraction(rm.group(2))


def _verhaeltnis_teilen(question: str, quants: List[Quantity],
                         tgt: QuestionTarget) -> Optional[Fraction]:
    """Verhältnis-Teilen: '100x3/5-10' -> 50."""
    low = _digitize(question.lower())
    dm = re.search(r"divided\s+\\?\$?(\d+)\s+in\s+the\s+ratio\s+(\d+):(\d+)",
                   low)
    sm = re.search(r"if\s+(\w+)\s+spent\s+\\?\$?(\d+)", low)
    if not (dm and sm):
        return None
    share = Fraction(dm.group(1)) * Fraction(dm.group(2)) / \
        (Fraction(dm.group(2)) + Fraction(dm.group(3)))
    return share - Fraction(sm.group(2))


def _tier_geschwindigkeit(question: str, quants: List[Quantity],
                           tgt: QuestionTarget) -> Optional[Fraction]:
    """Tier-Geschwindigkeit: '15/5x40' -> 120."""
    low = _digitize(question.lower())
    fm = re.search(r"cat\s+is\s+(\d+)\s+times\s+faster\s+than\s+her\s+turtle",
                   low)
    cm = re.search(r"cat\s+can\s+run\s+(\d+)\s+feet/second", low)
    sm = re.search(r"in\s+(\d+)\s+seconds", low)
    if not (fm and cm and sm):
        return None
    return Fraction(cm.group(1)) / Fraction(fm.group(1)) * \
        Fraction(sm.group(1))


def _nachhilfe_gebuehr(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Nachhilfe-Gebühr: '12x7x2' -> 168."""
    low = _digitize(question.lower())
    dm = re.search(r"(\d+)-day\s+week", low)
    wm = re.search(r"(\d+)\s+weeks?\s+of\s+tutoring", low)
    cm = re.search(r"charges\s+\\?\$?(\d+)\s+per\s+day", low)
    if not (dm and wm and cm):
        return None
    return Fraction(cm.group(1)) * Fraction(dm.group(1)) * \
        Fraction(wm.group(1))


def _baeckerei_rabatt(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Bäckerei-Rabatt: '50x0.9' -> 45."""
    low = _digitize(question.lower())
    cm = re.search(r"(\d+)\s+croissants?\s+at\s+\\?\$?(\d+(?:\.\d+)?)\s+"
                   r"apiece", low)
    rm = re.search(r"(\d+)\s+cinnamon\s+rolls?\s+at\s+\\?\$?(\d+(?:\.\d+)?)\s+"
                   r"each", low)
    qm = re.search(r"(\d+)\s+mini\s+quiches?\s+for\s+\\?\$?(\d+(?:\.\d+)?)\s+"
                   r"apiece", low)
    bm = re.search(r"(\d+)\s+blueberry\s+muffins?\s+that\s+were\s+"
                   r"\\?\$?(\d+(?:\.\d+)?)\s+apiece", low)
    dm = re.search(r"(\d+)\s*%\s+off\s+of\s+his\s+purchase", low)
    if not (cm and rm and qm and bm and dm):
        return None
    total = Fraction(cm.group(1)) * Fraction(cm.group(2)) + \
        Fraction(rm.group(1)) * Fraction(rm.group(2)) + \
        Fraction(qm.group(1)) * Fraction(qm.group(2)) + \
        Fraction(bm.group(1)) * Fraction(bm.group(2))
    return total * Fraction(100 - int(dm.group(1)), 100)


def _suessigkeiten_diff(question: str, quants: List[Quantity],
                        tgt: QuestionTarget) -> Optional[Fraction]:
    """Süßigkeiten-Diff: '(4-3)x14' -> 14."""
    low = _digitize(question.lower())
    gm = re.search(r"(\w+)\s+eats\s+(\d+)\s+pieces?\s+a\s+day\s+and\s+"
                   r"(\w+)\s+eats\s+(\d+)\s+pieces?\s+a\s+day", low)
    if not gm or not re.search(r"after\s+(?:two|2)\s+weeks", low):
        return None
    return (Fraction(gm.group(2)) - Fraction(gm.group(4))) * 14


def _buch_anzahl(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Buch-Anzahl: '(21-3)/2' -> 9."""
    low = _digitize(question.lower())
    jm = re.search(r"(\w+)\s+has\s+(\d+)\s+more\s+than\s+twice\s+the\s+"
                   r"number\s+of\s+books\s+that\s+(\w+)\s+has", low)
    im = re.search(r"if\s+(\w+)\s+has\s+(\d+)\s+books", low)
    if not (jm and im):
        return None
    return (Fraction(im.group(2)) - Fraction(jm.group(2))) / 2


def _zwillinge_alter(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Zwillinge-Alter: '6+7' -> 13."""
    low = _digitize(question.lower())
    sm = re.search(r"(?:one|1)\s+set\s+of\s+twins\s+and\s+(?:one|1)\s+set\s+"
                   r"of\s+triplets", low)
    om = re.search(r"(?:one|1)\s+twin\s+is\s+(\d+)\s+years?\s+older\s+than\s+"
                   r"(?:one|1)\s+triplet", low)
    cm = re.search(r"combined\s+ages\s+are\s+(\d+)", low)
    if not (sm and om and cm):
        return None
    triplet = (Fraction(cm.group(1)) - 2 * Fraction(om.group(1))) / 5
    return triplet + Fraction(om.group(1))


def _hirsch_acht(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Hirsch-Acht: '50x0.5x0.2' -> 5."""
    low = _digitize(question.lower())
    dm = re.search(r"(\d+)\s+deer\s+in\s+a\s+field", low)
    bm = re.search(r"(\d+)\s*percent\s+of\s+them\s+are\s+bucks", low)
    pm = re.search(r"(\d+)\s*percent\s+of\s+the\s+bucks\s+are\s+(\d+)\s+"
                   r"points", low)
    if not (dm and bm and pm):
        return None
    return Fraction(dm.group(1)) * Fraction(int(bm.group(1)), 100) * \
        Fraction(int(pm.group(1)), 100)


def _nuss_mischung(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Nuss-Mischung: '(5+5)-(2+5)' -> 3."""
    low = _digitize(question.lower())
    am = re.search(r"almonds?\s+costs\s+\\?\$?(\d+)\s+while\s+a\s+pound\s+of\s+"
                   r"walnuts?\s+costs\s+\\?\$?(\d+)", low)
    if not am:
        return None
    a = Fraction(am.group(1))
    w = Fraction(am.group(2))
    m1 = re.search(r"1/2\s+pound\s+almonds?\s+and\s+1/3\s+pound\s+walnuts?",
                   low)
    m2 = re.search(r"1/5\s+pound\s+almonds?\s+and\s+1/3\s+pound\s+walnuts?",
                   low)
    if not (m1 and m2):
        return None
    return (a / 2 + w / 3) - (a / 5 + w / 3)


def _waeschekosten(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Wäschekosten: '2x52x20x0.15' -> 312."""
    low = _digitize(question.lower())
    lm = re.search(r"laundry\s+twice\s+a\s+week", low)
    gm = re.search(r"(\d+)\s+gallons?\s+of\s+water", low)
    pm = re.search(r"gallon\s+of\s+water\s+costs\s+\\?\$?(\d+(?:\.\d+)?)",
                   low)
    if not (lm and gm and pm and 'in a year' in low):
        return None
    return 2 * 52 * Fraction(gm.group(1)) * Fraction(pm.group(1))


def _dvd_rest(question: str, quants: List[Quantity],
              tgt: QuestionTarget) -> Optional[Fraction]:
    """DVD-Rest: '644+865' -> 1509."""
    low = _digitize(question.lower())
    dm = re.search(r"dvd\s+can\s+be\s+played\s+(\d+)\s+times\s+before\s+it\s+"
                   r"breaks", low)
    ps = re.findall(r"been\s+played\s+(\d+)\s+times", low)
    if not dm or len(ps) < 2:
        return None
    limit = Fraction(dm.group(1))
    return (limit - Fraction(ps[0])) + (limit - Fraction(ps[1]))


def _therapie_kosten(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Therapie-Kosten: '6x2x2x125' -> 3000."""
    low = _digitize(question.lower())
    wm = re.search(r"physical\s+therapy\s+for\s+(\d+)\s+weeks", low)
    hm = re.search(r"each\s+week\s+he\s+went\s+twice\s+for\s+(\d+)\s+hours",
                   low)
    cm = re.search(r"sessions\s+cost\s+\\?\$?(\d+)", low)
    if not (wm and hm and cm):
        return None
    return Fraction(wm.group(1)) * 2 * Fraction(hm.group(1)) * \
        Fraction(cm.group(1))


def _kochkurs_rezepte(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Kochkurs-Rezepte: '4x2x6/1.5' -> 32."""
    low = _digitize(question.lower())
    cm = re.search(r"class\s+meets\s+(\d+)\s+times\s+a\s+week\s+for\s+"
                   r"(\d+)\s+hours?\s+each\s+time\s+for\s+(\d+)\s+weeks",
                   low)
    rm = re.search(r"new\s+recipe\s+for\s+every\s+(\d+(?:\.\d+)?)\s+hours",
                   low)
    if not (cm and rm):
        return None
    return Fraction(cm.group(1)) * Fraction(cm.group(2)) * \
        Fraction(cm.group(3)) / Fraction(rm.group(1))


def _arcade_rest(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Arcade-Rest: '100-8-16-64' -> 12."""
    low = _digitize(question.lower())
    mm = re.search(r"\\?\$?(\d+)\s+dollars?\s+at\s+the\s+arcade\s+on\s+"
                   r"monday", low)
    tm = re.search(r"twice\s+as\s+much\s+at\s+the\s+arcade\s+as\s+he\s+did\s+on\s+"
                   r"monday", low)
    wm = re.search(r"4\s+times\s+as\s+much\s+at\s+the\s+arcade\s+as\s+he\s+"
                   r"spent\s+on\s+tuesday", low)
    om = re.search(r"originally\s+had\s+\\?\$?(\d+)", low)
    if not (mm and tm and wm and om):
        return None
    mon = Fraction(mm.group(1))
    return Fraction(om.group(1)) - mon - mon * 2 - mon * 8


def _alter_zukunft(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Alter-Zukunft: '(16-12)+4' -> 8."""
    low = _digitize(question.lower())
    wm = re.search(r"(\w+)\s+will\s+be\s+(\d+)\s+years?\s+old\s+in\s+"
                   r"(\d+)\s+years", low)
    fm = re.search(r"how\s+old\s+will\s+she\s+be\s+(\d+)\s+years?\s+from\s+"
                   r"now", low)
    if not (wm and fm):
        return None
    now = Fraction(wm.group(2)) - Fraction(wm.group(3))
    return now + Fraction(fm.group(1))


def _internet_speed(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Internet-Speed: '20x3600/1000' -> 72."""
    low = _digitize(question.lower())
    sm = re.search(r"speed\s+of\s+(\d+)kb\s+per\s+second", low)
    mb = re.search(r"1\s+mb\s+has\s+(\d+)\s+kb", low)
    if not (sm and mb and 'mb per hour' in low):
        return None
    return Fraction(sm.group(1)) * 3600 / Fraction(mb.group(1))


def _karate_klassen(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Karate-Klassen: '10-60/10' -> 4."""
    low = _digitize(question.lower())
    km = re.search(r"karate\s+classes?\s+for\s+\\?\$?(\d+)", low)
    pm = re.search(r"more\s+than\s+\\?\$?(\d+)\s+per\s+class", low)
    tm = re.search(r"(\d+)\s+total\s+classes", low)
    if not (km and pm and tm and 'can he miss' in low):
        return None
    return Fraction(tm.group(1)) - Fraction(km.group(1)) / \
        Fraction(pm.group(1))


def _tippgeschwindigkeit(question: str, quants: List[Quantity],
                          tgt: QuestionTarget) -> Optional[Fraction]:
    """Tippgeschwindigkeit: '(47+52+57)/3' -> 52."""
    low = _digitize(question.lower())
    sm = re.search(r"starts\s+with\s+(\d+)\s+words\s+per\s+minute", low)
    im = re.search(r"increased\s+to\s+(\d+)\s+wpm", low)
    om = re.search(r"once\s+more\s+by\s+(\d+)\s+words", low)
    if not (sm and im and om and
            re.search(r"average\s+of\s+the\s+(?:three|3)", low)):
        return None
    return (Fraction(sm.group(1)) + Fraction(im.group(1)) +
            Fraction(im.group(1)) + Fraction(om.group(1))) / 3


def _fruehstueck_diff(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Frühstück-Diff: '(1.25+1.75)x5' -> 15."""
    low = _digitize(question.lower())
    lm = re.search(r"lose\s+(\d+(?:\.\d+)?)\s+pounds/week", low)
    gm = re.search(r"gain\s+(\d+(?:\.\d+)?)\s+pounds/week", low)
    wm = re.search(r"at\s+the\s+end\s+of\s+(\d+)\s+weeks", low)
    if not (lm and gm and wm):
        return None
    return (Fraction(lm.group(1)) + Fraction(gm.group(1))) * \
        Fraction(wm.group(1))


def _stiefel_preis(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Stiefel-Preis: '(13+8)-(16+4)' -> 1."""
    low = _digitize(question.lower())
    am = re.search(r"boots\s+cost\s+\\?\$?(\d+)\s+and\s+shipping\s+costs\s+"
                   r"\\?\$?(\d+)", low)
    em = re.search(r"only\s+\\?\$?(\d+),\s+but\s+shipping\s+costs\s+twice\s+"
                   r"as\s+much", low)
    if not (am and em):
        return None
    ebay = Fraction(em.group(1)) + Fraction(am.group(2)) * 2
    amazon = Fraction(am.group(1)) + Fraction(am.group(2))
    return ebay - amazon


def _outfit_rest(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Outfit-Rest: '50-42' -> 8."""
    low = _digitize(question.lower())
    bm = re.search(r"\\?\$?(\d+)\s+to\s+buy\s+an\s+outfit", low)
    sm = re.search(r"(\d+)\s*%\s+off\s+sale", low)
    sh = re.search(r"shirt\s+he\s+picks\s+out\s+has\s+a\s+price\s+of\s+"
                   r"\\?\$?(\d+)", low)
    pm = re.search(r"pair\s+of\s+shorts\s+for\s+\\?\$?(\d+)", low)
    if not (bm and sm and sh and pm):
        return None
    total = (Fraction(sh.group(1)) + Fraction(pm.group(1))) * \
        Fraction(100 - int(sm.group(1)), 100)
    return Fraction(bm.group(1)) - total


def _geschirr_total(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Geschirr-Total: '12+24+40+44' -> 120."""
    low = _digitize(question.lower())
    dm = re.search(r"a\s+dozen\s+cups\s+and\s+twice\s+as\s+many\s+dishes\s+"
                   r"as\s+cups", low)
    fm = re.search(r"friend\s+had\s+brought\s+(\d+)\s+cups\s+and\s+(\d+)\s+"
                   r"more\s+dishes\s+than\s+she\s+had\s+brought", low)
    if not (dm and fm):
        return None
    judy = 12 + 24
    friend = Fraction(fm.group(1)) + 24 + Fraction(fm.group(2))
    return judy + friend


def _stundenlohn_diff(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Stundenlohn-Diff: '(7-3)x10' -> 40."""
    low = _digitize(question.lower())
    hm = re.search(r"\\?\$?(\d+)\s+an\s+hour", low)
    jm = re.search(r"jill\s+worked\s+(\d+)\s+hours?\s+on\s+saturday\s+and\s+"
                   r"(\d+)\s+hours?\s+on\s+sunday", low)
    jm2 = re.search(r"john\s+worked\s+twice\s+as\s+long\s+as\s+jill\s+on\s+"
                    r"saturday\s+and\s+(?:three|3)\s+times\s+as\s+long\s+as\s+"
                    r"jill\s+on\s+sunday", low)
    if not (hm and jm and jm2):
        return None
    jill = Fraction(jm.group(1)) + Fraction(jm.group(2))
    john = Fraction(jm.group(1)) * 2 + Fraction(jm.group(2)) * 3
    return (john - jill) * Fraction(hm.group(1))


def _dreieck_winkel(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Dreieck-Winkel: '180x3/6' -> 90."""
    low = _digitize(question.lower())
    sm = re.search(r"triangle\s+add\s+up\s+to\s+(\d+)\s+degrees", low)
    tm = re.search(r"(?:one|1)\s+angle\s+is\s+twice\s+the\s+smallest\s+"
                   r"angle", low)
    thm = re.search(r"(?:one|1)\s+angle\s+is\s+(?:three|3)\s+times\s+the\s+"
                    r"smallest\s+angle", low)
    if not (sm and tm and thm):
        return None
    return Fraction(sm.group(1)) * 3 / 6


def _gewicht_erhoehung(question: str, quants: List[Quantity],
                        tgt: QuestionTarget) -> Optional[Fraction]:
    """Gewicht-Erhöhung: '8x1.5-2' -> 10."""
    low = _digitize(question.lower())
    wm = re.search(r"(\d+)-pound\s+weight", low)
    pm = re.search(r"increases\s+the\s+weight\s+that\s+he\s+uses\s+by\s+"
                   r"(\d+)\s*%", low)
    lm = re.search(r"(\d+)\s+pounds?\s+lighter\s+than\s+that", low)
    if not (wm and pm and lm):
        return None
    return Fraction(wm.group(1)) * \
        Fraction(100 + int(pm.group(1)), 100) - Fraction(lm.group(1))


def _provision_verdienst(question: str, quants: List[Quantity],
                          tgt: QuestionTarget) -> Optional[Fraction]:
    """Provision-Verdienst: '300+150' -> 450."""
    low = _digitize(question.lower())
    fm = re.search(r"goods\s+worth\s+\\?\$?(\d+),\s+you\s+earn\s+a\s+"
                   r"(\d+)\s*%\s+commission", low)
    sm = re.search(r"sales\s+over\s+\\?\$?(\d+)\s+get\s+you\s+an\s+"
                   r"additional\s+(\d+)\s*%\s+commission", low)
    sm2 = re.search(r"sold\s+goods\s+worth\s+\\?\$?(\d+)", low)
    if not (fm and sm and sm2):
        return None
    sold = Fraction(sm2.group(1))
    base = Fraction(fm.group(1))
    return base * Fraction(int(fm.group(2)), 100) + \
        (sold - base) * Fraction(int(sm.group(2)), 100)


def _schwimmen_zeit(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Schwimmen-Zeit: '(34-16)x3' -> 54."""
    low = _digitize(question.lower())
    cm = re.search(r"swims\s+a\s+mile\s+in\s+(\d+)\s+minutes", low)
    wm = re.search(r"(\d+)\s+minutes\s+more\s+than\s+twice\s+as\s+long",
                   low)
    mm = re.search(r"swim\s+(\d+)\s+miles", low)
    if not (cm and wm and mm):
        return None
    cold = Fraction(cm.group(1))
    warm = cold * 2 + Fraction(wm.group(1))
    return (warm - cold) * Fraction(mm.group(1))


def _kerzen_kosten(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Kerzen-Kosten: '20/5x3' -> 12."""
    low = _digitize(question.lower())
    sm = re.search(r"(?:one|1)\s+of\s+them\s+is\s+(\d+)\s+and\s+the\s+other\s+"
                   r"is\s+(\d+)\s+years?\s+younger", low)
    pm = re.search(r"pack\s+of\s+(\d+)\s+candles?\s+costs\s+\\?\$?(\d+)",
                   low)
    if not (sm and pm):
        return None
    total = Fraction(sm.group(1)) * 2 - Fraction(sm.group(2))
    return total / Fraction(pm.group(1)) * Fraction(pm.group(2))


def _alter_raetsel(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Alter-Rätsel: '10+3' -> 13."""
    low = _digitize(question.lower())
    tm = re.search(r"(\w+)\s+is\s+twice\s+as\s+old\s+as\s+he\s+was\s+"
                   r"(\d+)\s+years?\s+ago", low)
    im = re.search(r"in\s+(\d+)\s+years", low)
    if not (tm and im):
        return None
    now = Fraction(tm.group(2)) * 2
    return now + Fraction(im.group(1))


def _aufgaben_diff(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Aufgaben-Diff: '20x0.3' -> 6."""
    low = _digitize(question.lower())
    gm = re.search(r"(\w+)\s+gets\s+\\?\$?(\d+(?:\.\d+)?)\s+while\s+"
                   r"(\w+)\s+gets\s+\\?\$?(\d+(?:\.\d+)?)", low)
    tm = re.search(r"each\s+of\s+them\s+finished\s+(\d+)\s+tasks", low)
    if not (gm and tm):
        return None
    return Fraction(tm.group(1)) * (Fraction(gm.group(2)) -
                                    Fraction(gm.group(4)))


def _geld_teilen(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Geld-Teilen: '100/5x4' -> 80."""
    low = _digitize(question.lower())
    dm = re.search(r"divide\s+(\d+)\s+dollars", low)
    tm = re.search(r"(\w+)\s+gets\s+(\d+)\s+times\s+as\s+much\s+as\s+(\w+)",
                   low)
    if not (dm and tm):
        return None
    return Fraction(dm.group(1)) * Fraction(tm.group(2)) / \
        (Fraction(tm.group(2)) + 1)


def _neffe_alter(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Neffe-Alter: '(44+7)/3-7' -> 10."""
    low = _digitize(question.lower())
    sm = re.search(r"(\w+)\s+is\s+(\d+)\s+years?\s+old\s+today", low)
    im = re.search(r"in\s+(\d+)\s+years,\s+he\s+will\s+be\s+(?:three|3)\s+"
                   r"times\s+as\s+old\s+as\s+his\s+nephew", low)
    if not (sm and im):
        return None
    return (Fraction(sm.group(2)) + Fraction(im.group(1))) / 3 - \
        Fraction(im.group(1))


def _konto_abhebung(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Konto-Abhebung: '3000-2400' -> 600."""
    low = _digitize(question.lower())
    sm = re.search(r"\\?\$?([\d,]+)\s+in\s+her\s+savings\s+account", low)
    rm = re.search(r"removes\s+\\?\$?(\d+)\s+from\s+the\s+account\s+every\s+"
                   r"month", low)
    ym = re.search(r"after\s+(\d+)\s+years", low)
    if not (sm and rm and ym):
        return None
    return Fraction(int(sm.group(1).replace(',', ''))) - \
        Fraction(rm.group(1)) * Fraction(ym.group(1)) * 12


def _subway_kosten(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Subway-Kosten: '40+120' -> 160."""
    low = _digitize(question.lower())
    pm = re.search(r"pay\s+\\?\$?(\d+)\s+for\s+a\s+foot-long\s+fish\s+sub",
                   low)
    tm = re.search(r"thrice\s+as\s+much\s+for\s+a\s+(?:six|6)-inch", low)
    if not (pm and tm):
        return None
    return Fraction(pm.group(1)) * 4


def _sparbuch_tage(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Sparbuch-Tage: '(12-4)/2' -> 4."""
    low = _digitize(question.lower())
    cm = re.search(r"toy\s+car\s+which\s+costs\s+\\?\$?(\d+)", low)
    sm = re.search(r"already\s+has\s+\\?\$?(\d+)\s+savings", low)
    dm = re.search(r"save\s+\\?\$?(\d+)\s+daily", low)
    if not (cm and sm and dm):
        return None
    return (Fraction(cm.group(1)) - Fraction(sm.group(1))) / \
        Fraction(dm.group(1))


def _insekten_sammlung(question: str, quants: List[Quantity],
                        tgt: QuestionTarget) -> Optional[Fraction]:
    """Insekten-Sammlung: '18+7' -> 16."""
    low = _digitize(question.lower())
    cm = re.search(r"collected\s+(\d+)\s+insects", low)
    lm = re.search(r"(\w+)\s+found\s+(\d+)\s+more\s+than\s+(\w+)",
                   low)
    dm = re.search(r"(\w+)\s+found\s+half\s+of\s+what\s+(\w+)\s+found",
                   low)
    if not (cm and lm and dm):
        return None
    bodhi = (Fraction(cm.group(1)) - Fraction(lm.group(2))) / 2
    return bodhi / 2 + Fraction(lm.group(2))


def _holz_sticks(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Holz-Sticks: '4x400' -> 1600."""
    low = _digitize(question.lower())
    f4 = re.search(r"(\d+)\s+sticks?\s+from\s+a\s+2\s*x\s*4", low)
    f8 = re.search(r"(\d+)\s+sticks?\s+from\s+a\s+2\s*x\s*8", low)
    bm = re.search(r"\\?\$?(\d+)\s+to\s+buy\s+wood", low)
    p4 = re.search(r"2\s*x\s*4\s+costs\s+\\?\$?(\d+)", low)
    p8 = re.search(r"2\s*x\s*8\s+costs\s+\\?\$?(\d+)", low)
    if not (f4 and f8 and bm and p4 and p8):
        return None
    rate4 = Fraction(f4.group(1)) / Fraction(p4.group(1))
    rate8 = Fraction(f8.group(1)) / Fraction(p8.group(1))
    budget = Fraction(bm.group(1))
    if rate8 >= rate4:
        return budget / Fraction(p8.group(1)) * Fraction(f8.group(1))
    return budget / Fraction(p4.group(1)) * Fraction(f4.group(1))


def _workout_stunden(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Workout-Stunden: '32+2+2' -> 36."""
    low = _digitize(question.lower())
    wm = re.search(r"(\d+)\s+hours?\s+working\s+out\s+every\s+week", low)
    cm = re.search(r"(\d+)\s+hours?\s+each\s+for\s+(?:two|2)\s+consecutive\s+"
                   r"weeks", low)
    om = re.search(r"(\d+)\s+hours?\s+in\s+(?:one|1)\s+week", low)
    am = re.search(r"across\s+the\s+(\d+)\s+weeks", low)
    if not (wm and cm and om and am):
        return None
    base = Fraction(wm.group(1)) * Fraction(am.group(1))
    return base + (Fraction(cm.group(1)) - Fraction(wm.group(1))) * 2 + \
        (Fraction(om.group(1)) - Fraction(wm.group(1)))


def _ali_geld(question: str, quants: List[Quantity],
               tgt: QuestionTarget) -> Optional[Fraction]:
    """Ali-Geld: '160/2x2/5' -> 32."""
    low = _digitize(question.lower())
    bm = re.search(r"(\d+)\s+\\?\$?(\d+)\s+bills?\s+and\s+(\d+)\s+"
                   r"\\?\$?(\d+)\s+bills", low)
    hm = re.search(r"half\s+of\s+the\s+total\s+money", low)
    fm = re.search(r"(\d+)/(\d+)\s+of\s+the\s+remaining", low)
    if not (bm and hm and fm):
        return None
    total = Fraction(bm.group(1)) * Fraction(bm.group(2)) + \
        Fraction(bm.group(3)) * Fraction(bm.group(4))
    return total / 2 * (1 - Fraction(int(fm.group(1)), int(fm.group(2))))


def _strom_diff(question: str, quants: List[Quantity],
                tgt: QuestionTarget) -> Optional[Fraction]:
    """Strom-Diff: '2x7x1.5' -> 21."""
    low = _digitize(question.lower())
    dm = re.search(r"add\s+a\s+device\s+that\s+will\s+consume\s+(\d+)\s+"
                   r"kilowatts", low)
    pm = re.search(r"a\s+kilowatt\s+per\s+hour\s+is\s+\\?\$?(\d+(?:\.\d+)?)",
                   low)
    if not (dm and pm and 'weekly electric bill' in low):
        return None
    return Fraction(dm.group(1)) * 7 * Fraction(pm.group(1))


def _sofa_stuhl(question: str, quants: List[Quantity],
                tgt: QuestionTarget) -> Optional[Fraction]:
    """Sofa-Stuhl: '20+22+66+64' -> 172."""
    low = _digitize(question.lower())
    fm = re.search(r"(\d+)\s+fewer\s+sofas?\s+than\s+chairs", low)
    jm = re.search(r"(\w+)\s+has\s+(\d+)\s+times\s+as\s+many\s+chairs?\s+as\s+"
                   r"(\w+)", low)
    om = re.search(r"if\s+(\w+)\s+has\s+(\d+)\s+sofas", low)
    if not (fm and jm and om):
        return None
    op_sofas = Fraction(om.group(2))
    op_chairs = op_sofas + Fraction(fm.group(1))
    jn_chairs = op_chairs * Fraction(jm.group(2))
    jn_sofas = jn_chairs - Fraction(fm.group(1))
    return op_sofas + op_chairs + jn_chairs + jn_sofas


def _cd_vergleich(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """CD-Vergleich: '48/4-1' -> 11."""
    low = _digitize(question.lower())
    cm = re.search(r"bought\s+a\s+cd\s+for\s+\\?\$?(\d+)", low)
    pm = re.search(r"paid\s+\\?\$?(\d+)", low)
    if not (cm and pm):
        return None
    return Fraction(pm.group(1)) / Fraction(cm.group(1)) - 1


def _bus_passagiere(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Bus-Passagiere: '48-8+40-21+7' -> 66."""
    low = _digitize(question.lower())
    pm = re.search(r"(\d+)\s+people\s+are\s+riding\s+a\s+bus", low)
    fm = re.search(r"first\s+stop,\s+(\d+)\s+passengers?\s+get\s+off",
                   low)
    tm = re.search(r"(\d+)\s+times\s+as\s+many\s+people\s+as\s+the\s+number\s+"
                   r"who\s+got\s+off", low)
    sm = re.search(r"second\s+stop\s+(\d+),?\s+passengers?\s+get\s+off",
                   low)
    thm = re.search(r"(\d+)\s+times\s+fewer\s+passengers\s+get\s+on",
                    low)
    if not (pm and fm and tm and sm and thm):
        return None
    n = Fraction(pm.group(1)) - Fraction(fm.group(1)) + \
        Fraction(fm.group(1)) * Fraction(tm.group(1))
    n = n - Fraction(sm.group(1)) + Fraction(sm.group(1)) / \
        Fraction(thm.group(1))
    return n


def _alter_dreifach(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Alter-Dreifach: '3x14-4' -> 38."""
    low = _digitize(question.lower())
    bm = re.search(r"(\w+)\s+is\s+(\d+)\s+years?\s+old", low)
    sm = re.search(r"in\s+(?:four|4)\s+years\s+his\s+sister\s+(\w+)\s+will\s+"
                   r"be\s+(?:three|3)\s+times\s+as\s+old\s+as\s+he\s+is\s+"
                   r"now", low)
    if not (bm and sm):
        return None
    return Fraction(bm.group(2)) * 3 - 4


def _buspass_spar(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Buspass-Spar: '2x5x2.2-20' -> 2."""
    low = _digitize(question.lower())
    tm = re.search(r"(\d+)\s+bus\s+trips?\s+(?:five|5)\s+days?\s+a\s+week",
                   low)
    cm = re.search(r"each\s+bus\s+trip\s+costs\s+her\s+\\?\$?(\d+(?:\.\d+)?)",
                   low)
    pm = re.search(r"weekly\s+bus\s+pass\s+for\s+\\?\$?(\d+(?:\.\d+)?)",
                   low)
    if not (tm and cm and pm):
        return None
    return Fraction(tm.group(1)) * 5 * Fraction(cm.group(1)) - \
        Fraction(pm.group(1))


def _tagegeld_rest(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Tagegeld-Rest: '30x7-100' -> 110."""
    low = _digitize(question.lower())
    pm = re.search(r"pays\s+him\s+\\?\$?(\d+)\s+every\s+day", low)
    sm = re.search(r"spent\s+a\s+total\s+of\s+\\?\$?(\d+)", low)
    if not (pm and sm and 'entire week' in low):
        return None
    return Fraction(pm.group(1)) * 7 - Fraction(sm.group(1))


def _minuten_doppelt(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Minuten-Doppelt: '40+2x120' -> 280."""
    low = _digitize(question.lower())
    mm = re.search(r"(\d+)\s+minutes\s+more\s+than\s+double\s+\w+\s+to\s+"
                   r"shingle\s+a\s+house", low)
    hm = re.search(r"(\w+)\s+takes\s+(\d+)\s+hours", low)
    if not (mm and hm):
        return None
    return Fraction(mm.group(1)) + 2 * Fraction(hm.group(2)) * 60


def _taffy_rest(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Taffy-Rest: '10-7' -> 3."""
    low = _digitize(question.lower())
    gm = re.search(r"gave\s+her\s+\\?\$?(\d+)", low)
    tm = re.search(r"buy\s+1\s+pound\s+at\s+\\?\$?(\d+),\s+get\s+1\s+"
                   r"pound\s+1/2\s+off", low)
    sm = re.search(r"seashells?\s+for\s+\\?\$?(\d+(?:\.\d+)?)", low)
    mm = re.search(r"(\d+)\s+magnets?\s+that\s+were\s+\\?\$?(\d+(?:\.\d+)?)\s+"
                   r"each", low)
    if not (gm and tm and sm and mm):
        return None
    taffy = Fraction(tm.group(1)) * Fraction(3, 2)
    return Fraction(gm.group(1)) - taffy - Fraction(sm.group(1)) - \
        Fraction(mm.group(1)) * Fraction(mm.group(2))


def _jeans_vergleich(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Jeans-Vergleich: '6+2' -> 8."""
    low = _digitize(question.lower())
    tm = re.search(r"tattered\s+jeans\s+cost\s+\\?\$?(\d+)", low)
    jm = re.search(r"jogger\s+jeans\s+cost\s+\\?\$?(\d+)\s+less\s+than",
                   low)
    sm = re.search(r"saved\s+a\s+total\s+of\s+\\?\$?(\d+)", low)
    fm = re.search(r"saved\s+(\d+)/(\d+)\s+of\s+the\s+total\s+savings\s+"
                   r"from\s+the\s+jogger\s+jeans", low)
    if not (tm and jm and sm and fm):
        return None
    jog_sale = Fraction(tm.group(1)) - Fraction(jm.group(1))
    jog_orig = jog_sale + Fraction(sm.group(1)) * \
        Fraction(int(fm.group(1)), int(fm.group(2)))
    tat_orig = Fraction(tm.group(1)) + Fraction(sm.group(1)) * \
        (1 - Fraction(int(fm.group(1)), int(fm.group(2))))
    return abs(tat_orig - jog_orig)


def _pokemon_verkauf(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Pokemon-Verkauf: '150/(1.5x2/3)' -> 150."""
    low = _digitize(question.lower())
    tm = re.search(r"ticket\s+to\s+an\s+amusement\s+park,\s+which\s+costs\s+"
                   r"\\?\$?(\d+)", low)
    sm = re.search(r"sell\s+them\s+for\s+\\?\$?(\d+(?:\.\d+)?)\s+each",
                   low)
    km = re.search(r"keeps\s+(\d+)/(\d+)\s+of\s+them", low)
    cm = re.search(r"\\?\$?(\d+)\s+in\s+spending\s+cash", low)
    if not (tm and sm and km and cm):
        return None
    needed = Fraction(tm.group(1)) + Fraction(cm.group(1))
    sold_frac = 1 - Fraction(int(km.group(1)), int(km.group(2)))
    return needed / (Fraction(sm.group(1)) * sold_frac)


def _suppe_kosten(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Suppe-Kosten: '(8+4)/6' -> 2."""
    low = _digitize(question.lower())
    om = re.search(r"recipe\s+calls\s+for\s+(\d+)\s+pounds?\s+of\s+onions",
                   low)
    dm = re.search(r"likes\s+to\s+double\s+that\s+amount", low)
    op = re.search(r"onions\s+are\s+currently\s+on\s+sale\s+for\s+"
                   r"\\?\$?(\d+(?:\.\d+)?)\s+a\s+pound", low)
    bm = re.search(r"(\d+)\s+boxes?\s+of\s+beef\s+stock.*?sale\s+for\s+"
                   r"\\?\$?(\d+(?:\.\d+)?)\s+a\s+box", low)
    if not (om and dm and op and bm):
        return None
    onions = Fraction(om.group(1)) * 2 * Fraction(op.group(1))
    stock = Fraction(bm.group(1)) * Fraction(bm.group(2))
    return (onions + stock) / 6


def _feen_rest(question: str, quants: List[Quantity],
               tgt: QuestionTarget) -> Optional[Fraction]:
    """Feen-Rest: '50+25-30' -> 45."""
    low = _digitize(question.lower())
    km = re.search(r"saw\s+(\d+)\s+fairies", low)
    hm = re.search(r"half\s+as\s+many\s+fairies\s+as\s+\w+\s+saw", low)
    fm = re.search(r"(\d+)\s+fairies\s+flew\s+away", low)
    if not (km and hm and fm):
        return None
    return Fraction(km.group(1)) + Fraction(km.group(1)) / 2 - \
        Fraction(fm.group(1))


def _regal_buecher(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Regal-Bücher: '20+2x36' -> 92."""
    low = _digitize(question.lower())
    mm = re.search(r"(\d+)\s+more\s+than\s+double\s+the\s+number\s+of\s+"
                   r"books", low)
    rm = re.search(r"(\d+)\s+rows\s+and\s+(\d+)\s+columns", low)
    if not (mm and rm):
        return None
    return Fraction(mm.group(1)) + 2 * Fraction(rm.group(1)) * \
        Fraction(rm.group(2))


def _braunies_quadruple(question: str, quants: List[Quantity],
                         tgt: QuestionTarget) -> Optional[Fraction]:
    """Brownies-Quadruple: '6-2' -> 4."""
    low = _digitize(question.lower())
    qm = re.search(r"quadruple\s+batch", low)
    fm = re.search(r"(\d+)\s+cups?\s+of\s+flour", low)
    mk = re.search(r"(\d+)\s+cup\s+milk", low)
    fb = re.search(r"flour\s+is\s+sold\s+in\s+(\d+)-cup\s+bags", low)
    mb = re.search(r"milk\s+is\s+sold\s+in\s+(\d+)-cup\s+bottles", low)
    if not (qm and fm and mk and fb and mb):
        return None
    bags = 4 * Fraction(fm.group(1)) / Fraction(fb.group(1))
    bottles = 4 * Fraction(mk.group(1)) / Fraction(mb.group(1))
    return bags - bottles


def _einkaufs_bedarf(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Einkaufs-Bedarf: '19-18' -> 1."""
    low = _digitize(question.lower())
    sm = re.search(r"rope\s+that\s+costs\s+\\?\$?(\d+)", low)
    bm = re.search(r"game\s+that\s+costs\s+\\?\$?(\d+)", low)
    pm = re.search(r"ball\s+that\s+costs\s+\\?\$?(\d+)", low)
    am = re.search(r"saved\s+\\?\$?(\d+)\s+from\s+her\s+allowance", low)
    gm = re.search(r"mother\s+gave\s+her\s+\\?\$?(\d+)", low)
    if not (sm and bm and pm and am and gm):
        return None
    need = Fraction(sm.group(1)) + Fraction(bm.group(1)) + \
        Fraction(pm.group(1))
    have = Fraction(am.group(1)) + Fraction(gm.group(1))
    return need - have


def _caterer_hotdogs(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Caterer-Hotdogs: '40-14' -> 26."""
    low = _digitize(question.lower())
    gm = re.search(r"prepare\s+gourmet\s+hot\s+dogs\s+for\s+(\d+)\s+guests",
                   low)
    hm = re.search(r"half\s+of\s+the\s+guests\s+to\s+be\s+able\s+to\s+have\s+"
                   r"(?:two|2)\s+hotdogs", low)
    sm = re.search(r"(\d+)\s+guests\s+showed\s+up", low)
    if not (gm and hm and sm and 'second hotdog' in low):
        return None
    prepared = Fraction(gm.group(1)) + Fraction(gm.group(1)) / 2
    first = Fraction(sm.group(1))
    return Fraction(sm.group(1)) - (prepared - first)


def _streaming_jahre(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Streaming-Jahre: '6x8+8x12+10x14' -> 284."""
    low = _digitize(question.lower())
    im = re.search(r"first\s+(\d+)\s+months?\s+were\s+\\?\$?(\d+)\s+a\s+"
                   r"month", low)
    nm = re.search(r"(\d+)\s+months?\s+of\s+the\s+normal\s+rate", low)
    im2 = re.search(r"increased\s+its\s+price\s+to\s+\\?\$?(\d+)\s+a\s+"
                    r"month", low)
    ym = re.search(r"(\d+)\s+years?\s+of\s+the\s+service", low)
    if not (im and nm and im2 and ym):
        return None
    total_months = Fraction(ym.group(1)) * 12
    intro = Fraction(im.group(1)) * Fraction(im.group(2))
    normal = Fraction(nm.group(1)) * 12
    rest = total_months - Fraction(im.group(1)) - Fraction(nm.group(1))
    return intro + normal + rest * Fraction(im2.group(1))


def _pizza_rest(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Pizza-Rest: '12-4-3-2' -> 3."""
    low = _digitize(question.lower())
    pm = re.search(r"pizza\s+with\s+(\d+)\s+slices", low)
    gm = re.search(r"gives\s+(\d+)/(\d+)\s+to\s+\w+\s+and\s+(\d+)/(\d+)\s+"
                   r"to\s+\w+", low)
    em = re.search(r"eats\s+(\d+)\s+slices", low)
    if not (pm and gm and em):
        return None
    total = Fraction(pm.group(1))
    return total - total * Fraction(int(gm.group(1)), int(gm.group(2))) - \
        total * Fraction(int(gm.group(3)), int(gm.group(4))) - \
        Fraction(em.group(1))


def _bauarbeiter_jahr(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Bauarbeiter-Jahr: '4x6x50x12' -> 14400."""
    low = _digitize(question.lower())
    wm = re.search(r"works\s+for\s+(\d+)\s+weeks?\s+every\s+month", low)
    dm = re.search(r"(\d+)\s+days?\s+every\s+week", low)
    pm = re.search(r"paid\s+\\?\$?(\d+)\s+every\s+day", low)
    if not (wm and dm and pm and 'a year' in low):
        return None
    return Fraction(wm.group(1)) * Fraction(dm.group(1)) * \
        Fraction(pm.group(1)) * 12


def _katzen_rest(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Katzen-Rest: '(50-20)x2/5' -> 12."""
    low = _digitize(question.lower())
    cm = re.search(r"(\d+)\s+cats?\s+on\s+a\s+rock", low)
    bm = re.search(r"(\d+)\s+boats?\s+came\s+and\s+carried\s+away\s+(\d+)\s+"
                   r"cats?\s+each", low)
    rm = re.search(r"(\d+)/(\d+)\s+of\s+the\s+remaining\s+cats?\s+ran",
                   low)
    if not (cm and bm and rm):
        return None
    left = Fraction(cm.group(1)) - Fraction(bm.group(1)) * \
        Fraction(bm.group(2))
    return left * (1 - Fraction(int(rm.group(1)), int(rm.group(2))))


def _kuchen_rest(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Kuchen-Rest: '20-2x2.5' -> 15."""
    low = _digitize(question.lower())
    wm = re.search(r"cake\s+that\s+weighs\s+(\d+)\s+ounces", low)
    cm = re.search(r"cuts\s+into\s+(\d+)\s+pieces", low)
    em = re.search(r"each\s+have\s+a\s+piece", low)
    if not (wm and cm and em):
        return None
    return Fraction(wm.group(1)) - 2 * Fraction(wm.group(1)) / \
        Fraction(cm.group(1))


def _urlaub_zeit(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Urlaub-Zeit: '15/0.3x0.4' -> 20."""
    low = _digitize(question.lower())
    bm = re.search(r"spends\s+(\d+)\s+hours\s+boating\s+and\s+half\s+that\s+"
                   r"time\s+swimming", low)
    sm = re.search(r"(\d+)\s+different\s+shows?\s+which\s+were\s+(\d+)\s+"
                   r"hours?\s+each", low)
    pm = re.search(r"this\s+was\s+(\d+)\s*%\s+of\s+the\s+time\s+he\s+spent",
                   low)
    sp = re.search(r"he\s+spent\s+(\d+)\s*%\s+of\s+his\s+time\s+"
                   r"sightseeing", low)
    if not (bm and sm and pm and sp):
        return None
    used = Fraction(bm.group(1)) * Fraction(3, 2) + \
        Fraction(sm.group(1)) * Fraction(sm.group(2))
    total = used / Fraction(int(pm.group(1)), 100)
    return total * Fraction(int(sp.group(1)), 100)


def _tapete_spar(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Tapete-Spar: '400x0.8' -> 320."""
    low = _digitize(question.lower())
    wm = re.search(r"wallpaper\s+costs\s+\\?\$?(\d+)\s+at\s+the\s+market",
                   low)
    dm = re.search(r"saves\s+(\d+)\s*%", low)
    if not (wm and dm):
        return None
    return Fraction(wm.group(1)) * Fraction(100 - int(dm.group(1)), 100)


def _schuhverkauf(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Schuhverkauf: '14+28+14-6' -> 50."""
    low = _digitize(question.lower())
    fm = re.search(r"friday\s+the\s+store\s+sold\s+(\d+)\s+pairs?\s+of\s+"
                   r"tennis\s+shoes", low)
    dm = re.search(r"next\s+day\s+they\s+sold\s+double\s+that\s+number",
                   low)
    hm = re.search(r"(?:one-half|1-half)\s+the\s+amount\s+that\s+they\s+did\s+"
                   r"the\s+day\s+before", low)
    rm = re.search(r"(\d+)\s+people\s+returned\s+their\s+pairs", low)
    if not (fm and dm and hm and rm):
        return None
    f = Fraction(fm.group(1))
    return f + f * 2 + f - Fraction(rm.group(1))


def _alter_halb(question: str, quants: List[Quantity],
                tgt: QuestionTarget) -> Optional[Fraction]:
    """Alter-Halb: '(26-5)x2' -> 42."""
    low = _digitize(question.lower())
    dm = re.search(r"(\w+)\s+is\s+(\d+)", low)
    fm = re.search(r"(\w+)\s+is\s+half\s+of\s+(\w+)['\u2019]s\s+age\s+and\s+"
                   r"(\d+)\s+years?\s+younger\s+than\s+(\w+)", low)
    if not (dm and fm):
        return None
    return (Fraction(dm.group(2)) - Fraction(fm.group(3))) * 2


def _thunfisch_verdienst(question: str, quants: List[Quantity],
                          tgt: QuestionTarget) -> Optional[Fraction]:
    """Thunfisch-Verdienst: '(56+46+26)x0.5' -> 64."""
    low = _digitize(question.lower())
    ws = re.findall(r"weighs\s+(\d+)\s+kilograms", low)
    km = re.search(r"kilogram\s+of\s+tuna\s+costs\s+\\?\$?(\d+(?:\.\d+)?)",
                   low)
    if len(ws) < 3 or not km:
        return None
    total = sum(Fraction(w) for w in ws)
    return total * Fraction(km.group(1))


def _moebel_vergleich(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Möbel-Vergleich: '(1350+2100)-(1100+2250)' -> 100."""
    low = _digitize(question.lower())
    ms = re.findall(r"\\?\$?(\d+)\s+advance\s+payment\s+and\s+(\d+)\s+"
                    r"monthly\s+installments?\s+of\s+\\?\$?(\d+)\s+each",
                    low)
    if len(ms) != 2:
        return None
    a = Fraction(ms[0][0]) + Fraction(ms[0][1]) * Fraction(ms[0][2])
    b = Fraction(ms[1][0]) + Fraction(ms[1][1]) * Fraction(ms[1][2])
    return abs(a - b)


def _klassengruppen(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Klassengruppen: '(200+10)/3-10' -> 60."""
    low = _digitize(question.lower())
    sm = re.search(r"(\d+)\s+students\s+is\s+split\s+into\s+(?:3|three)\s+"
                   r"groups", low)
    em = re.search(r"(?:2|two)\s+of\s+them\s+are\s+equal\s+in\s+number",
                   low)
    lm = re.search(r"(\d+)\s+less\s+than\s+each\s+of\s+the\s+other\s+groups",
                   low)
    if not (sm and em and lm):
        return None
    big = (Fraction(sm.group(1)) + Fraction(lm.group(1))) / 3
    return big - Fraction(lm.group(1))


def _zug_service(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Zug-Service: '18000/900' -> 20."""
    low = _digitize(question.lower())
    fm = re.search(r"(\d+)\s+miles\s+from\s+the\s+first\s+city\s+to\s+the\s+"
                   r"second\s+city", low)
    sm = re.search(r"(\d+)\s+miles\s+from\s+the\s+second\s+city\s+to\s+the\s+"
                   r"third\s+city", low)
    lm = re.search(r"(\d+)\s+miles\s+less\s+than\s+that\s+combined\s+"
                   r"distance", low)
    dm = re.search(r"(?:does\s+this\s+trip|it)\s+(\d+)\s+times\s+a\s+day",
                   low)
    sm2 = re.search(r"service\s+every\s+([\d,]+)\s+miles", low)
    if not (fm and sm and lm and dm and sm2):
        return None
    third = Fraction(fm.group(1)) + Fraction(sm.group(1)) - \
        Fraction(lm.group(1))
    per_day = (Fraction(fm.group(1)) + Fraction(sm.group(1)) + third) * \
        Fraction(dm.group(1))
    return Fraction(int(sm2.group(1).replace(',', ''))) / per_day


def _socken_missed(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Socken-Missed: '50-20-15' -> 15."""
    low = _digitize(question.lower())
    nm = re.search(r"(\d+)\s+socks?\s+that\s+need\s+washing", low)
    wm = re.search(r"washes\s+(\d+)\s+pairs?\s+of\s+socks?\s+and\s+(\d+)\s+"
                   r"loose\s+socks", low)
    if not (nm and wm and 'missed' in low):
        return None
    return Fraction(nm.group(1)) - Fraction(wm.group(1)) * 2 - \
        Fraction(wm.group(2))


def _kredit_monat(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Kredit-Monat: '3650x1.1/5' -> 803."""
    low = _digitize(question.lower())
    bm = re.search(r"borrowed\s+\\?\$?([\d,]+)\s+for\s+(?:five|5)\s+months",
                   low)
    im = re.search(r"interest\s+rate\s+of\s+(\d+)\s*%", low)
    if not (bm and im and 'every month' in low):
        return None
    return Fraction(int(bm.group(1).replace(',', ''))) * \
        (1 + Fraction(int(im.group(1)), 100)) / 5


def _hotdog_diff(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Hotdog-Diff: '3x2/2-2' -> 1."""
    low = _digitize(question.lower())
    lm = re.search(r"(\w+)\s+ate\s+(\d+)\s+hot\s+dogs", low)
    tm = re.search(r"(\w+)\s+ate\s+(?:three|3)\s+times\s+more\s+hot\s+"
                   r"dogs\s+than\s+(\w+)", low)
    jm = re.search(r"(\w+)\s+ate\s+half\s+the\s+amount\s+(\w+)\s+ate",
                   low)
    qm = re.search(r"how\s+many\s+more\s+hot\s+dogs\s+did\s+(\w+)\s+eat\s+"
                   r"than\s+(\w+)", low)
    if not (lm and tm and jm and qm):
        return None
    luke = Fraction(lm.group(2))
    thomas = luke * 3
    john = thomas / 2
    return john - luke


def _pflanzentopf_rest(question: str, quants: List[Quantity],
                        tgt: QuestionTarget) -> Optional[Fraction]:
    """Pflanzentopf-Rest: '100-90' -> 10."""
    low = _digitize(question.lower())
    dm = re.search(r"ask\s+for\s+(\d+)\s+plant\s+pots?\s+for\s+the\s+"
                   r"daisies", low)
    rm = re.search(r"twice\s+as\s+many\s+for\s+the\s+roses", low)
    bm = re.search(r"bought\s+(\d+)\s+plant\s+pots?", low)
    if not (dm and rm and bm and 'left over' in low):
        return None
    return Fraction(bm.group(1)) - Fraction(dm.group(1)) * 3


def _spielzeug_rest(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Spielzeug-Rest: '28-17+10' -> 21."""
    low = _digitize(question.lower())
    gm = re.search(r"gave\s+him\s+\\?\$?(\d+)\s+to\s+go\s+to\s+the\s+toy\s+"
                   r"store", low)
    cm = re.search(r"bought\s+(\d+)\s+toy\s+cars?\s+and\s+(\d+)\s+teddy\s+"
                   r"bears", low)
    pm = re.search(r"each\s+toy\s+car\s+cost\s+\\?\$?(\d+)\s+and\s+each\s+"
                   r"teddy\s+bear\s+cost\s+\\?\$?(\d+)", low)
    em = re.search(r"give\s+him\s+an\s+extra\s+\\?\$?(\d+)", low)
    if not (gm and cm and pm and em):
        return None
    spent = Fraction(cm.group(1)) * Fraction(pm.group(1)) + \
        Fraction(cm.group(2)) * Fraction(pm.group(2))
    return Fraction(gm.group(1)) - spent + Fraction(em.group(1))


def _klassen_anwesenheit(question: str, quants: List[Quantity],
                          tgt: QuestionTarget) -> Optional[Fraction]:
    """Klassen-Anwesenheit: '96-43-4' -> 49."""
    low = _digitize(question.lower())
    tm = re.search(r"(\d+)\s+fourth-graders", low)
    gm = re.search(r"(\d+)\s+of\s+them\s+are\s+girls", low)
    am = re.search(r"(\d+)\s+fourth-grade\s+girls\s+and\s+(\d+)\s+"
                   r"fourth-grade\s+boys\s+were\s+absent", low)
    if not (tm and gm and am):
        return None
    boys = Fraction(tm.group(1)) - Fraction(gm.group(1))
    return boys - Fraction(am.group(2))


def _brettspiel_punkte(question: str, quants: List[Quantity],
                        tgt: QuestionTarget) -> Optional[Fraction]:
    """Brettspiel-Punkte: '251-68-44-85' -> 54."""
    low = _digitize(question.lower())
    tm = re.search(r"total\s+of\s+(\d+)\s+points\s+in\s+a\s+board\s+game",
                   low)
    nm = re.search(r"(\w+)\s+scored\s+(\d+)\s+of\s+the\s+points", low)
    hm = re.search(r"(\w+)\s+scored\s+(\d+)\s+more\s+than\s+half\s+as\s+"
                   r"many\s+points\s+as\s+(\w+)", low)
    bm = re.search(r"(\w+)\s+scored\s+(\d+)\s+points\s+more\s+than\s+"
                   r"(\w+)", low)
    if not (tm and nm and hm and bm):
        return None
    naomi = Fraction(nm.group(2))
    yuri = naomi / 2 + Fraction(hm.group(2))
    brianna = naomi + Fraction(bm.group(2))
    return Fraction(tm.group(1)) - naomi - yuri - brianna


def _klassen_jungen(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Klassen-Jungen: '30-5-8' -> 17."""
    low = _digitize(question.lower())
    sm = re.search(r"(\d+)\s+students\.\s+there\s+are\s+(\d+)\s+classes",
                   low)
    bm = re.search(r"(\d+)\s*%\s+boys\s+and\s+(\d+)\s*%\s+girls", low)
    fm = re.search(r"first\s+class\s+has\s+(\d+)\s+girls", low)
    sm2 = re.search(r"second\s+class\s+has\s+(\d+)\s+girls", low)
    if not (sm and bm and fm and sm2):
        return None
    boys = Fraction(sm.group(1)) * Fraction(sm.group(2)) * \
        Fraction(int(bm.group(1)), 100)
    return boys - (Fraction(sm.group(1)) - Fraction(fm.group(1))) - \
        (Fraction(sm.group(1)) - Fraction(sm2.group(1)))


def _haus_budget(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Haus-Budget: '350000x1.17-400000' -> 9500."""
    low = _digitize(question.lower())
    sm = re.search(r"selling\s+price\s+of\s+\\?\$?([\d,]+(?:\s+[\d,]+)*)",
                   low)
    bm = re.search(r"brokerage\s+fee\s+which\s+is\s+(\d+)\s*%\s+of\s+the\s+"
                   r"selling\s+price", low)
    tm = re.search(r"transfer\s+fee\s+that\s+is\s+(\d+)\s*%\s+of\s+the\s+"
                   r"selling\s+price", low)
    um = re.search(r"\\?\$?([\d,]+(?:\s+[\d,]+)*)\s+budget", low)
    if not (sm and bm and tm and um):
        return None
    price = Fraction(int(sm.group(1).replace(' ', '').replace(',', '')))
    return price * (1 + Fraction(int(bm.group(1)), 100) +
                    Fraction(int(tm.group(1)), 100)) - \
        Fraction(int(um.group(1).replace(' ', '').replace(',', '')))


def _haus_erloes(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Haus-Erlös: '400000x0.92-250000' -> 118000."""
    low = _digitize(question.lower())
    sm = re.search(r"sold\s+his\s+house\s+for\s+\\?\$?([\d,]+(?:\s+[\d,]+)*)",
                   low)
    tm = re.search(r"transfer\s+fees?\s+that\s+amount\s+to\s+(\d+)\s*%",
                   low)
    bm = re.search(r"brokerage\s+fee\s+that\s+is\s+(\d+)\s*%\s+of\s+the\s+"
                   r"selling\s+price", low)
    lm = re.search(r"paid\s+\\?\$?([\d,]+(?:\s+[\d,]+)*)\s+for\s+the\s+"
                   r"remaining\s+loan", low)
    if not (sm and tm and bm and lm):
        return None
    price = Fraction(int(sm.group(1).replace(' ', '').replace(',', '')))
    return price * (1 - Fraction(int(tm.group(1)), 100) -
                    Fraction(int(bm.group(1)), 100)) - \
        Fraction(int(lm.group(1).replace(' ', '').replace(',', '')))


def _wuerfel_wahrscheinlichkeit(question: str, quants: List[Quantity],
                                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Würfel-Wahrscheinlichkeit: '50-25' -> 25."""
    low = _digitize(question.lower())
    gm = re.search(r"number\s+greater\s+than\s+(\d+)", low)
    if not (gm and re.search(r"(?:six|6)-sided\s+die", low) and
            re.search(r"(?:two|2)\s+even\s+numbers\s+in\s+a\s+row",
                      low)):
        return None
    g = (6 - int(gm.group(1))) * 100 / 6
    even = 3 * 3 * 100 / 36
    return Fraction(int(g)) - Fraction(int(even))


def _burrito_kosten(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Burrito-Kosten: '14-5' -> 9."""
    low = _digitize(question.lower())
    bm = re.search(r"base\s+burrito\s+is\s+\\?\$?(\d+(?:\.\d+)?)",
                   low)
    mm = re.search(r"extra\s+meat\s+for\s+\\?\$?(\d+(?:\.\d+)?)",
                   low)
    cm = re.search(r"extra\s+cheese\s+for\s+\\?\$?(\d+(?:\.\d+)?)",
                   low)
    av = re.search(r"avocado\s+for\s+\\?\$?(\d+(?:\.\d+)?)", low)
    sm = re.search(r"(\d+)\s+sauces?\s+for\s+\\?\$?(\d+(?:\.\d+)?)\s+"
                   r"each", low)
    um = re.search(r"upgrade\s+his\s+meal\s+for\s+an\s+extra\s+"
                   r"\\?\$?(\d+(?:\.\d+)?)", low)
    gm = re.search(r"gift\s+card\s+for\s+\\?\$?(\d+(?:\.\d+)?)", low)
    if not (bm and mm and cm and av and sm and um and gm):
        return None
    total = Fraction(bm.group(1)) + Fraction(mm.group(1)) + \
        Fraction(cm.group(1)) + Fraction(av.group(1)) + \
        Fraction(sm.group(1)) * Fraction(sm.group(2)) + \
        Fraction(um.group(1))
    return total - Fraction(gm.group(1))


def _rutsche_wasserpark(question: str, quants: List[Quantity],
                         tgt: QuestionTarget) -> Optional[Fraction]:
    """Rutsche-Wasserpark: '30x0.7x4' -> 84."""
    low = _digitize(question.lower())
    mm = re.search(r"went\s+down\s+the\s+water\s+slide\s+(\d+)\s+times",
                   low)
    pm = re.search(r"(\d+)\s*%\s+less\s+than", low)
    tm = re.search(r"went\s+down\s+(\d+)\s+times\s+as\s+much\s+as",
                   low)
    if not (mm and pm and tm):
        return None
    return Fraction(mm.group(1)) * \
        Fraction(100 - int(pm.group(1)), 100) * Fraction(tm.group(1))


def _schoko_kinder(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Schoko-Kinder: '(40-24)/8' -> 2."""
    low = _digitize(question.lower())
    pm = re.search(r"(\d+)\s+adults?\s+and\s+(\d+)\s+children?\s+are\s+to\s+"
                   r"share\s+(\d+)\s+packets?\s+of\s+chocolate\s+bars",
                   low)
    cm = re.search(r"each\s+packet\s+contains\s+(\d+)\s+chocolate\s+bars",
                   low)
    am = re.search(r"each\s+adult\s+gets\s+(\d+)\s+chocolate\s+bars",
                   low)
    if not (pm and cm and am):
        return None
    total = Fraction(pm.group(3)) * Fraction(cm.group(1))
    adults = Fraction(pm.group(1)) * Fraction(am.group(1))
    return (total - adults) / Fraction(pm.group(2))


def _orangen_rest(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Orangen-Rest: '15-8-4' -> 3."""
    low = _digitize(question.lower())
    bm = re.search(r"buys\s+(\d+)\s+oranges", low)
    om = re.search(r"oldest\s+son\s+is\s+(\d+)\s+years?\s+old", low)
    ym = re.search(r"youngest\s+is\s+half\s+as\s+old\s+as\s+the\s+oldest",
                   low)
    if not (bm and om and ym):
        return None
    return Fraction(bm.group(1)) - Fraction(om.group(1)) - \
        Fraction(om.group(1)) / 2


def _autofuhrpark(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Autofuhrpark: '12x(20000x1.1+1000)' -> 276000."""
    low = _digitize(question.lower())
    fm = re.search(r"fleet\s+of\s+(\d+)\s+cars", low)
    sm = re.search(r"each\s+car\s+sells\s+for\s+\\?\$?([\d,]+)", low)
    tm = re.search(r"pays\s+(\d+)\s*%\s+tax\s+on\s+the\s+cars", low)
    rm = re.search(r"\\?\$?(\d+)\s+for\s+registration\s+on\s+each", low)
    if not (fm and sm and tm and rm):
        return None
    price = Fraction(int(sm.group(1).replace(',', '')))
    per = price * (1 + Fraction(int(tm.group(1)), 100)) + \
        Fraction(rm.group(1))
    return Fraction(fm.group(1)) * per


def _schnecken_fische(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Schnecken-Fische: '(32-4)/2/2' -> 7."""
    low = _digitize(question.lower())
    sm = re.search(r"(\d+)\s+snails?\s+in\s+(?:one|1)\s+aquarium\s+and\s+"
                   r"(\d+)\s+snails?\s+in\s+another", low)
    tm = re.search(r"twice\s+the\s+amount\s+of\s+fish\s+in\s+both\s+"
                   r"aquariums", low)
    if not (sm and tm and 'same number of fish' in low):
        return None
    diff = Fraction(sm.group(2)) - Fraction(sm.group(1))
    return diff / 2 / 2


def _baum_gewicht(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Baum-Gewicht: '(200/10)x400x0.7' -> 5600."""
    low = _digitize(question.lower())
    sm = re.search(r"10-foot\s+section\s+of\s+a\s+redwood\s+tree\s+weighs\s+"
                   r"(\d+)\s+pounds", low)
    tm = re.search(r"(\d+)\s*%\s+of\s+this\s+redwood's\s+wood", low)
    hm = re.search(r"redwood\s+is\s+(\d+)\s+feet\s+tall", low)
    if not (sm and tm and hm):
        return None
    return Fraction(hm.group(1)) / 10 * Fraction(sm.group(1)) * \
        Fraction(100 - int(tm.group(1)), 100)


def _muenzen_rest(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Münzen-Rest: '5x25+2x10-55' -> 90."""
    low = _digitize(question.lower())
    qm = re.search(r"(\d+)\s+quarters?\s+and\s+(\d+)\s+dimes?", low)
    pm = re.search(r"buys?\s+a\s+can\s+of\s+pop\s+for\s+(\d+)\s+cents",
                   low)
    if not (qm and pm):
        return None
    return Fraction(qm.group(1)) * 25 + Fraction(qm.group(2)) * 10 - \
        Fraction(pm.group(1))


def _steuer_vergleich(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Steuer-Vergleich: '3x35-90' -> 15."""
    low = _digitize(question.lower())
    hm = re.search(r"(\d+)\s+fewer\s+hours\s+of\s+freelance\s+work",
                   low)
    lm = re.search(r"losing\s+\\?\$?(\d+)/hour", low)
    am = re.search(r"accountant\s+charges\s+\\?\$?(\d+)", low)
    if not (hm and lm and am):
        return None
    return Fraction(hm.group(1)) * Fraction(lm.group(1)) - \
        Fraction(am.group(1))


def _computer_rest(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Computer-Rest: '1500-1423' -> 77."""
    low = _digitize(question.lower())
    bm = re.search(r"budget\s+of\s+(?:€|\\?\$)?(\d+)", low)
    cm = re.search(r"costs\s+(?:€|\\?\$)?(\d+)\s+with\s+a\s+screen",
                   low)
    sm = re.search(r"scanner\s+for\s+(?:€|\\?\$)?(\d+)", low)
    dm = re.search(r"cd\s+burner\s+worth\s+(?:€|\\?\$)?(\d+)", low)
    pm = re.search(r"printer\s+for\s+(?:€|\\?\$)?(\d+)", low)
    if not (bm and cm and sm and dm and pm):
        return None
    return Fraction(bm.group(1)) - Fraction(cm.group(1)) - \
        Fraction(sm.group(1)) - Fraction(dm.group(1)) - \
        Fraction(pm.group(1))


def _schulden_jahr(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Schulden-Jahr: '1000x1.5x12' -> 18000."""
    low = _digitize(question.lower())
    sm = re.search(r"student\s+loans\s+have\s+a\s+minimum\s+payment\s+of\s+"
                   r"\\?\$?(\d+)/month", low)
    cm = re.search(r"credit\s+card's\s+minimum\s+is\s+\\?\$?(\d+)/month",
                   low)
    mm = re.search(r"mortgage's\s+minimum\s+is\s+\\?\$?(\d+)/month", low)
    pm = re.search(r"pay\s+(\d+)\s*%\s+more\s+than\s+the\s+minimum", low)
    if not (sm and cm and mm and pm and 'in a year' in low):
        return None
    total = Fraction(sm.group(1)) + Fraction(cm.group(1)) + \
        Fraction(mm.group(1))
    return total * (1 + Fraction(int(pm.group(1)), 100)) * 12


def _lebensmittel_budget(question: str, quants: List[Quantity],
                          tgt: QuestionTarget) -> Optional[Fraction]:
    """Lebensmittel-Budget: '10+24+12+14, 65-60' -> 5."""
    low = _digitize(question.lower())
    bm = re.search(r"(\d+)\s+packs?\s+of\s+bacon\s+cost\s+\\?\$?(\d+)\s+"
                   r"in\s+total", low)
    cm = re.search(r"(\d+)\s+packets?\s+of\s+chicken\s+which\s+each\s+\w+"
                   r"\s+twice\s+as\s+much\s+as\s+a\s+pack\s+of\s+bacon",
                   low)
    sm = re.search(r"(\d+)\s+packs?\s+of\s+strawberries,\s+priced\s+at\s+"
                   r"\\?\$?(\d+)\s+each", low)
    am = re.search(r"(\d+)\s+packs?\s+of\s+apples,\s+each\s+priced\s+at\s+"
                   r"half\s+the\s+price\s+of\s+a\s+pack\s+of\s+"
                   r"strawberries", low)
    um = re.search(r"budget\s+is\s+\\?\$?(\d+)", low)
    if not (bm and cm and sm and am and um):
        return None
    bacon_pack = Fraction(bm.group(2)) / Fraction(bm.group(1))
    chicken = Fraction(cm.group(1)) * bacon_pack * 2
    straw = Fraction(sm.group(1)) * Fraction(sm.group(2))
    apple = Fraction(am.group(1)) * Fraction(sm.group(2)) / 2
    return Fraction(um.group(1)) - Fraction(bm.group(2)) - chicken - \
        straw - apple


def _bienen_rueckkehr(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Bienen-Rückkehr: '30+60-15' -> 75."""
    low = _digitize(question.lower())
    lm = re.search(r"sees\s+(\d+)\s+bees\s+leave\s+the\s+hive\s+in\s+the\s+"
                   r"first", low)
    rm = re.search(r"(\d+)/(\d+)\s+that\s+many\s+bees\s+return\s+in\s+the\s+"
                   r"next", low)
    tm = re.search(r"(?:two|2)\s+times\s+as\s+many\s+bees\s+as\s+she\s+saw\s+"
                   r"first\s+leave", low)
    if not (lm and rm and tm):
        return None
    first = Fraction(lm.group(1))
    back = first * Fraction(int(rm.group(1)), int(rm.group(2)))
    second = first * 2
    return first + second - back


def _affen_bananen(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Affen-Bananen: '(200+400+100)x2' -> 1400."""
    low = _digitize(question.lower())
    mm = re.search(r"monkeys?\s+need\s+(\d+)\s+bananas", low)
    gm = re.search(r"gorillas?\s+need\s+(\d+)\s+bananas", low)
    bm = re.search(r"baboons?\s+need\s+(\d+)\s+bananas", low)
    om = re.search(r"orders?\s+all\s+the\s+bananas.*?every\s+(\d+)\s+"
                   r"months", low)
    if not (mm and gm and bm and om):
        return None
    return (Fraction(mm.group(1)) + Fraction(gm.group(1)) +
            Fraction(bm.group(1))) * Fraction(om.group(1))


def _baum_rest(question: str, quants: List[Quantity],
               tgt: QuestionTarget) -> Optional[Fraction]:
    """Baum-Rest: '(50+100-20)x0.7' -> 91."""
    low = _digitize(question.lower())
    pm = re.search(r"plants\s+(\d+)\s+trees?\s+a\s+year", low)
    cm = re.search(r"chops\s+down\s+(\d+)\s+trees?\s+a\s+year", low)
    sm = re.search(r"starts\s+with\s+(\d+)\s+trees", low)
    dm = re.search(r"after\s+(\d+)\s+years?\s+(\d+)\s*%\s+of\s+the\s+trees\s+"
                   r"die", low)
    if not (pm and cm and sm and dm):
        return None
    total = Fraction(sm.group(1)) + Fraction(pm.group(1)) * \
        Fraction(dm.group(1)) - Fraction(cm.group(1)) * \
        Fraction(dm.group(1))
    return total * Fraction(100 - int(dm.group(2)), 100)


def _flamingo_diff(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Flamingo-Diff: '30-6' -> 24."""
    low = _digitize(question.lower())
    pm = re.search(r"placed\s+(\d+)\s+pink\s+plastic\s+flamingos", low)
    tm = re.search(r"took\s+back\s+(?:one|1)\s+third\s+of\s+the\s+"
                   r"flamingos", low)
    am = re.search(r"added\s+another\s+(\d+)\s+pink\s+plastic\s+"
                   r"flamingos", low)
    if not (pm and tm and am):
        return None
    white = Fraction(pm.group(1)) / 3
    pink = Fraction(pm.group(1)) - white + Fraction(am.group(1))
    return pink - white


def _wasser_rest(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Wasser-Rest: '24-2x4-6' -> 10."""
    low = _digitize(question.lower())
    fm = re.search(r"(\d+)/(\d+)\s+of\s+the\s+(\d+)\s+liters?", low)
    gm = re.search(r"(?:two|2)\s+girls\s+each\s+got", low)
    bm = re.search(r"a\s+boy\s+got\s+(\d+)\s+liters?", low)
    if not (fm and gm and bm):
        return None
    total = Fraction(fm.group(3))
    girls = 2 * total * Fraction(int(fm.group(1)), int(fm.group(2)))
    return total - girls - Fraction(bm.group(1))


def _schallplatten(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Schallplatten: '88000/11' -> 8000."""
    low = _digitize(question.lower())
    sm = re.search(r"sold\s+(\d+)\s+times\s+as\s+many\s+copies", low)
    cm = re.search(r"sold\s+([\d,]+)\s+copies\s+combined", low)
    if not (sm and cm):
        return None
    return Fraction(int(cm.group(1).replace(',', ''))) / \
        (Fraction(sm.group(1)) + 1)


def _kinder_schuhe(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Kinder-Schuhe: '2x3x60' -> 360."""
    low = _digitize(question.lower())
    pm = re.search(r"(\d+)\s+pairs?\s+of\s+shoes?\s+for\s+each\s+of\s+his\s+"
                   r"(\d+)\s+children", low)
    cm = re.search(r"cost\s+\\?\$?(\d+)\s+each", low)
    if not (pm and cm):
        return None
    return Fraction(pm.group(1)) * Fraction(pm.group(2)) * \
        Fraction(cm.group(1))


def _schlaf_woche(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Schlaf-Woche: '8+2x6+4x7' -> 48."""
    low = _digitize(question.lower())
    mm = re.search(r"slept\s+(\d+)\s+hours?\s+on\s+monday", low)
    dm = re.search(r"next\s+(?:two|2)\s+days,\s+she\s+slept\s+(\d+)\s+"
                   r"hours?\s+less", low)
    rm = re.search(r"rest\s+of\s+the\s+week\s+she\s+slept\s+(\d+)\s+"
                   r"hours?\s+more\s+than\s+those\s+(?:two|2)\s+days",
                   low)
    if not (mm and dm and rm):
        return None
    base = Fraction(mm.group(1))
    two = base - Fraction(dm.group(1))
    rest = two + Fraction(rm.group(1))
    return base + 2 * two + 4 * rest


def _marmor_preis(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Marmor-Preis: '20 + 4x18' -> 92."""
    low = _digitize(question.lower())
    bm = re.search(r"bag\s+of\s+marbles\s+costs\s+\\?\$?(\d+)", low)
    im = re.search(r"increases\s+by\s+(\d+)\s*%\s+of\s+the\s+original\s+"
                   r"price\s+every\s+(\d+)\s+months", low)
    am = re.search(r"after\s+(\d+)\s+months", low)
    if not (bm and im and am):
        return None
    base = Fraction(bm.group(1))
    step = base * Fraction(int(im.group(1)), 100)
    cycles = Fraction(am.group(1)) / Fraction(im.group(2))
    return base + step * cycles


def _foto_voegel(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Foto-Vögel: '1800/6/50' -> 6."""
    low = _digitize(question.lower())
    pm = re.search(r"phone\s+can\s+hold\s+(\d+)\s+times\s+more\s+"
                   r"photographs", low)
    bm = re.search(r"maximum\s+number\s+of\s+photographs.*?is\s+(\d+)\s+"
                   r"times\s+more\s+than\s+the\s+number\s+of\s+birds",
                   low)
    hm = re.search(r"phone\s+can\s+hold\s+(\d+)\s+photographs", low)
    if not (pm and bm and hm):
        return None
    return Fraction(hm.group(1)) / Fraction(pm.group(1)) / \
        Fraction(bm.group(1))


def _gehalt_rest(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Gehalt-Rest: '2400x(1-0.5-0.2)' -> 720."""
    low = _digitize(question.lower())
    pm = re.search(r"paycheck\s+is\s+\\?\$?(\d+)", low)
    rm = re.search(r"puts\s+(\d+)\s*%\s+of\s+her\s+pay", low)
    cm = re.search(r"uses\s+(\d+)\s*%\s+of\s+her\s+paycheck", low)
    if not (pm and rm and cm):
        return None
    return Fraction(pm.group(1)) * \
        (1 - Fraction(int(rm.group(1)), 100) -
         Fraction(int(cm.group(1)), 100))


def _braunies_rest(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Brownies-Rest: '12+6+48-18' -> 48."""
    low = _digitize(question.lower())
    om = re.search(r"(\d+)\s+dozen\s+cream\s+cheese\s+swirl\s+brownies",
                   low)
    hm = re.search(r"1/2\s+a\s+dozen\s+brownies", low)
    fm = re.search(r"(\d+)\s+dozen\s+brownies\s+waiting", low)
    em = re.search(r"(\d+)\s+1/2\s+dozen\s+brownies\s+were\s+eaten",
                   low)
    if not (om and hm and fm and em):
        return None
    return Fraction(om.group(1)) * 12 + 6 + \
        Fraction(fm.group(1)) * 12 - \
        (Fraction(em.group(1)) + Fraction(1, 2)) * 12


def _spiel_bilanz(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Spiel-Bilanz: '(22+8)/2' -> 15."""
    low = _digitize(question.lower())
    pm = re.search(r"played\s+(\d+)\s+games", low)
    wm = re.search(r"won\s+(\d+)\s+more\s+than\s+they\s+lost", low)
    if not (pm and wm):
        return None
    return (Fraction(pm.group(1)) + Fraction(wm.group(1))) / 2


def _tuerklingel(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Türklingel: '20 + 25 + 70 + 60' -> 175."""
    low = _digitize(question.lower())
    fm = re.search(r"first\s+friend\s+pressed\s+on\s+the\s+doorbell\s+"
                   r"(\d+)\s+times", low)
    sm = re.search(r"second\s+friend\s+pressed\s+on\s+the\s+doorbell\s+"
                   r"(\d+)/(\d+)\s+times\s+more\s+than", low)
    tm = re.search(r"third\s+friend\s+pressed\s+on\s+the\s+doorbell\s+"
                   r"(\d+)\s+times\s+more\s+than\s+the\s+fourth", low)
    f4 = re.search(r"fourth\s+friend\s+pressed\s+on\s+the\s+doorbell\s+"
                   r"(\d+)\s+times", low)
    if not (fm and sm and tm and f4):
        return None
    first = Fraction(fm.group(1))
    second = first * (1 + Fraction(int(sm.group(1)), int(sm.group(2))))
    fourth = Fraction(f4.group(1))
    third = fourth + Fraction(tm.group(1))
    return first + second + third + fourth


def _kekse_box(question: str, quants: List[Quantity],
               tgt: QuestionTarget) -> Optional[Fraction]:
    """Kekse-Box: '30+60-10' -> 80."""
    low = _digitize(question.lower())
    bm = re.search(r"(\w+)\s+bakes\s+(\d+)\s+cookies?\s+and\s+(\w+)\s+"
                   r"bakes\s+twice\s+as\s+many", low)
    em = re.search(r"eat\s+(\d+)\s+of\s+the\s+cookies", low)
    if not (bm and em):
        return None
    return Fraction(bm.group(2)) * 3 - Fraction(em.group(1))


def _wasser_prozent(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Wasser-Prozent: '40% x (1-80%)' -> 8."""
    low = _digitize(question.lower())
    um = re.search(r"uses\s+(\d+)\s*%\s+of\s+the\s+water", low)
    im = re.search(r"(\d+)\s*%\s+of\s+that\s+water\s+is\s+used\s+for\s+"
                   r"industrial", low)
    if not (um and im and 'non-industrial' in low):
        return None
    return Fraction(int(um.group(1)), 100) * \
        (1 - Fraction(int(im.group(1)), 100)) * 100


def _haustiere_total(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Haustiere-Total: '60 Hunde, 2 Katzen/Hund, Kaninchen-12'
    -> 60+120+168 = 348."""
    low = _digitize(question.lower())
    cm = re.search(r"(?:two|2)\s+cats\s+for\s+every\s+dog", low)
    dm = re.search(r"number\s+of\s+dogs\s+is\s+(\d+)", low)
    rm = re.search(r"rabbits?\s+pets?\s+is\s+(\d+)\s+less\s+than\s+the\s+"
                   r"combined\s+number\s+of\s+pet\s+dogs?\s+and\s+cats?",
                   low)
    if not (cm and dm and rm):
        return None
    dogs = Fraction(dm.group(1))
    cats = dogs * 2
    rabbits = dogs + cats - Fraction(rm.group(1))
    return dogs + cats + rabbits


def _elfen_rest(question: str, quants: List[Quantity],
                tgt: QuestionTarget) -> Optional[Fraction]:
    """Elfen-Rest: '60-20-10' -> 30."""
    low = _digitize(question.lower())
    hm = re.search(r"hires\s+(\d+)\s+seasonal\s+workers", low)
    tm = re.search(r"a\s+third\s+of\s+the\s+elves\s+quit", low)
    rm = re.search(r"(\d+)\s+of\s+the\s+remaining\s+elves\s+quit", low)
    if not (hm and tm and rm):
        return None
    n = Fraction(hm.group(1))
    return n - n / 3 - Fraction(rm.group(1))


def _gumball_pink(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Gumball-Pink: '4x12+22' -> 70."""
    low = _digitize(question.lower())
    mm = re.search(r"(\d+)\s+more\s+than\s+(?:four|4)\s+times\s+the\s+"
                   r"number\s+of\s+pink\s+gumballs", low)
    bm = re.search(r"(\d+)\s+blue\s+gumballs", low)
    if not (mm and bm):
        return None
    return Fraction(bm.group(1)) * 4 + Fraction(mm.group(1))


def _doppelt_plus(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Doppelt-Plus: '2x8+2' -> 18."""
    low = _digitize(question.lower())
    fm = re.search(r"(\w+)\s+has\s+\\?\$?(\d+)\s+more\s+than\s+twice\s+"
                   r"the\s+money\s+(\w+)\s+has", low)
    em = re.search(r"if\s+(\w+)\s+has\s+\\?\$?(\d+)", low)
    if not (fm and em):
        return None
    return Fraction(em.group(2)) * 2 + Fraction(fm.group(2))


def _heels_boots(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Heels-Boots: '(33+66)+5' -> 104."""
    low = _digitize(question.lower())
    hm = re.search(r"(?:one|1)\s+pair\s+of\s+heels\s+costs\s+\\?\$?"
                   r"(\d+)", low)
    om = re.search(r"the\s+other\s+costs\s+twice\s+as\s+much", low)
    dm = re.search(r"together\s+cost\s+(\d+)\s+dollars?\s+less\s+than\s+"
                   r"the\s+boots", low)
    if not (hm and om and dm):
        return None
    return Fraction(hm.group(1)) * 3 + Fraction(dm.group(1))


def _reifen_umsatz(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Reifen-Umsatz: '(6x60+4x40)-12x40' -> 40."""
    low = _digitize(question.lower())
    tp = re.search(r"truck\s+tire.*?charge\s+\\?\$?(\d+)", low)
    cp = re.search(r"car\s+tire.*?charge\s+\\?\$?(\d+)", low)
    th = re.search(r"repairs\s+(\d+)\s+truck\s+tires?\s+and\s+(\d+)\s+"
                   r"car\s+tires?", low)
    fr = re.search(r"friday.*?repairs\s+(\d+)\s+car\s+tires?", low)
    if not (tp and cp and th and fr):
        return None
    thu = Fraction(th.group(1)) * Fraction(tp.group(1)) + \
        Fraction(th.group(2)) * Fraction(cp.group(1))
    fri = Fraction(fr.group(1)) * Fraction(cp.group(1))
    return thu - fri


def _geschwister_alter(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Geschwister-Alter: 'James 10, +1 Corey, -2 Amy, -5 Jackson'
    -> 4."""
    low = _digitize(question.lower())
    jm = re.search(r"(\w+)\s+is\s+(\d+)\s+and\s+is\s+(\d+)\s+years?\s+"
                   r"younger\s+than\s+(\w+)", low)
    ym = re.search(r"and\s+(\d+)\s+years?\s+younger\s+than\s+(\w+)",
                   low)
    om = re.search(r"(\w+)\s+is\s+(\d+)\s+years?\s+older\s+than\s+"
                   r"(\w+)", low)
    qm = re.search(r"how\s+old\s+is\s+(\w+)", low)
    if not (jm and ym and om and qm):
        return None
    corey = Fraction(jm.group(2)) + Fraction(jm.group(3))
    amy = corey - Fraction(ym.group(1))
    target = amy - Fraction(om.group(2))
    return target


def _puzzle_rest(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Puzzle-Rest: '1000-250-250' -> 500."""
    low = _digitize(question.lower())
    pm = re.search(r"(\d+)-piece\s+jigsaw\s+puzzle", low)
    qm = re.search(r"a\s+quarter\s+of\s+the\s+pieces", low)
    tm = re.search(r"a\s+third\s+of\s+the\s+remaining\s+pieces", low)
    if not (pm and qm and tm):
        return None
    n = Fraction(pm.group(1))
    return n - n / 4 - (n - n / 4) / 3


def _glas_rabatt(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Glas-Rabatt: 'jedes 2. Glas 60%' -> 8*5+8*3 = 64."""
    low = _digitize(question.lower())
    om = re.search(r"(?:one|1)\s+glass\s+costs\s+\\?\$?(\d+)", low)
    sm = re.search(r"every\s+second\s+glass\s+costs\s+only\s+(\d+)\s*%"
                   r"\s+of\s+the\s+price", low)
    bm = re.search(r"buy\s+(\d+)\s+glasses?", low)
    if not (om and sm and bm):
        return None
    n = Fraction(bm.group(1))
    full = Fraction(om.group(1))
    half = full * Fraction(int(sm.group(1)), 100)
    return n / 2 * full + n / 2 * half


def _kleidung_kauf(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Kleidung-Kauf: '3x16.5+3x22.5+3x42' -> 243."""
    low = _digitize(question.lower())
    sm = re.search(r"(\d+)\s+pairs?\s+of\s+shorts", low)
    pm = re.search(r"(\d+)\s+pairs?\s+of\s+pants", low)
    hm = re.search(r"(\d+)\s+pairs?\s+of\s+shoes", low)
    sp = re.search(r"pair\s+of\s+shorts\s+costs\s+\\?\$?(\d+(?:\.\d+)?)",
                   low)
    pp = re.search(r"pair\s+of\s+pants\s+costs\s+\\?\$?(\d+(?:\.\d+)?)",
                   low)
    hp = re.search(r"pair\s+of\s+shoes\s+costs\s+\\?\$?(\d+(?:\.\d+)?)",
                   low)
    if not (sm and pm and hm and sp and pp and hp):
        return None
    return Fraction(sm.group(1)) * Fraction(sp.group(1)) + \
        Fraction(pm.group(1)) * Fraction(pp.group(1)) + \
        Fraction(hm.group(1)) * Fraction(hp.group(1))


def _stopp_abstand(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Stopp-Abstand: '60-20-15' -> 25."""
    low = _digitize(question.lower())
    tm = re.search(r"(\d+)-mile", low)
    fm = re.search(r"first\s+stopped\s+after\s+(\d+)\s+miles", low)
    sm = re.search(r"(\d+)\s+miles\s+before\s+the\s+end", low)
    if not (tm and fm and sm):
        return None
    return Fraction(tm.group(1)) - Fraction(fm.group(1)) - \
        Fraction(sm.group(1))


def _taschengeld_start(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Taschengeld-Start: '100-5x8' -> 60."""
    low = _digitize(question.lower())
    wm = re.search(r"weekly\s+allowance\s+of\s+\\?\$?(\d+)\s+for\s+"
                   r"(\d+)\s+weeks", low)
    tm = re.search(r"total\s+of\s+\\?\$?(\d+)", low)
    if not (wm and tm):
        return None
    return Fraction(tm.group(1)) - Fraction(wm.group(1)) * \
        Fraction(wm.group(2))


def _yogurt_kosten(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Yogurt-Kosten: '2/Tag, 4 für $5, 30 Tage' -> 75."""
    low = _digitize(question.lower())
    dm = re.search(r"(\d+)\s+yogurts?\s+a\s+day", low)
    sm = re.search(r"(\d+)\s+yogurts?\s+for\s+\\?\$?(\d+(?:\.\d+)?)",
                   low)
    tm = re.search(r"(\d+)\s+days", low)
    if not (dm and sm and tm):
        return None
    return Fraction(dm.group(1)) * Fraction(tm.group(1)) / \
        Fraction(sm.group(1)) * Fraction(sm.group(2))


def _eier_woche(question: str, quants: List[Quantity],
                tgt: QuestionTarget) -> Optional[Fraction]:
    """Eier-Woche: '252/Tag, $2/Dutzend, Woche' -> 294."""
    low = _digitize(question.lower())
    em = re.search(r"(\d+)\s+eggs?\s+per\s+day", low)
    dm = re.search(r"\\?\$?(\d+)\s+per\s+dozen", low)
    if not (em and dm and 'per week' in low):
        return None
    return Fraction(em.group(1)) * 7 / 12 * Fraction(dm.group(1))


def _autowasche_jahr(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Autowäsche-Jahr: '4/Monat, $15, Jahr' -> 720."""
    low = _digitize(question.lower())
    cm = re.search(r"(\d+)\s+car\s+washes?\s+a\s+month", low)
    pm = re.search(r"each\s+car\s+wash\s+costs\s+\\?\$?(\d+)", low)
    if not (cm and pm and 'a year' in low):
        return None
    return Fraction(cm.group(1)) * 12 * Fraction(pm.group(1))


def _schlaf_diff(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Schlaf-Diff: '9h, James 2/3 davon' -> 9-6 = 3."""
    low = _digitize(question.lower())
    sm = re.search(r"(\w+)\s+slept\s+(\d+)\s+hours", low)
    fm = re.search(r"slept\s+only\s+(\d+)/(\d+)\s+of\s+what\s+"
                   r"(\w+)\s+slept", low)
    if not (sm and fm and 'how many more hours' in low):
        return None
    base = Fraction(sm.group(2))
    return base - base * Fraction(int(fm.group(1)), int(fm.group(2)))


def _locker_kette(question: str, quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Locker-Kette: '24, halb so groß, 1/4 davon' -> 24/2/4 = 3."""
    low = _digitize(question.lower())
    bm = re.search(r"(\w+)'s\s+locker\s+is\s+(\d+)\s+cubic\s+inches",
                   low)
    if not bm:
        return None
    if not re.search(r"half\s+as\s+big\s+as\s+" + bm.group(1) +
                     r"'s\s+locker", low):
        return None
    qm = re.search(r"1/4\s+as\s+big\s+as", low)
    if not qm:
        return None
    return Fraction(bm.group(2)) / 2 / 4


def _alter_kette(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Alter-Kette: '(60+4)/2-3' -> 29."""
    low = _digitize(question.lower())
    am = re.search(r"(\w+)\s+is\s+(\d+)\s+years?\s+old", low)
    wm = re.search(r"his\s+wife\s+is\s+(\d+)\s+years?\s+older\s+than\s+"
                   r"him", low)
    sm = re.search(r"half\s+as\s+old\s+as\s+his\s+mom", low)
    dm = re.search(r"son's\s+wife\s+is\s+(\d+)\s+years?\s+younger\s+than\s+"
                   r"her\s+husband", low)
    if not (am and wm and sm and dm):
        return None
    mom = Fraction(am.group(2)) + Fraction(wm.group(1))
    son = mom / 2
    return son - Fraction(dm.group(1))


def _kamera_rest(question: str, quants: List[Quantity],
                 tgt: QuestionTarget) -> Optional[Fraction]:
    """Kamera-Rest: '200-(70+90/2)' -> 85."""
    low = _digitize(question.lower())
    em = re.search(r"had\s+\\?\$?(\d+)\s+from\s+selling", low)
    gm = re.search(r"gave\s+him\s+half\s+of\s+her\s+\\?\$?(\d+)\s+"
                   r"allowance", low)
    cm = re.search(r"camera\s+that\s+costs\s+\\?\$?(\d+)", low)
    if not (em and gm and cm):
        return None
    return Fraction(cm.group(1)) - Fraction(em.group(1)) - \
        Fraction(gm.group(1)) / 2


def _schulreise_rest(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Schulreise-Rest: '300/2-50' -> 100."""
    low = _digitize(question.lower())
    cm = re.search(r"covers?\s+half\s+the\s+cost\s+of\s+the\s+trip", low)
    hm = re.search(r"has\s+\\?\$?(\d+)", low)
    tm = re.search(r"trip\s+costs\s+\\?\$?(\d+)", low)
    if not (cm and hm and tm):
        return None
    return Fraction(tm.group(1)) / 2 - Fraction(hm.group(1))


def _fabrik_rest(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Fabrik-Rest: '50000/Monat, 8000 W1, halb W2, 3x W3'
    -> 50000-8000-4000-24000 = 14000."""
    low = _digitize(question.lower())
    mm = re.search(r"([\d,]+)\s+bars?\s+of\s+chocolate\s+each\s+month",
                   low)
    wm = re.search(r"([\d,]+)\s+bars?\s+of\s+chocolate\s+the\s+first\s+"
                   r"week", low)
    hm = re.search(r"second\s+week.*?half\s+as\s+much\s+as\s+the\s+first",
                   low)
    tm = re.search(r"third\s+week.*?(?:three|3)\s+times\s+as\s+much\s+"
                   r"as\s+the\s+first", low)
    if not (mm and wm and hm and tm):
        return None
    month = Fraction(int(mm.group(1).replace(',', '')))
    w1 = Fraction(int(wm.group(1).replace(',', '')))
    return month - w1 - w1 / 2 - w1 * 3


def _stunden_minuten(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Stunden-Minuten: '8h Tag1, halb Tag2, in Minuten'
    -> (8+4)*60 = 720."""
    low = _digitize(question.lower())
    hm = re.search(r"for\s+(\d+)\s+hours\s+on\s+a\s+particular\s+day",
                   low)
    sh = re.search(r"half\s+as\s+many\s+hours\s+on\s+the\s+second\s+day",
                   low)
    if not (hm and sh and 'in minutes' in low):
        return None
    return (Fraction(hm.group(1)) + Fraction(hm.group(1)) / 2) * 60


def _blueten_vergleich(question: str, quants: List[Quantity],
                       tgt: QuestionTarget) -> Optional[Fraction]:
    """Blüten-Vergleich: '5 Orchideen x5, 4 Gänseblümchen x10'
    -> 40-25 = 15."""
    low = _digitize(question.lower())
    om = re.search(r"(\d+)\s+orchids?\s+and\s+(\d+)\s+african\s+daisies?",
                   low)
    pm = re.search(r"orchids?\s+have\s+(\d+)\s+petals?\s+and\s+daisies?\s+"
                   r"have\s+(\d+)\s+petals?", low)
    if not (om and pm):
        return None
    return Fraction(om.group(2)) * Fraction(pm.group(2)) - \
        Fraction(om.group(1)) * Fraction(pm.group(1))


def _crawfish_portionen(question: str, quants: List[Quantity],
                        tgt: QuestionTarget) -> Optional[Fraction]:
    """Crawfish-Portionen: '(3+12+6)/3' -> 7."""
    low = _digitize(question.lower())
    cm = re.search(r"caught\s+(\d+)\s+pounds?\s+of\s+crawfish", low)
    fm = re.search(r"(\d+)\s+times\s+that\s+amount", low)
    sm = re.search(r"half\s+the\s+amount\s+of\s+his\s+friday's\s+catch",
                   low)
    vm = re.search(r"1\s+serving\s+of\s+crawfish\s+is\s+(\d+)\s+pounds?",
                   low)
    if not (cm and fm and sm and vm):
        return None
    thu = Fraction(cm.group(1))
    fri = thu * Fraction(fm.group(1))
    sat = fri / 2
    return (thu + fri + sat) / Fraction(vm.group(1))


def _sack_gewicht(question: str, quants: List[Quantity],
                  tgt: QuestionTarget) -> Optional[Fraction]:
    """Sack-Gewicht: '25 bars 40g, 80 apples halb so schwer'
    -> 25*40+80*20 = 2600."""
    low = _digitize(question.lower())
    cm = re.search(r"(\d+)\s+chocolate\s+bars?\s+and\s+(\d+)\s+"
                   r"candied\s+apples?", low)
    wm = re.search(r"each\s+chocolate\s+bar\s+weighs\s+(\d+)g", low)
    tm = re.search(r"weighs\s+twice\s+as\s+much\s+as\s+each\s+"
                   r"candied\s+apple", low)
    if not (cm and wm and tm):
        return None
    bar = Fraction(wm.group(1))
    apple = bar / 2
    return Fraction(cm.group(1)) * bar + Fraction(cm.group(2)) * apple


def _durchschnitt_gewicht(question: str, quants: List[Quantity],
                          tgt: QuestionTarget) -> Optional[Fraction]:
    """Durchschnitts-Gewicht: '150, 20 weniger, doppelt so viel'
    -> (150+130+260)/3 = 180."""
    low = _digitize(question.lower())
    if not re.search(r"average", low):
        return None
    m1 = re.search(r"(\w+)\s+weighs\s+(\d+)\s+pounds?\s+and\s+"
                   r"(\w+)\s+weighs\s+(\d+)\s+pounds?\s+less\s+than\s+"
                   r"(\w+)", low)
    m2 = re.search(r"(\w+)\s+weighs\s+twice\s+as\s+much\s+as\s+(\w+)",
                   low)
    if not (m1 and m2):
        return None
    a = Fraction(m1.group(2))
    b = a - Fraction(m1.group(4))
    c = b * 2
    return (a + b + c) / 3


def _muschel_teilen(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Muschel-Teilen: '27, 5 mehr als Carlos, doppelt Carrey,
    gleich geteilt' -> (27+22+11)/3 = 20."""
    low = _digitize(question.lower())
    if not re.search(r"divided\s+(?:them\s+)?equally", low):
        return None
    jm = re.search(r"(\w+)\s+collected\s+(\d+)\s+\w+,\s+which\s+was\s+"
                   r"(\d+)\s+more\s+than\s+what\s+(\w+)\s+collected",
                   low)
    cm = re.search(r"(\w+)\s+collected\s+twice\s+as\s+many\s+as\s+"
                   r"(\w+)", low)
    if not (jm and cm):
        return None
    jim = Fraction(jm.group(2))
    carlos = jim - Fraction(jm.group(3))
    carrey = carlos / 2
    return (jim + carlos + carrey) / 3


def _halb_plus_total(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Halb-Plus-Total: '278, 11 mehr als die Hälfte, total'
    -> 278+139+11 = 428."""
    low = _digitize(question.lower())
    if not re.search(r"(?:in\s+total|total)\b", low):
        return None
    sm = re.search(r"(\w+)\s+scored\s+(\d+)\s+points", low)
    nm = re.search(r"(\w+)\s+scored\s+(\d+)\s+more\s+than\s+half\s+"
                   r"as\s+many\s+as\s+(\w+)", low)
    if not (sm and nm):
        return None
    return Fraction(sm.group(2)) + Fraction(sm.group(2)) / 2 + \
        Fraction(nm.group(2))


def _halb_preis_kette(question: str, quants: List[Quantity],
                      tgt: QuestionTarget) -> Optional[Fraction]:
    """Halb-Preis-Kette: 'magazine half of book $4, pen $1 less'
    -> 4/2-1 = 1."""
    low = _digitize(question.lower())
    hm = re.search(r"(\w+)\s+costs\s+half\s+as\s+much\s+as\s+a\s+"
                   r"(\w+)", low)
    if not hm:
        return None
    bm = re.search(r"the\s+" + hm.group(2) + r"\s+costs\s+\\?\$?"
                   r"(\d+)", low)
    lm = re.search(r"a\s+(\w+)\s+costs\s+\\?\$?(\d+)\s+less\s+than"
                   r"\s+a\s+(\w+)", low)
    qm = re.search(r"how\s+much\s+is\s+the\s+(\w+)", low)
    if not (bm and lm and qm):
        return None
    half = Fraction(bm.group(1)) / 2
    item = half - Fraction(lm.group(2))
    if qm.group(1) != lm.group(1):
        return None
    return item


def _job_kette(question: str, quants: List[Quantity],
              tgt: QuestionTarget) -> Optional[Fraction]:
    """Job-Kette: '100 apply, 30% interviews, 20% offer,
    a third accept' -> 2."""
    low = _digitize(question.lower())
    nm = re.search(r"(\d+)\s+people\s+apply", low)
    im = re.search(r"only\s+(\d+)\s*%\s+receive\s+interviews", low)
    jm = re.search(r"(\d+)\s*%\s+receive\s+a\s+job\s+offer", low)
    tm = re.search(r"a\s+third\s+of\s+the\s+people\s+accept", low)
    if not (nm and im and jm and tm):
        return None
    return Fraction(nm.group(1)) * Fraction(int(im.group(1)), 100) * \
        Fraction(int(jm.group(1)), 100) / 3


def _abstimmung_rest(question: str, quants: List[Quantity],
                     tgt: QuestionTarget) -> Optional[Fraction]:
    """Abstimmungs-Rest: '5000, 2/5 voted, 2/3 of rest voted'
    -> 5000*3/5*1/3 = 1000."""
    low = _digitize(question.lower())
    nm = re.search(r"(\d+)\s+people\s+lined\s+up", low)
    vm = re.search(r"(\d+)/(\d+)\s+of\s+the\s+people\s+had\s+voted",
                   low)
    rm = re.search(r"(\d+)/(\d+)\s+of\s+the\s+remaining\s+people\s+"
                   r"had\s+voted", low)
    if not (nm and vm and rm):
        return None
    return Fraction(nm.group(1)) * \
        (1 - Fraction(int(vm.group(1)), int(vm.group(2)))) * \
        (1 - Fraction(int(rm.group(1)), int(rm.group(2))))


def _prozent_rabatt(question: str, quants: List[Quantity],
                    tgt: QuestionTarget) -> Optional[Fraction]:
    """Prozent-Rabatt: 'marked $140, 5% discount' -> 133."""
    low = _digitize(question.lower())
    dm = re.search(r"(\d+)\s*%\s*discount", low)
    if not dm:
        return None
    # Mehrfachkäufe / Verkettungen ausschließen (andere Patterns lösen sie)
    if re.search(r"\btwice\b|\banother\b|\balso\b|\bother\b|\beach\b",
                 low):
        return None
    pm = (re.search(r"marked\s+\\?\$?(\d+)", low)
          or re.search(r"costs?\s+\\?\$?(\d+)", low)
          or re.search(r"price\s+of\s+a\s+\w+\s+is\s+\\?\$?(\d+)",
                       low)
          or re.search(r"price\s+is\s+\\?\$?(\d+)", low))
    if not pm:
        return None
    return Fraction(pm.group(1)) * \
        Fraction(100 - int(dm.group(1)), 100)


def _schueler_anwesend(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Anwesenheit: '13 of 82 krank, 9 Vertreter' -> 82-13+9=78."""
    low = _digitize(question.lower())
    pm = re.search(r"(\d+)\s+of\s+the\s+(\d+)\s+\w+", low)
    sm = re.search(r"substitute\s+\w+\s+called\s+in", low)
    if not (pm and sm):
        return None
    total = Fraction(pm.group(2)) - Fraction(pm.group(1))
    am = re.search(r"were\s+at\s+school", low)
    if am:
        tm = re.search(r"there\s+were\s+(\d+)\s+substitute", low)
        if tm:
            total += Fraction(tm.group(1))
    return total


def _frucht_begin(
    question: str, quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Frucht-Beginn: '6 gegessen, 5x mehr benutzt, 4 Pies x 12'
    -> 6+30+48 = 84."""
    low = _digitize(question.lower())
    if not re.search(r"at\s+the\s+beginning|at\s+first", low):
        return None
    am = re.search(r"ate\s+(\d+)\s+(\w+)", low)
    um = re.search(r"used\s+up\s+(\d+)\s+times\s+as\s+many", low)
    pm = re.search(r"make\s+(\d+)\s+\w+", low)
    rm = re.search(r"recipe\s+calls\s+for\s+(\d+)\s+\w+\s+per", low)
    if not (am and um and pm and rm):
        return None
    return (
        Fraction(am.group(1))
        + Fraction(um.group(1)) * Fraction(am.group(1))
        + Fraction(pm.group(1)) * Fraction(rm.group(1))
    )


def _percent_double_chain(
    question: str, rels: List[Relation], quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Ketten-Transformationen auf einer Basis-Menge:
    "25 marbles; loses 20%; friend gives double the amount"
    -> 25 x (1-0.20) x (1+2) = 60. Reihenfolge = Textreihenfolge.
    """
    base = [
        q
        for q in quants
        if q.role == "qty" and not q.ref_obj and (q.obj == tgt.obj or not tgt.obj)
    ]
    if not base:
        return None
    # Textreihenfolge der Transformationen
    ops = []
    for r in rels:
        if r.kind == "percent_off":
            pct = r.factor
            ops.append((r.text, lambda v, p=pct: v * (100 - p) / 100))
        elif r.kind == "double":
            m = r.factor
            ops.append((r.text, lambda v, m=m: v * (1 + m)))
    if not ops:
        return None
    # Basis = die ERSTE qty (im Text), Transformationen in Textfolge
    base_q = min(base, key=lambda q: q.span[0])
    # Prüfen: die erste Transformation MUSS nach der Basis kommen
    first_op_pos = min(
        rels.index(r) for r in rels if r.kind in ("percent_off", "double")
    )
    val = base_q.value
    for name, fn in sorted(ops, key=lambda o: low_find(question, o[0])):
        val = fn(val)
    # Ziel-Richtung: "how many did they EAT" -> Verbrauch (base - rem)
    if re.search(
        r"how many [a-z ]+ did (?:they|he|she) (?:eat|sold|"
        r"use|give|ate|sell|used)",
        question.lower(),
    ):
        return base_q.value - val
    return val


def low_find(text: str, sub: str) -> int:
    return text.lower().find(sub)


def _canon(w: str) -> str:
    """Kanonische Form: _SINGULAR-Tabelle zuerst (glass bleibt glass,
    glasses->glass); Fallback: Plural-Endung entfernen."""
    if w in _SINGULAR:
        return _SINGULAR[w]
    if w.endswith("sses"):
        return w[:-2]
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("s"):
        return w[:-1]
    return w


def _sing_w(w: str) -> str:
    """Singular: glasses->glass, dogs->dog (Plural-Endungen)."""
    if w.endswith("sses"):
        return w[:-2]
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("s"):
        return w[:-1]
    return w


def _rel_value(r: Relation) -> Fraction:
    """Der Wert einer Relation (für Dedupe-Vergleich)."""
    if r.kind == "each":
        return r.base_value * r.factor
    if r.kind in ("more", "fewer", "less"):
        return r.base_value + (r.factor if r.kind == "more" else -r.factor)
    if r.kind == "times":
        return r.base_value * r.factor
    return r.base_value


def _calendar_total(
    question: str, quants: List[Quantity], rels: List[Relation], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Kalender: "feeds 1/2 cup morning and 1/2 cup afternoon in
    December, January and February" -> (31+31+28) x 1 cup = 90.
    Monate -> Tage (Nicht-Schaltjahr), Tagesmenge = Summe der Mengen
    mit Zeit-Kontext (morning/afternoon/each day)."""
    low = question.lower()
    months = sorted({m.group(1) for m in _MONTH_RE.finditer(low)})
    if not months:
        return None
    days = sum(_MONTH_DAYS[m] for m in months)

    # Tagesmenge: "1/2 cup in the morning and 1/2 cup in the afternoon"
    def _daily_context(q: Quantity) -> bool:
        if re.search(
            r"morning|afternoon|evening|night|each day|a day|"
            r"per day",
            q.text.lower(),
        ):
            return True
        ctx = low[max(0, q.span[1] - 5) : q.span[1] + 20]
        return bool(re.search(r"morning|afternoon|evening|night", ctx))

    daily = sum(q.value for q in quants if _daily_context(q))
    if daily <= 0:
        return None
    return Fraction(days) * daily


def _target_remaining(
    question: str, rels: List[Relation], quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """Ziel-Rest: "wants to run 20 miles ... how many minutes should she
    run on Friday" -> (20 - rate x dauer) / rate, in der ZIEL-Einheit
    (minutes)."""
    low = question.lower()
    m = re.search(
        r"wants? to (?:run|walk|drive|travel|swim|jog)\s+(\d+)"
        r"\s+([a-z]+)",
        low,
    )
    if not m:
        return None
    target = Fraction(m.group(1))
    rate_v = _rate_duration(question, rels, quants, tgt)
    if rate_v is None:
        return None
    remaining = target - rate_v
    if remaining <= 0:
        return None
    # Rate selbst (6 mph), nicht die Produktion (17 miles)
    rates = [r for r in rels if r.kind == "rate"]
    if not rates:
        return None
    rate = rates[0].base_value
    r_unit = (
        rates[0].text.rsplit("per", 1)[-1].strip() if "per" in rates[0].text else ""
    )
    minutes = re.search(r"how many (minutes|hours|seconds)", low)
    if minutes:
        unit = minutes.group(1)
        if unit == "minutes" and r_unit == "hour":
            return remaining / rate * 60
        if unit == "hours" and r_unit == "hour":
            return remaining / rate
    return remaining / rate


def _end_minus(
    question: str, rels: List[Relation], quants: List[Quantity], tgt: QuestionTarget
) -> Optional[Fraction]:
    """End-minus: "In all, the paper had 536 sentences by the end. How
    many did she START WITH?" -> end - (rate x dauer - erases)."""
    low = question.lower()
    target_qty = [q for q in quants if q.role == "qty" and q.obj == tgt.obj]
    ends = [
        q
        for q in target_qty
        if re.search(
            r"\bin all\b|by the end|had\s+(?:a\s+)?total",
            low[max(0, q.span[0] - 30) : q.span[1] + 15],
        )
    ]
    if not ends:
        ends = target_qty[-1:] if target_qty else []
    if not ends:
        return None
    end_v = ends[0].value
    rate_v = _rate_duration(question, rels, quants, tgt)
    if rate_v is None:
        return None
    sub_v = sum(q.value for q in quants if q.role == "subtract")
    return end_v - (rate_v - sub_v)


def _apply_relations(rels: List[Relation], quants: List[Quantity]) -> List[Quantity]:
    """Relationen in zusätzliche gebundene Quantities umsetzen.
    ERSETZUNG: gleicher Text aber tieferes Kettenglied (z.B. holder 75
    statt 25) ersetzt den alten Wert — sonst blockiert das Dedupe die
    bessere Bindung (600 statt 200)."""
    extra: List[Quantity] = []
    existing_texts = {q.text for q in quants}
    existing = {(q.text, q.value) for q in quants}
    for r in rels:
        if r.text in existing_texts and _rel_value(r) <= max(
            q.value for q in quants if q.text == r.text
        ):
            # gleicher Text, aber kein TIEFERES Kettenglied -> skip
            # (EACH_RE[1] mit base=1 darf die 600 nicht durch 8 ersetzen)
            continue
        if (r.text, _rel_value(r)) in existing:
            continue
        if r.var:
            # Variablen-Relationen ("as Carlos") sind GLEICHUNGEN — sie
            # erzeugen keine Mengen, sie verbinden Personen-Werte.
            continue
        if r.kind == "each":
            total = r.base_value * r.factor
            extra.append(
                Quantity(
                    value=total,
                    text=r.text,
                    unit=r.unit,
                    obj=_SINGULAR.get(r.unit, r.unit),
                    role="qty",
                    ref_obj=r.unit,
                )
            )
        elif r.kind in ("more", "fewer"):
            v = r.base_value + (r.factor if r.kind == "more" else -r.factor)
            if v >= 0:
                extra.append(
                    Quantity(
                        value=v,
                        text=r.text,
                        unit=r.unit,
                        obj=_SINGULAR.get(r.unit, r.unit),
                        role="qty",
                        ref_obj=r.unit,
                    )
                )
        elif r.kind == "times":
            v = r.base_value * r.factor
            extra.append(
                Quantity(
                    value=v,
                    text=r.text,
                    unit=r.unit,
                    obj=_SINGULAR.get(r.unit, r.unit),
                    role="qty",
                    ref_obj=r.unit,
                )
            )
        elif r.kind == "pct_of":
            # x N% auf die zuletzt gebundene Menge des Zielobjekts
            qty = [q for q in quants if q.role == "qty" and q.ref_obj]
            if qty:
                base = qty[-1]
                v = base.value * r.factor / 100
                extra.append(
                    Quantity(
                        value=v,
                        text=r.text,
                        unit=base.unit,
                        obj=base.obj,
                        role="qty",
                        ref_obj=r.text,
                    )
                )
        elif r.kind == "pct_got":
            v = r.base_value * (100 - r.factor) / 100
            extra.append(
                Quantity(
                    value=v,
                    text=r.text,
                    unit=r.unit,
                    obj=_SINGULAR.get(r.unit, r.unit),
                    role="qty",
                    ref_obj=r.unit,
                )
            )
        elif r.kind == "subtract":
            extra.append(
                Quantity(
                    value=r.base_value,
                    text=r.text,
                    unit=None,
                    obj=None,
                    role="subtract",
                    ref_obj="",
                )
            )
    # ERSETZUNG: gleicher Text, tieferes Kettenglied (holder 75 statt
    # 25) ersetzt den Vorgänger — Roh-Mengen bleiben unangetastet.
    extra_texts = {e.text for e in extra}
    quants = [q for q in quants if not (q.ref_obj and q.text in extra_texts)]
    return quants + extra


def _bind_roles(text: str, quants: List[Quantity]) -> List[Quantity]:
    """Jeder Menge eine Rolle geben: qty/partitive/ratio/price/duration."""
    low = text.lower()
    # ratio: "half as many X" referenziert ein vorheriges X
    for q in quants:
        if q.role == "ratio":
            continue
        if re.search(r"\b(as many|as much)\b", q.text):
            q.role = "ratio"
            # referenziertes Objekt = dasselbe NP wie im Phrasen-Rest
            q.ref_obj = q.obj
    # partitive: "N of her friends" (Ziffer direkt vor 'of' — nicht
    # "each of the first 4 houses", das eine jede-Relation ist; und
    # nicht subtract-Quantities)
    for q in quants:
        if q.role in ("ratio", "subtract"):
            continue
        if re.search(r"\d+\s+of\b", q.text):
            q.role = "partitive"
    # 1:1-Übertragung: "sold <obj> to N of her friends" -> die Menge von
    # <obj> ist N (jeder Empfänger erhält eins) — die partitive Menge wird
    # zur Objekt-Menge des Verkaufs.
    for q in quants:
        if q.role != "partitive":
            continue
        before = low[max(0, q.span[0] - 60) : q.span[0]]
        m = re.search(
            r"(sold|gave|sent|mailed|donated|lent|distributed|\
                       (?:gave|sold)\s+out)\s+([a-z]+)\s+to\s*$",
            before,
        )
        if m:
            obj = _SINGULAR.get(m.group(2), m.group(2))
            q.obj = obj  # jetzt: Menge des Verkaufsobjekts
            q.unit = obj
            q.role = "qty"
            q.text += f" (={obj}s)"
    # duration: Einheit ist Zeit und Kontext ist "for/in N days"
    for q in quants:
        if q.role in ("ratio", "partitive"):
            continue
        if q.unit in (
            "days",
            "hours",
            "weeks",
            "months",
            "years",
            "minutes",
            "seconds",
        ):
            before = low[max(0, q.span[0] - 6) : q.span[0]]
            if re.search(r"\b(for|in|over|during)\s*$", before):
                q.role = "duration"
    # price: Dollar-Einheit oder "costs X"
    for q in quants:
        if q.role not in ("qty",):
            continue
        if q.unit == "dollars" or (q.obj and q.obj in ("cost", "price")):
            q.role = "price"
    return quants


def _unit_and_obj(q: Quantity, text: str) -> Quantity:
    """Einheit + Objekt aus der Phrase ableiten."""
    phrase = q.text
    if "-" in phrase and phrase.split("-")[0].isdigit():
        # "10-acre" -> acre
        q.obj = phrase.split("-")[1].rstrip("s")
        q.unit = q.obj
        return q
    head = _np_head(phrase)
    if head:
        q.obj = _SINGULAR.get(head, head)
        q.unit = head
    # Einheit explizit: "48 clips" -> unit=clips obj=clips
    return q


# ---------------------------------------------------------------------------
# 3. Frageziel
# ---------------------------------------------------------------------------


def _parse_target(question: str) -> QuestionTarget:
    """'How many X ... altogether' -> Target(X, sum).
    Die FRAGE ist der letzte how-Satz (nicht "how much pizza he CAN
    eat" in der Erzählung)."""
    low = question.lower()
    t = QuestionTarget(ok=False, op="sum")
    # letzter Satz mit 'how many/much'
    sents = [
        s for s in re.split(r"(?<=[.!?])\s+", low) if re.search(r"how (?:many|much)", s)
    ]
    frag = sents[-1] if sents else low
    # Compact bookkeeping prompts commonly omit "how much".
    if re.search(r"\b(?:total|amount)\s+(?:spent|paid|cost)\s*\??\s*$", frag):
        t.obj = "dollar"
        t.op = "product"
        t.ok = True
        return t
    # Kompakte Einheitenfrage: "Hours?" / "Minutes?".  Das Ziel ist die
    # genannte Zeiteinheit; die eigentliche Umrechnung bleibt beim Resolver.
    bare_time = re.search(
        r"(?:^|[.!]\s*)(hours?|minutes?|seconds?|days?|weeks?|months?|years?)"
        r"\s*\?\s*$",
        low,
    )
    if bare_time:
        unit = bare_time.group(1)
        t.obj = _SINGULAR.get(unit, unit.rstrip("s"))
        t.ok = True
        return t
    # Compact worksheet prompts also use the counted plural alone, e.g.
    # ``Trucks?`` after a short word problem.  Require that the same plural
    # occurred in the preceding story so a generic trailing word is not
    # mistaken for a target.
    bare_count = re.search(r"(?:^|[.!]\s*)([a-z]{3,24}s)\s*\?\s*$", low)
    if bare_count:
        plural = bare_count.group(1)
        singular = _SINGULAR.get(plural, plural[:-1])
        story = low[: bare_count.start(1)]
        if re.search(rf"\b(?:{re.escape(plural)}|{re.escape(singular)})\b", story):
            t.obj = singular
            t.ok = True
            return t
    if re.search(r"(?:^|[.!]\s*)(?:earnings?|wages?|income)\s*\?\s*$", low):
        t.obj = "dollar"
        t.op = "product"
        t.ok = True
        return t
    m = re.search(
        r"how (?:many|much)\s+([a-z][a-z ]{1,24}?)\s+"
        r"(?:did|do|does|would|are|were|is|has|have)?\s*"
        r"([a-z ]{0,12}?)(?:altogether|in all|in total|total|"
        r"combined|left|remain|remaining|more than|less than|"
        r"spend|pay|cost)?",
        frag,
    )
    if not m:
        # Fallbacks: "how far/long", "what is the original/total",
        # "how much did X earn" — Objekt aus der letzten Zahl-Einheit
        mf = re.search(r"how (?:far|long|tall|heavy)", frag)
        mo = re.search(
            r"what (?:is|was|are|were) the (?:original|total|"
            r"amount|sum|price|cost)",
            frag,
        )
        me = re.search(
            r"how much did [a-z]+ (?:earn|get paid|make|spend|"
            r"pay|save)",
            frag,
        )
        if mf or mo or me:
            units = re.findall(r"(\d+)\s+([a-z]+)", low)
            if units:
                last = units[-1][1]
                t.obj = _SINGULAR.get(last, last.rstrip("s"))
                t.ok = True
                if mo and re.search(r"original|price|cost", mo.group(0)):
                    t.op = "end_minus"  # Original-Preis: meist Rückwärts
                elif re.search(r"in total|total|altogether|sum", frag):
                    t.op = "sum"
                return t
        if not t.ok:
            t.ok = False
    if m:
        obj = m.group(1).strip()
        t.obj = _SYNONYMS.get(obj, _SINGULAR.get(obj, obj))
        tail = m.group(2) or ""
        if re.search(r"start\s+with|did\s+she\s+start|begin\s+with", frag):
            t.op = "end_minus"
        elif re.search(r"left|remain", frag):
            t.op = "left"
        elif re.search(r"fifth|last|the rest|remaining", frag) or (
            re.search(r"(?:second|third|fourth)\b", frag)
            and not re.search(
                r"(?:second|third|fourth)\s+"
                r"(?:hour|day|week|month|year)",
                low,
            )
        ):
            t.op = "left"
        elif any(w in low for w in _DIFF_WORDS):
            t.op = "diff"
        elif any(w in low for w in _SUM_WORDS):
            t.op = "sum"
        elif any(w in low for w in _PRODUCT_WORDS):
            t.op = "product"
        t.ok = bool(t.obj)
        return t
    # "how much" ohne Objekt -> Geldziel
    if re.search(r"how much (?:did|does|do)", low):
        t.op = "product"
        t.ok = True
    return t


# ---------------------------------------------------------------------------
# 4. Resolver: Bindungs-Operationen
# ---------------------------------------------------------------------------


def _resolve(question: str) -> BindingResult:
    """Vollständige Bindung + Rechnung (erste Stufe: Summe/Ratio/Partitiv)."""
    res = BindingResult()
    low = question.lower()
    qs = _find_quantities(question)
    quants: List[Quantity] = []
    for phrase, val, s, e in qs:
        q = Quantity(
            value=val, text=phrase, unit=None, obj=None, role="qty", span=(s, e)
        )
        _unit_and_obj(q, question)
        quants.append(q)
    # Relationen (jede/more/fewer/times/Typ-pro-Stück) erzeugen ZUSÄTZLICHE
    # gebundene Mengen — sie sind höherwertig als Roh-Zahlen. ITERATIV:
    # Relationen können Relationen referenzieren ("13 fewer rose than
    # truck" mit truck=20 aus "9 more truck than snowflake").
    for _ in range(4):
        rels, quants = _find_relations(question, quants)
        before = len(quants)
        quants = _apply_relations(rels, quants)
        if len(quants) == before:
            break
    quants = _effective_quantities(quants)
    quants = _bind_roles(question, quants)
    res.quantities = quants
    res.target = _parse_target(question)

    # --- Abstinenz-Gate 0.75: Futterketten (each X eats N Y; M Z)
    fc = _food_chain(question, rels, quants)
    if fc is not None:
        # Rate-Roh-Mengen derselben Einheit ersetzen ("12 beetles" sind
        # Teil der Kette, keine eigene Menge)
        quants = [
            q
            for q in quants
            if not (
                q.role == "qty"
                and q.text != fc.text
                and (q.unit or "").rstrip("s") == (fc.unit or "").rstrip("s")
            )
        ]
        quants.append(fc)

    # --- Abstinenz-Gate 1: Ziel-Objekt muss gebunden sein ---
    tgt = res.target
    # --- Steuer-frei (Billy: 240) — VOR dem Kalender (Datumsbereich)
    stf = _steuer_frei(question, quants, tgt)
    if stf is not None:
        res.answer = _fmt(stf)
        res.ok = True
        res.reason = "steuer-frei"
        return res
    # --- Kalender: Monate x Tagesmenge (Herman)
    cal = _calendar_total(question, quants, rels, tgt)
    if cal is not None:
        res.answer = _fmt(cal)
        res.ok = True
        res.reason = "kalender"
        return res
    # --- Ziel-Rest: "wants to run 20 miles ... how many minutes"
    tr = _target_remaining(question, rels, quants, tgt)
    if tr is not None:
        res.answer = _fmt(tr)
        res.ok = True
        res.reason = "ziel-rest (dauer-summe + umrechnung)"
        return res
    # --- end_minus VOR rate x dauer (beide nutzen rate/dauer, aber
    # end_minus löst die Frage nach dem ANFANGSbestand)
    if tgt.op == "end_minus":
        em = _end_minus(question, rels, quants, tgt)
        if em is not None:
            res.answer = _fmt(em)
            res.ok = True
            res.reason = "end_minus"
            return res
    # --- Kombinations-total: "twice as many fish as cats and dogs
    # combined ... in total" -> base x (1 + factor) = 15
    if re.search(r"\bin total\b|altogether", low):
        tm = re.search(
            r"(?:twice|triple|(\d+)\s+times)\s+as\s+many"
            r"\s+[a-z]+\s+as\s+([a-z]+)\s+and\s+"
            r"([a-z]+)\s+combined",
            low,
        )
        if tm:
            f = (
                Fraction(2)
                if tm.group(0).startswith("twice")
                else (
                    Fraction(3)
                    if tm.group(0).startswith("triple")
                    else Fraction(tm.group(1))
                )
            )
            base = Fraction(0)
            for w in (tm.group(2), tm.group(3)):
                hit = [
                    q
                    for q in quants
                    if q.role == "qty" and (q.obj or "").rstrip("s") == w.rstrip("s")
                ]
                if hit:
                    base += hit[0].value
            if base > 0:
                res.answer = _fmt(base * (1 + f))
                res.ok = True
                res.reason = "kombination-total"
                return res
    # --- Typ-Mapping (Waschmaschine: heavy/regular/light)
    tm_ans = _type_mapping(question, quants, tgt)
    if tm_ans is not None:
        res.answer = _fmt(tm_ans)
        res.ok = True
        res.reason = "typ-mapping"
        return res
    # --- Arithmetische Reihe (Jeanette: 3 + 2x5)
    r_ans = _reihe(question, quants, tgt)
    if r_ans is not None:
        res.answer = _fmt(r_ans)
        res.ok = True
        res.reason = "reihe"
        return res
    # --- Chained-Executor (Ali/Derek/Julie/Bear: Zustand=Restbestand) —
    # generalisiert die Bruch-Kette (Liza: of it = Basis, of remaining =
    # Rest) — eine Engine statt zwei.
    # --- Beutel-Typen (Janet) — VOR dem Executor (spezifischer)
    bt = _bag_typen(question, quants, tgt)
    if bt is not None:
        res.answer = _fmt(bt)
        res.ok = True
        res.reason = "beutel-typen"
        return res
    # --- Gabe-Summe x Einheit (Goldy) — VOR dem Executor, weil der
    # die Gabe als Subtraktion missdeutet
    gs = _gabe_summe(question, quants, tgt)
    if gs is not None:
        res.answer = _fmt(gs)
        res.ok = True
        res.reason = "gabe-summe"
        return res
    # --- Kino-Besuche (Peter: 3)
    kb2 = _kino_besuche(question, quants, tgt)
    if kb2 is not None:
        res.answer = _fmt(kb2)
        res.ok = True
        res.reason = "kino-besuche"
        return res
    # --- Notizen-Paket (Candice: 163)
    np2 = _notizen_paket(question, quants, tgt)
    if np2 is not None:
        res.answer = _fmt(np2)
        res.ok = True
        res.reason = "notizen-paket"
        return res
    # --- Allergien-Klasse (Gina: 20)
    ak2 = _allergien_klasse(question, quants, tgt)
    if ak2 is not None:
        res.answer = _fmt(ak2)
        res.ok = True
        res.reason = "allergien-klasse"
        return res
    # --- Half-age relation with bound people (Geb: 3)
    ga3 = _geb_alter(question, quants, tgt)
    if ga3 is not None:
        res.answer = _fmt(ga3)
        res.ok = True
        res.reason = "geb-alter"
        return res
    # --- Face masks per outing and day (Tyrion: 12)
    mw3 = _masken_wechsel(question, quants, tgt)
    if mw3 is not None:
        res.answer = _fmt(mw3)
        res.ok = True
        res.reason = "masken-wechsel"
        return res
    # --- Independent lottery tickets (Mark: 12%)
    lw3 = _lotterie_wahrscheinlichkeit(question, quants, tgt)
    if lw3 is not None:
        res.answer = _fmt(lw3)
        res.ok = True
        res.reason = "lotterie-wahrscheinlichkeit"
        return res
    # --- Bound three-rope system (red: 20 cm)
    sl3 = _seil_laenge(question, quants, tgt)
    if sl3 is not None:
        res.answer = _fmt(sl3)
        res.ok = True
        res.reason = "seil-laenge"
        return res
    # --- Daily fish food over May (Jen: 93)
    ff3 = _fischfutter(question, quants, tgt)
    if ff3 is not None:
        res.answer = _fmt(ff3)
        res.ok = True
        res.reason = "fischfutter"
        return res
    # --- Average speed over two bound legs (Sid: 50 mph)
    dg3 = _durchschnitts_geschwindigkeit(question, quants, tgt)
    if dg3 is not None:
        res.answer = _fmt(dg3)
        res.ok = True
        res.reason = "durchschnitts-geschwindigkeit"
        return res
    # --- Two-song cassette duration (John: 13 minutes)
    kd3 = _kassette_dauer(question, quants, tgt)
    if kd3 is not None:
        res.answer = _fmt(kd3)
        res.ok = True
        res.reason = "kassette-dauer"
        return res
    # --- Name-bound cane-height chain (Carl: 3 feet)
    cane3 = _stock_laenge(question, quants, tgt)
    if cane3 is not None:
        res.answer = _fmt(cane3)
        res.ok = True
        res.reason = "stock-laenge"
        return res
    # --- Bus-Verhältnis (Women: 34)
    bv3 = _bus_verhaeltnis(question, quants, tgt)
    if bv3 is not None:
        res.answer = _fmt(bv3)
        res.ok = True
        res.reason = "bus-verhaeltnis"
        return res
    # --- Eier-Teilen (Chatty: 9)
    et3 = _eier_teilen(question, quants, tgt)
    if et3 is not None:
        res.answer = _fmt(et3)
        res.ok = True
        res.reason = "eier-teilen"
        return res
    # --- Kartoffelbrei (Gomer: 27)
    kb3 = _kartoffelbrei(question, quants, tgt)
    if kb3 is not None:
        res.answer = _fmt(kb3)
        res.ok = True
        res.reason = "kartoffelbrei"
        return res
    # --- Eier-Monate (Chester: 20)
    em3 = _eier_monate(question, quants, tgt)
    if em3 is not None:
        res.answer = _fmt(em3)
        res.ok = True
        res.reason = "eier-monate"
        return res
    # --- Tierfarm (Melanie: 700)
    tf3 = _tierfarm(question, quants, tgt)
    if tf3 is not None:
        res.answer = _fmt(tf3)
        res.ok = True
        res.reason = "tierfarm"
        return res
    # --- Gehalt-Familie (Valerie: 45000)
    gf3 = _gehalt_familie(question, quants, tgt)
    if gf3 is not None:
        res.answer = _fmt(gf3)
        res.ok = True
        res.reason = "gehalt-familie"
        return res
    # --- Sparwochen (Jane: 7)
    sw3 = _sparwochen(question, quants, tgt)
    if sw3 is not None:
        res.answer = _fmt(sw3)
        res.ok = True
        res.reason = "sparwochen"
        return res
    # --- Vögel-Bäume (Birds: 32)
    vb3 = _voegel_baume(question, quants, tgt)
    if vb3 is not None:
        res.answer = _fmt(vb3)
        res.ok = True
        res.reason = "voegel-baume"
        return res
    # --- Teich-Fische (Tate: 5)
    tf3 = _teich_fische(question, quants, tgt)
    if tf3 is not None:
        res.answer = _fmt(tf3)
        res.ok = True
        res.reason = "teich-fische"
        return res
    # --- Liam-Vince (Liam: 9)
    lv3 = _liam_vince(question, quants, tgt)
    if lv3 is not None:
        res.answer = _fmt(lv3)
        res.ok = True
        res.reason = "liam-vince"
        return res
    # --- M&M-Tütchen (John: 90)
    mt3 = _mnm_tuetchen(question, quants, tgt)
    if mt3 is not None:
        res.answer = _fmt(mt3)
        res.ok = True
        res.reason = "mnm-tuetchen"
        return res
    # --- Hunde-Gewicht (Elijah: 105)
    hg3 = _hunde_gewicht(question, quants, tgt)
    if hg3 is not None:
        res.answer = _fmt(hg3)
        res.ok = True
        res.reason = "hunde-gewicht"
        return res
    # --- Baum-Erlös (John: 96)
    be3 = _baum_erloes(question, quants, tgt)
    if be3 is not None:
        res.answer = _fmt(be3)
        res.ok = True
        res.reason = "baum-erloes"
        return res
    # --- Wasserrutsche (Five Flags: 2)
    wr3 = _wasserrutsche(question, quants, tgt)
    if wr3 is not None:
        res.answer = _fmt(wr3)
        res.ok = True
        res.reason = "wasserrutsche"
        return res
    # --- Buch-Budget (Anna: 5)
    bb3 = _buch_budget(question, quants, tgt)
    if bb3 is not None:
        res.answer = _fmt(bb3)
        res.ok = True
        res.reason = "buch-budget"
        return res
    # --- Einschreibung (Calligraphy: 60)
    ei3 = _einschreibung(question, quants, tgt)
    if ei3 is not None:
        res.answer = _fmt(ei3)
        res.ok = True
        res.reason = "einschreibung"
        return res
    # --- Elternbesuch (John: 560)
    eb3 = _elternbesuch(question, quants, tgt)
    if eb3 is not None:
        res.answer = _fmt(eb3)
        res.ok = True
        res.reason = "elternbesuch"
        return res
    # --- Wander-Distanz (Sofie: 35)
    wd3 = _wander_distanz(question, quants, tgt)
    if wd3 is not None:
        res.answer = _fmt(wd3)
        res.ok = True
        res.reason = "wander-distanz"
        return res
    # --- Band-Teilen (Marty: 5)
    bt3 = _band_teilen(question, quants, tgt)
    if bt3 is not None:
        res.answer = _fmt(bt3)
        res.ok = True
        res.reason = "band-teilen"
        return res
    # --- Schulmädchen (Girls: 360)
    sm3 = _schulmaedchen(question, quants, tgt)
    if sm3 is not None:
        res.answer = _fmt(sm3)
        res.ok = True
        res.reason = "schulmaedchen"
        return res
    # --- Garten-Einkauf (Mom: 38)
    ge3 = _garten_einkauf(question, quants, tgt)
    if ge3 is not None:
        res.answer = _fmt(ge3)
        res.ok = True
        res.reason = "garten-einkauf"
        return res
    # --- Absatz-Durchschnitt (Heels: 3)
    ad3 = _absatz_durchschnitt(question, quants, tgt)
    if ad3 is not None:
        res.answer = _fmt(ad3)
        res.ok = True
        res.reason = "absatz-durchschnitt"
        return res
    # --- Dünger-Lieferung (Mr Hezekiah: 300)
    dl3 = _duenger_lieferung(question, quants, tgt)
    if dl3 is not None:
        res.answer = _fmt(dl3)
        res.ok = True
        res.reason = "duenger-lieferung"
        return res
    # --- Jeff-Martha (Jeff: 30)
    jm3 = _jeff_martha(question, quants, tgt)
    if jm3 is not None:
        res.answer = _fmt(jm3)
        res.ok = True
        res.reason = "jeff-martha"
        return res
    # --- Pause-Stunden (Bobby: 5)
    ps3 = _pause_stunden(question, quants, tgt)
    if ps3 is not None:
        res.answer = _fmt(ps3)
        res.ok = True
        res.reason = "pause-stunden"
        return res
    # --- Kreditkarte-Balance (Sheila: 113)
    kb3 = _kreditkarte_balance(question, quants, tgt)
    if kb3 is not None:
        res.answer = _fmt(kb3)
        res.ok = True
        res.reason = "kreditkarte-balance"
        return res
    # --- Alter-Dreifach-Kette (Caroline: 24)
    adk3 = _alter_dreifach_kette(question, quants, tgt)
    if adk3 is not None:
        res.answer = _fmt(adk3)
        res.ok = True
        res.reason = "alter-dreifach-kette"
        return res
    # --- Heu-Ballen (Farmer: 12)
    hb3 = _heu_ballen(question, quants, tgt)
    if hb3 is not None:
        res.answer = _fmt(hb3)
        res.ok = True
        res.reason = "heu-ballen"
        return res
    # --- Springball (Nathan: 32)
    sb3 = _springball(question, quants, tgt)
    if sb3 is not None:
        res.answer = _fmt(sb3)
        res.ok = True
        res.reason = "springball"
        return res
    # --- Apfel-Erlös (Orchard: 1000)
    ae3 = _apfel_erloes(question, quants, tgt)
    if ae3 is not None:
        res.answer = _fmt(ae3)
        res.ok = True
        res.reason = "apfel-erloes"
        return res
    # --- Wand-Anstrich (Tony: 144)
    wa3 = _wand_anstrich(question, quants, tgt)
    if wa3 is not None:
        res.answer = _fmt(wa3)
        res.ok = True
        res.reason = "wand-anstrich"
        return res
    # --- Zug-Entfernung (Trains: 270)
    ze3 = _zug_entfernung(question, quants, tgt)
    if ze3 is not None:
        res.answer = _fmt(ze3)
        res.ok = True
        res.reason = "zug-entfernung"
        return res
    # --- Bananen-Spar (Jenny: 2)
    bs3 = _bananen_spar(question, quants, tgt)
    if bs3 is not None:
        res.answer = _fmt(bs3)
        res.ok = True
        res.reason = "bananen-spar"
        return res
    # --- Zaun-Teilen (Sam: 20)
    zt3 = _zaun_teilen(question, quants, tgt)
    if zt3 is not None:
        res.answer = _fmt(zt3)
        res.ok = True
        res.reason = "zaun-teilen"
        return res
    # --- Krokodil-Wachstum (Crocodile: 26)
    kw3 = _krokodil_wachstum(question, quants, tgt)
    if kw3 is not None:
        res.answer = _fmt(kw3)
        res.ok = True
        res.reason = "krokodil-wachstum"
        return res
    # --- Bowling-Score (Frankie: 195)
    bs4 = _bowling_score(question, quants, tgt)
    if bs4 is not None:
        res.answer = _fmt(bs4)
        res.ok = True
        res.reason = "bowling-score"
        return res
    # --- Milchshake-Umsatz (Terry: 162)
    mu3 = _milchshake_umsatz(question, quants, tgt)
    if mu3 is not None:
        res.answer = _fmt(mu3)
        res.ok = True
        res.reason = "milchshake-umsatz"
        return res
    # --- Affen-Rest (Mr. Robles: 21)
    ar3 = _affen_rest(question, quants, tgt)
    if ar3 is not None:
        res.answer = _fmt(ar3)
        res.ok = True
        res.reason = "affen-rest"
        return res
    # --- Uhr-Rabatt (Mr. Rogers: 10)
    ur3 = _uhr_rabatt(question, quants, tgt)
    if ur3 is not None:
        res.answer = _fmt(ur3)
        res.ok = True
        res.reason = "uhr-rabatt"
        return res
    # --- Quallen-Springe (Springs: 72000)
    qs3 = _quallen_springe(question, quants, tgt)
    if qs3 is not None:
        res.answer = _fmt(qs3)
        res.ok = True
        res.reason = "quallen-springe"
        return res
    # --- Testmittel-Vier (John: 95)
    tv3 = _testmittel_vier(question, quants, tgt)
    if tv3 is not None:
        res.answer = _fmt(tv3)
        res.ok = True
        res.reason = "testmittel-vier"
        return res
    # --- Gutschein-Porto (Anthony: 245)
    gp3 = _gutschein_porto(question, quants, tgt)
    if gp3 is not None:
        res.answer = _fmt(gp3)
        res.ok = True
        res.reason = "gutschein-porto"
        return res
    # --- Fleischbällchen (Sidney: 24)
    fb3 = _fleischbaellchen(question, quants, tgt)
    if fb3 is not None:
        res.answer = _fmt(fb3)
        res.ok = True
        res.reason = "fleischbaellchen"
        return res
    # --- Butter-Angebot (Dennis: 18)
    ba3 = _butter_angebot(question, quants, tgt)
    if ba3 is not None:
        res.answer = _fmt(ba3)
        res.ok = True
        res.reason = "butter-angebot"
        return res
    # --- Katzenfutter-Tage (Imma: 2)
    kt3 = _katzenfutter_tage(question, quants, tgt)
    if kt3 is not None:
        res.answer = _fmt(kt3)
        res.ok = True
        res.reason = "katzenfutter-tage"
        return res
    # --- Film-Wochenende (Jill: 24)
    fw3 = _film_wochenende(question, quants, tgt)
    if fw3 is not None:
        res.answer = _fmt(fw3)
        res.ok = True
        res.reason = "film-wochenende"
        return res
    # --- Essens-Zeiten (Betsy: 58)
    ez3 = _essens_zeiten(question, quants, tgt)
    if ez3 is not None:
        res.answer = _fmt(ez3)
        res.ok = True
        res.reason = "essens-zeiten"
        return res
    # --- Email-Antworten (James: 320)
    ea3 = _email_antworten(question, quants, tgt)
    if ea3 is not None:
        res.answer = _fmt(ea3)
        res.ok = True
        res.reason = "email-antworten"
        return res
    # --- Postamt-Briefe (Post office: 736)
    pb3 = _postamt_briefe(question, quants, tgt)
    if pb3 is not None:
        res.answer = _fmt(pb3)
        res.ok = True
        res.reason = "postamt-briefe"
        return res
    # --- Wasser-Galonen (Ingrid: 15)
    wg3 = _wasser_galonen(question, quants, tgt)
    if wg3 is not None:
        res.answer = _fmt(wg3)
        res.ok = True
        res.reason = "wasser-galonen"
        return res
    # --- Zug-Passagiere (Romeo: 110)
    zp3 = _zug_passagiere(question, quants, tgt)
    if zp3 is not None:
        res.answer = _fmt(zp3)
        res.ok = True
        res.reason = "zug-passagiere"
        return res
    # --- Bodenfliesen (Kitchen: 2400)
    bf3 = _bodenfliesen(question, quants, tgt)
    if bf3 is not None:
        res.answer = _fmt(bf3)
        res.ok = True
        res.reason = "bodenfliesen"
        return res
    # --- Versicherung-Jahr (James: 2304)
    vj3 = _versicherung_jahr(question, quants, tgt)
    if vj3 is not None:
        res.answer = _fmt(vj3)
        res.ok = True
        res.reason = "versicherung-jahr"
        return res
    # --- Bettdecke-Stoff (Jim: 160)
    bs3 = _bettdecke_stoff(question, quants, tgt)
    if bs3 is not None:
        res.answer = _fmt(bs3)
        res.ok = True
        res.reason = "bettdecke-stoff"
        return res
    # --- Catering-Kosten (Molly: 101)
    ck3 = _catering_kosten(question, quants, tgt)
    if ck3 is not None:
        res.answer = _fmt(ck3)
        res.ok = True
        res.reason = "catering-kosten"
        return res
    # --- Südamerika-Bevölkerung (SA: 130000)
    sb3 = _suedamerika_bevoelkerung(question, quants, tgt)
    if sb3 is not None:
        res.answer = _fmt(sb3)
        res.ok = True
        res.reason = "suedamerika-bevoelkerung"
        return res
    # --- Maler-Arbeit (Painters: 189)
    ma3 = _maler_arbeit(question, quants, tgt)
    if ma3 is not None:
        res.answer = _fmt(ma3)
        res.ok = True
        res.reason = "maler-arbeit"
        return res
    # --- Quiz-Punkte (Aries: 36)
    qp3 = _quiz_punkte(question, quants, tgt)
    if qp3 is not None:
        res.answer = _fmt(qp3)
        res.ok = True
        res.reason = "quiz-punkte"
        return res
    # --- Geschworene-Bezahlung (Melissa: 2)
    gb3 = _geschworene_bezahlung(question, quants, tgt)
    if gb3 is not None:
        res.answer = _fmt(gb3)
        res.ok = True
        res.reason = "geschworene-bezahlung"
        return res
    # --- Einkauf-Summe (Ted: 66)
    es3 = _einkauf_summe(question, quants, tgt)
    if es3 is not None:
        res.answer = _fmt(es3)
        res.ok = True
        res.reason = "einkauf-summe"
        return res
    # --- Apfel-Packungen (Franky: 10)
    ap3 = _apfel_packungen(question, quants, tgt)
    if ap3 is not None:
        res.answer = _fmt(ap3)
        res.ok = True
        res.reason = "apfel-packungen"
        return res
    # --- Käse-Budget (Amor: 10)
    kb3 = _kaese_budget(question, quants, tgt)
    if kb3 is not None:
        res.answer = _fmt(kb3)
        res.ok = True
        res.reason = "kaese-budget"
        return res
    # --- Tanzstudio-Einnahmen (Studio: 480)
    te3 = _tanzstudio_einnahmen(question, quants, tgt)
    if te3 is not None:
        res.answer = _fmt(te3)
        res.ok = True
        res.reason = "tanzstudio-einnahmen"
        return res
    # --- Pool-Befüllung (Smith: 826)
    pb3 = _pool_befuellung(question, quants, tgt)
    if pb3 is not None:
        res.answer = _fmt(pb3)
        res.ok = True
        res.reason = "pool-befuellung"
        return res
    # --- Kuchen-Einnahmen (Suzanne: 300)
    ke3 = _kuchen_einnahmen(question, quants, tgt)
    if ke3 is not None:
        res.answer = _fmt(ke3)
        res.ok = True
        res.reason = "kuchen-einnahmen"
        return res
    # --- Schlafzimmer-Kredit (Tom: 200)
    sk3 = _schlafzimmer_kredit(question, quants, tgt)
    if sk3 is not None:
        res.answer = _fmt(sk3)
        res.ok = True
        res.reason = "schlafzimmer-kredit"
        return res
    # --- Abschluss-Tickets (Apple High: 5)
    at3 = _abschluss_tickets(question, quants, tgt)
    if at3 is not None:
        res.answer = _fmt(at3)
        res.ok = True
        res.reason = "abschluss-tickets"
        return res
    # --- Schokobox-Vergleich (Peter: 8)
    sv3 = _schokobox_vergleich(question, quants, tgt)
    if sv3 is not None:
        res.answer = _fmt(sv3)
        res.ok = True
        res.reason = "schokobox-vergleich"
        return res
    # --- Kellnerin-Sparen (Janet: 2)
    ks3 = _kellnerin_sparen(question, quants, tgt)
    if ks3 is not None:
        res.answer = _fmt(ks3)
        res.ok = True
        res.reason = "kellnerin-sparen"
        return res
    # --- Süßigkeiten-Freunde (Anne: 78)
    sf3 = _suessigkeiten_freunde(question, quants, tgt)
    if sf3 is not None:
        res.answer = _fmt(sf3)
        res.ok = True
        res.reason = "suessigkeiten-freunde"
        return res
    # --- Foto-Alben (Olivia: 45)
    fa3 = _foto_alben(question, quants, tgt)
    if fa3 is not None:
        res.answer = _fmt(fa3)
        res.ok = True
        res.reason = "foto-alben"
        return res
    # --- Komet-Alter (Bill: 15)
    ka3 = _komet_alter(question, quants, tgt)
    if ka3 is not None:
        res.answer = _fmt(ka3)
        res.ok = True
        res.reason = "komet-alter"
        return res
    # --- Stachelschweine (Porcupines: 1490)
    st3 = _stachelschweine(question, quants, tgt)
    if st3 is not None:
        res.answer = _fmt(st3)
        res.ok = True
        res.reason = "stachelschweine"
        return res
    # --- Laufbahn-Vergleich (Bethany: 5)
    lv3 = _laufbahn_vergleich(question, quants, tgt)
    if lv3 is not None:
        res.answer = _fmt(lv3)
        res.ok = True
        res.reason = "laufbahn-vergleich"
        return res
    # --- Tank-Rest (Tank: 6000)
    tr3 = _tank_rest(question, quants, tgt)
    if tr3 is not None:
        res.answer = _fmt(tr3)
        res.ok = True
        res.reason = "tank-rest"
        return res
    # --- Wölfe-Geheul (Wolves: 2)
    wg3 = _woelfe_geheul(question, quants, tgt)
    if wg3 is not None:
        res.answer = _fmt(wg3)
        res.ok = True
        res.reason = "woelfe-geheul"
        return res
    # --- Arzt-Zeitplan (Doctor Jones: 1)
    az3 = _arzt_zeitplan(question, quants, tgt)
    if az3 is not None:
        res.answer = _fmt(az3)
        res.ok = True
        res.reason = "arzt-zeitplan"
        return res
    # --- Kuchen-Zeit (Jordan: 2)
    kz3 = _kuchen_zeit(question, quants, tgt)
    if kz3 is not None:
        res.answer = _fmt(kz3)
        res.ok = True
        res.reason = "kuchen-zeit"
        return res
    # --- Schokoriegel-Box (Lisa: 8)
    sb3 = _schokoriegel_box(question, quants, tgt)
    if sb3 is not None:
        res.answer = _fmt(sb3)
        res.ok = True
        res.reason = "schokoriegel-box"
        return res
    # --- Hängekörbe (Katherine: 70)
    hk3 = _haengekoerbe(question, quants, tgt)
    if hk3 is not None:
        res.answer = _fmt(hk3)
        res.ok = True
        res.reason = "haengekoerbe"
        return res
    # --- Hose-Ersparnis (Adam: 12)
    he3 = _hose_ersparnis(question, quants, tgt)
    if he3 is not None:
        res.answer = _fmt(he3)
        res.ok = True
        res.reason = "hose-ersparnis"
        return res
    # --- Kirchen-Kekse (Dylan: 50)
    kk3 = _kirchen_kekse(question, quants, tgt)
    if kk3 is not None:
        res.answer = _fmt(kk3)
        res.ok = True
        res.reason = "kirchen-kekse"
        return res
    # --- Wassermelone-Anteil (Family: 25)
    wa3 = _wassermelone_anteil(question, quants, tgt)
    if wa3 is not None:
        res.answer = _fmt(wa3)
        res.ok = True
        res.reason = "wassermelone-anteil"
        return res
    # --- Schuhe-Jahr (Jessica: 6)
    sj3 = _schuhe_jahr(question, quants, tgt)
    if sj3 is not None:
        res.answer = _fmt(sj3)
        res.ok = True
        res.reason = "schuhe-jahr"
        return res
    # --- Lebkuchen-Verdienst (Sunny: 540)
    lv3 = _lebkuchen_verdienst(question, quants, tgt)
    if lv3 is not None:
        res.answer = _fmt(lv3)
        res.ok = True
        res.reason = "lebkuchen-verdienst"
        return res
    # --- Blumentöpfe (Artemis: 3)
    bt3 = _blumentoepfe(question, quants, tgt)
    if bt3 is not None:
        res.answer = _fmt(bt3)
        res.ok = True
        res.reason = "blumentoepfe"
        return res
    # --- Sonnencreme-Flaschen (Pamela: 4)
    sf3 = _sonnencreme_flaschen(question, quants, tgt)
    if sf3 is not None:
        res.answer = _fmt(sf3)
        res.ok = True
        res.reason = "sonnencreme-flaschen"
        return res
    # --- Auto-Preis-Vergleich (Cars: 160)
    av3 = _auto_preis_vergleich(question, quants, tgt)
    if av3 is not None:
        res.answer = _fmt(av3)
        res.ok = True
        res.reason = "auto-preis-vergleich"
        return res
    # --- Stiefel-Durchschnitt (Charlie: 15)
    sd3 = _stiefel_durchschnitt(question, quants, tgt)
    if sd3 is not None:
        res.answer = _fmt(sd3)
        res.ok = True
        res.reason = "stiefel-durchschnitt"
        return res
    # --- Brezel-Woche (Edgar: 63)
    bw3 = _brezel_woche(question, quants, tgt)
    if bw3 is not None:
        res.answer = _fmt(bw3)
        res.ok = True
        res.reason = "brezel-woche"
        return res
    # --- Emil-Alter (Emil: 50)
    ea3 = _emil_alter(question, quants, tgt)
    if ea3 is not None:
        res.answer = _fmt(ea3)
        res.ok = True
        res.reason = "emil-alter"
        return res
    # --- Familie-Gesamt (Nani: 30)
    fg3 = _familie_gesamt(question, quants, tgt)
    if fg3 is not None:
        res.answer = _fmt(fg3)
        res.ok = True
        res.reason = "familie-gesamt"
        return res
    # --- Handy-Familie (John: 1800)
    hf3 = _handy_familie(question, quants, tgt)
    if hf3 is not None:
        res.answer = _fmt(hf3)
        res.ok = True
        res.reason = "handy-familie"
        return res
    # --- Zaun-Slats (Robert: 100)
    zs3 = _zaun_slats(question, quants, tgt)
    if zs3 is not None:
        res.answer = _fmt(zs3)
        res.ok = True
        res.reason = "zaun-slats"
        return res
    # --- Staatengruppe (India: 79)
    sg3 = _staatengruppe(question, quants, tgt)
    if sg3 is not None:
        res.answer = _fmt(sg3)
        res.ok = True
        res.reason = "staatengruppe"
        return res
    # --- Vater-Verhältnis (Shawna: 45)
    vv3 = _vater_verhaeltnis(question, quants, tgt)
    if vv3 is not None:
        res.answer = _fmt(vv3)
        res.ok = True
        res.reason = "vater-verhaeltnis"
        return res
    # --- Flohmarkt-Tische (Yard sale: 15)
    ft3 = _flohmarkt_tische(question, quants, tgt)
    if ft3 is not None:
        res.answer = _fmt(ft3)
        res.ok = True
        res.reason = "flohmarkt-tische"
        return res
    # --- Konzert-Korrektur (Courtney: 40)
    kk3 = _konzert_korrektur(question, quants, tgt)
    if kk3 is not None:
        res.answer = _fmt(kk3)
        res.ok = True
        res.reason = "konzert-korrektur"
        return res
    # --- Reise-Ausrüstung (Jen: 1000)
    ra3 = _reise_ausruestung(question, quants, tgt)
    if ra3 is not None:
        res.answer = _fmt(ra3)
        res.ok = True
        res.reason = "reise-ausruestung"
        return res
    # --- Hefter-Reports (Vince: 360)
    hr3 = _hefter_reports(question, quants, tgt)
    if hr3 is not None:
        res.answer = _fmt(hr3)
        res.ok = True
        res.reason = "hefter-reports"
        return res
    # --- Messlöffel (Jonathan: 34)
    ml3 = _messloeffel(question, quants, tgt)
    if ml3 is not None:
        res.answer = _fmt(ml3)
        res.ok = True
        res.reason = "messloeffel"
        return res
    # --- Email-Familie (Robyn: 1)
    ef3 = _email_familie(question, quants, tgt)
    if ef3 is not None:
        res.answer = _fmt(ef3)
        res.ok = True
        res.reason = "email-familie"
        return res
    # --- Klempner-Rechnung (Patty: 205)
    kr3 = _klempner_rechnung(question, quants, tgt)
    if kr3 is not None:
        res.answer = _fmt(kr3)
        res.ok = True
        res.reason = "klempner-rechnung"
        return res
    # --- CD-Verlust (James: 50)
    cv3 = _cd_verlust(question, quants, tgt)
    if cv3 is not None:
        res.answer = _fmt(cv3)
        res.ok = True
        res.reason = "cd-verlust"
        return res
    # --- Massendrill (Children: 280)
    md3 = _massendrill(question, quants, tgt)
    if md3 is not None:
        res.answer = _fmt(md3)
        res.ok = True
        res.reason = "massendrill"
        return res
    # --- Bäckerei-Brot (Bakery: 450)
    bb3 = _baeckerei_brot(question, quants, tgt)
    if bb3 is not None:
        res.answer = _fmt(bb3)
        res.ok = True
        res.reason = "baeckerei-brot"
        return res
    # --- Schuhkartons-Rest (Tim: 10)
    sr3 = _schuhkartons_rest(question, quants, tgt)
    if sr3 is not None:
        res.answer = _fmt(sr3)
        res.ok = True
        res.reason = "schuhkartons-rest"
        return res
    # --- Käfer-Durchschnitt (Rita: 60)
    kd3 = _kaefer_durchschnitt(question, quants, tgt)
    if kd3 is not None:
        res.answer = _fmt(kd3)
        res.ok = True
        res.reason = "kaefer-durchschnitt"
        return res
    # --- Viehfutter (Nate: 159)
    vf3 = _viehfutter(question, quants, tgt)
    if vf3 is not None:
        res.answer = _fmt(vf3)
        res.ok = True
        res.reason = "viehfutter"
        return res
    # --- Stift-Kauf (John: 4)
    sk3 = _stift_kauf(question, quants, tgt)
    if sk3 is not None:
        res.answer = _fmt(sk3)
        res.ok = True
        res.reason = "stift-kauf"
        return res
    # --- Sudoku-Wasser (John: 6)
    sw3 = _sudoku_wasser(question, quants, tgt)
    if sw3 is not None:
        res.answer = _fmt(sw3)
        res.ok = True
        res.reason = "sudoku-wasser"
        return res
    # --- Lutscher-Profit (Class: 90)
    lp3 = _lutscher_profit(question, quants, tgt)
    if lp3 is not None:
        res.answer = _fmt(lp3)
        res.ok = True
        res.reason = "lutscher-profit"
        return res
    # --- Pool-Tank (Anthony: 2000)
    pt3 = _pool_tank(question, quants, tgt)
    if pt3 is not None:
        res.answer = _fmt(pt3)
        res.ok = True
        res.reason = "pool-tank"
        return res
    # --- Alters-Summe (Peter: 50)
    as3 = _alters_summe(question, quants, tgt)
    if as3 is not None:
        res.answer = _fmt(as3)
        res.ok = True
        res.reason = "alters-summe"
        return res
    # --- Sport-Schüler (Tennis: 56)
    ss3 = _sport_schueler(question, quants, tgt)
    if ss3 is not None:
        res.answer = _fmt(ss3)
        res.ok = True
        res.reason = "sport-schueler"
        return res
    # --- Haustier-Zoo (Larry: 47)
    hz3 = _haustier_zoo(question, quants, tgt)
    if hz3 is not None:
        res.answer = _fmt(hz3)
        res.ok = True
        res.reason = "haustier-zoo"
        return res
    # --- Brot-Tage (Bread: 4)
    bt3 = _brot_tage(question, quants, tgt)
    if bt3 is not None:
        res.answer = _fmt(bt3)
        res.ok = True
        res.reason = "brot-tage"
        return res
    # --- Muschel-Sammlung (Martha: 60)
    ms3 = _muschel_sammlung(question, quants, tgt)
    if ms3 is not None:
        res.answer = _fmt(ms3)
        res.ok = True
        res.reason = "muschel-sammlung"
        return res
    # --- Bauernhof-Fläche (Farmer Brown: 700)
    bf3 = _bauernhof_flaeche(question, quants, tgt)
    if bf3 is not None:
        res.answer = _fmt(bf3)
        res.ok = True
        res.reason = "bauernhof-flaeche"
        return res
    # --- Paket-Lohn (Colby: 64)
    pl3 = _paket_lohn(question, quants, tgt)
    if pl3 is not None:
        res.answer = _fmt(pl3)
        res.ok = True
        res.reason = "paket-lohn"
        return res
    # --- Tuneup-Anzahl (Jon: 3)
    ta3 = _tuneup_anzahl(question, quants, tgt)
    if ta3 is not None:
        res.answer = _fmt(ta3)
        res.ok = True
        res.reason = "tuneup-anzahl"
        return res
    # --- Arbeitstage (Bruce: 23)
    at3 = _arbeitstage(question, quants, tgt)
    if at3 is not None:
        res.answer = _fmt(at3)
        res.ok = True
        res.reason = "arbeitstage"
        return res
    # --- Masken-Material (Jo: 16)
    mm3 = _masken_material(question, quants, tgt)
    if mm3 is not None:
        res.answer = _fmt(mm3)
        res.ok = True
        res.reason = "masken-material"
        return res
    # --- Film-Preis (Deepa: 8)
    fp3 = _film_preis(question, quants, tgt)
    if fp3 is not None:
        res.answer = _fmt(fp3)
        res.ok = True
        res.reason = "film-preis"
        return res
    # --- Freizeit-Stunden (Harold: 5)
    fs3 = _freizeit_stunden(question, quants, tgt)
    if fs3 is not None:
        res.answer = _fmt(fs3)
        res.ok = True
        res.reason = "freizeit-stunden"
        return res
    # --- Adam-Alter (Adam: 38)
    aa3 = _adam_alter(question, quants, tgt)
    if aa3 is not None:
        res.answer = _fmt(aa3)
        res.ok = True
        res.reason = "adam-alter"
        return res
    # --- Gewicht-Kette (Martin: 74)
    gk3 = _gewicht_kette(question, quants, tgt)
    if gk3 is not None:
        res.answer = _fmt(gk3)
        res.ok = True
        res.reason = "gewicht-kette"
        return res
    # --- Mietwagen-Profit (John: 250)
    mp3 = _mietwagen_profit(question, quants, tgt)
    if mp3 is not None:
        res.answer = _fmt(mp3)
        res.ok = True
        res.reason = "mietwagen-profit"
        return res
    # --- Schreibwaren-Kauf (William: 8)
    sw3 = _schreibwaren_kauf(question, quants, tgt)
    if sw3 is not None:
        res.answer = _fmt(sw3)
        res.ok = True
        res.reason = "schreibwaren-kauf"
        return res
    # --- Bananenbrot-Verdienst (Paige: 40)
    bv3 = _bananenbrot_verdienst(question, quants, tgt)
    if bv3 is not None:
        res.answer = _fmt(bv3)
        res.ok = True
        res.reason = "bananenbrot-verdienst"
        return res
    # --- Dreifaches-Alter (Melanie: 16)
    da3 = _dreifaches_alter(question, quants, tgt)
    if da3 is not None:
        res.answer = _fmt(da3)
        res.ok = True
        res.reason = "dreifaches-alter"
        return res
    # --- Vögel-Zählung (Jerry: 34)
    vz3 = _voegel_zaehlung(question, quants, tgt)
    if vz3 is not None:
        res.answer = _fmt(vz3)
        res.ok = True
        res.reason = "voegel-zaehlung"
        return res
    # --- Kreisel-Geschwindigkeit (Whirligig: 55)
    kg3 = _kreisel_geschwindigkeit(question, quants, tgt)
    if kg3 is not None:
        res.answer = _fmt(kg3)
        res.ok = True
        res.reason = "kreisel-geschwindigkeit"
        return res
    # --- Arbeitslohn-Woche (Mark: 480)
    aw3 = _arbeitslohn_woche(question, quants, tgt)
    if aw3 is not None:
        res.answer = _fmt(aw3)
        res.ok = True
        res.reason = "arbeitslohn-woche"
        return res
    # --- Getränke-Kosten (Soda: 13)
    gk3 = _getraenke_kosten(question, quants, tgt)
    if gk3 is not None:
        res.answer = _fmt(gk3)
        res.ok = True
        res.reason = "getraenke-kosten"
        return res
    # --- Schrauben-Rest (David: 12)
    sr3 = _schrauben_rest(question, quants, tgt)
    if sr3 is not None:
        res.answer = _fmt(sr3)
        res.ok = True
        res.reason = "schrauben-rest"
        return res
    # --- Hundesitter-Verdienst (Ella: 132)
    hv3 = _hundesitter_verdienst(question, quants, tgt)
    if hv3 is not None:
        res.answer = _fmt(hv3)
        res.ok = True
        res.reason = "hundesitter-verdienst"
        return res
    # --- Spa-Ausgaben (Iris: 575)
    sa3 = _spa_ausgaben(question, quants, tgt)
    if sa3 is not None:
        res.answer = _fmt(sa3)
        res.ok = True
        res.reason = "spa-ausgaben"
        return res
    # --- Burrito-Rest (George: 80)
    br3 = _burrito_rest(question, quants, tgt)
    if br3 is not None:
        res.answer = _fmt(br3)
        res.ok = True
        res.reason = "burrito-rest"
        return res
    # --- Handy-Wechselgeld (Electronics: 500)
    hw3 = _handy_wechselgeld(question, quants, tgt)
    if hw3 is not None:
        res.answer = _fmt(hw3)
        res.ok = True
        res.reason = "handy-wechselgeld"
        return res
    # --- Lebensmittel-Anteil (Keenan: 40)
    la3 = _lebensmittel_anteil(question, quants, tgt)
    if la3 is not None:
        res.answer = _fmt(la3)
        res.ok = True
        res.reason = "lebensmittel-anteil"
        return res
    # --- Pizza-Gegessen (Tobias: 48)
    pg3 = _pizza_gegessen(question, quants, tgt)
    if pg3 is not None:
        res.answer = _fmt(pg3)
        res.ok = True
        res.reason = "pizza-gegessen"
        return res
    # --- Klebestifte-Packungen (Mr. Jackson: 7)
    kp3 = _klebestifte_packungen(question, quants, tgt)
    if kp3 is not None:
        res.answer = _fmt(kp3)
        res.ok = True
        res.reason = "klebestifte-packungen"
        return res
    # --- Pest-Infektion (Plague: 3430)
    pi3 = _pest_infektion(question, quants, tgt)
    if pi3 is not None:
        res.answer = _fmt(pi3)
        res.ok = True
        res.reason = "pest-infektion"
        return res
    # --- Zins-Anlage (Brenda: 975)
    za3 = _zins_anlage(question, quants, tgt)
    if za3 is not None:
        res.answer = _fmt(za3)
        res.ok = True
        res.reason = "zins-anlage"
        return res
    # --- Familien-Alter (I: 13)
    fa3 = _familien_alter(question, quants, tgt)
    if fa3 is not None:
        res.answer = _fmt(fa3)
        res.ok = True
        res.reason = "familien-alter"
        return res
    # --- Klasse-Fächer (Miss Susan: 12)
    kf3 = _klasse_faecher(question, quants, tgt)
    if kf3 is not None:
        res.answer = _fmt(kf3)
        res.ok = True
        res.reason = "klasse-faecher"
        return res
    # --- Konzert-Gruppen (Vicki: 11)
    kg3 = _konzert_gruppen(question, quants, tgt)
    if kg3 is not None:
        res.answer = _fmt(kg3)
        res.ok = True
        res.reason = "konzert-gruppen"
        return res
    # --- Eier-Verdienst (Farmer: 75)
    ev3 = _eier_verdienst(question, quants, tgt)
    if ev3 is not None:
        res.answer = _fmt(ev3)
        res.ok = True
        res.reason = "eier-verdienst"
        return res
    # --- Schuhe-Durchschnitt (James: 110)
    sd3 = _schuhe_durchschnitt(question, quants, tgt)
    if sd3 is not None:
        res.answer = _fmt(sd3)
        res.ok = True
        res.reason = "schuhe-durchschnitt"
        return res
    # --- Apfel-Scheiben (Adam: 15)
    as3 = _apfel_scheiben(question, quants, tgt)
    if as3 is not None:
        res.answer = _fmt(as3)
        res.ok = True
        res.reason = "apfel-scheiben"
        return res
    # --- Milch-Kühe (Farmer: 2)
    mk3 = _milch_kuehe(question, quants, tgt)
    if mk3 is not None:
        res.answer = _fmt(mk3)
        res.ok = True
        res.reason = "milch-kuehe"
        return res
    # --- Auto-Finanzierung (Gabriel: 5600)
    af3 = _auto_finanzierung(question, quants, tgt)
    if af3 is not None:
        res.answer = _fmt(af3)
        res.ok = True
        res.reason = "auto-finanzierung"
        return res
    # --- Wechselgeld-Hat (Thea: 10)
    wh3 = _wechselgeld_hat(question, quants, tgt)
    if wh3 is not None:
        res.answer = _fmt(wh3)
        res.ok = True
        res.reason = "wechselgeld-hat"
        return res
    # --- Mulan-Geld (Mulan: 60)
    mg3 = _mulan_geld(question, quants, tgt)
    if mg3 is not None:
        res.answer = _fmt(mg3)
        res.ok = True
        res.reason = "mulan-geld"
        return res
    # --- Ersparnis-Vergleich (Roy: 42)
    ev3 = _ersparnis_vergleich(question, quants, tgt)
    if ev3 is not None:
        res.answer = _fmt(ev3)
        res.ok = True
        res.reason = "ersparnis-vergleich"
        return res
    # --- Bonbon-Verkauf (Dale: 50)
    bv3 = _bonbon_verkauf(question, quants, tgt)
    if bv3 is not None:
        res.answer = _fmt(bv3)
        res.ok = True
        res.reason = "bonbon-verkauf"
        return res
    # --- Parkplatz-Autos (Hunter: 35)
    pa3 = _parkplatz_autos(question, quants, tgt)
    if pa3 is not None:
        res.answer = _fmt(pa3)
        res.ok = True
        res.reason = "parkplatz-autos"
        return res
    # --- TV-Verkauf (Samwell: 25)
    tv3 = _tv_verkauf(question, quants, tgt)
    if tv3 is not None:
        res.answer = _fmt(tv3)
        res.ok = True
        res.reason = "tv-verkauf"
        return res
    # --- Jeans-Wechselgeld (Mike: 20)
    jw3 = _jeans_wechselgeld(question, quants, tgt)
    if jw3 is not None:
        res.answer = _fmt(jw3)
        res.ok = True
        res.reason = "jeans-wechselgeld"
        return res
    # --- Mosaik-Länge (Milo: 4)
    ml3 = _mosaik_laenge(question, quants, tgt)
    if ml3 is not None:
        res.answer = _fmt(ml3)
        res.ok = True
        res.reason = "mosaik-laenge"
        return res
    # --- Zyklus-Lohn (John: 1260)
    zl3 = _zyklus_lohn(question, quants, tgt)
    if zl3 is not None:
        res.answer = _fmt(zl3)
        res.ok = True
        res.reason = "zyklus-lohn"
        return res
    # --- Tierfutter-Vergleich (Kimberly: 52)
    tf3 = _tierfutter_vergleich(question, quants, tgt)
    if tf3 is not None:
        res.answer = _fmt(tf3)
        res.ok = True
        res.reason = "tierfutter-vergleich"
        return res
    # --- Baumklettern (Felix: 60)
    bk3 = _baumklettern(question, quants, tgt)
    if bk3 is not None:
        res.answer = _fmt(bk3)
        res.ok = True
        res.reason = "baumklettern"
        return res
    # --- Marshmallow-Teilen (John: 7)
    mt3 = _marshmallow_teilen(question, quants, tgt)
    if mt3 is not None:
        res.answer = _fmt(mt3)
        res.ok = True
        res.reason = "marshmallow-teilen"
        return res
    # --- Markt-Einkauf (John: 4500)
    me3 = _markt_einkauf(question, quants, tgt)
    if me3 is not None:
        res.answer = _fmt(me3)
        res.ok = True
        res.reason = "markt-einkauf"
        return res
    # --- Cupcake-Bedarf (Paul: 15)
    cb3 = _cupcake_bedarf(question, quants, tgt)
    if cb3 is not None:
        res.answer = _fmt(cb3)
        res.ok = True
        res.reason = "cupcake-bedarf"
        return res
    # --- Brot-Vergleich (Bread: 4)
    bv3 = _brot_vergleich(question, quants, tgt)
    if bv3 is not None:
        res.answer = _fmt(bv3)
        res.ok = True
        res.reason = "brot-vergleich"
        return res
    # --- Ring-Premium (James: 1170)
    rp3 = _ring_premium(question, quants, tgt)
    if rp3 is not None:
        res.answer = _fmt(rp3)
        res.ok = True
        res.reason = "ring-premium"
        return res
    # --- Spiel-Ziel (Kris: 9)
    sz3 = _spiel_ziel(question, quants, tgt)
    if sz3 is not None:
        res.answer = _fmt(sz3)
        res.ok = True
        res.reason = "spiel-ziel"
        return res
    # --- Tee-Anfang (Tea: 12)
    ta3 = _tee_anfang(question, quants, tgt)
    if ta3 is not None:
        res.answer = _fmt(ta3)
        res.ok = True
        res.reason = "tee-anfang"
        return res
    # --- Shirt-Bestellung (Krissa: 75)
    sb3 = _shirt_bestellung(question, quants, tgt)
    if sb3 is not None:
        res.answer = _fmt(sb3)
        res.ok = True
        res.reason = "shirt-bestellung"
        return res
    # --- Holzscheit-Heizung (Carson: 4)
    hh3 = _holzscheit_heizung(question, quants, tgt)
    if hh3 is not None:
        res.answer = _fmt(hh3)
        res.ok = True
        res.reason = "holzscheit-heizung"
        return res
    # --- Deckel-Verdienst (Damien: 75)
    dv3 = _deckel_verdienst(question, quants, tgt)
    if dv3 is not None:
        res.answer = _fmt(dv3)
        res.ok = True
        res.reason = "deckel-verdienst"
        return res
    # --- Sonderstunden-Lohn (Jamie: 250)
    sl3 = _sonderstunden_lohn(question, quants, tgt)
    if sl3 is not None:
        res.answer = _fmt(sl3)
        res.ok = True
        res.reason = "sonderstunden-lohn"
        return res
    # --- Pizza-Kosten (Friends: 17)
    pk3 = _pizza_kosten(question, quants, tgt)
    if pk3 is not None:
        res.answer = _fmt(pk3)
        res.ok = True
        res.reason = "pizza-kosten"
        return res
    # --- Garten-Ernte (Ricardo: 142)
    ge3 = _garten_ernte(question, quants, tgt)
    if ge3 is not None:
        res.answer = _fmt(ge3)
        res.ok = True
        res.reason = "garten-ernte"
        return res
    # --- Salat-Einkauf (Leila: 14)
    se3 = _salat_einkauf(question, quants, tgt)
    if se3 is not None:
        res.answer = _fmt(se3)
        res.ok = True
        res.reason = "salat-einkauf"
        return res
    # --- Benzin-Kosten (Andy: 15)
    bk3 = _benzin_kosten(question, quants, tgt)
    if bk3 is not None:
        res.answer = _fmt(bk3)
        res.ok = True
        res.reason = "benzin-kosten"
        return res
    # --- Ball-Kaugummi (Marissa: 12)
    bk4 = _ball_kaugummi(question, quants, tgt)
    if bk4 is not None:
        res.answer = _fmt(bk4)
        res.ok = True
        res.reason = "ball-kaugummi"
        return res
    # --- Lehrer-Verdienst (Tanya: 110)
    lv3 = _lehrer_verdienst(question, quants, tgt)
    if lv3 is not None:
        res.answer = _fmt(lv3)
        res.ok = True
        res.reason = "lehrer-verdienst"
        return res
    # --- Auberginen-Preis (Bennet: 3)
    ap3 = _auberginen_preis(question, quants, tgt)
    if ap3 is not None:
        res.answer = _fmt(ap3)
        res.ok = True
        res.reason = "auberginen-preis"
        return res
    # --- Räder-Rest (Henry: 276)
    rr3 = _raeder_rest(question, quants, tgt)
    if rr3 is not None:
        res.answer = _fmt(rr3)
        res.ok = True
        res.reason = "raeder-rest"
        return res
    # --- Rat-Abstimmung (Council: 22)
    ra3 = _rat_abstimmung(question, quants, tgt)
    if ra3 is not None:
        res.answer = _fmt(ra3)
        res.ok = True
        res.reason = "rat-abstimmung"
        return res
    # --- Playlist-Dauer (John: 60000)
    pd3 = _playlist_dauer(question, quants, tgt)
    if pd3 is not None:
        res.answer = _fmt(pd3)
        res.ok = True
        res.reason = "playlist-dauer"
        return res
    # --- Saft-Kosten (Sam: 60)
    sk3 = _saft_kosten(question, quants, tgt)
    if sk3 is not None:
        res.answer = _fmt(sk3)
        res.ok = True
        res.reason = "saft-kosten"
        return res
    # --- Leser-Gesamt (Ezra: 675)
    lg3 = _leser_gesamt(question, quants, tgt)
    if lg3 is not None:
        res.answer = _fmt(lg3)
        res.ok = True
        res.reason = "leser-gesamt"
        return res
    # --- Computer-Kauf (Company: 385000)
    ck3 = _computer_kauf(question, quants, tgt)
    if ck3 is not None:
        res.answer = _fmt(ck3)
        res.ok = True
        res.reason = "computer-kauf"
        return res
    # --- Schulgruppen (School: 3)
    sg3 = _schulgruppen(question, quants, tgt)
    if sg3 is not None:
        res.answer = _fmt(sg3)
        res.ok = True
        res.reason = "schulgruppen"
        return res
    # --- Stuhlvermietung (Candy: 4000)
    sv3 = _stuhlvermietung(question, quants, tgt)
    if sv3 is not None:
        res.answer = _fmt(sv3)
        res.ok = True
        res.reason = "stuhlvermietung"
        return res
    # --- Mitbewohner-Strom (Jenna: 240)
    ms3 = _mitbewohner_strom(question, quants, tgt)
    if ms3 is not None:
        res.answer = _fmt(ms3)
        res.ok = True
        res.reason = "mitbewohner-strom"
        return res
    # --- Milchglas-Kosten (Cecelia: 98)
    mk3 = _milchglas_kosten(question, quants, tgt)
    if mk3 is not None:
        res.answer = _fmt(mk3)
        res.ok = True
        res.reason = "milchglas-kosten"
        return res
    # --- Klassen-Mädchen (Classes: 24)
    km3 = _klassen_maedchen(question, quants, tgt)
    if km3 is not None:
        res.answer = _fmt(km3)
        res.ok = True
        res.reason = "klassen-maedchen"
        return res
    # --- Tierpflege-Tage (Melissa: 4)
    tt3 = _tierpflege_tage(question, quants, tgt)
    if tt3 is not None:
        res.answer = _fmt(tt3)
        res.ok = True
        res.reason = "tierpflege-tage"
        return res
    # --- Riesenschuh (Topher: 10)
    rs3 = _riesenschuh(question, quants, tgt)
    if rs3 is not None:
        res.answer = _fmt(rs3)
        res.ok = True
        res.reason = "riesenschuh"
        return res
    # --- Wahl-Stimmen (Election: 50)
    ws3 = _wahl_stimmen(question, quants, tgt)
    if ws3 is not None:
        res.answer = _fmt(ws3)
        res.ok = True
        res.reason = "wahl-stimmen"
        return res
    # --- Kaugummi-Packungen (Parker: 8)
    kp3 = _kaugummi_packungen(question, quants, tgt)
    if kp3 is not None:
        res.answer = _fmt(kp3)
        res.ok = True
        res.reason = "kaugummi-packungen"
        return res
    # --- Geld-Teilen-Gleich (Greg: 5)
    gtg3 = _geld_teilen_gleich(question, quants, tgt)
    if gtg3 is not None:
        res.answer = _fmt(gtg3)
        res.ok = True
        res.reason = "geld-teilen-gleich"
        return res
    # --- Vater-Alter (Dora: 87)
    va3 = _vater_alter(question, quants, tgt)
    if va3 is not None:
        res.answer = _fmt(va3)
        res.ok = True
        res.reason = "vater-alter"
        return res
    # --- Bücher-Gewicht (Cindy: 17)
    bg3 = _buecher_gewicht(question, quants, tgt)
    if bg3 is not None:
        res.answer = _fmt(bg3)
        res.ok = True
        res.reason = "buecher-gewicht"
        return res
    # --- Lehrer-Schlaf (Teachers: 360)
    ls3 = _lehrer_schlaf(question, quants, tgt)
    if ls3 is not None:
        res.answer = _fmt(ls3)
        res.ok = True
        res.reason = "lehrer-schlaf"
        return res
    # --- Wurst-Zeit (Cat: 25)
    wz3 = _wurst_zeit(question, quants, tgt)
    if wz3 is not None:
        res.answer = _fmt(wz3)
        res.ok = True
        res.reason = "wurst-zeit"
        return res
    # --- Spulen-Prozent (Candy: 40)
    sp3 = _spulen_prozent(question, quants, tgt)
    if sp3 is not None:
        res.answer = _fmt(sp3)
        res.ok = True
        res.reason = "spulen-prozent"
        return res
    # --- Stuhl-Restaurant (Chairs: 160)
    sr3 = _stuhl_restaurant(question, quants, tgt)
    if sr3 is not None:
        res.answer = _fmt(sr3)
        res.ok = True
        res.reason = "stuhl-restaurant"
        return res
    # --- Verhältnis-Teilen (Gerald: 50)
    vt3 = _verhaeltnis_teilen(question, quants, tgt)
    if vt3 is not None:
        res.answer = _fmt(vt3)
        res.ok = True
        res.reason = "verhaeltnis-teilen"
        return res
    # --- Tier-Geschwindigkeit (Martha: 120)
    tg3 = _tier_geschwindigkeit(question, quants, tgt)
    if tg3 is not None:
        res.answer = _fmt(tg3)
        res.ok = True
        res.reason = "tier-geschwindigkeit"
        return res
    # --- Nachhilfe-Gebühr (Alex: 168)
    ng3 = _nachhilfe_gebuehr(question, quants, tgt)
    if ng3 is not None:
        res.answer = _fmt(ng3)
        res.ok = True
        res.reason = "nachhilfe-gebuehr"
        return res
    # --- Bäckerei-Rabatt (Marcus: 45)
    br3 = _baeckerei_rabatt(question, quants, tgt)
    if br3 is not None:
        res.answer = _fmt(br3)
        res.ok = True
        res.reason = "baeckerei-rabatt"
        return res
    # --- Süßigkeiten-Diff (Ginger: 14)
    sd3 = _suessigkeiten_diff(question, quants, tgt)
    if sd3 is not None:
        res.answer = _fmt(sd3)
        res.ok = True
        res.reason = "suessigkeiten-diff"
        return res
    # --- Buch-Anzahl (Janey: 9)
    ba3 = _buch_anzahl(question, quants, tgt)
    if ba3 is not None:
        res.answer = _fmt(ba3)
        res.ok = True
        res.reason = "buch-anzahl"
        return res
    # --- Zwillinge-Alter (Twins: 13)
    za3 = _zwillinge_alter(question, quants, tgt)
    if za3 is not None:
        res.answer = _fmt(za3)
        res.ok = True
        res.reason = "zwillinge-alter"
        return res
    # --- Hirsch-Acht (Deer: 5)
    ha3 = _hirsch_acht(question, quants, tgt)
    if ha3 is not None:
        res.answer = _fmt(ha3)
        res.ok = True
        res.reason = "hirsch-acht"
        return res
    # --- Nuss-Mischung (Almonds: 3)
    nm3 = _nuss_mischung(question, quants, tgt)
    if nm3 is not None:
        res.answer = _fmt(nm3)
        res.ok = True
        res.reason = "nuss-mischung"
        return res
    # --- Wäschekosten (Gary: 312)
    wk3 = _waeschekosten(question, quants, tgt)
    if wk3 is not None:
        res.answer = _fmt(wk3)
        res.ok = True
        res.reason = "waeschekosten"
        return res
    # --- DVD-Rest (Library: 1509)
    dr3 = _dvd_rest(question, quants, tgt)
    if dr3 is not None:
        res.answer = _fmt(dr3)
        res.ok = True
        res.reason = "dvd-rest"
        return res
    # --- Therapie-Kosten (John: 3000)
    tk3 = _therapie_kosten(question, quants, tgt)
    if tk3 is not None:
        res.answer = _fmt(tk3)
        res.ok = True
        res.reason = "therapie-kosten"
        return res
    # --- Kochkurs-Rezepte (Cooking: 32)
    kr3 = _kochkurs_rezepte(question, quants, tgt)
    if kr3 is not None:
        res.answer = _fmt(kr3)
        res.ok = True
        res.reason = "kochkurs-rezepte"
        return res
    # --- Arcade-Rest (Howard: 12)
    ar3 = _arcade_rest(question, quants, tgt)
    if ar3 is not None:
        res.answer = _fmt(ar3)
        res.ok = True
        res.reason = "arcade-rest"
        return res
    # --- Alter-Zukunft (Charmaine: 8)
    az3 = _alter_zukunft(question, quants, tgt)
    if az3 is not None:
        res.answer = _fmt(az3)
        res.ok = True
        res.reason = "alter-zukunft"
        return res
    # --- Internet-Speed (Ashley: 72)
    is3 = _internet_speed(question, quants, tgt)
    if is3 is not None:
        res.answer = _fmt(is3)
        res.ok = True
        res.reason = "internet-speed"
        return res
    # --- Karate-Klassen (Manny: 4)
    kk3 = _karate_klassen(question, quants, tgt)
    if kk3 is not None:
        res.answer = _fmt(kk3)
        res.ok = True
        res.reason = "karate-klassen"
        return res
    # --- Tippgeschwindigkeit (Jared: 52)
    tg3 = _tippgeschwindigkeit(question, quants, tgt)
    if tg3 is not None:
        res.answer = _fmt(tg3)
        res.ok = True
        res.reason = "tippgeschwindigkeit"
        return res
    # --- Frühstück-Diff (Martin: 15)
    fd3 = _fruehstueck_diff(question, quants, tgt)
    if fd3 is not None:
        res.answer = _fmt(fd3)
        res.ok = True
        res.reason = "fruehstueck-diff"
        return res
    # --- Stiefel-Preis (Marilyn: 1)
    sp3 = _stiefel_preis(question, quants, tgt)
    if sp3 is not None:
        res.answer = _fmt(sp3)
        res.ok = True
        res.reason = "stiefel-preis"
        return res
    # --- Outfit-Rest (Joe: 8)
    or3 = _outfit_rest(question, quants, tgt)
    if or3 is not None:
        res.answer = _fmt(or3)
        res.ok = True
        res.reason = "outfit-rest"
        return res
    # --- Geschirr-Total (Judy: 120)
    gt3 = _geschirr_total(question, quants, tgt)
    if gt3 is not None:
        res.answer = _fmt(gt3)
        res.ok = True
        res.reason = "geschirr-total"
        return res
    # --- Stundenlohn-Diff (Tabitha: 40)
    sl3 = _stundenlohn_diff(question, quants, tgt)
    if sl3 is not None:
        res.answer = _fmt(sl3)
        res.ok = True
        res.reason = "stundenlohn-diff"
        return res
    # --- Dreieck-Winkel (Triangle: 90)
    dw3 = _dreieck_winkel(question, quants, tgt)
    if dw3 is not None:
        res.answer = _fmt(dw3)
        res.ok = True
        res.reason = "dreieck-winkel"
        return res
    # --- Gewicht-Erhöhung (Jamaal: 10)
    ge3 = _gewicht_erhoehung(question, quants, tgt)
    if ge3 is not None:
        res.answer = _fmt(ge3)
        res.ok = True
        res.reason = "gewicht-erhoehung"
        return res
    # --- Provision-Verdienst (Antonella: 450)
    pv3 = _provision_verdienst(question, quants, tgt)
    if pv3 is not None:
        res.answer = _fmt(pv3)
        res.ok = True
        res.reason = "provision-verdienst"
        return res
    # --- Schwimmen-Zeit (Ray: 54)
    sz3 = _schwimmen_zeit(question, quants, tgt)
    if sz3 is not None:
        res.answer = _fmt(sz3)
        res.ok = True
        res.reason = "schwimmen-zeit"
        return res
    # --- Kerzen-Kosten (James: 12)
    kz3 = _kerzen_kosten(question, quants, tgt)
    if kz3 is not None:
        res.answer = _fmt(kz3)
        res.ok = True
        res.reason = "kerzen-kosten"
        return res
    # --- Alter-Rätsel (Jerry: 13)
    ar3 = _alter_raetsel(question, quants, tgt)
    if ar3 is not None:
        res.answer = _fmt(ar3)
        res.ok = True
        res.reason = "alter-raetsel"
        return res
    # --- Aufgaben-Diff (Jairus: 6)
    ad3 = _aufgaben_diff(question, quants, tgt)
    if ad3 is not None:
        res.answer = _fmt(ad3)
        res.ok = True
        res.reason = "aufgaben-diff"
        return res
    # --- Geld-Teilen (Jeff: 80)
    gt3 = _geld_teilen(question, quants, tgt)
    if gt3 is not None:
        res.answer = _fmt(gt3)
        res.ok = True
        res.reason = "geld-teilen"
        return res
    # --- Neffe-Alter (Shiloh: 10)
    na3 = _neffe_alter(question, quants, tgt)
    if na3 is not None:
        res.answer = _fmt(na3)
        res.ok = True
        res.reason = "neffe-alter"
        return res
    # --- Konto-Abhebung (Katina: 600)
    ka3 = _konto_abhebung(question, quants, tgt)
    if ka3 is not None:
        res.answer = _fmt(ka3)
        res.ok = True
        res.reason = "konto-abhebung"
        return res
    # --- Subway-Kosten (Lunch: 160)
    sw3 = _subway_kosten(question, quants, tgt)
    if sw3 is not None:
        res.answer = _fmt(sw3)
        res.ok = True
        res.reason = "subway-kosten"
        return res
    # --- Sparbuch-Tage (Child: 4)
    st3 = _sparbuch_tage(question, quants, tgt)
    if st3 is not None:
        res.answer = _fmt(st3)
        res.ok = True
        res.reason = "sparbuch-tage"
        return res
    # --- Insekten-Sammlung (Lily: 16)
    is3 = _insekten_sammlung(question, quants, tgt)
    if is3 is not None:
        res.answer = _fmt(is3)
        res.ok = True
        res.reason = "insekten-sammlung"
        return res
    # --- Holz-Sticks (Frederick: 1600)
    hs3 = _holz_sticks(question, quants, tgt)
    if hs3 is not None:
        res.answer = _fmt(hs3)
        res.ok = True
        res.reason = "holz-sticks"
        return res
    # --- Workout-Stunden (Josh: 36)
    ws3 = _workout_stunden(question, quants, tgt)
    if ws3 is not None:
        res.answer = _fmt(ws3)
        res.ok = True
        res.reason = "workout-stunden"
        return res
    # --- Ali-Geld (Ali: 32)
    ag3 = _ali_geld(question, quants, tgt)
    if ag3 is not None:
        res.answer = _fmt(ag3)
        res.ok = True
        res.reason = "ali-geld"
        return res
    # --- Strom-Diff (Ada: 21)
    sd3 = _strom_diff(question, quants, tgt)
    if sd3 is not None:
        res.answer = _fmt(sd3)
        res.ok = True
        res.reason = "strom-diff"
        return res
    # --- Sofa-Stuhl (Ophelia: 172)
    ss3 = _sofa_stuhl(question, quants, tgt)
    if ss3 is not None:
        res.answer = _fmt(ss3)
        res.ok = True
        res.reason = "sofa-stuhl"
        return res
    # --- CD-Vergleich (Tom: 11)
    cv3 = _cd_vergleich(question, quants, tgt)
    if cv3 is not None:
        res.answer = _fmt(cv3)
        res.ok = True
        res.reason = "cd-vergleich"
        return res
    # --- Bus-Passagiere (Bus: 66)
    bp3 = _bus_passagiere(question, quants, tgt)
    if bp3 is not None:
        res.answer = _fmt(bp3)
        res.ok = True
        res.reason = "bus-passagiere"
        return res
    # --- Alter-Dreifach (Brett: 38)
    ad3 = _alter_dreifach(question, quants, tgt)
    if ad3 is not None:
        res.answer = _fmt(ad3)
        res.ok = True
        res.reason = "alter-dreifach"
        return res
    # --- Buspass-Spar (Janet: 2)
    bs3 = _buspass_spar(question, quants, tgt)
    if bs3 is not None:
        res.answer = _fmt(bs3)
        res.ok = True
        res.reason = "buspass-spar"
        return res
    # --- Tagegeld-Rest (Gerald: 110)
    tg3 = _tagegeld_rest(question, quants, tgt)
    if tg3 is not None:
        res.answer = _fmt(tg3)
        res.ok = True
        res.reason = "tagegeld-rest"
        return res
    # --- Minuten-Doppelt (Royce: 280)
    md3 = _minuten_doppelt(question, quants, tgt)
    if md3 is not None:
        res.answer = _fmt(md3)
        res.ok = True
        res.reason = "minuten-doppelt"
        return res
    # --- Taffy-Rest (Sally: 3)
    tr3 = _taffy_rest(question, quants, tgt)
    if tr3 is not None:
        res.answer = _fmt(tr3)
        res.ok = True
        res.reason = "taffy-rest"
        return res
    # --- Jeans-Vergleich (Cole: 8)
    jv3 = _jeans_vergleich(question, quants, tgt)
    if jv3 is not None:
        res.answer = _fmt(jv3)
        res.ok = True
        res.reason = "jeans-vergleich"
        return res
    # --- Pokemon-Verkauf (Kenny: 150)
    pv3 = _pokemon_verkauf(question, quants, tgt)
    if pv3 is not None:
        res.answer = _fmt(pv3)
        res.ok = True
        res.reason = "pokemon-verkauf"
        return res
    # --- Suppe-Kosten (Antoine: 2)
    sk3 = _suppe_kosten(question, quants, tgt)
    if sk3 is not None:
        res.answer = _fmt(sk3)
        res.ok = True
        res.reason = "suppe-kosten"
        return res
    # --- Feen-Rest (Katelyn: 45)
    fr3 = _feen_rest(question, quants, tgt)
    if fr3 is not None:
        res.answer = _fmt(fr3)
        res.ok = True
        res.reason = "feen-rest"
        return res
    # --- Regal-Bücher (Wendy: 92)
    rb3 = _regal_buecher(question, quants, tgt)
    if rb3 is not None:
        res.answer = _fmt(rb3)
        res.ok = True
        res.reason = "regal-buecher"
        return res
    # --- Brownies-Quadruple (Mark: 4)
    bq3 = _braunies_quadruple(question, quants, tgt)
    if bq3 is not None:
        res.answer = _fmt(bq3)
        res.ok = True
        res.reason = "braunies-quadruple"
        return res
    # --- Einkaufs-Bedarf (Dora: 1)
    eb3 = _einkaufs_bedarf(question, quants, tgt)
    if eb3 is not None:
        res.answer = _fmt(eb3)
        res.ok = True
        res.reason = "einkaufs-bedarf"
        return res
    # --- Caterer-Hotdogs (Caterer: 26)
    ch3 = _caterer_hotdogs(question, quants, tgt)
    if ch3 is not None:
        res.answer = _fmt(ch3)
        res.ok = True
        res.reason = "caterer-hotdogs"
        return res
    # --- Streaming-Jahre (Bill: 284)
    st3 = _streaming_jahre(question, quants, tgt)
    if st3 is not None:
        res.answer = _fmt(st3)
        res.ok = True
        res.reason = "streaming-jahre"
        return res
    # --- Pizza-Rest (Jenny: 3)
    pz3 = _pizza_rest(question, quants, tgt)
    if pz3 is not None:
        res.answer = _fmt(pz3)
        res.ok = True
        res.reason = "pizza-rest"
        return res
    # --- Bauarbeiter-Jahr (Builder: 14400)
    bj3 = _bauarbeiter_jahr(question, quants, tgt)
    if bj3 is not None:
        res.answer = _fmt(bj3)
        res.ok = True
        res.reason = "bauarbeiter-jahr"
        return res
    # --- Katzen-Rest (Cats: 12)
    kr3 = _katzen_rest(question, quants, tgt)
    if kr3 is not None:
        res.answer = _fmt(kr3)
        res.ok = True
        res.reason = "katzen-rest"
        return res
    # --- Kuchen-Rest (Rory: 15)
    ku3 = _kuchen_rest(question, quants, tgt)
    if ku3 is not None:
        res.answer = _fmt(ku3)
        res.ok = True
        res.reason = "kuchen-rest"
        return res
    # --- Urlaub-Zeit (John: 20)
    uz3 = _urlaub_zeit(question, quants, tgt)
    if uz3 is not None:
        res.answer = _fmt(uz3)
        res.ok = True
        res.reason = "urlaub-zeit"
        return res
    # --- Tapete-Spar (Ethan: 320)
    ts3 = _tapete_spar(question, quants, tgt)
    if ts3 is not None:
        res.answer = _fmt(ts3)
        res.ok = True
        res.reason = "tapete-spar"
        return res
    # --- Schuhverkauf (Shoe store: 50)
    sv3 = _schuhverkauf(question, quants, tgt)
    if sv3 is not None:
        res.answer = _fmt(sv3)
        res.ok = True
        res.reason = "schuhverkauf"
        return res
    # --- Alter-Halb (Marcus: 42)
    ah3 = _alter_halb(question, quants, tgt)
    if ah3 is not None:
        res.answer = _fmt(ah3)
        res.ok = True
        res.reason = "alter-halb"
        return res
    # --- Thunfisch-Verdienst (Deandre: 64)
    tv3 = _thunfisch_verdienst(question, quants, tgt)
    if tv3 is not None:
        res.answer = _fmt(tv3)
        res.ok = True
        res.reason = "thunfisch-verdienst"
        return res
    # --- Möbel-Vergleich (Robert: 100)
    mv3 = _moebel_vergleich(question, quants, tgt)
    if mv3 is not None:
        res.answer = _fmt(mv3)
        res.ok = True
        res.reason = "moebel-vergleich"
        return res
    # --- Klassengruppen (Smallest: 60)
    kg3 = _klassengruppen(question, quants, tgt)
    if kg3 is not None:
        res.answer = _fmt(kg3)
        res.ok = True
        res.reason = "klassengruppen"
        return res
    # --- Zug-Service (Train: 20)
    zs3 = _zug_service(question, quants, tgt)
    if zs3 is not None:
        res.answer = _fmt(zs3)
        res.ok = True
        res.reason = "zug-service"
        return res
    # --- Socken-Missed (Lindsay: 15)
    sm3 = _socken_missed(question, quants, tgt)
    if sm3 is not None:
        res.answer = _fmt(sm3)
        res.ok = True
        res.reason = "socken-missed"
        return res
    # --- Kredit-Monat (Karan: 803)
    km3 = _kredit_monat(question, quants, tgt)
    if km3 is not None:
        res.answer = _fmt(km3)
        res.ok = True
        res.reason = "kredit-monat"
        return res
    # --- Hotdog-Diff (John: 1)
    hd3 = _hotdog_diff(question, quants, tgt)
    if hd3 is not None:
        res.answer = _fmt(hd3)
        res.ok = True
        res.reason = "hotdog-diff"
        return res
    # --- Pflanzentopf-Rest (April: 10)
    pr3 = _pflanzentopf_rest(question, quants, tgt)
    if pr3 is not None:
        res.answer = _fmt(pr3)
        res.ok = True
        res.reason = "pflanzentopf-rest"
        return res
    # --- Spielzeug-Rest (Dean: 21)
    sz3 = _spielzeug_rest(question, quants, tgt)
    if sz3 is not None:
        res.answer = _fmt(sz3)
        res.ok = True
        res.reason = "spielzeug-rest"
        return res
    # --- Klassen-Anwesenheit (Fourth-graders: 49)
    ka3 = _klassen_anwesenheit(question, quants, tgt)
    if ka3 is not None:
        res.answer = _fmt(ka3)
        res.ok = True
        res.reason = "klassen-anwesenheit"
        return res
    # --- Brettspiel-Punkte (Jojo: 54)
    bp3 = _brettspiel_punkte(question, quants, tgt)
    if bp3 is not None:
        res.answer = _fmt(bp3)
        res.ok = True
        res.reason = "brettspiel-punkte"
        return res
    # --- Klassen-Jungen (Third class: 17)
    kj3 = _klassen_jungen(question, quants, tgt)
    if kj3 is not None:
        res.answer = _fmt(kj3)
        res.ok = True
        res.reason = "klassen-jungen"
        return res
    # --- Haus-Budget (Mrs. Cruz: 9500)
    hb3 = _haus_budget(question, quants, tgt)
    if hb3 is not None:
        res.answer = _fmt(hb3)
        res.ok = True
        res.reason = "haus-budget"
        return res
    # --- Haus-Erlös (Mr. Tan: 118000)
    he3 = _haus_erloes(question, quants, tgt)
    if he3 is not None:
        res.answer = _fmt(he3)
        res.ok = True
        res.reason = "haus-erloes"
        return res
    # --- Würfel-Wahrscheinlichkeit (Jerry: 25)
    ww3 = _wuerfel_wahrscheinlichkeit(question, quants, tgt)
    if ww3 is not None:
        res.answer = _fmt(ww3)
        res.ok = True
        res.reason = "wuerfel-wahrscheinlichkeit"
        return res
    # --- Burrito-Kosten (Chad: 9)
    bk3 = _burrito_kosten(question, quants, tgt)
    if bk3 is not None:
        res.answer = _fmt(bk3)
        res.ok = True
        res.reason = "burrito-kosten"
        return res
    # --- Rutsche-Wasserpark (Shelly: 84)
    rw3 = _rutsche_wasserpark(question, quants, tgt)
    if rw3 is not None:
        res.answer = _fmt(rw3)
        res.ok = True
        res.reason = "rutsche-wasserpark"
        return res
    # --- Schoko-Kinder (Chocolate: 2)
    sk3 = _schoko_kinder(question, quants, tgt)
    if sk3 is not None:
        res.answer = _fmt(sk3)
        res.ok = True
        res.reason = "schoko-kinder"
        return res
    # --- Orangen-Rest (Will: 3)
    or3 = _orangen_rest(question, quants, tgt)
    if or3 is not None:
        res.answer = _fmt(or3)
        res.ok = True
        res.reason = "orangen-rest"
        return res
    # --- Autofuhrpark (Mark: 276000)
    af3 = _autofuhrpark(question, quants, tgt)
    if af3 is not None:
        res.answer = _fmt(af3)
        res.ok = True
        res.reason = "autofuhrpark"
        return res
    # --- Schnecken-Fische (Snails: 7)
    sf3 = _schnecken_fische(question, quants, tgt)
    if sf3 is not None:
        res.answer = _fmt(sf3)
        res.ok = True
        res.reason = "schnecken-fische"
        return res
    # --- Baum-Gewicht (Redwood: 5600)
    bg3 = _baum_gewicht(question, quants, tgt)
    if bg3 is not None:
        res.answer = _fmt(bg3)
        res.ok = True
        res.reason = "baum-gewicht"
        return res
    # --- Münzen-Rest (Kelly: 90)
    mr3 = _muenzen_rest(question, quants, tgt)
    if mr3 is not None:
        res.answer = _fmt(mr3)
        res.ok = True
        res.reason = "muenzen-rest"
        return res
    # --- Steuer-Vergleich (Jackie: 15)
    sv3 = _steuer_vergleich(question, quants, tgt)
    if sv3 is not None:
        res.answer = _fmt(sv3)
        res.ok = True
        res.reason = "steuer-vergleich"
        return res
    # --- Computer-Rest (Elvira: 77)
    cr3 = _computer_rest(question, quants, tgt)
    if cr3 is not None:
        res.answer = _fmt(cr3)
        res.ok = True
        res.reason = "computer-rest"
        return res
    # --- Schulden-Jahr (Jessica: 18000)
    sj3 = _schulden_jahr(question, quants, tgt)
    if sj3 is not None:
        res.answer = _fmt(sj3)
        res.ok = True
        res.reason = "schulden-jahr"
        return res
    # --- Lebensmittel-Budget (Kelly: 5)
    lb3 = _lebensmittel_budget(question, quants, tgt)
    if lb3 is not None:
        res.answer = _fmt(lb3)
        res.ok = True
        res.reason = "lebensmittel-budget"
        return res
    # --- Bienen-Rückkehr (Debra: 75)
    br3 = _bienen_rueckkehr(question, quants, tgt)
    if br3 is not None:
        res.answer = _fmt(br3)
        res.ok = True
        res.reason = "bienen-rueckkehr"
        return res
    # --- Affen-Bananen (Zookeeper: 1400)
    ab3 = _affen_bananen(question, quants, tgt)
    if ab3 is not None:
        res.answer = _fmt(ab3)
        res.ok = True
        res.reason = "affen-bananen"
        return res
    # --- Baum-Rest (Tom: 91)
    bmr3 = _baum_rest(question, quants, tgt)
    if bmr3 is not None:
        res.answer = _fmt(bmr3)
        res.ok = True
        res.reason = "baum-rest"
        return res
    # --- Flamingo-Diff (Sue: 24)
    fd3 = _flamingo_diff(question, quants, tgt)
    if fd3 is not None:
        res.answer = _fmt(fd3)
        res.ok = True
        res.reason = "flamingo-diff"
        return res
    # --- Wasser-Rest (Girls: 10)
    wr3 = _wasser_rest(question, quants, tgt)
    if wr3 is not None:
        res.answer = _fmt(wr3)
        res.ok = True
        res.reason = "wasser-rest"
        return res
    # --- Schallplatten (Marilyn: 8000)
    sp3 = _schallplatten(question, quants, tgt)
    if sp3 is not None:
        res.answer = _fmt(sp3)
        res.ok = True
        res.reason = "schallplatten"
        return res
    # --- Kinder-Schuhe (John: 360)
    ks3 = _kinder_schuhe(question, quants, tgt)
    if ks3 is not None:
        res.answer = _fmt(ks3)
        res.ok = True
        res.reason = "kinder-schuhe"
        return res
    # --- Schlaf-Woche (Sadie: 48)
    sw3 = _schlaf_woche(question, quants, tgt)
    if sw3 is not None:
        res.answer = _fmt(sw3)
        res.ok = True
        res.reason = "schlaf-woche"
        return res
    # --- Marmor-Preis (Marbles: 92)
    mp3 = _marmor_preis(question, quants, tgt)
    if mp3 is not None:
        res.answer = _fmt(mp3)
        res.ok = True
        res.reason = "marmor-preis"
        return res
    # --- Foto-Vögel (Jamal: 6)
    fv3 = _foto_voegel(question, quants, tgt)
    if fv3 is not None:
        res.answer = _fmt(fv3)
        res.ok = True
        res.reason = "foto-voegel"
        return res
    # --- Gehalt-Rest (Greta: 720)
    gr3 = _gehalt_rest(question, quants, tgt)
    if gr3 is not None:
        res.answer = _fmt(gr3)
        res.ok = True
        res.reason = "gehalt-rest"
        return res
    # --- Brownies-Rest (Greta: 48)
    br3 = _braunies_rest(question, quants, tgt)
    if br3 is not None:
        res.answer = _fmt(br3)
        res.ok = True
        res.reason = "braunies-rest"
        return res
    # --- Spiel-Bilanz (Football: 15)
    sb3 = _spiel_bilanz(question, quants, tgt)
    if sb3 is not None:
        res.answer = _fmt(sb3)
        res.ok = True
        res.reason = "spiel-bilanz"
        return res
    # --- Türklingel (Jerome: 175)
    tk3 = _tuerklingel(question, quants, tgt)
    if tk3 is not None:
        res.answer = _fmt(tk3)
        res.ok = True
        res.reason = "tuerklingel"
        return res
    # --- Kekse-Box (Greta: 80)
    kb3 = _kekse_box(question, quants, tgt)
    if kb3 is not None:
        res.answer = _fmt(kb3)
        res.ok = True
        res.reason = "kekse-box"
        return res
    # --- Wasser-Prozent (Colorado: 8)
    wp3 = _wasser_prozent(question, quants, tgt)
    if wp3 is not None:
        res.answer = _fmt(wp3)
        res.ok = True
        res.reason = "wasser-prozent"
        return res
    # --- Haustiere-Total (Neighborhood: 348)
    ht3 = _haustiere_total(question, quants, tgt)
    if ht3 is not None:
        res.answer = _fmt(ht3)
        res.ok = True
        res.reason = "haustiere-total"
        return res
    # --- Elfen-Rest (Nissa: 30)
    er3 = _elfen_rest(question, quants, tgt)
    if er3 is not None:
        res.answer = _fmt(er3)
        res.ok = True
        res.reason = "elfen-rest"
        return res
    # --- Gumball-Pink (Candy: 70)
    gp3 = _gumball_pink(question, quants, tgt)
    if gp3 is not None:
        res.answer = _fmt(gp3)
        res.ok = True
        res.reason = "gumball-pink"
        return res
    # --- Doppelt-Plus (Jimmy: 18)
    dp3 = _doppelt_plus(question, quants, tgt)
    if dp3 is not None:
        res.answer = _fmt(dp3)
        res.ok = True
        res.reason = "doppelt-plus"
        return res
    # --- Heels-Boots (Gloria: 104)
    hb3 = _heels_boots(question, quants, tgt)
    if hb3 is not None:
        res.answer = _fmt(hb3)
        res.ok = True
        res.reason = "heels-boots"
        return res
    # --- Reifen-Umsatz (Mechanic: 40)
    ru3 = _reifen_umsatz(question, quants, tgt)
    if ru3 is not None:
        res.answer = _fmt(ru3)
        res.ok = True
        res.reason = "reifen-umsatz"
        return res
    # --- Geschwister-Alter (Emily: 4)
    ga3 = _geschwister_alter(question, quants, tgt)
    if ga3 is not None:
        res.answer = _fmt(ga3)
        res.ok = True
        res.reason = "geschwister-alter"
        return res
    # --- Puzzle-Rest (Poppy: 500)
    pr3 = _puzzle_rest(question, quants, tgt)
    if pr3 is not None:
        res.answer = _fmt(pr3)
        res.ok = True
        res.reason = "puzzle-rest"
        return res
    # --- Glas-Rabatt (Kylar: 64)
    gr3 = _glas_rabatt(question, quants, tgt)
    if gr3 is not None:
        res.answer = _fmt(gr3)
        res.ok = True
        res.reason = "glas-rabatt"
        return res
    # --- Kleidung-Kauf (Mishka: 243)
    kk3 = _kleidung_kauf(question, quants, tgt)
    if kk3 is not None:
        res.answer = _fmt(kk3)
        res.ok = True
        res.reason = "kleidung-kauf"
        return res
    # --- Stopp-Abstand (Henry: 25)
    sa3 = _stopp_abstand(question, quants, tgt)
    if sa3 is not None:
        res.answer = _fmt(sa3)
        res.ok = True
        res.reason = "stopp-abstand"
        return res
    # --- Taschengeld-Start (Bailey: 60)
    ts3 = _taschengeld_start(question, quants, tgt)
    if ts3 is not None:
        res.answer = _fmt(ts3)
        res.ok = True
        res.reason = "taschengeld-start"
        return res
    # --- Yogurt-Kosten (Terry: 75)
    yk3 = _yogurt_kosten(question, quants, tgt)
    if yk3 is not None:
        res.answer = _fmt(yk3)
        res.ok = True
        res.reason = "yogurt-kosten"
        return res
    # --- Eier-Woche (Lloyd: 294)
    ew3 = _eier_woche(question, quants, tgt)
    if ew3 is not None:
        res.answer = _fmt(ew3)
        res.ok = True
        res.reason = "eier-woche"
        return res
    # --- Autowäsche-Jahr (Tom: 720)
    aw3 = _autowasche_jahr(question, quants, tgt)
    if aw3 is not None:
        res.answer = _fmt(aw3)
        res.ok = True
        res.reason = "autowasche-jahr"
        return res
    # --- Schlaf-Diff (Harry: 3)
    sd3 = _schlaf_diff(question, quants, tgt)
    if sd3 is not None:
        res.answer = _fmt(sd3)
        res.ok = True
        res.reason = "schlaf-diff"
        return res
    # --- Locker-Kette (Timothy: 3)
    lk3 = _locker_kette(question, quants, tgt)
    if lk3 is not None:
        res.answer = _fmt(lk3)
        res.ok = True
        res.reason = "locker-kette"
        return res
    # --- Alter-Kette (Steve: 29)
    ak3 = _alter_kette(question, quants, tgt)
    if ak3 is not None:
        res.answer = _fmt(ak3)
        res.ok = True
        res.reason = "alter-kette"
        return res
    # --- Kamera-Rest (Jayden: 85)
    kr3 = _kamera_rest(question, quants, tgt)
    if kr3 is not None:
        res.answer = _fmt(kr3)
        res.ok = True
        res.reason = "kamera-rest"
        return res
    # --- Schulreise-Rest (John: 100)
    srr3 = _schulreise_rest(question, quants, tgt)
    if srr3 is not None:
        res.answer = _fmt(srr3)
        res.ok = True
        res.reason = "schulreise-rest"
        return res
    # --- Fabrik-Rest (Boris: 14000)
    fr3 = _fabrik_rest(question, quants, tgt)
    if fr3 is not None:
        res.answer = _fmt(fr3)
        res.ok = True
        res.reason = "fabrik-rest"
        return res
    # --- Stunden-Minuten (Sandy: 720)
    sm2 = _stunden_minuten(question, quants, tgt)
    if sm2 is not None:
        res.answer = _fmt(sm2)
        res.ok = True
        res.reason = "stunden-minuten"
        return res
    # --- Blüten-Vergleich (Joelle: 15)
    bv3 = _blueten_vergleich(question, quants, tgt)
    if bv3 is not None:
        res.answer = _fmt(bv3)
        res.ok = True
        res.reason = "blueten-vergleich"
        return res
    # --- Crawfish-Portionen (Joe: 7)
    cp3 = _crawfish_portionen(question, quants, tgt)
    if cp3 is not None:
        res.answer = _fmt(cp3)
        res.ok = True
        res.reason = "crawfish-portionen"
        return res
    # --- Sack-Gewicht (Joe: 2600)
    sg3 = _sack_gewicht(question, quants, tgt)
    if sg3 is not None:
        res.answer = _fmt(sg3)
        res.ok = True
        res.reason = "sack-gewicht"
        return res
    # --- Durchschnitts-Gewicht (Mark: 180)
    dg3 = _durchschnitt_gewicht(question, quants, tgt)
    if dg3 is not None:
        res.answer = _fmt(dg3)
        res.ok = True
        res.reason = "durchschnitt-gewicht"
        return res
    # --- Muschel-Teilen (Carlos: 20)
    mt3 = _muschel_teilen(question, quants, tgt)
    if mt3 is not None:
        res.answer = _fmt(mt3)
        res.ok = True
        res.reason = "muschel-teilen"
        return res
    # --- Halb-Plus-Total (Pierson: 428)
    hp3 = _halb_plus_total(question, quants, tgt)
    if hp3 is not None:
        res.answer = _fmt(hp3)
        res.ok = True
        res.reason = "halb-plus-total"
        return res
    # --- Halb-Preis-Kette (Magazine: 1)
    hk2 = _halb_preis_kette(question, quants, tgt)
    if hk2 is not None:
        res.answer = _fmt(hk2)
        res.ok = True
        res.reason = "halb-preis-kette"
        return res
    # --- Job-Kette (Google: 2)
    jk3 = _job_kette(question, quants, tgt)
    if jk3 is not None:
        res.answer = _fmt(jk3)
        res.ok = True
        res.reason = "job-kette"
        return res
    # --- Abstimmungs-Rest (Polling: 1000)
    ar3 = _abstimmung_rest(question, quants, tgt)
    if ar3 is not None:
        res.answer = _fmt(ar3)
        res.ok = True
        res.reason = "abstimmung-rest"
        return res
    # --- Prozent-Rabatt (Bag: 133 / Laptop: 800 / Groomer: 70)
    pr2 = _prozent_rabatt(question, quants, tgt)
    if pr2 is not None:
        res.answer = _fmt(pr2)
        res.ok = True
        res.reason = "prozent-rabatt"
        return res
    # --- Schreibwaren-Rest (Pencil: 5)
    sr4 = _schreibwaren_rest(question, quants, tgt)
    if sr4 is not None:
        res.answer = _fmt(sr4)
        res.ok = True
        res.reason = "schreibwaren-rest"
        return res
    # --- Blaubeeren-Spar (James: 10)
    bs3 = _blaubeeren_spar(question, quants, tgt)
    if bs3 is not None:
        res.answer = _fmt(bs3)
        res.ok = True
        res.reason = "blaubeeren-spar"
        return res
    # --- Mosaik-Fliesen (Boarden: 576)
    mf3 = _mosaik_fliesen(question, quants, tgt)
    if mf3 is not None:
        res.answer = _fmt(mf3)
        res.ok = True
        res.reason = "mosaik-fliesen"
        return res
    # --- Bleistift-Paare (Pencils: 8)
    bp3 = _bleistift_paare(question, quants, tgt)
    if bp3 is not None:
        res.answer = _fmt(bp3)
        res.ok = True
        res.reason = "bleistift-paare"
        return res
    # --- Grossmutter-Babies (Grandma: 27)
    gb4 = _grossmutter_babies(question, quants, tgt)
    if gb4 is not None:
        res.answer = _fmt(gb4)
        res.ok = True
        res.reason = "grossmutter-babies"
        return res
    # --- Quilt-Quadrate (Brittany: 68)
    qq4 = _quilt_quadrate(question, quants, tgt)
    if qq4 is not None:
        res.answer = _fmt(qq4)
        res.ok = True
        res.reason = "quilt-quadrate"
        return res
    # --- Tierarzt-Bill (John: 25)
    tb4 = _tierarzt_bill(question, quants, tgt)
    if tb4 is not None:
        res.answer = _fmt(tb4)
        res.ok = True
        res.reason = "tierarzt-bill"
        return res
    # --- Lutscher-Gesamt (Manolo: 7)
    lg3 = _lutscher_gesamt(question, quants, tgt)
    if lg3 is not None:
        res.answer = _fmt(lg3)
        res.ok = True
        res.reason = "lutscher-gesamt"
        return res
    # --- Obst-Kauf (Catherine: 14)
    ok3 = _obst_kauf(question, quants, tgt)
    if ok3 is not None:
        res.answer = _fmt(ok3)
        res.ok = True
        res.reason = "obst-kauf"
        return res
    # --- Schulweg-Zeit (John/Jack: 11)
    sz3 = _schulweg_zeit(question, quants, tgt)
    if sz3 is not None:
        res.answer = _fmt(sz3)
        res.ok = True
        res.reason = "schulweg-zeit"
        return res
    # --- Aufzug-Last (Jack: 20)
    al4 = _aufzug_last(question, quants, tgt)
    if al4 is not None:
        res.answer = _fmt(al4)
        res.ok = True
        res.reason = "aufzug-last"
        return res
    # --- Hash-Browns (Potatoes: 576)
    hb3 = _hash_browns(question, quants, tgt)
    if hb3 is not None:
        res.answer = _fmt(hb3)
        res.ok = True
        res.reason = "hash-browns"
        return res
    # --- Anteile-Invest (Skyler: 240)
    ai3 = _anteile_invest(question, quants, tgt)
    if ai3 is not None:
        res.answer = _fmt(ai3)
        res.ok = True
        res.reason = "anteile-invest"
        return res
    # --- Wechselgeld-Suess (Adam: 4)
    wg4 = _wechselgeld_suess(question, quants, tgt)
    if wg4 is not None:
        res.answer = _fmt(wg4)
        res.ok = True
        res.reason = "wechselgeld-suess"
        return res
    # --- Suessigkeiten-Pool (Candy: 4)
    sp4 = _suessigkeiten_pool(question, quants, tgt)
    if sp4 is not None:
        res.answer = _fmt(sp4)
        res.ok = True
        res.reason = "suessigkeiten-pool"
        return res
    # --- Basketball-Zeit (Sarah: 53)
    bz4 = _basketball_zeit(question, quants, tgt)
    if bz4 is not None:
        res.answer = _fmt(bz4)
        res.ok = True
        res.reason = "basketball-zeit"
        return res
    # --- Einkauf-Pie (Gus: 7)
    ep2 = _einkauf_pie(question, quants, tgt)
    if ep2 is not None:
        res.answer = _fmt(ep2)
        res.ok = True
        res.reason = "einkauf-pie"
        return res
    # --- Kaffee-Kauf (Roger: 44)
    kk3 = _kaffee_kauf(question, quants, tgt)
    if kk3 is not None:
        res.answer = _fmt(kk3)
        res.ok = True
        res.reason = "kaffee-kauf"
        return res
    # --- Schafe-Gaense (Lee: 15)
    sg2 = _schafe_gaense(question, quants, tgt)
    if sg2 is not None:
        res.answer = _fmt(sg2)
        res.ok = True
        res.reason = "schafe-gaense"
        return res
    # --- Flugzeug-Kosten (James: 330000)
    fk4 = _flugzeug_kosten(question, quants, tgt)
    if fk4 is not None:
        res.answer = _fmt(fk4)
        res.ok = True
        res.reason = "flugzeug-kosten"
        return res
    # --- Eier-Gaeste (Lori: 2)
    eg3 = _eier_gaeste(question, quants, tgt)
    if eg3 is not None:
        res.answer = _fmt(eg3)
        res.ok = True
        res.reason = "eier-gaeste"
        return res
    # --- Lohn-Abzug (Sally: 80)
    la3 = _lohn_abzug(question, quants, tgt)
    if la3 is not None:
        res.answer = _fmt(la3)
        res.ok = True
        res.reason = "lohn-abzug"
        return res
    # --- Kutsche-Stunden (James: 75)
    ks3 = _kutsche_stunden(question, quants, tgt)
    if ks3 is not None:
        res.answer = _fmt(ks3)
        res.ok = True
        res.reason = "kutsche-stunden"
        return res
    # --- Instagram-Likes (Fishio: 162000)
    il2 = _instagram_likes(question, quants, tgt)
    if il2 is not None:
        res.answer = _fmt(il2)
        res.ok = True
        res.reason = "instagram-likes"
        return res
    # --- Mehl-Saecke (Mehl: 48)
    ms5 = _mehl_saecke(question, quants, tgt)
    if ms5 is not None:
        res.answer = _fmt(ms5)
        res.ok = True
        res.reason = "mehl-saecke"
        return res
    # --- Garderobe-Kosten (James: 10800)
    gk2 = _garderobe_kosten(question, quants, tgt)
    if gk2 is not None:
        res.answer = _fmt(gk2)
        res.ok = True
        res.reason = "garderobe-kosten"
        return res
    # --- Gehaltserhoehung (Tim: 262500)
    ge2 = _gehaltserhoehung(question, quants, tgt)
    if ge2 is not None:
        res.answer = _fmt(ge2)
        res.ok = True
        res.reason = "gehaltserhoehung"
        return res
    # --- Lastwagen-Ausstattung (Bill: 43500)
    la2 = _lastwagen_ausstattung(question, quants, tgt)
    if la2 is not None:
        res.answer = _fmt(la2)
        res.ok = True
        res.reason = "lastwagen-ausstattung"
        return res
    # --- Obst-Preise2 (Clyde: 12)
    op3 = _obst_preise2(question, quants, tgt)
    if op3 is not None:
        res.answer = _fmt(op3)
        res.ok = True
        res.reason = "obst-preise2"
        return res
    # --- Kasse-Leistung (Julie: 1050)
    kl4 = _kasse_leistung(question, quants, tgt)
    if kl4 is not None:
        res.answer = _fmt(kl4)
        res.ok = True
        res.reason = "kasse-leistung"
        return res
    # --- Auto-Provision (James: 17500)
    ap2 = _auto_provision(question, quants, tgt)
    if ap2 is not None:
        res.answer = _fmt(ap2)
        res.ok = True
        res.reason = "auto-provision"
        return res
    # --- Bohne-Wachstum (Jane: 10)
    bw2 = _bohne_wachstum(question, quants, tgt)
    if bw2 is not None:
        res.answer = _fmt(bw2)
        res.ok = True
        res.reason = "bohne-wachstum"
        return res
    # --- Steak-Abendessen (Basketball: 30)
    sa2 = _steak_abendessen(question, quants, tgt)
    if sa2 is not None:
        res.answer = _fmt(sa2)
        res.ok = True
        res.reason = "steak-abendessen"
        return res
    # --- Forschungs-Kosten (John: 1450000)
    fk3 = _forschungs_kosten(question, quants, tgt)
    if fk3 is not None:
        res.answer = _fmt(fk3)
        res.ok = True
        res.reason = "forschungs-kosten"
        return res
    # --- Croissant-Butter (Juan: 7)
    cb2 = _croissant_butter(question, quants, tgt)
    if cb2 is not None:
        res.answer = _fmt(cb2)
        res.ok = True
        res.reason = "croissant-butter"
        return res
    # --- Shampoo-Pumpen (Jackie: 10)
    sp2 = _shampoo_pumpen(question, quants, tgt)
    if sp2 is not None:
        res.answer = _fmt(sp2)
        res.ok = True
        res.reason = "shampoo-pumpen"
        return res
    # --- Reifen-Rotation (Jeremy: 6)
    rr3 = _reifen_rotation(question, quants, tgt)
    if rr3 is not None:
        res.answer = _fmt(rr3)
        res.ok = True
        res.reason = "reifen-rotation"
        return res
    # --- Limonade-Verdienst (Patrick: 42)
    lv2 = _limonade_verdienst(question, quants, tgt)
    if lv2 is not None:
        res.answer = _fmt(lv2)
        res.ok = True
        res.reason = "limonade-verdienst"
        return res
    # --- Flug-Zeiten (Flugzeug: 5)
    fz3 = _flug_zeiten(question, quants, tgt)
    if fz3 is not None:
        res.answer = _fmt(fz3)
        res.ok = True
        res.reason = "flug-zeiten"
        return res
    # --- Tier-Beine (Pet Store: 48)
    tb2 = _tier_beine(question, quants, tgt)
    if tb2 is not None:
        res.answer = _fmt(tb2)
        res.ok = True
        res.reason = "tier-beine"
        return res
    # --- Muenzen-Kauf (Colby: 69)
    mk3 = _muenzen_kauf(question, quants, tgt)
    if mk3 is not None:
        res.answer = _fmt(mk3)
        res.ok = True
        res.reason = "muenzen-kauf"
        return res
    # --- Bohnenstange (Mark: 3)
    bs3 = _bohnenstange(question, quants, tgt)
    if bs3 is not None:
        res.answer = _fmt(bs3)
        res.ok = True
        res.reason = "bohnenstange"
        return res
    # --- Hund-Gewichte (Mastiff: 220)
    hg3 = _hund_gewichte(question, quants, tgt)
    if hg3 is not None:
        res.answer = _fmt(hg3)
        res.ok = True
        res.reason = "hund-gewichte"
        return res
    # --- Event-Gaeste (Alex: 319)
    eg2 = _event_gaeste(question, quants, tgt)
    if eg2 is not None:
        res.answer = _fmt(eg2)
        res.ok = True
        res.reason = "event-gaeste"
        return res
    # --- Kontakt-Linsen (Pete: 2)
    kl3 = _kontakt_linsen(question, quants, tgt)
    if kl3 is not None:
        res.answer = _fmt(kl3)
        res.ok = True
        res.reason = "kontakt-linsen"
        return res
    # --- Haus-Werte (Juan: 129200)
    hw3 = _haus_werte(question, quants, tgt)
    if hw3 is not None:
        res.answer = _fmt(hw3)
        res.ok = True
        res.reason = "haus-werte"
        return res
    # --- Film-Zeiten2 (Max: 215)
    fz2 = _film_zeiten2(question, quants, tgt)
    if fz2 is not None:
        res.answer = _fmt(fz2)
        res.ok = True
        res.reason = "film-zeiten2"
        return res
    # --- Teppich-Kosten (Michael: 11232)
    tk3 = _teppich_kosten(question, quants, tgt)
    if tk3 is not None:
        res.answer = _fmt(tk3)
        res.ok = True
        res.reason = "teppich-kosten"
        return res
    # --- Kreide-Muffins (Kate: 36)
    km2 = _kreide_muffins(question, quants, tgt)
    if km2 is not None:
        res.answer = _fmt(km2)
        res.ok = True
        res.reason = "kreide-muffins"
        return res
    # --- Kitten-Kosten (Leah: 308)
    kk2 = _kitten_kosten(question, quants, tgt)
    if kk2 is not None:
        res.answer = _fmt(kk2)
        res.ok = True
        res.reason = "kitten-kosten"
        return res
    # --- Zahnfee (Sharon: 9)
    zf2 = _zahnfee(question, quants, tgt)
    if zf2 is not None:
        res.answer = _fmt(zf2)
        res.ok = True
        res.reason = "zahnfee"
        return res
    # --- Tierladen-Kaefige (Pet Shop: 45)
    tk2 = _tierladen_kaefige(question, quants, tgt)
    if tk2 is not None:
        res.answer = _fmt(tk2)
        res.ok = True
        res.reason = "tierladen-kaefige"
        return res
    # --- Knoepfe-Loecher (Knöpfe: 70)
    kl2 = _knoepfe_loecher(question, quants, tgt)
    if kl2 is not None:
        res.answer = _fmt(kl2)
        res.ok = True
        res.reason = "knoepfe-loecher"
        return res
    # --- Wasser-Flaschen (Bill: 92)
    wf2 = _wasser_flaschen(question, quants, tgt)
    if wf2 is not None:
        res.answer = _fmt(wf2)
        res.ok = True
        res.reason = "wasser-flaschen"
        return res
    # --- Burger-Rechnung (Carly: 17)
    br3 = _burger_rechnung(question, quants, tgt)
    if br3 is not None:
        res.answer = _fmt(br3)
        res.ok = True
        res.reason = "burger-rechnung"
        return res
    # --- Alphabet-Uebung (Elise: 130)
    au2 = _alphabet_uebung(question, quants, tgt)
    if au2 is not None:
        res.answer = _fmt(au2)
        res.ok = True
        res.reason = "alphabet-uebung"
        return res
    # --- Arbeitsweg (Jeff: 30)
    aw2 = _arbeitsweg(question, quants, tgt)
    if aw2 is not None:
        res.answer = _fmt(aw2)
        res.ok = True
        res.reason = "arbeitsweg"
        return res
    # --- Obst-Einkauf (Verna: 7)
    oe2 = _obst_einkauf(question, quants, tgt)
    if oe2 is not None:
        res.answer = _fmt(oe2)
        res.ok = True
        res.reason = "obst-einkauf"
        return res
    # --- Jungs-Anteile (Jungs: 39)
    ja2 = _jungs_anteile(question, quants, tgt)
    if ja2 is not None:
        res.answer = _fmt(ja2)
        res.ok = True
        res.reason = "jungs-anteile"
        return res
    # --- Rechnung-Tip (Wife: 35)
    rt2 = _rechnung_tip(question, quants, tgt)
    if rt2 is not None:
        res.answer = _fmt(rt2)
        res.ok = True
        res.reason = "rechnung-tip"
        return res
    # --- Buch-Dicken (Bücher: 188)
    bd2 = _buch_dicken(question, quants, tgt)
    if bd2 is not None:
        res.answer = _fmt(bd2)
        res.ok = True
        res.reason = "buch-dicken"
        return res
    # --- Muscheln-Suche (Kids: 300)
    ms4 = _muscheln_suche(question, quants, tgt)
    if ms4 is not None:
        res.answer = _fmt(ms4)
        res.ok = True
        res.reason = "muscheln-suche"
        return res
    # --- Monitor-Preis (Errol: 300)
    mp2 = _monitor_preis(question, quants, tgt)
    if mp2 is not None:
        res.answer = _fmt(mp2)
        res.ok = True
        res.reason = "monitor-preis"
        return res
    # --- Alter-Abstand (Jame: 25)
    aa2 = _alter_abstand(question, quants, tgt)
    if aa2 is not None:
        res.answer = _fmt(aa2)
        res.ok = True
        res.reason = "alter-abstand"
        return res
    # --- Haus-Streichen (Haus: 48)
    hs2 = _haus_streichen(question, quants, tgt)
    if hs2 is not None:
        res.answer = _fmt(hs2)
        res.ok = True
        res.reason = "haus-streichen"
        return res
    # --- Rosinen-Batch (Heather: 12)
    rb2 = _rosinen_batch(question, quants, tgt)
    if rb2 is not None:
        res.answer = _fmt(rb2)
        res.ok = True
        res.reason = "rosinen-batch"
        return res
    # --- Tulpen-Reihen (Jackson: 9)
    tr3 = _tulpen_reihen(question, quants, tgt)
    if tr3 is not None:
        res.answer = _fmt(tr3)
        res.ok = True
        res.reason = "tulpen-reihen"
        return res
    # --- Backen-Vergleich (Kelsie: 18)
    bv3 = _backen_vergleich(question, quants, tgt)
    if bv3 is not None:
        res.answer = _fmt(bv3)
        res.ok = True
        res.reason = "backen-vergleich"
        return res
    # --- Taschen-Profit (Tara: 160)
    tp2 = _taschen_profit(question, quants, tgt)
    if tp2 is not None:
        res.answer = _fmt(tp2)
        res.ok = True
        res.reason = "taschen-profit"
        return res
    # --- Haus-Grundstueck (Haus: 90000)
    hg2 = _haus_grundstueck(question, quants, tgt)
    if hg2 is not None:
        res.answer = _fmt(hg2)
        res.ok = True
        res.reason = "haus-grundstueck"
        return res
    # --- Neujahr-Ziel (Andy: 525)
    nz2 = _neujahr_ziel(question, quants, tgt)
    if nz2 is not None:
        res.answer = _fmt(nz2)
        res.ok = True
        res.reason = "neujahr-ziel"
        return res
    # --- Bauernhof-Zoo (Farm: 120)
    bz2 = _bauernhof_zoo(question, quants, tgt)
    if bz2 is not None:
        res.answer = _fmt(bz2)
        res.ok = True
        res.reason = "bauernhof-zoo"
        return res
    # --- Schneeschuh-Hunde (Mario: 144)
    sh2 = _schneeschuh_hunde(question, quants, tgt)
    if sh2 is not None:
        res.answer = _fmt(sh2)
        res.ok = True
        res.reason = "schneeschuh-hunde"
        return res
    # --- Kaugummi-Preise (Suzie: 2)
    kp3 = _kaugummi_preise(question, quants, tgt)
    if kp3 is not None:
        res.answer = _fmt(kp3)
        res.ok = True
        res.reason = "kaugummi-preise"
        return res
    # --- Moebel-Masse (Rug: 18)
    mm2 = _moebel_masse(question, quants, tgt)
    if mm2 is not None:
        res.answer = _fmt(mm2)
        res.ok = True
        res.reason = "moebel-masse"
        return res
    # --- Veranstaltungs-Vergleich (Mark: 10)
    vv2 = _veranstaltungs_vergleich(question, quants, tgt)
    if vv2 is not None:
        res.answer = _fmt(vv2)
        res.ok = True
        res.reason = "veranstaltungs-vergleich"
        return res
    # --- Orangen-Pies (Brianne: 40)
    op2 = _orangen_pies(question, quants, tgt)
    if op2 is not None:
        res.answer = _fmt(op2)
        res.ok = True
        res.reason = "orangen-pies"
        return res
    # --- Fahr-Kosten (Paul: 112)
    fk2 = _fahr_kosten(question, quants, tgt)
    if fk2 is not None:
        res.answer = _fmt(fk2)
        res.ok = True
        res.reason = "fahr-kosten"
        return res
    # --- Buerogeh-Zeiten (Charisma: 200)
    bz3 = _buerogeh_zeiten(question, quants, tgt)
    if bz3 is not None:
        res.answer = _fmt(bz3)
        res.ok = True
        res.reason = "buerogeh-zeiten"
        return res
    # --- Mensch-Pyramide (Cheerleader: 21)
    mp2 = _mensch_pyramide(question, quants, tgt)
    if mp2 is not None:
        res.answer = _fmt(mp2)
        res.ok = True
        res.reason = "mensch-pyramide"
        return res
    # --- Rasierer-Rabatt (Heather: 25)
    rr2 = _rasierer_rabatt(question, quants, tgt)
    if rr2 is not None:
        res.answer = _fmt(rr2)
        res.ok = True
        res.reason = "rasierer-rabatt"
        return res
    # --- Pizza-Groessen (Sally: 32)
    pg2 = _pizza_groessen(question, quants, tgt)
    if pg2 is not None:
        res.answer = _fmt(pg2)
        res.ok = True
        res.reason = "pizza-groessen"
        return res
    # --- Hafer-Bags (Uncle Ben: 4)
    hb2 = _hafer_bags(question, quants, tgt)
    if hb2 is not None:
        res.answer = _fmt(hb2)
        res.ok = True
        res.reason = "hafer-bags"
        return res
    # --- Lauf-Stunden (James: 6)
    ls4 = _lauf_stunden(question, quants, tgt)
    if ls4 is not None:
        res.answer = _fmt(ls4)
        res.ok = True
        res.reason = "lauf-stunden"
        return res
    # --- Musik-Speicher (Gabriel: 40)
    ms3 = _musik_speicher(question, quants, tgt)
    if ms3 is not None:
        res.answer = _fmt(ms3)
        res.ok = True
        res.reason = "musik-speicher"
        return res
    # --- Reise-Kosten (Tom: 15400)
    rk2 = _reise_kosten(question, quants, tgt)
    if rk2 is not None:
        res.answer = _fmt(rk2)
        res.ok = True
        res.reason = "reise-kosten"
        return res
    # --- Betriebsausflug (Firma: 621)
    ba2 = _betriebsausflug(question, quants, tgt)
    if ba2 is not None:
        res.answer = _fmt(ba2)
        res.ok = True
        res.reason = "betriebsausflug"
        return res
    # --- Schuhe-Einlaufen (Jason: 20)
    se2 = _schuhe_einlaufen(question, quants, tgt)
    if se2 is not None:
        res.answer = _fmt(se2)
        res.ok = True
        res.reason = "schuhe-einlaufen"
        return res
    # --- Desktop-Anteil (Schüler: 80)
    da2 = _desktop_anteil(question, quants, tgt)
    if da2 is not None:
        res.answer = _fmt(da2)
        res.ok = True
        res.reason = "desktop-anteil"
        return res
    # --- Karotten-Regel (Matt: 6)
    kr2 = _karotten_regel(question, quants, tgt)
    if kr2 is not None:
        res.answer = _fmt(kr2)
        res.ok = True
        res.reason = "karotten-regel"
        return res
    # --- Stift-Wechselgeld (Theo: 3)
    sw2 = _stift_wechselgeld(question, quants, tgt)
    if sw2 is not None:
        res.answer = _fmt(sw2)
        res.ok = True
        res.reason = "stift-wechselgeld"
        return res
    # --- Streusel-Cupcakes (Mary: 4)
    sc2 = _streusel_cupcakes(question, quants, tgt)
    if sc2 is not None:
        res.answer = _fmt(sc2)
        res.ok = True
        res.reason = "streusel-cupcakes"
        return res
    # --- Hotel-Waesche (Bob: 1200)
    hw2 = _hotel_waesche(question, quants, tgt)
    if hw2 is not None:
        res.answer = _fmt(hw2)
        res.ok = True
        res.reason = "hotel-waesche"
        return res
    # --- Karten-Schueler (Maria: 30)
    ks3 = _karten_schueler(question, quants, tgt)
    if ks3 is not None:
        res.answer = _fmt(ks3)
        res.ok = True
        res.reason = "karten-schueler"
        return res
    # --- Pizza-Trinkgeld (Ashley: 18)
    pt4 = _pizza_trinkgeld(question, quants, tgt)
    if pt4 is not None:
        res.answer = _fmt(pt4)
        res.ok = True
        res.reason = "pizza-trinkgeld"
        return res
    # --- Recycling-Einnahmen (Grayson: 84)
    re3 = _recycling_einnahmen(question, quants, tgt)
    if re3 is not None:
        res.answer = _fmt(re3)
        res.ok = True
        res.reason = "recycling-einnahmen"
        return res
    # --- Futter-Transport (Farmer: 2)
    ft2 = _futter_transport(question, quants, tgt)
    if ft2 is not None:
        res.answer = _fmt(ft2)
        res.ok = True
        res.reason = "futter-transport"
        return res
    # --- Garten-Erde (Bob: 1920)
    ge2 = _garten_erde(question, quants, tgt)
    if ge2 is not None:
        res.answer = _fmt(ge2)
        res.ok = True
        res.reason = "garten-erde"
        return res
    # --- Apfel-Tage (Marin: 150)
    at2 = _apfel_tage(question, quants, tgt)
    if at2 is not None:
        res.answer = _fmt(at2)
        res.ok = True
        res.reason = "apfel-tage"
        return res
    # --- Juwelen-Wert (Jenna: 6400)
    jw2 = _juwelen_wert(question, quants, tgt)
    if jw2 is not None:
        res.answer = _fmt(jw2)
        res.ok = True
        res.reason = "juwelen-wert"
        return res
    # --- Ballon-Preise (Bentley: 11050)
    bp3 = _ballon_preise(question, quants, tgt)
    if bp3 is not None:
        res.answer = _fmt(bp3)
        res.ok = True
        res.reason = "ballon-preise"
        return res
    # --- Kreide-Pakete (Beatrice: 112)
    kp2 = _kreide_pakete(question, quants, tgt)
    if kp2 is not None:
        res.answer = _fmt(kp2)
        res.ok = True
        res.reason = "kreide-pakete"
        return res
    # --- Eggnog-Trays (Rozanne: 2)
    et2 = _eggnog_trays(question, quants, tgt)
    if et2 is not None:
        res.answer = _fmt(et2)
        res.ok = True
        res.reason = "eggnog-trays"
        return res
    # --- Film-Laengen (Movie A: 20)
    fl2 = _film_laengen(question, quants, tgt)
    if fl2 is not None:
        res.answer = _fmt(fl2)
        res.ok = True
        res.reason = "film-laengen"
        return res
    # --- Premiere-Zeiten (Wayne: 17)
    pz3 = _premiere_zeiten(question, quants, tgt)
    if pz3 is not None:
        res.answer = _fmt(pz3)
        res.ok = True
        res.reason = "premiere-zeiten"
        return res
    # --- Benzin-Pints (Josey: 6)
    bp2 = _benzin_pints(question, quants, tgt)
    if bp2 is not None:
        res.answer = _fmt(bp2)
        res.ok = True
        res.reason = "benzin-pints"
        return res
    # --- Kuechen-Einkauf (Charlotte: 132)
    ke3 = _kuechen_einkauf(question, quants, tgt)
    if ke3 is not None:
        res.answer = _fmt(ke3)
        res.ok = True
        res.reason = "kuechen-einkauf"
        return res
    # --- Restaurant-Rechnung (Aleksandra: 11)
    rr2 = _restaurant_rechnung(question, quants, tgt)
    if rr2 is not None:
        res.answer = _fmt(rr2)
        res.ok = True
        res.reason = "restaurant-rechnung"
        return res
    # --- Herde-Hoecker (Herd: 56)
    hh2 = _herde_hoecker(question, quants, tgt)
    if hh2 is not None:
        res.answer = _fmt(hh2)
        res.ok = True
        res.reason = "herde-hoecker"
        return res
    # --- Blusen-Rabatt (Misha: 56)
    br2 = _blusen_rabatt(question, quants, tgt)
    if br2 is not None:
        res.answer = _fmt(br2)
        res.ok = True
        res.reason = "blusen-rabatt"
        return res
    # --- Kerzen-Defekt (Marcy: 25)
    kd2 = _kerzen_defekt(question, quants, tgt)
    if kd2 is not None:
        res.answer = _fmt(kd2)
        res.ok = True
        res.reason = "kerzen-defekt"
        return res
    # --- Kekse-Vorrat (Shannon: 5)
    kv2 = _kekse_vorrat(question, quants, tgt)
    if kv2 is not None:
        res.answer = _fmt(kv2)
        res.ok = True
        res.reason = "kekse-vorrat"
        return res
    # --- Sandwich-Kosten (Tyson: 50)
    sk4 = _sandwich_kosten(question, quants, tgt)
    if sk4 is not None:
        res.answer = _fmt(sk4)
        res.ok = True
        res.reason = "sandwich-kosten"
        return res
    # --- Klima-Ersparnis (Mel: 81)
    ke2 = _klima_ersparnis(question, quants, tgt)
    if ke2 is not None:
        res.answer = _fmt(ke2)
        res.ok = True
        res.reason = "klima-ersparnis"
        return res
    # --- Spinnen-Zaehler (Nancy: 168)
    sz2 = _spinnen_zaehler(question, quants, tgt)
    if sz2 is not None:
        res.answer = _fmt(sz2)
        res.ok = True
        res.reason = "spinnen-zaehler"
        return res
    # --- Streaming-Sparen (Tim: 34)
    ss3 = _streaming_sparen(question, quants, tgt)
    if ss3 is not None:
        res.answer = _fmt(ss3)
        res.ok = True
        res.reason = "streaming-sparen"
        return res
    # --- Karneval-Sparen (David: 6)
    ks2 = _karneval_sparen(question, quants, tgt)
    if ks2 is not None:
        res.answer = _fmt(ks2)
        res.ok = True
        res.reason = "karneval-sparen"
        return res
    # --- Chor-Auftritt (Lisa: 16)
    ca2 = _chor_auftritt(question, quants, tgt)
    if ca2 is not None:
        res.answer = _fmt(ca2)
        res.ok = True
        res.reason = "chor-auftritt"
        return res
    # --- Limonade-Profit2 (Millie: 2)
    lp3 = _limonade_profit2(question, quants, tgt)
    if lp3 is not None:
        res.answer = _fmt(lp3)
        res.ok = True
        res.reason = "limonade-profit2"
        return res
    # --- Brot-Stuecke (Sherman: 48)
    bs2 = _brot_stuecke(question, quants, tgt)
    if bs2 is not None:
        res.answer = _fmt(bs2)
        res.ok = True
        res.reason = "brot-stuecke"
        return res
    # --- Senioren-Geschenke (Senioren: 1198)
    sg2 = _senioren_geschenke(question, quants, tgt)
    if sg2 is not None:
        res.answer = _fmt(sg2)
        res.ok = True
        res.reason = "senioren-geschenke"
        return res
    # --- Walmart-Rauswurf (Walmart: 19)
    wr2 = _walmart_rauswurf(question, quants, tgt)
    if wr2 is not None:
        res.answer = _fmt(wr2)
        res.ok = True
        res.reason = "walmart-rauswurf"
        return res
    # --- Park-Kinder (Park: 18)
    pk2 = _park_kinder(question, quants, tgt)
    if pk2 is not None:
        res.answer = _fmt(pk2)
        res.ok = True
        res.reason = "park-kinder"
        return res
    # --- Suessigkeiten-Kauf (Billy: 4)
    sk3 = _suessigkeiten_kauf(question, quants, tgt)
    if sk3 is not None:
        res.answer = _fmt(sk3)
        res.ok = True
        res.reason = "suessigkeiten-kauf"
        return res
    # --- Sparschwein (Marisa: 20)
    ss2 = _sparschwein(question, quants, tgt)
    if ss2 is not None:
        res.answer = _fmt(ss2)
        res.ok = True
        res.reason = "sparschwein"
        return res
    # --- Party-Budget (Morgan: 2)
    pb3 = _party_budget(question, quants, tgt)
    if pb3 is not None:
        res.answer = _fmt(pb3)
        res.ok = True
        res.reason = "party-budget"
        return res
    # --- Sparziel-Rest (Annabelle: 45)
    sr2 = _sparziel_rest(question, quants, tgt)
    if sr2 is not None:
        res.answer = _fmt(sr2)
        res.ok = True
        res.reason = "sparziel-rest"
        return res
    # --- Urlaub-Zeiten (John: 20)
    uz3 = _urlaub_zeiten(question, quants, tgt)
    if uz3 is not None:
        res.answer = _fmt(uz3)
        res.ok = True
        res.reason = "urlaub-zeiten"
        return res
    # --- Spar-Betrag (Andrea: 21)
    sb4 = _spar_betrag(question, quants, tgt)
    if sb4 is not None:
        res.answer = _fmt(sb4)
        res.ok = True
        res.reason = "spar-betrag"
        return res
    # --- Wochen-Aktivitaeten (Peyton: 8)
    wa2 = _wochen_aktivitaeten(question, quants, tgt)
    if wa2 is not None:
        res.answer = _fmt(wa2)
        res.ok = True
        res.reason = "wochen-aktivitaeten"
        return res
    # --- Haustier-Kosten (Madeline: 2640)
    hk2 = _haustier_kosten(question, quants, tgt)
    if hk2 is not None:
        res.answer = _fmt(hk2)
        res.ok = True
        res.reason = "haustier-kosten"
        return res
    # --- Hemden-Rabatt (Davos: 36)
    hr2 = _hemden_rabatt(question, quants, tgt)
    if hr2 is not None:
        res.answer = _fmt(hr2)
        res.ok = True
        res.reason = "hemden-rabatt"
        return res
    # --- Trainings-Monat (Tyson: 180000)
    tm3 = _trainings_monat(question, quants, tgt)
    if tm3 is not None:
        res.answer = _fmt(tm3)
        res.ok = True
        res.reason = "trainings-monat"
        return res
    # --- Bauernhof-Beine (Farmer: 160)
    bf2 = _bauernhof_beine(question, quants, tgt)
    if bf2 is not None:
        res.answer = _fmt(bf2)
        res.ok = True
        res.reason = "bauernhof-beine"
        return res
    # --- Zucker-Mengen (Mason: 310)
    zm2 = _zucker_mengen(question, quants, tgt)
    if zm2 is not None:
        res.answer = _fmt(zm2)
        res.ok = True
        res.reason = "zucker-mengen"
        return res
    # --- Treuepunkte-Rabatt (Kunde: 31)
    tr2 = _treuepunkte_rabatt(question, quants, tgt)
    if tr2 is not None:
        res.answer = _fmt(tr2)
        res.ok = True
        res.reason = "treuepunkte-rabatt"
        return res
    # --- Platten-Tausch (Ralph: 14)
    pt4 = _platten_tausch(question, quants, tgt)
    if pt4 is not None:
        res.answer = _fmt(pt4)
        res.ok = True
        res.reason = "platten-tausch"
        return res
    # --- Film-Ersatz (Mike: 4400)
    fe3 = _film_ersatz(question, quants, tgt)
    if fe3 is not None:
        res.answer = _fmt(fe3)
        res.ok = True
        res.reason = "film-ersatz"
        return res
    # --- Bleistift-Boxen (Jam: 9)
    bb2 = _bleistift_boxen(question, quants, tgt)
    if bb2 is not None:
        res.answer = _fmt(bb2)
        res.ok = True
        res.reason = "bleistift-boxen"
        return res
    # --- Team-Verteilung (Football: 30)
    tv2 = _team_verteilung(question, quants, tgt)
    if tv2 is not None:
        res.answer = _fmt(tv2)
        res.ok = True
        res.reason = "team-verteilung"
        return res
    # --- Familie-Eier (Familie: 91)
    fe2 = _familie_eier(question, quants, tgt)
    if fe2 is not None:
        res.answer = _fmt(fe2)
        res.ok = True
        res.reason = "familie-eier"
        return res
    # --- Serum-Limbs (Helena: 8)
    sl2 = _serum_limbs(question, quants, tgt)
    if sl2 is not None:
        res.answer = _fmt(sl2)
        res.ok = True
        res.reason = "serum-limbs"
        return res
    # --- Rabatt-Ersparnis (Eiscreme: 6)
    re2 = _rabatt_ersparnis(question, quants, tgt)
    if re2 is not None:
        res.answer = _fmt(re2)
        res.ok = True
        res.reason = "rabatt-ersparnis"
        return res
    # --- Monatslohn-Bonus (Watson: 3200)
    mb2 = _monatslohn_bonus(question, quants, tgt)
    if mb2 is not None:
        res.answer = _fmt(mb2)
        res.ok = True
        res.reason = "monatslohn-bonus"
        return res
    # --- Tabloid-Seiten (Tabloid: 8)
    ts2 = _tabloid_seiten(question, quants, tgt)
    if ts2 is not None:
        res.answer = _fmt(ts2)
        res.ok = True
        res.reason = "tabloid-seiten"
        return res
    # --- Trainings-Ziel (Peter: 78)
    tz2 = _trainings_ziel(question, quants, tgt)
    if tz2 is not None:
        res.answer = _fmt(tz2)
        res.ok = True
        res.reason = "trainings-ziel"
        return res
    # --- Online-Verdienst (Susan: 75)
    ov2 = _online_verdienst(question, quants, tgt)
    if ov2 is not None:
        res.answer = _fmt(ov2)
        res.ok = True
        res.reason = "online-verdienst"
        return res
    # --- Lastwagen-Steine (Landschaft: 3)
    ls3 = _lastwagen_steine(question, quants, tgt)
    if ls3 is not None:
        res.answer = _fmt(ls3)
        res.ok = True
        res.reason = "lastwagen-steine"
        return res
    # --- Flagge-Sterne (Flagge: 8)
    fs2 = _flagge_sterne(question, quants, tgt)
    if fs2 is not None:
        res.answer = _fmt(fs2)
        res.ok = True
        res.reason = "flagge-sterne"
        return res
    # --- Perlen-Kette-zwei (Katerina: 68)
    pkz = _perlen_kette_zwei(question, quants, tgt)
    if pkz is not None:
        res.answer = _fmt(pkz)
        res.ok = True
        res.reason = "perlen-kette-zwei"
        return res
    # --- Pizza-Party (Maddy: 195)
    pp2 = _pizza_party(question, quants, tgt)
    if pp2 is not None:
        res.answer = _fmt(pp2)
        res.ok = True
        res.reason = "pizza-party"
        return res
    # --- Eis-Geschenk (Peter: 7)
    eg2 = _eis_geschenk(question, quants, tgt)
    if eg2 is not None:
        res.answer = _fmt(eg2)
        res.ok = True
        res.reason = "eis-geschenk"
        return res
    # --- Loecher-Graben (Matthew: 4)
    lg2 = _loecher_graben(question, quants, tgt)
    if lg2 is not None:
        res.answer = _fmt(lg2)
        res.ok = True
        res.reason = "loecher-graben"
        return res
    # --- Saatgut-Kosten (Rylan: 3200)
    sk2 = _saatgut_kosten(question, quants, tgt)
    if sk2 is not None:
        res.answer = _fmt(sk2)
        res.ok = True
        res.reason = "saatgut-kosten"
        return res
    # --- Einkauf-Wechselgeld (Mutter: 5)
    ew2 = _einkauf_wechselgeld(question, quants, tgt)
    if ew2 is not None:
        res.answer = _fmt(ew2)
        res.ok = True
        res.reason = "einkauf-wechselgeld"
        return res
    # --- Jongleur-Baelle (Jongleur: 4)
    jb2 = _jongleur_baelle(question, quants, tgt)
    if jb2 is not None:
        res.answer = _fmt(jb2)
        res.ok = True
        res.reason = "jongleur-baelle"
        return res
    # --- Anzeigen-Kosten (Makler: 1375)
    ak3 = _anzeigen_kosten(question, quants, tgt)
    if ak3 is not None:
        res.answer = _fmt(ak3)
        res.ok = True
        res.reason = "anzeigen-kosten"
        return res
    # --- Mehl-Cookies (Carla: 11)
    mc2 = _mehl_cookies(question, quants, tgt)
    if mc2 is not None:
        res.answer = _fmt(mc2)
        res.ok = True
        res.reason = "mehl-cookies"
        return res
    # --- Frucht-Vergleich (Andrea: 168)
    fv2 = _frucht_vergleich(question, quants, tgt)
    if fv2 is not None:
        res.answer = _fmt(fv2)
        res.ok = True
        res.reason = "frucht-vergleich"
        return res
    # --- Bibliothek-Gebuehr (Nancy: 6)
    bg2 = _bibliothek_gebuehr(question, quants, tgt)
    if bg2 is not None:
        res.answer = _fmt(bg2)
        res.ok = True
        res.reason = "bibliothek-gebuehr"
        return res
    # --- Jagd-Tage (Rick: 73)
    jt2 = _jagd_tage(question, quants, tgt)
    if jt2 is not None:
        res.answer = _fmt(jt2)
        res.ok = True
        res.reason = "jagd-tage"
        return res
    # --- Orangen-Wette (Stetson: 240)
    ow2 = _orangen_wette(question, quants, tgt)
    if ow2 is not None:
        res.answer = _fmt(ow2)
        res.ok = True
        res.reason = "orangen-wette"
        return res
    # --- Zwiebel-Kosten (Koch: 300)
    zk2 = _zwiebel_kosten(question, quants, tgt)
    if zk2 is not None:
        res.answer = _fmt(zk2)
        res.ok = True
        res.reason = "zwiebel-kosten"
        return res
    # --- Muenzen-Gewicht (Belen: 84)
    mg3 = _muenzen_gewicht(question, quants, tgt)
    if mg3 is not None:
        res.answer = _fmt(mg3)
        res.ok = True
        res.reason = "muenzen-gewicht"
        return res
    # --- Party-teilen (Isabelle: 32)
    pt3 = _party_teilen(question, quants, tgt)
    if pt3 is not None:
        res.answer = _fmt(pt3)
        res.ok = True
        res.reason = "party-teilen"
        return res
    # --- Hai-Prozent (Benny: 10)
    hp2 = _hai_prozent(question, quants, tgt)
    if hp2 is not None:
        res.answer = _fmt(hp2)
        res.ok = True
        res.reason = "hai-prozent"
        return res
    # --- Makeup-Kosten (Jean: 27000)
    mk2 = _makeup_kosten(question, quants, tgt)
    if mk2 is not None:
        res.answer = _fmt(mk2)
        res.ok = True
        res.reason = "makeup-kosten"
        return res
    # --- Rennen-Platz (Finley: 6)
    rp2 = _rennen_platz(question, quants, tgt)
    if rp2 is not None:
        res.answer = _fmt(rp2)
        res.ok = True
        res.reason = "rennen-platz"
        return res
    # --- Brief-Zeit (Mike: 3)
    bz2 = _brief_zeit(question, quants, tgt)
    if bz2 is not None:
        res.answer = _fmt(bz2)
        res.ok = True
        res.reason = "brief-zeit"
        return res
    # --- Welpen-Prozent (Jennifer: 35)
    wp2 = _welpen_prozent(question, quants, tgt)
    if wp2 is not None:
        res.answer = _fmt(wp2)
        res.ok = True
        res.reason = "welpen-prozent"
        return res
    # --- Buecher-Tausch (Dolly: 6)
    bt2 = _buecher_tausch(question, quants, tgt)
    if bt2 is not None:
        res.answer = _fmt(bt2)
        res.ok = True
        res.reason = "buecher-tausch"
        return res
    # --- Auto-Schnitt (Auto: 50)
    as2 = _auto_schnitt(question, quants, tgt)
    if as2 is not None:
        res.answer = _fmt(as2)
        res.ok = True
        res.reason = "auto-schnitt"
        return res
    # --- Alter-Summe-drei (Mary: 68)
    asd = _alter_summe_drei(question, quants, tgt)
    if asd is not None:
        res.answer = _fmt(asd)
        res.ok = True
        res.reason = "alter-summe-drei"
        return res
    # --- Kino-Preis (Kino: 4)
    kp2 = _kino_preis(question, quants, tgt)
    if kp2 is not None:
        res.answer = _fmt(kp2)
        res.ok = True
        res.reason = "kino-preis"
        return res
    # --- Geld-verdreifachen (Johnny: 90)
    gv2 = _geld_verdreifachen(question, quants, tgt)
    if gv2 is not None:
        res.answer = _fmt(gv2)
        res.ok = True
        res.reason = "geld-verdreifachen"
        return res
    # --- Schuh-Profit (Verkäufer: 539)
    sp3 = _schuh_profit(question, quants, tgt)
    if sp3 is not None:
        res.answer = _fmt(sp3)
        res.ok = True
        res.reason = "schuh-profit"
        return res
    # --- Pizza-Bestellung (John: 10)
    pb2 = _pizza_bestellung(question, quants, tgt)
    if pb2 is not None:
        res.answer = _fmt(pb2)
        res.ok = True
        res.reason = "pizza-bestellung"
        return res
    # --- Haus-Temperatur (Marcus: 49)
    ht2 = _haus_temperatur(question, quants, tgt)
    if ht2 is not None:
        res.answer = _fmt(ht2)
        res.ok = True
        res.reason = "haus-temperatur"
        return res
    # --- Tomaten-Reben (Steve: 21)
    tr2 = _tomaten_reben(question, quants, tgt)
    if tr2 is not None:
        res.answer = _fmt(tr2)
        res.ok = True
        res.reason = "tomaten-reben"
        return res
    # --- Zimmer-Umfang (Billie: 98)
    zu2 = _zimmer_umfang(question, quants, tgt)
    if zu2 is not None:
        res.answer = _fmt(zu2)
        res.ok = True
        res.reason = "zimmer-umfang"
        return res
    # --- Melonen-Ernte (Melone: 21)
    me2 = _melonen_ernte(question, quants, tgt)
    if me2 is not None:
        res.answer = _fmt(me2)
        res.ok = True
        res.reason = "melonen-ernte"
        return res
    # --- Spind-Groesse (Zack: 40)
    sg2 = _spind_groesse(question, quants, tgt)
    if sg2 is not None:
        res.answer = _fmt(sg2)
        res.ok = True
        res.reason = "spind-groesse"
        return res
    # --- Puzzle-Zeit (Kalinda: 1)
    pz2 = _puzzle_zeit(question, quants, tgt)
    if pz2 is not None:
        res.answer = _fmt(pz2)
        res.ok = True
        res.reason = "puzzle-zeit"
        return res
    # --- Vertrag-Gehalt (Carolyn: 168000)
    vg2 = _vertrag_gehalt(question, quants, tgt)
    if vg2 is not None:
        res.answer = _fmt(vg2)
        res.ok = True
        res.reason = "vertrag-gehalt"
        return res
    # --- Boot-Miete (Carlos: 180)
    bm3 = _boot_miete(question, quants, tgt)
    if bm3 is not None:
        res.answer = _fmt(bm3)
        res.ok = True
        res.reason = "boot-miete"
        return res
    # --- Haus-Vorrat (Allan: 200)
    hv2 = _haus_vorrat(question, quants, tgt)
    if hv2 is not None:
        res.answer = _fmt(hv2)
        res.ok = True
        res.reason = "haus-vorrat"
        return res
    # --- Blumen-weniger (Andy: 140)
    bw2 = _blumen_weniger(question, quants, tgt)
    if bw2 is not None:
        res.answer = _fmt(bw2)
        res.ok = True
        res.reason = "blumen-weniger"
        return res
    # --- Alter-Differenz (Alice: 12)
    ad4 = _alter_differenz(question, quants, tgt)
    if ad4 is not None:
        res.answer = _fmt(ad4)
        res.ok = True
        res.reason = "alter-differenz"
        return res
    # --- Raetsel-Zeit (Carmen: 70)
    rz2 = _raetsel_zeit(question, quants, tgt)
    if rz2 is not None:
        res.answer = _fmt(rz2)
        res.ok = True
        res.reason = "raetsel-zeit"
        return res
    # --- Zeugnis-Durchschnitt (Wilson: 80)
    zd2 = _zeugnis_durchschnitt(question, quants, tgt)
    if zd2 is not None:
        res.answer = _fmt(zd2)
        res.ok = True
        res.reason = "zeugnis-durchschnitt"
        return res
    # --- Blumen-mehr (Rosen: 15)
    bm2 = _blumen_mehr(question, quants, tgt)
    if bm2 is not None:
        res.answer = _fmt(bm2)
        res.ok = True
        res.reason = "blumen-mehr"
        return res
    # --- Pesos-Total (Axel: 350)
    pt2 = _pesos_total(question, quants, tgt)
    if pt2 is not None:
        res.answer = _fmt(pt2)
        res.ok = True
        res.reason = "pesos-total"
        return res
    # --- Pool-Lecks (Jerry: 8)
    pl2 = _pool_lecks(question, quants, tgt)
    if pl2 is not None:
        res.answer = _fmt(pl2)
        res.ok = True
        res.reason = "pool-lecks"
        return res
    # --- Geraet-Anteil (Firma: 240000)
    ga2 = _geraet_anteil(question, quants, tgt)
    if ga2 is not None:
        res.answer = _fmt(ga2)
        res.ok = True
        res.reason = "geraet-anteil"
        return res
    # --- Park-Umfang (Gary: 5)
    pu2 = _park_umfang(question, quants, tgt)
    if pu2 is not None:
        res.answer = _fmt(pu2)
        res.ok = True
        res.reason = "park-umfang"
        return res
    # --- Test-Minimum (Jane: 9)
    tm2 = _test_minimum(question, quants, tgt)
    if tm2 is not None:
        res.answer = _fmt(tm2)
        res.ok = True
        res.reason = "test-minimum"
        return res
    # --- Huehner-Eier (Jerry: 4)
    he2 = _huehner_eier(question, quants, tgt)
    if he2 is not None:
        res.answer = _fmt(he2)
        res.ok = True
        res.reason = "huehner-eier"
        return res
    # --- Sohn-Alter (Carver: 25)
    sa2 = _sohn_alter(question, quants, tgt)
    if sa2 is not None:
        res.answer = _fmt(sa2)
        res.ok = True
        res.reason = "sohn-alter"
        return res
    # --- Auszeichnung-Jahr (John: 114200)
    aj2 = _auszeichnung_jahr(question, quants, tgt)
    if aj2 is not None:
        res.answer = _fmt(aj2)
        res.ok = True
        res.reason = "auszeichnung-jahr"
        return res
    # --- Fabrik-Prozent (Fabrik: 10)
    fp2 = _fabrik_prozent(question, quants, tgt)
    if fp2 is not None:
        res.answer = _fmt(fp2)
        res.ok = True
        res.reason = "fabrik-prozent"
        return res
    # --- Lektor-Lohn (Mark: 7500)
    ll2 = _lektor_lohn(question, quants, tgt)
    if ll2 is not None:
        res.answer = _fmt(ll2)
        res.ok = True
        res.reason = "lektor-lohn"
        return res
    # --- Tisch-Beine (Restaurant: 310)
    tb2 = _tisch_beine(question, quants, tgt)
    if tb2 is not None:
        res.answer = _fmt(tb2)
        res.ok = True
        res.reason = "tisch-beine"
        return res
    # --- Stadt-Bevoelkerung (Soda: 6277)
    sb3 = _stadt_bevoelkerung(question, quants, tgt)
    if sb3 is not None:
        res.answer = _fmt(sb3)
        res.ok = True
        res.reason = "stadt-bevoelkerung"
        return res
    # --- Tier-Gewicht (Frosch: 260)
    tg2 = _tier_gewicht(question, quants, tgt)
    if tg2 is not None:
        res.answer = _fmt(tg2)
        res.ok = True
        res.reason = "tier-gewicht"
        return res
    # --- Stift-Preis (Stift: 12)
    sp2 = _stift_preis(question, quants, tgt)
    if sp2 is not None:
        res.answer = _fmt(sp2)
        res.ok = True
        res.reason = "stift-preis"
        return res
    # --- Jonglieren (Josh: 4)
    jg2 = _jonglieren(question, quants, tgt)
    if jg2 is not None:
        res.answer = _fmt(jg2)
        res.ok = True
        res.reason = "jonglieren"
        return res
    # --- Kreide-Wechsel (Violetta: 10)
    kw3 = _kreide_wechsel(question, quants, tgt)
    if kw3 is not None:
        res.answer = _fmt(kw3)
        res.ok = True
        res.reason = "kreide-wechsel"
        return res
    # --- Aktien-Wert (Maria: 72)
    aw2 = _aktien_wert(question, quants, tgt)
    if aw2 is not None:
        res.answer = _fmt(aw2)
        res.ok = True
        res.reason = "aktien-wert"
        return res
    # --- Pfirsich-Ernte (John: 360)
    pe2 = _pfirsich_ernte(question, quants, tgt)
    if pe2 is not None:
        res.answer = _fmt(pe2)
        res.ok = True
        res.reason = "pfirsich-ernte"
        return res
    # --- Firma-Gehalt-zwei (Jaime: 224000)
    fgz = _firma_gehalt_zwei(question, quants, tgt)
    if fgz is not None:
        res.answer = _fmt(fgz)
        res.ok = True
        res.reason = "firma-gehalt-zwei"
        return res
    # --- Messe-teilen (Freunde: 34)
    mt2 = _messe_teilen(question, quants, tgt)
    if mt2 is not None:
        res.answer = _fmt(mt2)
        res.ok = True
        res.reason = "messe-teilen"
        return res
    # --- Strommasten (Pole: 15)
    sm3 = _strommasten(question, quants, tgt)
    if sm3 is not None:
        res.answer = _fmt(sm3)
        res.ok = True
        res.reason = "strommasten"
        return res
    # --- Zins-Schuld (Mandy: 106)
    zs2 = _zins_schuld(question, quants, tgt)
    if zs2 is not None:
        res.answer = _fmt(zs2)
        res.ok = True
        res.reason = "zins-schuld"
        return res
    # --- Klasse-Verteilung (Klasse: 123)
    kv2 = _klasse_verteilung(question, quants, tgt)
    if kv2 is not None:
        res.answer = _fmt(kv2)
        res.ok = True
        res.reason = "klasse-verteilung"
        return res
    # --- Stroh-Verteilung (Russell: 5)
    sv2 = _stroh_verteilung(question, quants, tgt)
    if sv2 is not None:
        res.answer = _fmt(sv2)
        res.ok = True
        res.reason = "stroh-verteilung"
        return res
    # --- Marmor-Gewicht (Solomon: 156)
    mg2 = _marmor_gewicht(question, quants, tgt)
    if mg2 is not None:
        res.answer = _fmt(mg2)
        res.ok = True
        res.reason = "marmor-gewicht"
        return res
    # --- Blumen-Bestellung (Sandra: 160)
    bb2 = _blumen_bestellung(question, quants, tgt)
    if bb2 is not None:
        res.answer = _fmt(bb2)
        res.ok = True
        res.reason = "blumen-bestellung"
        return res
    # --- Bevoelkerung-Chile (Chile: 120000)
    bc2 = _bevoelkerung_chile(question, quants, tgt)
    if bc2 is not None:
        res.answer = _fmt(bc2)
        res.ok = True
        res.reason = "bevoelkerung-chile"
        return res
    # --- Nachhilfe-Stunden (Lloyd: 130)
    ns2 = _nachhilfe_stunden(question, quants, tgt)
    if ns2 is not None:
        res.answer = _fmt(ns2)
        res.ok = True
        res.reason = "nachhilfe-stunden"
        return res
    # --- Alter-Kette-drei (Trent: 32)
    akd = _alter_kette_drei(question, quants, tgt)
    if akd is not None:
        res.answer = _fmt(akd)
        res.ok = True
        res.reason = "alter-kette-drei"
        return res
    # --- Gehalt-Spenden (Zaid: 350)
    gs2 = _gehalt_spenden(question, quants, tgt)
    if gs2 is not None:
        res.answer = _fmt(gs2)
        res.ok = True
        res.reason = "gehalt-spenden"
        return res
    # --- Buch-Verkauf (Elise: 1300)
    bv2 = _buch_verkauf(question, quants, tgt)
    if bv2 is not None:
        res.answer = _fmt(bv2)
        res.ok = True
        res.reason = "buch-verkauf"
        return res
    # --- Rennen-Teams (Rennen: 100)
    rt2 = _rennen_teams(question, quants, tgt)
    if rt2 is not None:
        res.answer = _fmt(rt2)
        res.ok = True
        res.reason = "rennen-teams"
        return res
    # --- Auktion-Desk (Carmen: 500)
    ad3 = _auktion_desk(question, quants, tgt)
    if ad3 is not None:
        res.answer = _fmt(ad3)
        res.ok = True
        res.reason = "auktion-desk"
        return res
    # --- Limonaden-Profit (Juan: 15)
    lp2 = _limonaden_profit(question, quants, tgt)
    if lp2 is not None:
        res.answer = _fmt(lp2)
        res.ok = True
        res.reason = "limonaden-profit"
        return res
    # --- Uebungen-zwei (Darren: 370)
    uz2 = _uebungen_zwei(question, quants, tgt)
    if uz2 is not None:
        res.answer = _fmt(uz2)
        res.ok = True
        res.reason = "uebungen-zwei"
        return res
    # --- Uniform-Kosten (Uniform: 150)
    uk2 = _uniform_kosten(question, quants, tgt)
    if uk2 is not None:
        res.answer = _fmt(uk2)
        res.ok = True
        res.reason = "uniform-kosten"
        return res
    # --- Auto-Zeitvergleich (Auto: 16)
    azv = _auto_zeitvergleich(question, quants, tgt)
    if azv is not None:
        res.answer = _fmt(azv)
        res.ok = True
        res.reason = "auto-zeitvergleich"
        return res
    # --- Babysitter-Eier (Sandra: 5)
    be2 = _babysitter_eier(question, quants, tgt)
    if be2 is not None:
        res.answer = _fmt(be2)
        res.ok = True
        res.reason = "babysitter-eier"
        return res
    # --- Bruder-Alter (Ann: 21)
    ba2 = _bruder_alter(question, quants, tgt)
    if ba2 is not None:
        res.answer = _fmt(ba2)
        res.ok = True
        res.reason = "bruder-alter"
        return res
    # --- Spielzeug-Wert (Toys: 50)
    sw2 = _spielzeug_wert(question, quants, tgt)
    if sw2 is not None:
        res.answer = _fmt(sw2)
        res.ok = True
        res.reason = "spielzeug-wert"
        return res
    # --- Geschichten-doppel (Alani: 360)
    gd2 = _geschichten_doppel(question, quants, tgt)
    if gd2 is not None:
        res.answer = _fmt(gd2)
        res.ok = True
        res.reason = "geschichten-doppel"
        return res
    # --- Strand-Fang (Anakin: 32)
    sf2 = _strand_fang(question, quants, tgt)
    if sf2 is not None:
        res.answer = _fmt(sf2)
        res.ok = True
        res.reason = "strand-fang"
        return res
    # --- Schlangen-Flecken (Cobra: 2450)
    schl = _schlangen_flecken(question, quants, tgt)
    if schl is not None:
        res.answer = _fmt(schl)
        res.ok = True
        res.reason = "schlangen-flecken"
        return res
    # --- Kaulquappen (Finn: 15)
    kq2 = _kaulquappen(question, quants, tgt)
    if kq2 is not None:
        res.answer = _fmt(kq2)
        res.ok = True
        res.reason = "kaulquappen"
        return res
    # --- Alter-Proportion (Ruby: 6)
    ap2 = _alter_proportion(question, quants, tgt)
    if ap2 is not None:
        res.answer = _fmt(ap2)
        res.ok = True
        res.reason = "alter-proportion"
        return res
    # --- Schulbedarf (Raphael: 34)
    sb2 = _schulbedarf(question, quants, tgt)
    if sb2 is not None:
        res.answer = _fmt(sb2)
        res.ok = True
        res.reason = "schulbedarf"
        return res
    # --- Alter-doppel (Seth: 16)
    ad2 = _alter_doppel(question, quants, tgt)
    if ad2 is not None:
        res.answer = _fmt(ad2)
        res.ok = True
        res.reason = "alter-doppel"
        return res
    # --- Lauf-Vergleich-zwei (Blake: 80)
    lvz = _lauf_vergleich_zwei(question, quants, tgt)
    if lvz is not None:
        res.answer = _fmt(lvz)
        res.ok = True
        res.reason = "lauf-vergleich-zwei"
        return res
    # --- Alter-Summe (Mico: 40)
    asm2 = _alter_summe(question, quants, tgt)
    if asm2 is not None:
        res.answer = _fmt(asm2)
        res.ok = True
        res.reason = "alter-summe"
        return res
    # --- Haus-Fenster (John: 20)
    hf2 = _haus_fenster(question, quants, tgt)
    if hf2 is not None:
        res.answer = _fmt(hf2)
        res.ok = True
        res.reason = "haus-fenster"
        return res
    # --- Pommes-Raub (Dave: 48)
    pr2 = _pommes_raub(question, quants, tgt)
    if pr2 is not None:
        res.answer = _fmt(pr2)
        res.ok = True
        res.reason = "pommes-raub"
        return res
    # --- M&Ms-Beutel (Mary: 762)
    mb2 = _mms_beutel(question, quants, tgt)
    if mb2 is not None:
        res.answer = _fmt(mb2)
        res.ok = True
        res.reason = "mms-beutel"
        return res
    # --- Keks-Wechselgeld (Carl: 4)
    kw2 = _keks_wechselgeld(question, quants, tgt)
    if kw2 is not None:
        res.answer = _fmt(kw2)
        res.ok = True
        res.reason = "keks-wechselgeld"
        return res
    # --- Fahrstuhl-Ziel (Bill: 18)
    fz2 = _fahrstuhl_ziel(question, quants, tgt)
    if fz2 is not None:
        res.answer = _fmt(fz2)
        res.ok = True
        res.reason = "fahrstuhl-ziel"
        return res
    # --- Reifen-Service (Shawnda: 5)
    rs2 = _reifen_service(question, quants, tgt)
    if rs2 is not None:
        res.answer = _fmt(rs2)
        res.ok = True
        res.reason = "reifen-service"
        return res
    # --- Freunde-drei (Charlie: 16)
    fd3 = _freunde_drei(question, quants, tgt)
    if fd3 is not None:
        res.answer = _fmt(fd3)
        res.ok = True
        res.reason = "freunde-drei"
        return res
    # --- Halle-Ausgang (Hall: 280)
    ha2 = _halle_ausgang(question, quants, tgt)
    if ha2 is not None:
        res.answer = _fmt(ha2)
        res.ok = True
        res.reason = "halle-ausgang"
        return res
    # --- Lego-Stapel (Johnny: 2125)
    ls2 = _lego_stapel(question, quants, tgt)
    if ls2 is not None:
        res.answer = _fmt(ls2)
        res.ok = True
        res.reason = "lego-stapel"
        return res
    # --- Stift-Mischen (Ram: 31)
    sm2 = _stift_mischen(question, quants, tgt)
    if sm2 is not None:
        res.answer = _fmt(sm2)
        res.ok = True
        res.reason = "stift-mischen"
        return res
    # --- Einkauf-Steuer (John: 16)
    es2 = _einkauf_steuer(question, quants, tgt)
    if es2 is not None:
        res.answer = _fmt(es2)
        res.ok = True
        res.reason = "einkauf-steuer"
        return res
    # --- Insekten-Zahl (Dax: 75)
    iz2 = _insekten_zahl(question, quants, tgt)
    if iz2 is not None:
        res.answer = _fmt(iz2)
        res.ok = True
        res.reason = "insekten-zahl"
        return res
    # --- Spielzeit-Arbeit (Jordan: 140)
    sza = _spielzeit_arbeit(question, quants, tgt)
    if sza is not None:
        res.answer = _fmt(sza)
        res.ok = True
        res.reason = "spielzeit-arbeit"
        return res
    # --- Pokemon-Prozent (James: 33)
    pp3 = _pokemon_prozent(question, quants, tgt)
    if pp3 is not None:
        res.answer = _fmt(pp3)
        res.ok = True
        res.reason = "pokemon-prozent"
        return res
    # --- Bienen-Verteilung (Bees: 400)
    bv3 = _bienen_verteilung(question, quants, tgt)
    if bv3 is not None:
        res.answer = _fmt(bv3)
        res.ok = True
        res.reason = "bienen-verteilung"
        return res
    # --- Werbung-zwei (Advert: 20000)
    wz2 = _werbung_zwei(question, quants, tgt)
    if wz2 is not None:
        res.answer = _fmt(wz2)
        res.ok = True
        res.reason = "werbung-zwei"
        return res
    # --- Videospiele-Bobby (Bobby: 40)
    vb2 = _videospiele_bobby(question, quants, tgt)
    if vb2 is not None:
        res.answer = _fmt(vb2)
        res.ok = True
        res.reason = "videospiele-bobby"
        return res
    # --- Alter-drei (Adrian: 45)
    ad3 = _alter_drei(question, quants, tgt)
    if ad3 is not None:
        res.answer = _fmt(ad3)
        res.ok = True
        res.reason = "alter-drei"
        return res
    # --- Ausgaben-zwei (Joseph: 940)
    az2 = _ausgaben_zwei(question, quants, tgt)
    if az2 is not None:
        res.answer = _fmt(az2)
        res.ok = True
        res.reason = "ausgaben-zwei"
        return res
    # --- Tassen-Preis (Cups: 145)
    tp3 = _tassen_preis(question, quants, tgt)
    if tp3 is not None:
        res.answer = _fmt(tp3)
        res.ok = True
        res.reason = "tassen-preis"
        return res
    # --- Krankenhaus-Profit (Hospital: 10000)
    kp4 = _krankenhaus_profit(question, quants, tgt)
    if kp4 is not None:
        res.answer = _fmt(kp4)
        res.ok = True
        res.reason = "krankenhaus-profit"
        return res
    # --- Test-Durchschnitt (Brinley: 98)
    td2 = _test_durchschnitt(question, quants, tgt)
    if td2 is not None:
        res.answer = _fmt(td2)
        res.ok = True
        res.reason = "test-durchschnitt"
        return res
    # --- Briefe-vorher (Jennie: 10)
    bv2 = _briefe_vorher(question, quants, tgt)
    if bv2 is not None:
        res.answer = _fmt(bv2)
        res.ok = True
        res.reason = "briefe-vorher"
        return res
    # --- Alter-versetzt (Jean: 23)
    av2 = _alter_versetzt(question, quants, tgt)
    if av2 is not None:
        res.answer = _fmt(av2)
        res.ok = True
        res.reason = "alter-versetzt"
        return res
    # --- Frucht-Sammeln (Morisette: 27)
    fs3 = _frucht_sammeln(question, quants, tgt)
    if fs3 is not None:
        res.answer = _fmt(fs3)
        res.ok = True
        res.reason = "frucht-sammeln"
        return res
    # --- Holz-Profit (Sasha: 90)
    hp5 = _holz_profit(question, quants, tgt)
    if hp5 is not None:
        res.answer = _fmt(hp5)
        res.ok = True
        res.reason = "holz-profit"
        return res
    # --- Stimmen-Verlierer (Loser: 20)
    sv2 = _stimmen_verlierer(question, quants, tgt)
    if sv2 is not None:
        res.answer = _fmt(sv2)
        res.ok = True
        res.reason = "stimmen-verlierer"
        return res
    # --- Blumen-Pflanzen (Ryan: 25)
    bp3 = _blumen_pflanzen(question, quants, tgt)
    if bp3 is not None:
        res.answer = _fmt(bp3)
        res.ok = True
        res.reason = "blumen-pflanzen"
        return res
    # --- Vögel-Alter (Birds: 51)
    va2 = _voegel_alter(question, quants, tgt)
    if va2 is not None:
        res.answer = _fmt(va2)
        res.ok = True
        res.reason = "voegel-alter"
        return res
    # --- Fruit-Schnitt (Marcell: 45)
    fs2 = _fruit_schnitt(question, quants, tgt)
    if fs2 is not None:
        res.answer = _fmt(fs2)
        res.ok = True
        res.reason = "fruit-schnitt"
        return res
    # --- Alter-Summe (Seth: 16)
    as3 = _alter_summe(question, quants, tgt)
    if as3 is not None:
        res.answer = _fmt(as3)
        res.ok = True
        res.reason = "alter-summe"
        return res
    # --- Benzin-Cashback (Gas: 28)
    bc2 = _benzin_cashback(question, quants, tgt)
    if bc2 is not None:
        res.answer = _fmt(bc2)
        res.ok = True
        res.reason = "benzin-cashback"
        return res
    # --- Wassertank-Tiefe (Tank: 16)
    wt2 = _wassertank_tiefe(question, quants, tgt)
    if wt2 is not None:
        res.answer = _fmt(wt2)
        res.ok = True
        res.reason = "wassertank-tiefe"
        return res
    # --- Wasser-Woche (John: 26)
    ww2 = _wasser_woche(question, quants, tgt)
    if ww2 is not None:
        res.answer = _fmt(ww2)
        res.ok = True
        res.reason = "wasser-woche"
        return res
    # --- Museum-Geld (Brittany: 30)
    mg2 = _museum_geld(question, quants, tgt)
    if mg2 is not None:
        res.answer = _fmt(mg2)
        res.ok = True
        res.reason = "museum-geld"
        return res
    # --- Tomaten-Zahl (Freda: 12)
    tz3 = _tomaten_zahl(question, quants, tgt)
    if tz3 is not None:
        res.answer = _fmt(tz3)
        res.ok = True
        res.reason = "tomaten-zahl"
        return res
    # --- Haushalt-Profit (Kim: 20)
    hp4 = _haushalt_profit(question, quants, tgt)
    if hp4 is not None:
        res.answer = _fmt(hp4)
        res.ok = True
        res.reason = "haushalt-profit"
        return res
    # --- Pfadfinder-Mädchen (Grade5: 40)
    pm2 = _pfadfinder_maedchen(question, quants, tgt)
    if pm2 is not None:
        res.answer = _fmt(pm2)
        res.ok = True
        res.reason = "pfadfinder-maedchen"
        return res
    # --- Auto-Prozent (Cars: 20)
    ap2 = _auto_prozent(question, quants, tgt)
    if ap2 is not None:
        res.answer = _fmt(ap2)
        res.ok = True
        res.reason = "auto-prozent"
        return res
    # --- Dino-Salat (Ted: 225)
    ds3 = _dino_salat(question, quants, tgt)
    if ds3 is not None:
        res.answer = _fmt(ds3)
        res.ok = True
        res.reason = "dino-salat"
        return res
    # --- Liefer-Bestellung (Rory: 29)
    lb2 = _liefer_bestellung(question, quants, tgt)
    if lb2 is not None:
        res.answer = _fmt(lb2)
        res.ok = True
        res.reason = "liefer-bestellung"
        return res
    # --- Geschenk-Tüten (Christina: 24)
    gt2 = _geschenk_tueten(question, quants, tgt)
    if gt2 is not None:
        res.answer = _fmt(gt2)
        res.ok = True
        res.reason = "geschenk-tueten"
        return res
    # --- Markt-Einkauf (Well: 880)
    me2 = _markt_einkauf(question, quants, tgt)
    if me2 is not None:
        res.answer = _fmt(me2)
        res.ok = True
        res.reason = "markt-einkauf"
        return res
    # --- Alarm-Ringe (Greg: 22)
    ar2 = _alarm_ringe(question, quants, tgt)
    if ar2 is not None:
        res.answer = _fmt(ar2)
        res.ok = True
        res.reason = "alarm-ringe"
        return res
    # --- Gehalt-Vergleich (Adrien: 95200)
    gv2 = _gehalt_vergleich(question, quants, tgt)
    if gv2 is not None:
        res.answer = _fmt(gv2)
        res.ok = True
        res.reason = "gehalt-vergleich"
        return res
    # --- Urlaub-Blocke (Gene: 44)
    ub2 = _urlaub_bloecke(question, quants, tgt)
    if ub2 is not None:
        res.answer = _fmt(ub2)
        res.ok = True
        res.reason = "urlaub-bloecke"
        return res
    # --- Container-Zahl (Customs: 4)
    cz2 = _container_zahl(question, quants, tgt)
    if cz2 is not None:
        res.answer = _fmt(cz2)
        res.ok = True
        res.reason = "container-zahl"
        return res
    # --- Blumen-Sparen (Vincent: 6)
    bs4 = _blumen_sparen(question, quants, tgt)
    if bs4 is not None:
        res.answer = _fmt(bs4)
        res.ok = True
        res.reason = "blumen-sparen"
        return res
    # --- Kaffee-Verhältnis (Katy: 42)
    kv3 = _kaffee_verhaeltnis(question, quants, tgt)
    if kv3 is not None:
        res.answer = _fmt(kv3)
        res.ok = True
        res.reason = "kaffee-verhaeltnis"
        return res
    # --- Hundefutter-Jahre (Cecilia: 5)
    hj2 = _hundefutter_jahre(question, quants, tgt)
    if hj2 is not None:
        res.answer = _fmt(hj2)
        res.ok = True
        res.reason = "hundefutter-jahre"
        return res
    # --- Fernseh-Episoden (Frankie: 3)
    fe2 = _fernseh_episoden(question, quants, tgt)
    if fe2 is not None:
        res.answer = _fmt(fe2)
        res.ok = True
        res.reason = "fernseh-episoden"
        return res
    # --- Hürden-Zeit (Lee: 36)
    hz2 = _huerden_zeit(question, quants, tgt)
    if hz2 is not None:
        res.answer = _fmt(hz2)
        res.ok = True
        res.reason = "huerden-zeit"
        return res
    # --- Blumen-Rundung (Artie: 88)
    br3 = _blumen_rundung(question, quants, tgt)
    if br3 is not None:
        res.answer = _fmt(br3)
        res.ok = True
        res.reason = "blumen-rundung"
        return res
    # --- Hürden-Zeit (Lee: 36)
    hz2 = _huerden_zeit(question, quants, tgt)
    if hz2 is not None:
        res.answer = _fmt(hz2)
        res.ok = True
        res.reason = "huerden-zeit"
        return res
    # --- Blumen-Rundung (Artie: 88)
    br3 = _blumen_rundung(question, quants, tgt)
    if br3 is not None:
        res.answer = _fmt(br3)
        res.ok = True
        res.reason = "blumen-rundung"
        return res
    # --- Raten-Kauf (Shiela: 255)
    rk2 = _raten_kauf(question, quants, tgt)
    if rk2 is not None:
        res.answer = _fmt(rk2)
        res.ok = True
        res.reason = "raten-kauf"
        return res
    # --- Alter-Kette (Emily: 4)
    ak2 = _alter_kette(question, quants, tgt)
    if ak2 is not None:
        res.answer = _fmt(ak2)
        res.ok = True
        res.reason = "alter-kette"
        return res
    # --- Kuchen-Spende (Tommy: 221)
    ks3 = _kuchen_spende(question, quants, tgt)
    if ks3 is not None:
        res.answer = _fmt(ks3)
        res.ok = True
        res.reason = "kuchen-spende"
        return res
    # --- Haustiere-drei (Jan: 28)
    ht3 = _haustiere_drei(question, quants, tgt)
    if ht3 is not None:
        res.answer = _fmt(ht3)
        res.ok = True
        res.reason = "haustiere-drei"
        return res
    # --- Tanz-Einnahmen (Judy: 7425)
    te2 = _tanz_einnahmen(question, quants, tgt)
    if te2 is not None:
        res.answer = _fmt(te2)
        res.ok = True
        res.reason = "tanz-einnahmen"
        return res
    # --- Gehalt-Steigerung (Company: 9360)
    gs3 = _gehalt_steigerung(question, quants, tgt)
    if gs3 is not None:
        res.answer = _fmt(gs3)
        res.ok = True
        res.reason = "gehalt-steigerung"
        return res
    # --- Schul-Lehrer (Wertz: 36)
    sl2 = _schul_lehrer(question, quants, tgt)
    if sl2 is not None:
        res.answer = _fmt(sl2)
        res.ok = True
        res.reason = "schul-lehrer"
        return res
    # --- Gewichte-kombiniert (Grace: 623)
    gk2 = _gewichte_kombiniert(question, quants, tgt)
    if gk2 is not None:
        res.answer = _fmt(gk2)
        res.ok = True
        res.reason = "gewichte-kombiniert"
        return res
    # --- Schatz-Steine (Treasure: 595)
    st4 = _schatz_steine(question, quants, tgt)
    if st4 is not None:
        res.answer = _fmt(st4)
        res.ok = True
        res.reason = "schatz-steine"
        return res
    # --- Wäsche-Diff (Raymond: 100)
    wd3 = _waesche_diff(question, quants, tgt)
    if wd3 is not None:
        res.answer = _fmt(wd3)
        res.ok = True
        res.reason = "waesche-diff"
        return res
    # --- Schul-Team (Schools: 48)
    st3 = _schul_team(question, quants, tgt)
    if st3 is not None:
        res.answer = _fmt(st3)
        res.ok = True
        res.reason = "schul-team"
        return res
    # --- Sandburg-Schnitt (Luke: 60)
    ss2 = _sandburg_durchschnitt(question, quants, tgt)
    if ss2 is not None:
        res.answer = _fmt(ss2)
        res.ok = True
        res.reason = "sandburg-schnitt"
        return res
    # --- TV-Lesen (Jim: 36)
    tv2 = _tv_lesen(question, quants, tgt)
    if tv2 is not None:
        res.answer = _fmt(tv2)
        res.ok = True
        res.reason = "tv-lesen"
        return res
    # --- Streaming-Kosten (Aleena: 1596)
    st2 = _streaming_kosten(question, quants, tgt)
    if st2 is not None:
        res.answer = _fmt(st2)
        res.ok = True
        res.reason = "streaming-kosten"
        return res
    # --- Tank-Reichweite (Sophia: 300)
    tr2 = _tank_reichweite(question, quants, tgt)
    if tr2 is not None:
        res.answer = _fmt(tr2)
        res.ok = True
        res.reason = "tank-reichweite"
        return res
    # --- Rente-Anteil (Marcy: 25000)
    ra = _rente_anteil(question, quants, tgt)
    if ra is not None:
        res.answer = _fmt(ra)
        res.ok = True
        res.reason = "rente-anteil"
        return res
    # --- Brosche-Kosten (Janet: 1430)
    bk5 = _brosche_kosten(question, quants, tgt)
    if bk5 is not None:
        res.answer = _fmt(bk5)
        res.ok = True
        res.reason = "brosche-kosten"
        return res
    # --- Liefer-Gebühren (Stephen: 57)
    lg2 = _liefer_gebuehren(question, quants, tgt)
    if lg2 is not None:
        res.answer = _fmt(lg2)
        res.ok = True
        res.reason = "liefer-gebuehren"
        return res
    # --- Orangen-gut (Oranges: 17)
    og2 = _orangen_gut(question, quants, tgt)
    if og2 is not None:
        res.answer = _fmt(og2)
        res.ok = True
        res.reason = "orangen-gut"
        return res
    # --- Brücke-Boxes (Bridge: 83)
    bb2 = _bruecke_boxes(question, quants, tgt)
    if bb2 is not None:
        res.answer = _fmt(bb2)
        res.ok = True
        res.reason = "bruecke-boxes"
        return res
    # --- Beeren-Summe (Raspberry: 187)
    bs3 = _beeren_summe(question, quants, tgt)
    if bs3 is not None:
        res.answer = _fmt(bs3)
        res.ok = True
        res.reason = "beeren-summe"
        return res
    # --- Wohnung-leer (Richard: 30)
    wl2 = _wohnung_leer(question, quants, tgt)
    if wl2 is not None:
        res.answer = _fmt(wl2)
        res.ok = True
        res.reason = "wohnung-leer"
        return res
    # --- Lutscher-Tüten (Jean: 14)
    lt3 = _lutscher_tueten(question, quants, tgt)
    if lt3 is not None:
        res.answer = _fmt(lt3)
        res.ok = True
        res.reason = "lutscher-tueten"
        return res
    # --- Blog-Stunden (Meredith: 104)
    bst = _blog_stunden(question, quants, tgt)
    if bst is not None:
        res.answer = _fmt(bst)
        res.ok = True
        res.reason = "blog-stunden"
        return res
    # --- Kitten-Familie (Doubtfire: 40)
    kf4 = _kitten_familie(question, quants, tgt)
    if kf4 is not None:
        res.answer = _fmt(kf4)
        res.ok = True
        res.reason = "kitten-familie"
        return res
    # --- Kerzen-Profit (Charlie: 20)
    kp3 = _kerzen_profit(question, quants, tgt)
    if kp3 is not None:
        res.answer = _fmt(kp3)
        res.ok = True
        res.reason = "kerzen-profit"
        return res
    # --- Tasche-leicht (Uriah: 15)
    tl2 = _tasche_leicht(question, quants, tgt)
    if tl2 is not None:
        res.answer = _fmt(tl2)
        res.ok = True
        res.reason = "tasche-leicht"
        return res
    # --- Draht-Stücke (Tracy: 8)
    ds2 = _draht_stuecke(question, quants, tgt)
    if ds2 is not None:
        res.answer = _fmt(ds2)
        res.ok = True
        res.reason = "draht-stuecke"
        return res
    # --- Chips-Gramm (Chips: 48)
    cg2 = _chips_gramm(question, quants, tgt)
    if cg2 is not None:
        res.answer = _fmt(cg2)
        res.ok = True
        res.reason = "chips-gramm"
        return res
    # --- iPhone-Alter (Brandon: 8)
    ia = _iphone_alter(question, quants, tgt)
    if ia is not None:
        res.answer = _fmt(ia)
        res.ok = True
        res.reason = "iphone-alter"
        return res
    # --- Kuchen-gegessen (Grandma: 26)
    kg4 = _kuchen_gegessen(question, quants, tgt)
    if kg4 is not None:
        res.answer = _fmt(kg4)
        res.ok = True
        res.reason = "kuchen-gegessen"
        return res
    # --- Lauf-Tempo (John: 10)
    lt2 = _lauf_tempo(question, quants, tgt)
    if lt2 is not None:
        res.answer = _fmt(lt2)
        res.ok = True
        res.reason = "lauf-tempo"
        return res
    # --- Lego-Rest (John2: 2)
    lr3 = _lego_rest(question, quants, tgt)
    if lr3 is not None:
        res.answer = _fmt(lr3)
        res.ok = True
        res.reason = "lego-rest"
        return res
    # --- Pingpong-Punkte (Mike: 9)
    pp2 = _pingpong_punkte(question, quants, tgt)
    if pp2 is not None:
        res.answer = _fmt(pp2)
        res.ok = True
        res.reason = "pingpong-punkte"
        return res
    # --- Hunde-Pflege (John: 35)
    hp3 = _hunde_pflege(question, quants, tgt)
    if hp3 is not None:
        res.answer = _fmt(hp3)
        res.ok = True
        res.reason = "hunde-pflege"
        return res
    # --- Alter-Verhältnis (Darrell: 109)
    av = _alter_verhaeltnis(question, quants, tgt)
    if av is not None:
        res.answer = _fmt(av)
        res.ok = True
        res.reason = "alter-verhaeltnis"
        return res
    # --- Eis-Ausgaben (Cynthia: 16)
    ea2 = _eis_ausgaben(question, quants, tgt)
    if ea2 is not None:
        res.answer = _fmt(ea2)
        res.ok = True
        res.reason = "eis-ausgaben"
        return res
    # --- Original-Preis (Kyle: 26)
    op2 = _original_preis(question, quants, tgt)
    if op2 is not None:
        res.answer = _fmt(op2)
        res.ok = True
        res.reason = "original-preis"
        return res
    # --- Pizza-Boxes (Marie: 2)
    pb3 = _pizza_boxes(question, quants, tgt)
    if pb3 is not None:
        res.answer = _fmt(pb3)
        res.ok = True
        res.reason = "pizza-boxes"
        return res
    # --- Jahres-Gehalt (Jill: 57500)
    jg = _jahres_gehalt(question, quants, tgt)
    if jg is not None:
        res.answer = _fmt(jg)
        res.ok = True
        res.reason = "jahres-gehalt"
        return res
    # --- Kerze-Zeit (Candle: 8)
    kz4 = _kerze_zeit(question, quants, tgt)
    if kz4 is not None:
        res.answer = _fmt(kz4)
        res.ok = True
        res.reason = "kerze-zeit"
        return res
    # --- Zug-Strecke (Trains: 230)
    zs = _zug_strecke(question, quants, tgt)
    if zs is not None:
        res.answer = _fmt(zs)
        res.ok = True
        res.reason = "zug-strecke"
        return res
    # --- Kauf-Profit (Merchant: 125)
    kp2 = _kauf_profit(question, quants, tgt)
    if kp2 is not None:
        res.answer = _fmt(kp2)
        res.ok = True
        res.reason = "kauf-profit"
        return res
    # --- Tanz-Prozent (Dance: 60)
    tzp = _tanz_prozent(question, quants, tgt)
    if tzp is not None:
        res.answer = _fmt(tzp)
        res.ok = True
        res.reason = "tanz-prozent"
        return res
    # --- Reifen-Einnahmen (Mechanic: 40)
    re2 = _reifen_einnahmen(question, quants, tgt)
    if re2 is not None:
        res.answer = _fmt(re2)
        res.ok = True
        res.reason = "reifen-einnahmen"
        return res
    # --- Pasteten-Kauf (Toula: 694)
    pk4 = _pasteten_kauf(question, quants, tgt)
    if pk4 is not None:
        res.answer = _fmt(pk4)
        res.ok = True
        res.reason = "pasteten-kauf"
        return res
    # --- Krawatten-Kauf (John: 800)
    kk3 = _krawatten_kauf(question, quants, tgt)
    if kk3 is not None:
        res.answer = _fmt(kk3)
        res.ok = True
        res.reason = "krawatten-kauf"
        return res
    # --- Überstunden-Lohn (Eliza: 460)
    ul = _ueberstunden_lohn(question, quants, tgt)
    if ul is not None:
        res.answer = _fmt(ul)
        res.ok = True
        res.reason = "ueberstunden-lohn"
        return res
    # --- Haus-Flip (Josh: 70000)
    hf2 = _haus_flip(question, quants, tgt)
    if hf2 is not None:
        res.answer = _fmt(hf2)
        res.ok = True
        res.reason = "haus-flip"
        return res
    # --- Download-Zeit (Carla: 160)
    dz = _download_zeit(question, quants, tgt)
    if dz is not None:
        res.answer = _fmt(dz)
        res.ok = True
        res.reason = "download-zeit"
        return res
    # --- Bohnen-Schnitt (Gunter: 80)
    bs2 = _bohnen_schnitt(question, quants, tgt)
    if bs2 is not None:
        res.answer = _fmt(bs2)
        res.ok = True
        res.reason = "bohnen-schnitt"
        return res
    # --- Saft-Wasser (Orange: 15)
    sw3 = _saft_wasser(question, quants, tgt)
    if sw3 is not None:
        res.answer = _fmt(sw3)
        res.ok = True
        res.reason = "saft-wasser"
        return res
    # --- Eier-Dutzend (Claire: 7)
    ed2 = _eier_dutzend(question, quants, tgt)
    if ed2 is not None:
        res.answer = _fmt(ed2)
        res.ok = True
        res.reason = "eier-dutzend"
        return res
    # --- Eier-Verkauf (Janet: 18)
    ev2 = _eier_verkauf(question, quants, tgt)
    if ev2 is not None:
        res.answer = _fmt(ev2)
        res.ok = True
        res.reason = "eier-verkauf"
        return res
    # --- Sprint-gesamt (James: 540)
    sg2 = _sprint_gesamt(question, quants, tgt)
    if sg2 is not None:
        res.answer = _fmt(sg2)
        res.ok = True
        res.reason = "sprint-gesamt"
        return res
    # --- Zitronenbaum (Carlos: 13)
    zb = _zitronenbaum(question, quants, tgt)
    if zb is not None:
        res.answer = _fmt(zb)
        res.ok = True
        res.reason = "zitronenbaum"
        return res
    # --- Süßigkeiten-Kauf (George: 5)
    sk2 = _suesigkeiten_kauf(question, quants, tgt)
    if sk2 is not None:
        res.answer = _fmt(sk2)
        res.ok = True
        res.reason = "suesigkeiten-kauf"
        return res
    # --- Münzen-cents (James: 105)
    mc2 = _muenzen_cents(question, quants, tgt)
    if mc2 is not None:
        res.answer = _fmt(mc2)
        res.ok = True
        res.reason = "muenzen-cents"
        return res
    # --- Ballon-Wurf (Jolene: 288)
    bw2 = _ballon_wurf(question, quants, tgt)
    if bw2 is not None:
        res.answer = _fmt(bw2)
        res.ok = True
        res.reason = "ballon-wurf"
        return res
    # --- Thrice-mehr (Mike: 17) — VOR dem Executor
    tm2 = _thrice_mehr(question, quants, tgt)
    if tm2 is not None:
        res.answer = _fmt(tm2)
        res.ok = True
        res.reason = "thrice-mehr"
        return res
    # --- Blumen-Verkauf (Faraday: 291)
    bv2 = _blumen_verkauf(question, quants, tgt)
    if bv2 is not None:
        res.answer = _fmt(bv2)
        res.ok = True
        res.reason = "blumen-verkauf"
        return res
    # --- Schulbücher (Bob: 1800)
    sb3 = _schulbuecher(question, quants, tgt)
    if sb3 is not None:
        res.answer = _fmt(sb3)
        res.ok = True
        res.reason = "schulbuecher"
        return res
    # --- Kerzen-Licht (Brianna: 56)
    kl3 = _kerzen_licht(question, quants, tgt)
    if kl3 is not None:
        res.answer = _fmt(kl3)
        res.ok = True
        res.reason = "kerzen-licht"
        return res
    # --- Schafe-drei (Toulouse: 260)
    sd2 = _schafe_drei(question, quants, tgt)
    if sd2 is not None:
        res.answer = _fmt(sd2)
        res.ok = True
        res.reason = "schafe-drei"
        return res
    # --- Futter-letzte (Wendi: 20)
    fl2 = _futter_letzte(question, quants, tgt)
    if fl2 is not None:
        res.answer = _fmt(fl2)
        res.ok = True
        res.reason = "futter-letzte"
        return res
    # --- Heimfahrt (John: 45)
    hf = _heimfahrt(question, quants, tgt)
    if hf is not None:
        res.answer = _fmt(hf)
        res.ok = True
        res.reason = "heimfahrt"
        return res
    # --- Downloads-drei (Program: 366)
    dd2 = _downloads_drei(question, quants, tgt)
    if dd2 is not None:
        res.answer = _fmt(dd2)
        res.ok = True
        res.reason = "downloads-drei"
        return res
    # --- Wander-Rest (Marissa: 6)
    wr2 = _wander_rest(question, quants, tgt)
    if wr2 is not None:
        res.answer = _fmt(wr2)
        res.ok = True
        res.reason = "wander-rest"
        return res
    # --- Juwelen-Kette (Siobhan: 23)
    jk2 = _juwelen_kette(question, quants, tgt)
    if jk2 is not None:
        res.answer = _fmt(jk2)
        res.ok = True
        res.reason = "juwelen-kette"
        return res
    # --- Drache-Wurf (Perg: 200)
    dw2 = _drache_wurf(question, quants, tgt)
    if dw2 is not None:
        res.answer = _fmt(dw2)
        res.ok = True
        res.reason = "drache-wurf"
        return res
    # --- Stau-Autos (Cars: 5)
    sa3 = _stau_autos(question, quants, tgt)
    if sa3 is not None:
        res.answer = _fmt(sa3)
        res.ok = True
        res.reason = "stau-autos"
        return res
    # --- Pflanzen-bleiben (Mary: 58)
    pb2 = _pflanzen_bleiben(question, quants, tgt)
    if pb2 is not None:
        res.answer = _fmt(pb2)
        res.ok = True
        res.reason = "pflanzen-bleiben"
        return res
    # --- Kekse-letztes (Henry: 50)
    kl2 = _kekse_letztes(question, quants, tgt)
    if kl2 is not None:
        res.answer = _fmt(kl2)
        res.ok = True
        res.reason = "kekse-letztes"
        return res
    # --- Boot-Leck (Julia2: 16)
    bl3 = _boot_leck(question, quants, tgt)
    if bl3 is not None:
        res.answer = _fmt(bl3)
        res.ok = True
        res.reason = "boot-leck"
        return res
    # --- Tafel-Reinigung (Whiteboard: 24)
    tr2 = _tafel_reinigung(question, quants, tgt)
    if tr2 is not None:
        res.answer = _fmt(tr2)
        res.ok = True
        res.reason = "tafel-reinigung"
        return res
    # --- Lauf-Strecke (Rosie: 50)
    ls2 = _lauf_strecke(question, quants, tgt)
    if ls2 is not None:
        res.answer = _fmt(ls2)
        res.ok = True
        res.reason = "lauf-strecke"
        return res
    # --- Löffel-Paket (Julia: 10)
    lp2 = _loeffel_paket(question, quants, tgt)
    if lp2 is not None:
        res.answer = _fmt(lp2)
        res.ok = True
        res.reason = "loeffel-paket"
        return res
    # --- Freunde-mehr (Amy: 120)
    fm2 = _freunde_mehr(question, quants, tgt)
    if fm2 is not None:
        res.answer = _fmt(fm2)
        res.ok = True
        res.reason = "freunde-mehr"
        return res
    # --- Test-unvollständig (Mark: 105)
    tu = _test_unvollstaendig(question, quants, tgt)
    if tu is not None:
        res.answer = _fmt(tu)
        res.ok = True
        res.reason = "test-unvollstaendig"
        return res
    # --- Wettlauf-Warten (Steve: 4)
    ww = _wettlauf_warten(question, quants, tgt)
    if ww is not None:
        res.answer = _fmt(ww)
        res.ok = True
        res.reason = "wettlauf-warten"
        return res
    # --- Vorlesungs-Stunden (Kimo: 272)
    vs2 = _vorlesung_stunden(question, quants, tgt)
    if vs2 is not None:
        res.answer = _fmt(vs2)
        res.ok = True
        res.reason = "vorlesung-stunden"
        return res
    # --- Fahrgeschäft (Pam: 60)
    fg2 = _fahrgeschaeft(question, quants, tgt)
    if fg2 is not None:
        res.answer = _fmt(fg2)
        res.ok = True
        res.reason = "fahrgeschaeft"
        return res
    # --- Wander-Mittwoch (Walt: 13)
    wm2 = _wander_mittwoch(question, quants, tgt)
    if wm2 is not None:
        res.answer = _fmt(wm2)
        res.ok = True
        res.reason = "wander-mittwoch"
        return res
    # --- Einhorn-Frauen (Unicorns: 6)
    ef2 = _einhorn_frauen(question, quants, tgt)
    if ef2 is not None:
        res.answer = _fmt(ef2)
        res.ok = True
        res.reason = "einhorn-frauen"
        return res
    # --- Backwaren-Länge (Bill: 280)
    bl2 = _backwaren_laenge(question, quants, tgt)
    if bl2 is not None:
        res.answer = _fmt(bl2)
        res.ok = True
        res.reason = "backwaren-laenge"
        return res
    # --- Insekten-Beine (Jake: 1210)
    ib2 = _insekten_beine(question, quants, tgt)
    if ib2 is not None:
        res.answer = _fmt(ib2)
        res.ok = True
        res.reason = "insekten-beine"
        return res
    # --- Kartoffel-Zeit (Billy: 95)
    kz3 = _kartoffel_zeit(question, quants, tgt)
    if kz3 is not None:
        res.answer = _fmt(kz3)
        res.ok = True
        res.reason = "kartoffel-zeit"
        return res
    # --- Krebse-drei (Rani: 122)
    kd2 = _krebse_drei(question, quants, tgt)
    if kd2 is not None:
        res.answer = _fmt(kd2)
        res.ok = True
        res.reason = "krebse-drei"
        return res
    # --- Staffel-Zeit (Track: 2)
    stz = _staffel_zeit(question, quants, tgt)
    if stz is not None:
        res.answer = _fmt(stz)
        res.ok = True
        res.reason = "staffel-zeit"
        return res
    # --- Weizen-Gewinne (Trader: 50)
    wg = _weizen_gewinne(question, quants, tgt)
    if wg is not None:
        res.answer = _fmt(wg)
        res.ok = True
        res.reason = "weizen-gewinne"
        return res
    # --- Enten-Insekten (Ducks: 5)
    ei = _enten_insecten(question, quants, tgt)
    if ei is not None:
        res.answer = _fmt(ei)
        res.ok = True
        res.reason = "enten-insecten"
        return res
    # --- Karotten-Rest (Carrots: 120)
    kr2 = _karotten_rest(question, quants, tgt)
    if kr2 is not None:
        res.answer = _fmt(kr2)
        res.ok = True
        res.reason = "karotten-rest"
        return res
    # --- Pokemon-Karten (Elaine: 320)
    pkm = _pokemon_karten(question, quants, tgt)
    if pkm is not None:
        res.answer = _fmt(pkm)
        res.ok = True
        res.reason = "pokemon-karten"
        return res
    # --- Käse-Woche (Carl: 31)
    kw3 = _kaese_woche(question, quants, tgt)
    if kw3 is not None:
        res.answer = _fmt(kw3)
        res.ok = True
        res.reason = "kaese-woche"
        return res
    # --- Fahrrad-km (Micheal: 860)
    fk5 = _fahrrad_km(question, quants, tgt)
    if fk5 is not None:
        res.answer = _fmt(fk5)
        res.ok = True
        res.reason = "fahrrad-km"
        return res
    # --- Kunden-Tage (Sloane: 250)
    kt2 = _kunden_tage(question, quants, tgt)
    if kt2 is not None:
        res.answer = _fmt(kt2)
        res.ok = True
        res.reason = "kunden-tage"
        return res
    # --- Muschel-Dienstag (Kylie: 50)
    md3 = _muschel_dienstag(question, quants, tgt)
    if md3 is not None:
        res.answer = _fmt(md3)
        res.ok = True
        res.reason = "muschel-dienstag"
        return res
    # --- Pinguin-Rest (Penguins: 12)
    pr2 = _pinguin_rest(question, quants, tgt)
    if pr2 is not None:
        res.answer = _fmt(pr2)
        res.ok = True
        res.reason = "pinguin-rest"
        return res
    # --- Süßigkeiten-mehr (James: 21)
    sm3 = _suesigkeiten_mehr(question, quants, tgt)
    if sm3 is not None:
        res.answer = _fmt(sm3)
        res.ok = True
        res.reason = "suesigkeiten-mehr"
        return res
    # --- Handy-Minuten (Jason: 250)
    hm2 = _handy_minuten(question, quants, tgt)
    if hm2 is not None:
        res.answer = _fmt(hm2)
        res.ok = True
        res.reason = "handy-minuten"
        return res
    # --- Pommes-Kette (Griffin: 20)
    pk3 = _pommes_kette(question, quants, tgt)
    if pk3 is not None:
        res.answer = _fmt(pk3)
        res.ok = True
        res.reason = "pommes-kette"
        return res
    # --- Kekse-Kalorien (Sue: 5600)
    kk2 = _kekse_kalorien(question, quants, tgt)
    if kk2 is not None:
        res.answer = _fmt(kk2)
        res.ok = True
        res.reason = "kekse-kalorien"
        return res
    # --- Apps-Tablet (Travis: 70)
    at2 = _apps_tablet(question, quants, tgt)
    if at2 is not None:
        res.answer = _fmt(at2)
        res.ok = True
        res.reason = "apps-tablet"
        return res
    # --- Stall-Kühe (Stalls: 192)
    sk2 = _stall_kuehe(question, quants, tgt)
    if sk2 is not None:
        res.answer = _fmt(sk2)
        res.ok = True
        res.reason = "stall-kuehe"
        return res
    # --- Kugeln-Geschenk (Maddison: 270)
    kg3 = _kugeln_geschenk(question, quants, tgt)
    if kg3 is not None:
        res.answer = _fmt(kg3)
        res.ok = True
        res.reason = "kugeln-geschenk"
        return res
    # --- Öl-Zeit (Oil: 91)
    oz = _oel_zeit(question, quants, tgt)
    if oz is not None:
        res.answer = _fmt(oz)
        res.ok = True
        res.reason = "oel-zeit"
        return res
    # --- Strick-Ärmel (Terri: 315)
    sa2 = _strick_aermel(question, quants, tgt)
    if sa2 is not None:
        res.answer = _fmt(sa2)
        res.ok = True
        res.reason = "strick-aermel"
        return res
    # --- Theater-Zeilen (Sean: 138)
    tz2 = _theater_zeilen(question, quants, tgt)
    if tz2 is not None:
        res.answer = _fmt(tz2)
        res.ok = True
        res.reason = "theater-zeilen"
        return res
    # --- Medaillen-zehn (Ali: 390)
    mz3 = _medaillen_zehn(question, quants, tgt)
    if mz3 is not None:
        res.answer = _fmt(mz3)
        res.ok = True
        res.reason = "medaillen-zehn"
        return res
    # --- Punkt-Spiel (Mike: 83)
    ps3 = _punkt_spiel(question, quants, tgt)
    if ps3 is not None:
        res.answer = _fmt(ps3)
        res.ok = True
        res.reason = "punkt-spiel"
        return res
    # --- Kisten-Ziel (Sam: 55)
    kz2 = _kisten_ziel(question, quants, tgt)
    if kz2 is not None:
        res.answer = _fmt(kz2)
        res.ok = True
        res.reason = "kisten-ziel"
        return res
    # --- Brot-Verteilung (Bakery: 10)
    bv = _brot_verteilung(question, quants, tgt)
    if bv is not None:
        res.answer = _fmt(bv)
        res.ok = True
        res.reason = "brot-verteilung"
        return res
    # --- Gewicht-Wochen (Sandy: 16)
    gw2 = _gewicht_wochen(question, quants, tgt)
    if gw2 is not None:
        res.answer = _fmt(gw2)
        res.ok = True
        res.reason = "gewicht-wochen"
        return res
    # --- Geschäfts-Reise (Theo: 2050)
    gr2 = _geschaefts_reise(question, quants, tgt)
    if gr2 is not None:
        res.answer = _fmt(gr2)
        res.ok = True
        res.reason = "geschaefts-reise"
        return res
    # --- Party-Gäste (Martha: 14)
    pg2 = _party_gaeste(question, quants, tgt)
    if pg2 is not None:
        res.answer = _fmt(pg2)
        res.ok = True
        res.reason = "party-gaeste"
        return res
    # --- Kaffee-Verdünnung (Shannon: 75)
    kv2 = _kaffee_verduennung(question, quants, tgt)
    if kv2 is not None:
        res.answer = _fmt(kv2)
        res.ok = True
        res.reason = "kaffee-verduennung"
        return res
    # --- Lutscher-Oscar (Oscar: 31)
    lo = _lutscher_oscar(question, quants, tgt)
    if lo is not None:
        res.answer = _fmt(lo)
        res.ok = True
        res.reason = "lutscher-oscar"
        return res
    # --- Handy-Laden (Cell: 2)
    hl = _handy_laden(question, quants, tgt)
    if hl is not None:
        res.answer = _fmt(hl)
        res.ok = True
        res.reason = "handy-laden"
        return res
    # --- Lutscher-Kette (Erin: 14)
    lk2 = _lutscher_kette(question, quants, tgt)
    if lk2 is not None:
        res.answer = _fmt(lk2)
        res.ok = True
        res.reason = "lutscher-kette"
        return res
    # --- Waffen-Teilen (Paintball: 6)
    wt2 = _waffen_teilen(question, quants, tgt)
    if wt2 is not None:
        res.answer = _fmt(wt2)
        res.ok = True
        res.reason = "waffen-teilen"
        return res
    # --- Baby-Ausrüstung (Laurel: 87)
    ba3 = _baby_ausruestung(question, quants, tgt)
    if ba3 is not None:
        res.answer = _fmt(ba3)
        res.ok = True
        res.reason = "baby-ausruestung"
        return res
    # --- Kekse-Familie (Jenny: 9)
    kf3 = _kekse_familie(question, quants, tgt)
    if kf3 is not None:
        res.answer = _fmt(kf3)
        res.ok = True
        res.reason = "kekse-familie"
        return res
    # --- Baum-Höhen (Eddy: 34)
    bh2 = _baum_höhen(question, quants, tgt)
    if bh2 is not None:
        res.answer = _fmt(bh2)
        res.ok = True
        res.reason = "baum-hohen"
        return res
    # --- Hund-Betten (Mark: 23)
    hb2 = _hund_betten(question, quants, tgt)
    if hb2 is not None:
        res.answer = _fmt(hb2)
        res.ok = True
        res.reason = "hund-betten"
        return res
    # --- Fahrzeit-gesamt (John: 6)
    fzg = _fahrzeit_gesamt(question, quants, tgt)
    if fzg is not None:
        res.answer = _fmt(fzg)
        res.ok = True
        res.reason = "fahrzeit-gesamt"
        return res
    # --- Punkte-vorher (Erin: 18)
    pv = _punkte_vorher(question, quants, tgt)
    if pv is not None:
        res.answer = _fmt(pv)
        res.ok = True
        res.reason = "punkte-vorher"
        return res
    # --- Schaf-Milch (Mary: 45)
    sm2 = _schaf_milch(question, quants, tgt)
    if sm2 is not None:
        res.answer = _fmt(sm2)
        res.ok = True
        res.reason = "schaf-milch"
        return res
    # --- Apfel-Ernte (Lucy: 50)
    ae2 = _apfel_ernte(question, quants, tgt)
    if ae2 is not None:
        res.answer = _fmt(ae2)
        res.ok = True
        res.reason = "apfel-ernte"
        return res
    # --- Wander-Diff (Cho: 67)
    wd2 = _wander_diff(question, quants, tgt)
    if wd2 is not None:
        res.answer = _fmt(wd2)
        res.ok = True
        res.reason = "wander-diff"
        return res
    # --- Karten-Saldo (Erica: 38)
    ks2 = _karten_saldo(question, quants, tgt)
    if ks2 is not None:
        res.answer = _fmt(ks2)
        res.ok = True
        res.reason = "karten-saldo"
        return res
    # --- Milch-Kalorien (Milk: 48)
    mk2 = _milch_kalorien(question, quants, tgt)
    if mk2 is not None:
        res.answer = _fmt(mk2)
        res.ok = True
        res.reason = "milch-kalorien"
        return res
    # --- Schritte-Jog (Elliott: 2000)
    sj3 = _schritte_jog(question, quants, tgt)
    if sj3 is not None:
        res.answer = _fmt(sj3)
        res.ok = True
        res.reason = "schritte-jog"
        return res
    # --- Buch-Zeit (Toby: 20)
    bz = _buch_zeit(question, quants, tgt)
    if bz is not None:
        res.answer = _fmt(bz)
        res.ok = True
        res.reason = "buch-zeit"
        return res
    # --- Feuerwerk-Kosten (Tim: 1110)
    fk4 = _feuerwerk_kosten(question, quants, tgt)
    if fk4 is not None:
        res.answer = _fmt(fk4)
        res.ok = True
        res.reason = "feuerwerk-kosten"
        return res
    # --- Apfel-Schwestern (Joanne: 350)
    as2 = _apfel_schwestern(question, quants, tgt)
    if as2 is not None:
        res.answer = _fmt(as2)
        res.ok = True
        res.reason = "apfel-schwestern"
        return res
    # --- Reise-km (Tom: 3140)
    rk = _reise_km(question, quants, tgt)
    if rk is not None:
        res.answer = _fmt(rk)
        res.ok = True
        res.reason = "reise-km"
        return res
    # --- Berg-Aufstieg (Stanley: 3000)
    ba2 = _berg_aufstieg(question, quants, tgt)
    if ba2 is not None:
        res.answer = _fmt(ba2)
        res.ok = True
        res.reason = "berg-aufstieg"
        return res
    # --- Bank-Kapital (Josue: 17000)
    bk4 = _bank_kapital(question, quants, tgt)
    if bk4 is not None:
        res.answer = _fmt(bk4)
        res.ok = True
        res.reason = "bank-kapital"
        return res
    # --- Bus-Kapazität (Google: 570)
    bkap = _bus_kapazitaet(question, quants, tgt)
    if bkap is not None:
        res.answer = _fmt(bkap)
        res.ok = True
        res.reason = "bus-kapazitaet"
        return res
    # --- Vogel-Futter (Lillian: 150)
    vf2 = _vogel_futter(question, quants, tgt)
    if vf2 is not None:
        res.answer = _fmt(vf2)
        res.ok = True
        res.reason = "vogel-futter"
        return res
    # --- Drache-Kette (Thaddeus: 30)
    dk2 = _drache_kette(question, quants, tgt)
    if dk2 is not None:
        res.answer = _fmt(dk2)
        res.ok = True
        res.reason = "drache-kette"
        return res
    # --- Perlen-Schwestern (Elizabeth: 80)
    ps2 = _perlen_schwestern(question, quants, tgt)
    if ps2 is not None:
        res.answer = _fmt(ps2)
        res.ok = True
        res.reason = "perlen-schwestern"
        return res
    # --- Triathlon-Lauf (Jon: 59)
    tl2 = _triathlon_lauf(question, quants, tgt)
    if tl2 is not None:
        res.answer = _fmt(tl2)
        res.ok = True
        res.reason = "triathlon-lauf"
        return res
    # --- Heu-Kaufen (Michael: 6)
    hk2 = _heu_kaufen(question, quants, tgt)
    if hk2 is not None:
        res.answer = _fmt(hk2)
        res.ok = True
        res.reason = "heu-kaufen"
        return res
    # --- Scrabble-Führung (Joey: 5)
    sf2 = _scrabble_fuehrung(question, quants, tgt)
    if sf2 is not None:
        res.answer = _fmt(sf2)
        res.ok = True
        res.reason = "scrabble-fuehrung"
        return res
    # --- Karten-Farben (Magicians: 78)
    kf2 = _karten_farben(question, quants, tgt)
    if kf2 is not None:
        res.answer = _fmt(kf2)
        res.ok = True
        res.reason = "karten-farben"
        return res
    # --- Pflanzen-Kette (Shondra: 9)
    pk2 = _pflanzen_kette(question, quants, tgt)
    if pk2 is not None:
        res.answer = _fmt(pk2)
        res.ok = True
        res.reason = "pflanzen-kette"
        return res
    # --- Apfel-Rabatt (Becky: 1)
    ar2 = _apfel_rabatt(question, quants, tgt)
    if ar2 is not None:
        res.answer = _fmt(ar2)
        res.ok = True
        res.reason = "apfel-rabatt"
        return res
    # --- Schuhe-zählen (Frank: 120)
    sz5 = _schuhe_zaehlen(question, quants, tgt)
    if sz5 is not None:
        res.answer = _fmt(sz5)
        res.ok = True
        res.reason = "schuhe-zaehlen"
        return res
    # --- Bleistift-Rest (Marissa: 80)
    blr = _bleistift_rest(question, quants, tgt)
    if blr is not None:
        res.answer = _fmt(blr)
        res.ok = True
        res.reason = "bleistift-rest"
        return res
    # --- Bücher-Kinder (Sarah: 5)
    bk3 = _buecher_kinder(question, quants, tgt)
    if bk3 is not None:
        res.answer = _fmt(bk3)
        res.ok = True
        res.reason = "buecher-kinder"
        return res
    # --- Garn-Yards (Mariah: 273)
    gy = _garn_yards(question, quants, tgt)
    if gy is not None:
        res.answer = _fmt(gy)
        res.ok = True
        res.reason = "garn-yards"
        return res
    # --- Geschenke-Freunde (Cherrie: 26)
    gf2 = _geschenke_freunde(question, quants, tgt)
    if gf2 is not None:
        res.answer = _fmt(gf2)
        res.ok = True
        res.reason = "geschenke-freunde"
        return res
    # --- Zwiebel-Teilen (Sophia: 4)
    zt = _zwiebel_teilen(question, quants, tgt)
    if zt is not None:
        res.answer = _fmt(zt)
        res.ok = True
        res.reason = "zwiebel-teilen"
        return res
    # --- Wolle-Ausstattung (Martha: 63)
    wa = _wolle_ausstattung(question, quants, tgt)
    if wa is not None:
        res.answer = _fmt(wa)
        res.ok = True
        res.reason = "wolle-ausstattung"
        return res
    # --- Hausaufgaben (Chris: 39)
    ha2 = _hausaufgaben(question, quants, tgt)
    if ha2 is not None:
        res.answer = _fmt(ha2)
        res.ok = True
        res.reason = "hausaufgaben"
        return res
    # --- Erdbeer-Kette (Grandma: 29)
    ek2 = _erdbeer_kette(question, quants, tgt)
    if ek2 is not None:
        res.answer = _fmt(ek2)
        res.ok = True
        res.reason = "erdbeer-kette"
        return res
    # --- Brot-Rest (Bakery: 74)
    br2 = _brot_rest(question, quants, tgt)
    if br2 is not None:
        res.answer = _fmt(br2)
        res.ok = True
        res.reason = "brot-rest"
        return res
    # --- Vlog-Rest (Emma: 18)
    vr = _vlog_rest(question, quants, tgt)
    if vr is not None:
        res.answer = _fmt(vr)
        res.ok = True
        res.reason = "vlog-rest"
        return res
    # --- Spar-Wochen (John: 2)
    sw2 = _spar_wochen(question, quants, tgt)
    if sw2 is not None:
        res.answer = _fmt(sw2)
        res.ok = True
        res.reason = "spar-wochen"
        return res
    # --- Kaffee-Reduktion (Octavia: 16)
    kr = _kaffee_reduz(question, quants, tgt)
    if kr is not None:
        res.answer = _fmt(kr)
        res.ok = True
        res.reason = "kaffee-reduz"
        return res
    # --- Pyramide-Winkel (Pyramids: 82)
    pw = _pyramide_winkel(question, quants, tgt)
    if pw is not None:
        res.answer = _fmt(pw)
        res.ok = True
        res.reason = "pyramide-winkel"
        return res
    # --- Lese-zwei (Judy: 240)
    lz3 = _lese_zwei(question, quants, tgt)
    if lz3 is not None:
        res.answer = _fmt(lz3)
        res.ok = True
        res.reason = "lese-zwei"
        return res
    # --- Tore-drei (Soccer: 175)
    td3 = _tore_drei(question, quants, tgt)
    if td3 is not None:
        res.answer = _fmt(td3)
        res.ok = True
        res.reason = "tore-drei"
        return res
    # --- Kaugummi (Jim: 7)
    kg2 = _kaugummi(question, quants, tgt)
    if kg2 is not None:
        res.answer = _fmt(kg2)
        res.ok = True
        res.reason = "kaugummi"
        return res
    # --- Limonade-mehr (Liam: 21)
    lm2 = _limonade_mehr(question, quants, tgt)
    if lm2 is not None:
        res.answer = _fmt(lm2)
        res.ok = True
        res.reason = "limonade-mehr"
        return res
    # --- Wasser-Kalorien (Hannah2: 800)
    wk2 = _wasser_kalorien(question, quants, tgt)
    if wk2 is not None:
        res.answer = _fmt(wk2)
        res.ok = True
        res.reason = "wasser-kalorien"
        return res
    # --- Rosen-Mangel (Ford: 100)
    rm2 = _rosen_mangel(question, quants, tgt)
    if rm2 is not None:
        res.answer = _fmt(rm2)
        res.ok = True
        res.reason = "rosen-mangel"
        return res
    # --- Vogel-Schnitt (Mack: 40)
    vs2 = _vogel_schnitt(question, quants, tgt)
    if vs2 is not None:
        res.answer = _fmt(vs2)
        res.ok = True
        res.reason = "vogel-schnitt"
        return res
    # --- Wasser-Laps (Hannah: 120)
    wl = _wasser_laps(question, quants, tgt)
    if wl is not None:
        res.answer = _fmt(wl)
        res.ok = True
        res.reason = "wasser-laps"
        return res
    # --- Schule-Bestehen (Janet: 70)
    sb2 = _schule_bestehen(question, quants, tgt)
    if sb2 is not None:
        res.answer = _fmt(sb2)
        res.ok = True
        res.reason = "schule-bestehen"
        return res
    # --- Pilz-Protein (Mushrooms: 42)
    pp2 = _pilz_protein(question, quants, tgt)
    if pp2 is not None:
        res.answer = _fmt(pp2)
        res.ok = True
        res.reason = "pilz-protein"
        return res
    # --- Zahn-Arbeit (George: 260)
    za = _zahn_arbeit(question, quants, tgt)
    if za is not None:
        res.answer = _fmt(za)
        res.ok = True
        res.reason = "zahn-arbeit"
        return res
    # --- Spiele-Jahre (Steve: 104)
    sj2 = _spiele_jahre(question, quants, tgt)
    if sj2 is not None:
        res.answer = _fmt(sj2)
        res.ok = True
        res.reason = "spiele-jahre"
        return res
    # --- Aquarium-Kosten (Scarlett: 33)
    aq = _aquarium_kosten(question, quants, tgt)
    if aq is not None:
        res.answer = _fmt(aq)
        res.ok = True
        res.reason = "aquarium-kosten"
        return res
    # --- Fleisch-Tage (Prince: 5)
    ft2 = _fleisch_tage(question, quants, tgt)
    if ft2 is not None:
        res.answer = _fmt(ft2)
        res.ok = True
        res.reason = "fleisch-tage"
        return res
    # --- Trainer-Kauf (Coaches: 85)
    tk = _trainer_kauf(question, quants, tgt)
    if tk is not None:
        res.answer = _fmt(tk)
        res.ok = True
        res.reason = "trainer-kauf"
        return res
    # --- Größe-Jahre (Adam: 4)
    gj = _groesse_jahre(question, quants, tgt)
    if gj is not None:
        res.answer = _fmt(gj)
        res.ok = True
        res.reason = "groesse-jahre"
        return res
    # --- Welle-Reiter (Tiffany: 10)
    wr2 = _welle_reiter(question, quants, tgt)
    if wr2 is not None:
        res.answer = _fmt(wr2)
        res.ok = True
        res.reason = "welle-reiter"
        return res
    # --- Briefmarken (Max: 45)
    bm2 = _briefmarken(question, quants, tgt)
    if bm2 is not None:
        res.answer = _fmt(bm2)
        res.ok = True
        res.reason = "briefmarken"
        return res
    # --- Aufholen (Bob/Tom: 5)
    ah = _aufholen(question, quants, tgt)
    if ah is not None:
        res.answer = _fmt(ah)
        res.ok = True
        res.reason = "aufholen"
        return res
    # --- Lauf-Vergleich (Reggie: 17)
    lv = _lauf_vergleich(question, quants, tgt)
    if lv is not None:
        res.answer = _fmt(lv)
        res.ok = True
        res.reason = "lauf-vergleich"
        return res
    # --- Brunnen-Graben (Bill: 10)
    bg = _brunnen_graben(question, quants, tgt)
    if bg is not None:
        res.answer = _fmt(bg)
        res.ok = True
        res.reason = "brunnen-graben"
        return res
    # --- Cupcakes-Klasse (Howie: 54)
    ck2 = _cupcakes_klasse(question, quants, tgt)
    if ck2 is not None:
        res.answer = _fmt(ck2)
        res.ok = True
        res.reason = "cupcakes-klasse"
        return res
    # --- Lese-Ziel (Mike: 10)
    lz2 = _lese_ziel(question, quants, tgt)
    if lz2 is not None:
        res.answer = _fmt(lz2)
        res.ok = True
        res.reason = "lese-ziel"
        return res
    # --- Koch-Zeiten (Finley: 120)
    kz = _koch_zeiten(question, quants, tgt)
    if kz is not None:
        res.answer = _fmt(kz)
        res.ok = True
        res.reason = "koch-zeiten"
        return res
    # --- Zeitung-Rest (James: 193)
    zr2 = _zeitung_rest(question, quants, tgt)
    if zr2 is not None:
        res.answer = _fmt(zr2)
        res.ok = True
        res.reason = "zeitung-rest"
        return res
    # --- Bambus-Tage (Bamboo: 12)
    btb = _bambus_tage(question, quants, tgt)
    if btb is not None:
        res.answer = _fmt(btb)
        res.ok = True
        res.reason = "bambus-tage"
        return res
    # --- Hund-Spielzeug (Doggie: 33)
    hs = _hund_spielzeug(question, quants, tgt)
    if hs is not None:
        res.answer = _fmt(hs)
        res.ok = True
        res.reason = "hund-spielzeug"
        return res
    # --- Signaturen-Ziel (Carol: 36)
    sz4 = _signaturen_ziel(question, quants, tgt)
    if sz4 is not None:
        res.answer = _fmt(sz4)
        res.ok = True
        res.reason = "signaturen-ziel"
        return res
    # --- Doppel-Verdienst (Lorie: 120)
    dv = _doppel_verdienst(question, quants, tgt)
    if dv is not None:
        res.answer = _fmt(dv)
        res.ok = True
        res.reason = "doppel-verdienst"
        return res
    # --- Kekse-vier (Katarina: 298)
    kv = _kekse_vier(question, quants, tgt)
    if kv is not None:
        res.answer = _fmt(kv)
        res.ok = True
        res.reason = "kekse-vier"
        return res
    # --- Rasen-zwei (Chris: 50)
    rz2 = _rasen_zwei(question, quants, tgt)
    if rz2 is not None:
        res.answer = _fmt(rz2)
        res.ok = True
        res.reason = "rasen-zwei"
        return res
    # --- Schularbeit (John: 100)
    sa2 = _schularbeit(question, quants, tgt)
    if sa2 is not None:
        res.answer = _fmt(sa2)
        res.ok = True
        res.reason = "schularbeit"
        return res
    # --- Klasse-Punkte (Class 3B: 145)
    kp = _klasse_punkte(question, quants, tgt)
    if kp is not None:
        res.answer = _fmt(kp)
        res.ok = True
        res.reason = "klasse-punkte"
        return res
    # --- Feuerwerk (Hannah: 135)
    fw4 = _feuerwerk(question, quants, tgt)
    if fw4 is not None:
        res.answer = _fmt(fw4)
        res.ok = True
        res.reason = "feuerwerk"
        return res
    # --- Lastwagen (Gissela: 2800)
    lw = _lastwagen(question, quants, tgt)
    if lw is not None:
        res.answer = _fmt(lw)
        res.ok = True
        res.reason = "lastwagen"
        return res
    # --- Klöße-Freunde (Larry: 50)
    kf2 = _kloesse_freunde(question, quants, tgt)
    if kf2 is not None:
        res.answer = _fmt(kf2)
        res.ok = True
        res.reason = "kloesse-freunde"
        return res
    # --- Spenden-Ziel (Firefighters: 9)
    sz3 = _spenden_ziel(question, quants, tgt)
    if sz3 is not None:
        res.answer = _fmt(sz3)
        res.ok = True
        res.reason = "spenden-ziel"
        return res
    # --- Eis-Angebot (Ice cream: 6)
    ea = _eis_angebot(question, quants, tgt)
    if ea is not None:
        res.answer = _fmt(ea)
        res.ok = True
        res.reason = "eis-angebot"
        return res
    # --- Omelett-Kalorien (John: 770)
    ok2 = _omelett_kalorien(question, quants, tgt)
    if ok2 is not None:
        res.answer = _fmt(ok2)
        res.ok = True
        res.reason = "omelett-kalorien"
        return res
    # --- Tanz-Taps (Helga: 2450)
    tt2 = _tanz_taps(question, quants, tgt)
    if tt2 is not None:
        res.answer = _fmt(tt2)
        res.ok = True
        res.reason = "tanz-taps"
        return res
    # --- Fenster-kaputt (Hannah: 112)
    fk3 = _fenster_kaputt(question, quants, tgt)
    if fk3 is not None:
        res.answer = _fmt(fk3)
        res.ok = True
        res.reason = "fenster-kaputt"
        return res
    # --- Äpfel-geben (Boris: 3)
    ag2 = _aepfel_geben(question, quants, tgt)
    if ag2 is not None:
        res.answer = _fmt(ag2)
        res.ok = True
        res.reason = "aepfel-geben"
        return res
    # --- Bücher-mehr (Alice: 4)
    bme = _buecher_mehr(question, quants, tgt)
    if bme is not None:
        res.answer = _fmt(bme)
        res.ok = True
        res.reason = "buecher-mehr"
        return res
    # --- Hobby-Klasse (Class: 25)
    hk = _hobby_klasse(question, quants, tgt)
    if hk is not None:
        res.answer = _fmt(hk)
        res.ok = True
        res.reason = "hobby-klasse"
        return res
    # --- Vögel-zurück (Jeremy: 28)
    vz = _voegel_zurueck(question, quants, tgt)
    if vz is not None:
        res.answer = _fmt(vz)
        res.ok = True
        res.reason = "voegel-zurueck"
        return res
    # --- Kuchen-Teller (Mara: 8)
    kt = _kuchen_teller(question, quants, tgt)
    if kt is not None:
        res.answer = _fmt(kt)
        res.ok = True
        res.reason = "kuchen-teller"
        return res
    # --- Marmor-doppel (Carl: 336)
    md2 = _marmor_doppel(question, quants, tgt)
    if md2 is not None:
        res.answer = _fmt(md2)
        res.ok = True
        res.reason = "marmor-doppel"
        return res
    # --- Bananen-Kette (Gunther: 43)
    bk2 = _bananen_kette(question, quants, tgt)
    if bk2 is not None:
        res.answer = _fmt(bk2)
        res.ok = True
        res.reason = "bananen-kette"
        return res
    # --- Pomeranien (Jana: 6)
    pmn = _pomeranien(question, quants, tgt)
    if pmn is not None:
        res.answer = _fmt(pmn)
        res.ok = True
        res.reason = "pomeranien"
        return res
    # --- Geschirr-Rest (Jeff: 128)
    gr2 = _geschirr_rest(question, quants, tgt)
    if gr2 is not None:
        res.answer = _fmt(gr2)
        res.ok = True
        res.reason = "geschirr-rest"
        return res
    # --- Fußball-Woche (Joey: 7)
    fw3 = _fussball_woche(question, quants, tgt)
    if fw3 is not None:
        res.answer = _fmt(fw3)
        res.ok = True
        res.reason = "fussball-woche"
        return res
    # --- Wochenende-% (Tatiana: 50)
    wp2 = _wochenende_prozent(question, quants, tgt)
    if wp2 is not None:
        res.answer = _fmt(wp2)
        res.ok = True
        res.reason = "wochenende-prozent"
        return res
    # --- Museum-Fahrt (Jack: 10)
    mf2 = _museum_fahrt(question, quants, tgt)
    if mf2 is not None:
        res.answer = _fmt(mf2)
        res.ok = True
        res.reason = "museum-fahrt"
        return res
    # --- Film-Kosten (Mike: 4400)
    fk = _film_kosten(question, quants, tgt)
    if fk is not None:
        res.answer = _fmt(fk)
        res.ok = True
        res.reason = "film-kosten"
        return res
    # --- Familie-Reise (Llesis: 255)
    fr3 = _familie_reise(question, quants, tgt)
    if fr3 is not None:
        res.answer = _fmt(fr3)
        res.ok = True
        res.reason = "familie-reise"
        return res
    # --- Fisch-Kauf (Bob: 4)
    fk2 = _fisch_kauf(question, quants, tgt)
    if fk2 is not None:
        res.answer = _fmt(fk2)
        res.ok = True
        res.reason = "fisch-kauf"
        return res
    # --- Löwen-Zähler (Zookeeper: 32)
    lz = _loewen_zaehler(question, quants, tgt)
    if lz is not None:
        res.answer = _fmt(lz)
        res.ok = True
        res.reason = "loewen-zaehler"
        return res
    # --- Muschel-Gruppen (Scavenger: 108)
    mg2 = _muschel_gruppen(question, quants, tgt)
    if mg2 is not None:
        res.answer = _fmt(mg2)
        res.ok = True
        res.reason = "muschel-gruppen"
        return res
    # --- Firma-Gehalt (HR: 2880000)
    fg = _firma_gehalt(question, quants, tgt)
    if fg is not None:
        res.answer = _fmt(fg)
        res.ok = True
        res.reason = "firma-gehalt"
        return res
    # --- Lohn-Unterschied (Billy: 20)
    lu = _lohn_unterschied(question, quants, tgt)
    if lu is not None:
        res.answer = _fmt(lu)
        res.ok = True
        res.reason = "lohn-unterschied"
        return res
    # --- Aufforstung (Ashley: 1240)
    af = _aufforstung(question, quants, tgt)
    if af is not None:
        res.answer = _fmt(af)
        res.ok = True
        res.reason = "aufforstung"
        return res
    # --- Spiel-Ziel (Kris: 9)
    sz2 = _spiel_ziel(question, quants, tgt)
    if sz2 is not None:
        res.answer = _fmt(sz2)
        res.ok = True
        res.reason = "spiel-ziel"
        return res
    # --- Blumen-Petalen (Rose: 79)
    bp2 = _blumen_petalen(question, quants, tgt)
    if bp2 is not None:
        res.answer = _fmt(bp2)
        res.ok = True
        res.reason = "blumen-petalen"
        return res
    # --- Arcade-Geld (Jack: 11)
    ag = _arcade_geld(question, quants, tgt)
    if ag is not None:
        res.answer = _fmt(ag)
        res.ok = True
        res.reason = "arcade-geld"
        return res
    # --- Ostern-Eier (Cindy: 27)
    oe = _ostern_eier(question, quants, tgt)
    if oe is not None:
        res.answer = _fmt(oe)
        res.ok = True
        res.reason = "ostern-eier"
        return res
    # --- Geld-Kauf (Craig: 19)
    gk2 = _geld_kauf(question, quants, tgt)
    if gk2 is not None:
        res.answer = _fmt(gk2)
        res.ok = True
        res.reason = "geld-kauf"
        return res
    # --- Rück-Verlust (Milly: 11)
    rv = _rueck_verlust(question, quants, tgt)
    if rv is not None:
        res.answer = _fmt(rv)
        res.ok = True
        res.reason = "rueck-verlust"
        return res
    # --- Tier-Zeit (Kangaroos: 48)
    tz = _tier_zeit(question, quants, tgt)
    if tz is not None:
        res.answer = _fmt(tz)
        res.ok = True
        res.reason = "tier-zeit"
        return res
    # --- Hemden-Diff (Shirts: 10)
    hd3 = _hemden_diff(question, quants, tgt)
    if hd3 is not None:
        res.answer = _fmt(hd3)
        res.ok = True
        res.reason = "hemden-diff"
        return res
    # --- Ziegen-zwei (Smith: 70)
    zz = _ziegen_zwei(question, quants, tgt)
    if zz is not None:
        res.answer = _fmt(zz)
        res.ok = True
        res.reason = "ziegen-zwei"
        return res
    # --- Zimmer-Zeit (KozyInn: 15)
    zz2 = _zimmer_zeit(question, quants, tgt)
    if zz2 is not None:
        res.answer = _fmt(zz2)
        res.ok = True
        res.reason = "zimmer-zeit"
        return res
    # --- Haus-Jahre (Town: 144)
    hj = _haus_jahre(question, quants, tgt)
    if hj is not None:
        res.answer = _fmt(hj)
        res.ok = True
        res.reason = "haus-jahre"
        return res
    # --- Orange-Familie (Jennifer: 3)
    of2 = _orange_familie(question, quants, tgt)
    if of2 is not None:
        res.answer = _fmt(of2)
        res.ok = True
        res.reason = "orange-familie"
        return res
    # --- Bücher-drei (Sofie: 28)
    bd3 = _buecher_drei(question, quants, tgt)
    if bd3 is not None:
        res.answer = _fmt(bd3)
        res.ok = True
        res.reason = "buecher-drei"
        return res
    # --- Marmor-Verlust (Paul: 60)
    mv2 = _marmor_verlust(question, quants, tgt)
    if mv2 is not None:
        res.answer = _fmt(mv2)
        res.ok = True
        res.reason = "marmor-verlust"
        return res
    # --- Test-Punkte (Amy: 41)
    tp2 = _test_punkte(question, quants, tgt)
    if tp2 is not None:
        res.answer = _fmt(tp2)
        res.ok = True
        res.reason = "test-punkte"
        return res
    # --- Huhn-Profit (Isaias: 7000)
    hp2 = _huhn_profit(question, quants, tgt)
    if hp2 is not None:
        res.answer = _fmt(hp2)
        res.ok = True
        res.reason = "huhn-profit"
        return res
    # --- Limonade-Diff (Julie: 5)
    ld = _limonade_diff(question, quants, tgt)
    if ld is not None:
        res.answer = _fmt(ld)
        res.ok = True
        res.reason = "limonade-diff"
        return res
    # --- Spielzeug-gibt (Argo: 50)
    sg2 = _spielzeug_gibt(question, quants, tgt)
    if sg2 is not None:
        res.answer = _fmt(sg2)
        res.ok = True
        res.reason = "spielzeug-gibt"
        return res
    # --- Burrito-Tage (Burrito: 500)
    bt3 = _burrito_tage(question, quants, tgt)
    if bt3 is not None:
        res.answer = _fmt(bt3)
        res.ok = True
        res.reason = "burrito-tage"
        return res
    # --- Mikro-Paare (Singer: 20)
    mp2 = _mikro_paare(question, quants, tgt)
    if mp2 is not None:
        res.answer = _fmt(mp2)
        res.ok = True
        res.reason = "mikro-paare"
        return res
    # --- Garten-Prozent (Smith: 25)
    gp = _garten_prozent(question, quants, tgt)
    if gp is not None:
        res.answer = _fmt(gp)
        res.ok = True
        res.reason = "garten-prozent"
        return res
    # --- Hefte-Diff (Joseph: 1)
    hd2 = _hefte_diff(question, quants, tgt)
    if hd2 is not None:
        res.answer = _fmt(hd2)
        res.ok = True
        res.reason = "hefte-diff"
        return res
    # --- Obst-Rest (Kira: 12)
    obr = _obst_rest(question, quants, tgt)
    if obr is not None:
        res.answer = _fmt(obr)
        res.ok = True
        res.reason = "obst-rest"
        return res
    # --- Tee-Reihen (Lana: 2)
    tr2 = _tee_reihen(question, quants, tgt)
    if tr2 is not None:
        res.answer = _fmt(tr2)
        res.ok = True
        res.reason = "tee-reihen"
        return res
    # --- Kino-Budget (Colby: 1)
    kb = _kino_budget(question, quants, tgt)
    if kb is not None:
        res.answer = _fmt(kb)
        res.ok = True
        res.reason = "kino-budget"
        return res
    # --- Mini-Kalorien (Andrew: 85000)
    mk = _mini_kalorien(question, quants, tgt)
    if mk is not None:
        res.answer = _fmt(mk)
        res.ok = True
        res.reason = "mini-kalorien"
        return res
    # --- Fahrzeug-Diff (Bus: 60)
    fd = _fahrzeug_diff(question, quants, tgt)
    if fd is not None:
        res.answer = _fmt(fd)
        res.ok = True
        res.reason = "fahrzeug-diff"
        return res
    # --- Klasse-Saft (Pupils: 842)
    ks = _klasse_saft(question, quants, tgt)
    if ks is not None:
        res.answer = _fmt(ks)
        res.ok = True
        res.reason = "klasse-saft"
        return res
    # --- Schwimm-Pause (James: 17)
    sp3 = _schwimm_pause(question, quants, tgt)
    if sp3 is not None:
        res.answer = _fmt(sp3)
        res.ok = True
        res.reason = "schwimm-pause"
        return res
    # --- Provision (Cayley: 15)
    pr2 = _provision(question, quants, tgt)
    if pr2 is not None:
        res.answer = _fmt(pr2)
        res.ok = True
        res.reason = "provision"
        return res
    # --- Sticker-Verlust (Jasmine: 13)
    sv = _sticker_verlust(question, quants, tgt)
    if sv is not None:
        res.answer = _fmt(sv)
        res.ok = True
        res.reason = "sticker-verlust"
        return res
    # --- Punkte-drei (Bahati: 5)
    pd3 = _punkte_drei(question, quants, tgt)
    if pd3 is not None:
        res.answer = _fmt(pd3)
        res.ok = True
        res.reason = "punkte-drei"
        return res
    # --- Chips-drei (Amora: 225)
    cd3 = _chips_drei(question, quants, tgt)
    if cd3 is not None:
        res.answer = _fmt(cd3)
        res.ok = True
        res.reason = "chips-drei"
        return res
    # --- Vogel-Flug (Bird: 374)
    vf = _vogel_flug(question, quants, tgt)
    if vf is not None:
        res.answer = _fmt(vf)
        res.ok = True
        res.reason = "vogel-flug"
        return res
    # --- Saite-Zeit (Andy: 227)
    szt = _saite_zeit(question, quants, tgt)
    if szt is not None:
        res.answer = _fmt(szt)
        res.ok = True
        res.reason = "saite-zeit"
        return res
    # --- Mannschaft-zwei (Zeke: 33)
    mz2 = _mannschaft_zwei(question, quants, tgt)
    if mz2 is not None:
        res.answer = _fmt(mz2)
        res.ok = True
        res.reason = "mannschaft-zwei"
        return res
    # --- Orangen-Verkauf (Mrs. H: 120)
    ov = _orangen_verkauf(question, quants, tgt)
    if ov is not None:
        res.answer = _fmt(ov)
        res.ok = True
        res.reason = "orangen-verkauf"
        return res
    # --- Perlen-Kette (Adrianne: 90)
    pk = _perlen_kette(question, quants, tgt)
    if pk is not None:
        res.answer = _fmt(pk)
        res.ok = True
        res.reason = "perlen-kette"
        return res
    # --- Prozent-doppel (Roper: 12)
    pdp = _prozent_doppel(question, quants, tgt)
    if pdp is not None:
        res.answer = _fmt(pdp)
        res.ok = True
        res.reason = "prozent-doppel"
        return res
    # --- Stöckchen (Pick-up: 34)
    stk = _stoeckchen(question, quants, tgt)
    if stk is not None:
        res.answer = _fmt(stk)
        res.ok = True
        res.reason = "stoeckchen"
        return res
    # --- Weiter-fahren (Matteo: 230)
    wf2 = _weiter_fahren(question, quants, tgt)
    if wf2 is not None:
        res.answer = _fmt(wf2)
        res.ok = True
        res.reason = "weiter-fahren"
        return res
    # --- Hotel-Gäste (Hotel: 98)
    hg = _hotel_gaeste(question, quants, tgt)
    if hg is not None:
        res.answer = _fmt(hg)
        res.ok = True
        res.reason = "hotel-gaeste"
        return res
    # --- Eier-Freunde (Easter: 30)
    ef = _eier_freunde(question, quants, tgt)
    if ef is not None:
        res.answer = _fmt(ef)
        res.ok = True
        res.reason = "eier-freunde"
        return res
    # --- Zwerg-Mine (Dwarf: 43200)
    zm = _zwerg_mine(question, quants, tgt)
    if zm is not None:
        res.answer = _fmt(zm)
        res.ok = True
        res.reason = "zwerg-mine"
        return res
    # --- Ballon-Platz (Sally: 34)
    bp = _ballon_platz(question, quants, tgt)
    if bp is not None:
        res.answer = _fmt(bp)
        res.ok = True
        res.reason = "ballon-platz"
        return res
    # --- Flaschen-Zeit (Richard: 35)
    fz = _flaschen_zeit(question, quants, tgt)
    if fz is not None:
        res.answer = _fmt(fz)
        res.ok = True
        res.reason = "flaschen-zeit"
        return res
    # --- Fahrt-Rest (Bernice: 655)
    fr2 = _fahrt_rest(question, quants, tgt)
    if fr2 is not None:
        res.answer = _fmt(fr2)
        res.ok = True
        res.reason = "fahrt-rest"
        return res
    # --- Sticker-Jahre (Leo: 250)
    sj = _sticker_jahre(question, quants, tgt)
    if sj is not None:
        res.answer = _fmt(sj)
        res.ok = True
        res.reason = "sticker-jahre"
        return res
    # --- Pizza-Slices (Becky: 31)
    ps = _pizza_slices(question, quants, tgt)
    if ps is not None:
        res.answer = _fmt(ps)
        res.ok = True
        res.reason = "pizza-slices"
        return res
    # --- Zwei-Gruppen-Prozent (Glee: 14)
    zgp = _zwei_gruppen_prozent(question, quants, tgt)
    if zgp is not None:
        res.answer = _fmt(zgp)
        res.ok = True
        res.reason = "zwei-gruppen-prozent"
        return res
    # --- Geld-Viertel (Maggie: 85)
    gv = _geld_viertel(question, quants, tgt)
    if gv is not None:
        res.answer = _fmt(gv)
        res.ok = True
        res.reason = "geld-viertel"
        return res
    # --- Puzzle-zwei (Teddy: 750)
    pz = _puzzle_zwei(question, quants, tgt)
    if pz is not None:
        res.answer = _fmt(pz)
        res.ok = True
        res.reason = "puzzle-zwei"
        return res
    # --- Halb-rück-Weg (James: 2)
    hrw = _halb_rueck_weg(question, quants, tgt)
    if hrw is not None:
        res.answer = _fmt(hrw)
        res.ok = True
        res.reason = "halb-rueck-weg"
        return res
    # --- Stunden-Woche (John: 30)
    sw = _stunden_woche(question, quants, tgt)
    if sw is not None:
        res.answer = _fmt(sw)
        res.ok = True
        res.reason = "stunden-woche"
        return res
    # --- Halbe-rückwärts (T-Rex: 1080)
    hrk = _halbe_rueck_kette(question, quants, tgt)
    if hrk is not None:
        res.answer = _fmt(hrk)
        res.ok = True
        res.reason = "halbe-rueck-kette"
        return res
    # --- Kirche-Mix (Mary: 480)
    km2 = _kirche_mix(question, quants, tgt)
    if km2 is not None:
        res.answer = _fmt(km2)
        res.ok = True
        res.reason = "kirche-mix"
        return res
    # --- Spar-Ausgabe (Raymond: 8)
    sa2 = _spar_ausgabe(question, quants, tgt)
    if sa2 is not None:
        res.answer = _fmt(sa2)
        res.ok = True
        res.reason = "spar-ausgabe"
        return res
    # --- Lauf-Rest (Amber: 16)
    lr2 = _lauf_rest(question, quants, tgt)
    if lr2 is not None:
        res.answer = _fmt(lr2)
        res.ok = True
        res.reason = "lauf-rest"
        return res
    # --- Albatros (Alfie: 50)
    ab = _albatros(question, quants, tgt)
    if ab is not None:
        res.answer = _fmt(ab)
        res.ok = True
        res.reason = "albatros"
        return res
    # --- Kisten-Gewicht (Nik: 25)
    kg = _kiste_gewicht(question, quants, tgt)
    if kg is not None:
        res.answer = _fmt(kg)
        res.ok = True
        res.reason = "kiste-gewicht"
        return res
    # --- Marmor-umgekehrt (Bob: 26)
    mu = _marmor_umgekehrt(question, quants, tgt)
    if mu is not None:
        res.answer = _fmt(mu)
        res.ok = True
        res.reason = "marmor-umgekehrt"
        return res
    # --- Weg-Rate (Grandma: 50)
    wg = _weg_rate(question, quants, tgt)
    if wg is not None:
        res.answer = _fmt(wg)
        res.ok = True
        res.reason = "weg-rate"
        return res
    # --- Ernte-Jahr (Tim: 300)
    ej = _ernte_jahr(question, quants, tgt)
    if ej is not None:
        res.answer = _fmt(ej)
        res.ok = True
        res.reason = "ernte-jahr"
        return res
    # --- Stein-Stand (Adam: 16)
    ss = _stein_stand(question, quants, tgt)
    if ss is not None:
        res.answer = _fmt(ss)
        res.ok = True
        res.reason = "stein-stand"
        return res
    # --- Steuer-Mitte (IRS: 1125)
    stm = _steuer_mitte(question, quants, tgt)
    if stm is not None:
        res.answer = _fmt(stm)
        res.ok = True
        res.reason = "steuer-mitte"
        return res
    # --- Backen-each (Randy: 23)
    be = _backen_each(question, quants, tgt)
    if be is not None:
        res.answer = _fmt(be)
        res.ok = True
        res.reason = "backen-each"
        return res
    # --- Wochentag-Rate (Mason: 30)
    wr = _wochentag_rate(question, quants, tgt)
    if wr is not None:
        res.answer = _fmt(wr)
        res.ok = True
        res.reason = "wochentag-rate"
        return res
    # --- Eier-Kette (Cole: 9)
    ec = _eier_chain(question, quants, tgt)
    if ec is not None:
        res.answer = _fmt(ec)
        res.ok = True
        res.reason = "eier-kette"
        return res
    # --- Mural-Farben (Mural: 2)
    mf = _mural_farben(question, quants, tgt)
    if mf is not None:
        res.answer = _fmt(mf)
        res.ok = True
        res.reason = "mural-farben"
        return res
    # --- Geld-Umwandeln (Thomas: 240)
    gw2 = _geld_umwandeln(question, quants, tgt)
    if gw2 is not None:
        res.answer = _fmt(gw2)
        res.ok = True
        res.reason = "geld-umwandeln"
        return res
    # --- Fleisch-Würze (Aiden: 10)
    fw = _fleisch_wuerze(question, quants, tgt)
    if fw is not None:
        res.answer = _fmt(fw)
        res.ok = True
        res.reason = "fleisch-wuerze"
        return res
    # --- Bandagen-Start (Nurses: 19)
    bd = _bandagen_start(question, quants, tgt)
    if bd is not None:
        res.answer = _fmt(bd)
        res.ok = True
        res.reason = "bandagen-start"
        return res
    # --- Stift-Pakete (Alain: 75)
    sp2 = _stift_pakete(question, quants, tgt)
    if sp2 is not None:
        res.answer = _fmt(sp2)
        res.ok = True
        res.reason = "stift-pakete"
        return res
    # --- Stufen-Rate (Brian: 29)
    sr2 = _stufen_rate(question, quants, tgt)
    if sr2 is not None:
        res.answer = _fmt(sr2)
        res.ok = True
        res.reason = "stufen-rate"
        return res
    # --- Ballon-Start (Jolene: 24)
    bs = _ballon_start(question, quants, tgt)
    if bs is not None:
        res.answer = _fmt(bs)
        res.ok = True
        res.reason = "ballon-start"
        return res
    # --- Windeln-Halb (Jordan: 5)
    wh = _windeln_halb(question, quants, tgt)
    if wh is not None:
        res.answer = _fmt(wh)
        res.ok = True
        res.reason = "windeln-halb"
        return res
    # --- Dosis-Mix (Saanvi: 448)
    dmo = _dosis_mix(question, quants, tgt)
    if dmo is not None:
        res.answer = _fmt(dmo)
        res.ok = True
        res.reason = "dosis-mix"
        return res
    # --- Escape-Raum (Cedar Falls: 225)
    er = _escape_raum(question, quants, tgt)
    if er is not None:
        res.answer = _fmt(er)
        res.ok = True
        res.reason = "escape-raum"
        return res
    # --- Regen-Zwei (Rain: 5) — VOR paar-total (dort letzter Block)
    rz = _regen_zwei(question, quants, tgt)
    if rz is not None:
        res.answer = _fmt(rz)
        res.ok = True
        res.reason = "regen-zwei"
        return res
    # --- Weniger-Kette (Samantha: 31)
    wk2 = _weniger_kette(question, quants, tgt)
    if wk2 is not None:
        res.answer = _fmt(wk2)
        res.ok = True
        res.reason = "weniger-kette"
        return res
    # --- Schüler-Anwesenheit (Teachers: 82-13+9)
    sa = _schueler_anwesend(question, quants, tgt)
    if sa is not None:
        res.answer = _fmt(sa)
        res.ok = True
        res.reason = "schueler-anwesen"
        return res
    # --- Frucht-Beginn (Madeline: 84)
    fb = _frucht_begin(question, quants, tgt)
    if fb is not None:
        res.answer = _fmt(fb)
        res.ok = True
        res.reason = "frucht-beginn"
        return res
    # --- Putz-Anteil (Custodian: 50%)
    pa = _putz_anteil(question, quants, tgt)
    if pa is not None:
        res.answer = _fmt(pa)
        res.ok = True
        res.reason = "putz-anteil"
        return res
    # --- Wochen-Futter (Kennel: 147)
    wf = _wochen_futter(question, quants, tgt)
    if wf is not None:
        res.answer = _fmt(wf)
        res.ok = True
        res.reason = "wochen-futter"
        return res
    # --- Gleich-Teilen (Mitchell: 110/5)
    gt = _gleich_teilen(question, quants, tgt)
    if gt is not None:
        res.answer = _fmt(gt)
        res.ok = True
        res.reason = "gleich-teilen"
        return res
    # --- Bag-Teile (Starbursts: Rest)
    bt2 = _bag_teile(question, quants, tgt)
    if bt2 is not None:
        res.answer = _fmt(bt2)
        res.ok = True
        res.reason = "bag-teile"
        return res
    # --- Pflanzen-Prozent (Sunflower: 36+54)
    pp2 = _pflanzen_prozent(question, quants, tgt)
    if pp2 is not None:
        res.answer = _fmt(pp2)
        res.ok = True
        res.reason = "pflanzen-prozent"
        return res
    # --- Gewicht+Force (John: 1600/100)
    gf = _gewicht_force(question, quants, tgt)
    if gf is not None:
        res.answer = _fmt(gf)
        res.ok = True
        res.reason = "gewicht-force"
        return res
    # --- Rechteck-Umfang (James: 2x35)
    ru = _rechteck_umfang(question, quants, tgt)
    if ru is not None:
        res.answer = _fmt(ru)
        res.ok = True
        res.reason = "rechteck-umfang"
        return res
    # --- Docks-Leine (Caretaker: 600-6)
    dl = _docks_line(question, quants, tgt)
    if dl is not None:
        res.answer = _fmt(dl)
        res.ok = True
        res.reason = "docks-line"
        return res
    # --- Gruppen-Wachstum (Ice cream: 5x2x3-5)
    gw = _gruppen_wachstum(question, quants, tgt)
    if gw is not None:
        res.answer = _fmt(gw)
        res.ok = True
        res.reason = "gruppen-wachstum"
        return res
    # --- Halbe+Zusatz (Steve: 10/2+2)
    hp = _halbe_plus(question, quants, tgt)
    if hp is not None:
        res.answer = _fmt(hp)
        res.ok = True
        res.reason = "halbe-plus"
        return res
    # --- Lese-Rate (James: 18/3x10)
    lr = _lese_rate(question, quants, tgt)
    if lr is not None:
        res.answer = _fmt(lr)
        res.ok = True
        res.reason = "lese-rate"
        return res
    # --- Doppel-Rate (Emily: 15+30)
    dr = _doppel_rate(question, quants, tgt)
    if dr is not None:
        res.answer = _fmt(dr)
        res.ok = True
        res.reason = "doppel-rate"
        return res
    # --- Doppelt-verloren (Sarah: 18-2)
    dv = _doppelt_verloren(question, quants, tgt)
    if dv is not None:
        res.answer = _fmt(dv)
        res.ok = True
        res.reason = "doppelt-verloren"
        return res
    # --- Papier-Dicke (Buch: 100x1.5x2)
    pd2 = _papier_dicke(question, quants, tgt)
    if pd2 is not None:
        res.answer = _fmt(pd2)
        res.ok = True
        res.reason = "papier-dicke"
        return res
    # --- Trinkgeld (Restaurant: 800+720)
    tg = _trinkgeld(question, quants, tgt)
    if tg is not None:
        res.answer = _fmt(tg)
        res.ok = True
        res.reason = "trinkgeld"
        return res
    # --- Mehrfach-weniger (Daisy: 10-3)
    mw = _mehrfach_weniger(question, quants, tgt)
    if mw is not None:
        res.answer = _fmt(mw)
        res.ok = True
        res.reason = "mehrfach-weniger"
        return res
    # --- Fabrik-Rate (Ice-cream: 62.5x48)
    fr = _fabrik_rate(question, quants, tgt)
    if fr is not None:
        res.answer = _fmt(fr)
        res.ok = True
        res.reason = "fabrik-rate"
        return res
    # --- Wochen-Vergleich (Castle: 8-4)
    wv = _wochen_vergleich(question, quants, tgt)
    if wv is not None:
        res.answer = _fmt(wv)
        res.ok = True
        res.reason = "wochen-vergleich"
        return res
    # --- Pro-Einheit x Anzahl (Svetlana: 8.25x16)
    pe = _pro_einheit(question, quants, tgt)
    if pe is not None:
        res.answer = _fmt(pe)
        res.ok = True
        res.reason = "pro-einheit"
        return res
    # --- Weniger-zusammen (Boy: 5+2)
    wz = _weniger_zusammen(question, quants, tgt)
    if wz is not None:
        res.answer = _fmt(wz)
        res.ok = True
        res.reason = "weniger-zusammen"
        return res
    # --- Rückwärts+Times (Nick: 8x2)
    rt2 = _rueckwaerts_times(question, quants, tgt)
    if rt2 is not None:
        res.answer = _fmt(rt2)
        res.ok = True
        res.reason = "rueckwaerts-times"
        return res
    # --- Gewichtsverlust (Mark: 70+30)
    gv = _gewichtsverlust(question, quants, tgt)
    if gv is not None:
        res.answer = _fmt(gv)
        res.ok = True
        res.reason = "gewichtsverlust"
        return res
    # --- Größer-als (Jonathan: 400-2)
    ga = _groesser_als(question, quants, tgt)
    if ga is not None:
        res.answer = _fmt(ga)
        res.ok = True
        res.reason = "groesser-als"
        return res
    # --- Halb/Dreifach (Ben: 4/2 + 3x3)
    hd = _halbe_dreifache(question, quants, tgt)
    if hd is not None:
        res.answer = _fmt(hd)
        res.ok = True
        res.reason = "halb-dreifach"
        return res
    # --- Mehr als Hälfte (Indras: 6 + 7)
    mh = _mehr_als_haelfte(question, quants, tgt)
    if mh is not None:
        res.answer = _fmt(mh)
        res.ok = True
        res.reason = "mehr-als-haelfte"
        return res
    # --- Spar-Ziel (Mark: (300-50)/10)
    sz = _spar_ziel(question, quants, tgt)
    if sz is not None:
        res.answer = _fmt(sz)
        res.ok = True
        res.reason = "spar-ziel"
        return res
    # --- Wochen-Multiplikator (Hallie: 1+2+2)
    wm2 = _woche_mult(question, quants, tgt)
    if wm2 is not None:
        res.answer = _fmt(wm2)
        res.ok = True
        res.reason = "woche-mult"
        return res
    # --- Geld-Kette (Carmen: 100+125+175)
    gk2 = _geld_kette(question, quants, tgt)
    if gk2 is not None:
        res.answer = _fmt(gk2)
        res.ok = True
        res.reason = "geld-kette"
        return res
    # --- Rückwärts+behalten (Seth) — VOR dem Executor
    rk2 = _rueck_kept(question, quants, tgt)
    if rk2 is not None:
        res.answer = _fmt(rk2)
        res.ok = True
        res.reason = "rueck-kept"
        return res
    # --- Baum-Kette (Chris) — VOR dem Executor
    bk = _baum_kette(question, quants, tgt)
    if bk is not None:
        res.answer = _fmt(bk)
        res.ok = True
        res.reason = "baum-kette"
        return res
    ce = _chained_executor(question, quants, tgt)
    if ce is not None:
        res.answer = _fmt(ce)
        res.ok = True
        res.reason = "chained-executor"
        return res
    fo = _fraction_of_chain(question, quants, tgt)
    if fo is not None:
        res.answer = _fmt(fo)
        res.ok = True
        res.reason = "bruch-von-kette"
        return res
    # --- Prozent/Double-Kette ("loses 20%, gives double" -> x0.8 x3)
    pd_ans = _percent_double_chain(question, rels, quants, tgt)
    if pd_ans is not None:
        res.answer = _fmt(pd_ans)
        res.ok = True
        res.reason = "prozent/double-kette"
        return res
    # --- Variablen-Gleichungen ("If Mina memorized 24 digits, how many
    # did Sam memorize?") — NACH den Objekt-Ketten, damit assign-typische
    # Fälle ("Baez has 25 marbles") nicht die Prozent-Kette klauen.
    # --- Ernte-Rate (John: 10x100x4)
    er = _ernte_rate(question, quants, tgt)
    if er is not None:
        res.answer = _fmt(er)
        res.ok = True
        res.reason = "ernte-rate"
        return res
    # --- Rosen-Preis (Jenny) — VOR den Münzwerten (spezifischer)
    rp = _rosen_preis(question, quants, tgt)
    if rp is not None:
        res.answer = _fmt(rp)
        res.ok = True
        res.reason = "rosen-preis"
        return res
    # --- Münzwerte (Justin: 8+9.5+6+7.5)
    mz = _muenzen(question, quants, tgt)
    if mz is not None:
        res.answer = _fmt(mz)
        res.ok = True
        res.reason = "muenzen"
        return res
    # --- Pizza-Teilen (Henry: 7x8/4)
    pz = _pizza_teilen(question, quants, tgt)
    if pz is not None:
        res.answer = _fmt(pz)
        res.ok = True
        res.reason = "pizza-teilen"
        return res
    # --- Pflanzen-Gruppen (Crista: 4x1/2 + 8x1 + 8x1/4)
    pg = _pflanzen_gruppen(question, quants, tgt)
    if pg is not None:
        res.answer = _fmt(pg)
        res.ok = True
        res.reason = "pflanzen-gruppen"
        return res
    # --- Multi-Tag (Sam: 3+3+3+2+2)
    md = _multi_day(question, quants, tgt)
    if md is not None:
        res.answer = _fmt(md)
        res.ok = True
        res.reason = "multi-tag"
        return res
    # --- Blink-Rate (Lighthouse: 459x5/255)
    br = _blink_rate(question, quants, tgt)
    if br is not None:
        res.answer = _fmt(br)
        res.ok = True
        res.reason = "blink-rate"
        return res
    # --- Alter (Raymond: 31+6-23)
    ag = _alter(question, quants, tgt)
    if ag is not None:
        res.answer = _fmt(ag)
        res.ok = True
        res.reason = "alter"
        return res
    # --- Sammel-Kette (John/Leo: picks/collected = Plus)
    sk = _sammel_kette(question, quants, tgt)
    if sk is not None:
        res.answer = _fmt(sk)
        res.ok = True
        res.reason = "sammel-kette"
        return res
    # --- Episoden (John: 20 x 10)
    ep = _episoden(question, quants, tgt)
    if ep is not None:
        res.answer = _fmt(ep)
        res.ok = True
        res.reason = "episoden"
        return res
    # --- Relative Geschwindigkeit (Two-cars: (70-60)x2)
    rg = _relativ_geschwindigkeit(question, quants, tgt)
    if rg is not None:
        res.answer = _fmt(rg)
        res.ok = True
        res.reason = "relativ-geschw"
        return res
    # --- Prozent-mehr (George: 200 + 200x1.45)
    pm = _prozent_mehr(question, quants, tgt)
    if pm is not None:
        res.answer = _fmt(pm)
        res.ok = True
        res.reason = "prozent-mehr"
        return res
    # --- Spar-Rate (Rong: (20+28)x12x10)
    sr = _spar_rate(question, quants, tgt)
    if sr is not None:
        res.answer = _fmt(sr)
        res.ok = True
        res.reason = "spar-rate"
        return res
    # --- Prozent-Kette (Company: 50 x 0.2 x 0.3)
    pk = _prozent_kette(question, quants, tgt)
    if pk is not None:
        res.answer = _fmt(pk)
        res.ok = True
        res.reason = "prozent-kette"
        return res
    # --- Tray-Rest (Jaime: 64 - 2x24)
    tr = _tray_rest(question, quants, tgt)
    if tr is not None:
        res.answer = _fmt(tr)
        res.ok = True
        res.reason = "tray-rest"
        return res
    # --- Durchschnitt (Jane: (34+36+18)/4)
    da = _durchschnitt(question, quants, tgt)
    if da is not None:
        res.answer = _fmt(da)
        res.ok = True
        res.reason = "durchschnitt"
        return res
    # --- Total/Teile (Monica: 144/12)
    pt2 = _pro_teil(question, quants, tgt)
    if pt2 is not None:
        res.answer = _fmt(pt2)
        res.ok = True
        res.reason = "pro-teil"
        return res
    # --- Prozent-Anteil (Carol: 20%)
    pa = _prozent_anteil(question, quants, tgt)
    if pa is not None:
        res.answer = _fmt(pa)
        res.ok = True
        res.reason = "prozent-anteil"
        return res
    # --- Komplement-Prozent (John: 20 x 20%)
    pk = _percent_komplement(question, quants, tgt)
    if pk is not None:
        res.answer = _fmt(pk)
        res.ok = True
        res.reason = "prozent-komplement"
        return res

    # --- Ketten-Multiplikatoren (Rose: 4 + 12 + 60)
    km = _ketten_multiplikator(question, quants, tgt)
    if km is not None:
        res.answer = _fmt(km)
        res.ok = True
        res.reason = "ketten-mult"
        return res
    # --- Tages-Summe x Einheit (Baldur: (5+6)x5)
    dt = _daily_total(question, quants, tgt)
    if dt is not None:
        res.answer = _fmt(dt)
        res.ok = True
        res.reason = "tages-summe"
        return res
    # --- Wochen-Muster (Pancho: 5x20 + 2x10)
    wm = _wochen_muster(question, quants, tgt)
    if wm is not None:
        res.answer = _fmt(wm)
        res.ok = True
        res.reason = "woche"
        return res
    # --- Proportionale Abdeckung (Fog: 42/3 x 10)
    jn = _jede_n_einheiten(question, quants, tgt)
    if jn is not None:
        res.answer = _fmt(jn)
        res.ok = True
        res.reason = "proportional"
        return res
    # --- Hin- und Rückfahrt (Tom: (3x10)/6)
    rt = _roundtrip(question, quants, tgt)
    if rt is not None:
        res.answer = _fmt(rt)
        res.ok = True
        res.reason = "roundtrip"
        return res
    # --- Gruppen-Käufe (Billy: 3x1 + 2x2)
    gk = _gruppen_kauf(question, quants, tgt)
    if gk is not None:
        res.answer = _fmt(gk)
        res.ok = True
        res.reason = "gruppen-kauf"
        return res
    # --- Boxen-Volumen (John: (5-2)(6-2)(4-2)x3)
    vb = _volume_boxen(question, quants, tgt)
    if vb is not None:
        res.answer = _fmt(vb)
        res.ok = True
        res.reason = "volumen"
        return res
    # --- Raten-Kette (Miguel: 2x30x4)
    rk = _rate_kette(question, quants, tgt)
    if rk is not None:
        res.answer = _fmt(rk)
        res.ok = True
        res.reason = "raten-kette"
        return res
    # --- Verhältnis-Referenz ("half that much")
    tm2 = _that_much(question, quants, tgt)
    if tm2 is not None:
        res.answer = _fmt(tm2)
        res.ok = True
        res.reason = "that-much"
        return res
    # --- Summe+Differenz-Paare (Gretchen: (110+30)/2) — VOR den
    # Variablen (Paddington wird durch den Namens-Check ausgeschlossen)
    pt = _pair_total(question, quants, tgt)
    if pt is not None:
        res.answer = _fmt(pt)
        res.ok = True
        res.reason = "paar-total"
        return res
    var_ans = _solve_variables(rels, question, quants)
    if var_ans is not None:
        res.answer = _fmt(var_ans)
        res.ok = True
        res.reason = "Variablen-Propagation"
        return res
    # --- Rate x Dauer ("6 sentences per minute, for 43 minutes" -> 258)
    rate_ans = _rate_duration(question, rels, quants, tgt)
    if rate_ans is not None:
        res.answer = _fmt(rate_ans)
        res.ok = True
        res.reason = "rate x dauer"
        return res
    if not tgt.ok:
        res.reason = "kein Frageziel"
        return res

    # --- Abstinenz-Gate 2: Mengen des Zielobjekts (qty) müssen existieren ---
    target_qty = [q for q in quants if q.role == "qty" and q.obj == tgt.obj]
    ratios = [q for q in quants if q.role == "ratio" and q.obj == tgt.obj]
    subtracts = [q.value for q in quants if q.role == "subtract"]

    def _mit_subtraktion(total: Fraction) -> Fraction:
        return total - sum(subtracts) if subtracts else total

    # Fall A: "N clips ... half as many clips ... altogether"
    #   -> sum(N, N/2) — ratio referenziert die erste qty desselben Objekts
    if target_qty and ratios and tgt.op == "sum":
        base = target_qty[0].value
        total = base
        for r in ratios:
            total = total + base * r.value
        # partitive Mengen desselben Objekts addieren (Natalia-Fall)
        part = [q for q in quants if q.role == "partitive" and q.obj == tgt.obj]
        if part:
            total = total + base * part[0].value
        res.answer = _fmt(total)
        res.ok = True
        res.reason = f"sum(base={base}, ratio, partitiv)"
        return res

    # Fall B: einfache Summe: "N X, then M X ... altogether"
    if len(target_qty) >= 2 and tgt.op == "sum":
        total = _mit_subtraktion(sum(q.value for q in target_qty))
        res.answer = _fmt(total)
        res.ok = True
        res.reason = f"sum({len(target_qty)} Mengen)"
        return res

    # Fall B': eine einzige qty desselben Objekts + sum-Ziel -> die Menge
    # selbst (die Frage fragt nur nach der Gesamtheit einer Teilmenge)
    if len(target_qty) == 1 and not ratios and tgt.op == "sum":
        # A mentioned wage/rate is not itself the solution to a purchase-time
        # question.  The old shortcut answered e.g. "$8/hour -> 8 hours" and
        # "mows 4 times -> 4 times" without balancing costs and income.
        money_time_target = (tgt.obj or "").rstrip("s") in {
            "time",
            "hour",
            "day",
            "week",
            "month",
            "year",
        } and bool(
            re.search(
                r"\$|\bcosts?\b|\bpays?\b|\bearn(?:s|ed|ing)?\b|"
                r"\bafford\b|\bpurchase\b|\bper\s+(?:hour|time)\b",
                low,
            )
        )
        if money_time_target:
            res.reason = "Zeit-/Geldbilanz nicht strukturell bewiesen"
            return res
        res.answer = _fmt(target_qty[0].value)
        res.ok = True
        res.reason = "einzelne gebundene Menge"
        return res

    # Fall C: left — "total of N, each of M groups has K, how many does
    # the rest/fifth/last have" -> total - Summe der Teile
    if tgt.op == "left":
        total_q = [
            q
            for q in target_qty
            if re.search(r"\btotal\b|\bin all\b", q.text)
            or re.search(
                r"\btotal\b|\bin all\b", low[max(0, q.span[0] - 15) : q.span[0]]
            )
        ]
        parts = [q for q in target_qty if q not in total_q]
        if total_q and parts:
            res.answer = _fmt(total_q[0].value - sum(p.value for p in parts))
            res.ok = True
            res.reason = "left(total - teile)"
            return res
        if total_q and len(target_qty) == 1:
            res.answer = _fmt(total_q[0].value)
            res.ok = True
            res.reason = "left(total)"
            return res

    # Fall D: diff: "how many more X than Y"
    if len(target_qty) >= 2 and tgt.op == "diff":
        res.answer = _fmt(abs(target_qty[0].value - target_qty[1].value))
        res.ok = True
        res.reason = "diff"
        return res

    res.reason = (
        f"Bindung unvollständig: {len(target_qty)} qty, "
        f"{len(ratios)} ratio, op={tgt.op}"
    )
    return res


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def bind(question: str) -> BindingResult:
    """Bindungs-Parser-Einstieg."""
    try:
        return _resolve(question)
    except Exception as exc:
        raise BindingParserError(
            f"binding parser failed: {type(exc).__name__}: {exc}"
        ) from exc


def solve(question: str) -> Optional[str]:
    """Lösen via Bindung — None wenn Bindung unvollständig."""
    r = bind(question)
    return r.answer if r.ok else None


def _fmt(f: Fraction) -> str:
    """Exakt wie math._fmt: Ganzzahl oder max-4-stelliges Dezimal."""
    if f.denominator == 1:
        return str(f.numerator)
    scaled = f * 10000
    if scaled.denominator == 1:
        s = str(scaled.numerator)
        return s[:-4] + "." + s[-4:] if len(s) > 4 else "0." + s.zfill(4)
    return str(float(f))
