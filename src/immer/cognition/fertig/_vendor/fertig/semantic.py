"""
fertig.semantic — semantischer Parser für Wortaufgaben (Entity-Relations-Graph).

Die schwere, richtige Lösung statt Templates:

  Sätze  ->  Entitäten binden  ("Randy hat 60 Mangobäume" -> mango=60)
  Sätze  ->  Relationen        ("5 weniger als halb so viele Kokosbäume
                                 wie Mangobäume" -> kokos = mango/2 - 5)
  Frage  ->  Query             ("wie viele insgesamt" -> summe)

Der Graph wird aufgebaut und ausgeführt — exakt (fractions), mit ehrlicher
Abstinenz (None), wenn eine Relation unklar bleibt.

Die Brücke zu FORGE: Der Graph IST ein Programm (Knoten = Werte,
Kanten = Operationen) — ausführbar, prüfbar, komponierbar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Tuple


@dataclass
class Entity:
    name: str
    value: Optional[Fraction] = None
    unit: str = ""
    known: bool = False


@dataclass
class Relation:
    """Kanten-Operation im Graph: type, Ziel, Quellen, Parameter."""
    op: str            # "half_of" | "adjust" | "percent_of" | "sum"
    target: str        # Ziel-Entität
    source: str = ""   # Quell-Entität (bei binären Op)
    param: Fraction = Fraction(0)


class SemanticGraph:
    """Entity-Relations-Graph einer Wortaufgabe."""

    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.query: Optional[str] = None   # Ziel-Entität oder "summe"

    def entity(self, name: str) -> Entity:
        name = name.strip().lower()
        if name not in self.entities:
            self.entities[name] = Entity(name)
        return self.entities[name]

    def solve(self) -> Optional[Fraction]:
        """Relationen anwenden, bis fixiert; dann Query auflösen.
        Abstinenz-Gate: Der Query-Wert MUSS durch mindestens eine
        angewendete Relation entstehen — eine zufällige Direkt-Bindung
        ist keine Antwort (GroundZero-Prinzip: lieber None als falsch)."""
        applied = 0
        # Relationen mehrfach anwenden (Ketten)
        for _ in range(len(self.relations) + 1):
            changed = False
            for rel in self.relations:
                tgt = self.entity(rel.target)
                src = self.entities.get(rel.source)
                if rel.op == "half_of":
                    if src and src.value is not None and tgt.value is None:
                        tgt.value = src.value / 2 + rel.param
                        applied += 1
                        changed = True
                elif rel.op == "percent_of":
                    if src and src.value is not None and tgt.value is None:
                        tgt.value = src.value * rel.param / 100
                        applied += 1
                        changed = True
                elif rel.op == "adjust":
                    if src and src.value is not None and tgt.value is None:
                        tgt.value = src.value + rel.param
                        applied += 1
                        changed = True
                elif rel.op == "times":
                    if src and src.value is not None and tgt.value is None:
                        tgt.value = src.value * rel.param
                        applied += 1
                        changed = True
            if not changed:
                break
        # Abstinenz: ohne angewendete Relation keine Antwort
        if applied == 0:
            return None
        # Query
        if self.query == "summe":
            vals = [e.value for e in self.entities.values()
                    if e.value is not None]
            return sum(vals) if vals else None
        tgt = self.entities.get(self.query or "")
        return tgt.value if tgt and tgt.value is not None else None


# ---------------------------------------------------------------------------
# Parser: Sätze -> Graph
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sprach-Schicht: semantische Primitive mit Oberflächen-Mustern pro Sprache.
# Eine neue Sprache = neue Muster im selben Primitive — keine neue Logik.
# ---------------------------------------------------------------------------

# Stoppwörter pro Sprache (Funktionswörter, keine Entitäts-Bestandteile)
_STOP = {
    "en": {"the", "a", "an", "his", "her", "their", "its", "our", "my",
           "your", "of", "in", "on", "at", "to", "for", "with", "and",
           "then", "also", "he", "she", "it", "they", "there", "are", "is",
           "was", "were", "has", "have", "had", "does", "do", "did", "than",
           "as", "how", "many", "much", "what", "total", "altogether",
           "combined", "all", "left", "remain", "remaining", "per", "each",
           "cost", "costs", "costing", "buys", "buy", "bought", "sells",
           "sold", "spends", "spent", "earns", "earned", "makes", "made",
           "gets", "got", "gives", "gave", "takes", "took", "puts", "put",
           "adds", "added", "subtracts", "subtracted", "multiplies",
           "divides", "farm", "april", "may", "june", "march", "january",
           "february", "july", "august", "september", "october", "november",
           "december", "his", "her"},
    "de": {"der", "die", "das", "ein", "eine", "einen", "sein", "seine",
           "ihr", "ihre", "auf", "in", "an", "zu", "für", "mit", "von",
           "und", "dann", "auch", "er", "sie", "es", "wie", "viele", "viel",
           "was", "insgesamt", "zusammen", "alle", "übrig", "bleiben",
           "bleibt", "hat", "haben", "hatte", "verkaufte", "verkauft",
           "kaufte", "kauft", "gekauft", "kostet", "kosten", "pro", "jeder",
           "jede", "jedes", "als", "halb", "weniger", "mehr", "farm",
           "hof", "im", "am", "april", "mai", "juni", "märz", "januar",
           "februar", "juli", "august", "september", "oktober", "november",
           "dezember", "clips", "bäume", "freunden", "freunde"},
}

# Entitäts-Suffixe (Wörter, die eine Entität abschließen können)
_ENTITY_END = {
    "en": {"trees", "clips", "books", "glasses", "apples", "cars", "eggs",
           "dollars", "meters", "liters", "hours", "days", "students"},
    "de": {"bäume", "bäumen", "clips", "bücher", "gläser", "äpfel", "autos",
           "eier", "dollar", "meter", "liter", "stunden", "tage", "tage",
           "schüler", "schülern"},
}

# Primitive: (op, Gruppen-Rolle) -> Muster pro Sprache.
# Rollen: subj=Besitzer, qty=Menge, obj=Objekt, cmp=Vergleichsbasis
_PATTERNS = {
    "bought": {
        "en": [r"([a-z]+)\s+(?:bought|buys|bought|purchased|purchases)"
               r"\s+(\d+)\s+([a-z][a-z ]*?)(?:\s+(?:for|at|to|of)|"
               r"[\.!,?]|$)"],
        "de": [r"([a-zäöüß]+)\s+(?:kaufte|kauft|kaufte)\s+(\d+)\s+"
               r"([a-zäöüß][a-zäöüß ]*?)(?:\s+(?:für|zu|an|von)|"
               r"[\.!,?]|$)"],
    },
    "needs": {
        "en": [r"([a-z]+)\s+(?:needs?|takes?|uses?)\s+(\d+)\s+"
               r"([a-z][a-z ]*?)(?:[\.!,?]|$)"],
        "de": [r"([a-zäöüß]+)\s+(?:braucht|benötigt)\s+(\d+)\s+"
               r"([a-zäöüß][a-zäöüß ]*?)(?:[\.!,?]|$)"],
    },
    "possess": {
        "en": [r"([a-z]+)\s+has\s+(\d+)\s+([a-z][a-z ]*?)"
               r"(?:\s+(?:on|in|at|for|to|with|of|and)|[.!?,]|$)"],
        "de": [r"([a-zäöüß]+)\s+hat\s+(\d+)\s+([a-zäöüß][a-zäöüß ]*?)"
               r"(?:\s+(?:auf|in|an|zu|für|mit|von|und)|[.!?,]|$)"],
    },
    "sold": {
        "en": [r"([a-z]+)\s+sold\s+([a-z][a-z ]*?)\s+to\s+(\d+)"],
        "de": [r"([a-zäöüß]+)\s+verkaufte\s+([a-zäöüß][a-zäöüß ]*?)"
               r"\s+an\s+(\d+)"],
    },
    "half_less": {
        "en": [r"(\d+)\s+(?:less|more)\s+than\s+half\s+as\s+many\s+"
               r"([a-z][a-z ]*?)\s+as\s+([a-z][a-z ]*?)(?:[.!?,]|$)"],
        "de": [r"(\d+)\s+(?:weniger|mehr)\s+als\s+halb\s+so\s+viele\s+"
               r"([a-zäöüß][a-zäöüß ]*?)\s+wie\s+"
               r"([a-zäöüß][a-zäöüß ]*?)(?:[.!?,]|$)"],
    },
    "half_as": {
        "en": [r"half\s+as\s+many\s+([a-z][a-z ]*?)\s+as\s+"
               r"([a-z][a-z ]*?)(?:[.!?,]|$)"],
        "de": [r"halb\s+so\s+viele\s+([a-zäöüß][a-zäöüß ]*?)\s+wie\s+"
               r"([a-zäöüß][a-zäöüß ]*?)(?:[.!?,]|$)"],
    },
    "half_later": {
        "en": [r"half\s+as\s+many\s+([a-z][a-z ]*?)"
               r"(?:\s+in\s+\w+|[.!?,]|$)"],
        "de": [r"halb\s+so\s+viele\s+([a-zäöüß][a-zäöüß ]*?)"
               r"(?:\s+im\s+\w+|[.!?,]|$)"],
    },
    "query_total": {
        "en": [r"in (?:all|total)|altogether|combined"],
        "de": [r"insgesamt|zusammen"],
    },
    "query_entity": {
        "en": [r"how many\s+([a-z][a-z ]*?)(?:\s+(?:does|do|are|is|in|"
               r"left|remain|remaining)|[.?!]|$)"],
        "de": [r"wie viele\s+([a-zäöüß][a-zäöüß ]*?)"
               r"(?:\s+(?:hat|gibt es|sind|übrig|bleiben)|[.?!]|$)"],
    },
}


def _clean_entity(phrase: str, lang: str = "en") -> str:
    """Nominalphrase -> kanonischer Entitätsname (Stoppwörter ab)."""
    toks = [t for t in phrase.lower().split()
            if t not in _STOP.get(lang, set())]
    return " ".join(toks)


def detect_language(question: str) -> str:
    """Sprach-Detektion: Umlaute/ß oder deutsche Stoppwörter -> de, sonst en."""
    q = question.lower()
    if re.search(r"[äöüß]", q):
        return "de"
    if re.search(r"\b(?:und|der|die|das|wie viele|insgesamt|hat)\b", q):
        return "de"
    return "en"


def _match(patterns: dict, lang: str, s: str):
    for pat in patterns.get(lang, []):
        m = re.search(pat, s)
        if m:
            return m
    return None


def parse_semantic(question: str, lang: str = "") -> SemanticGraph:
    """Wortaufgabe -> Entity-Relations-Graph (sprachagnostisch).

    Die Primitive (possess/sold/half/query) tragen Oberflächen-Muster
    pro Sprache — neue Sprachen erweitern die Muster, nie die Logik.
    Abstinenz-Gate: Antwort nur über angewendete Relationen."""
    lang = lang or detect_language(question)
    g = SemanticGraph()
    sents = re.split(r"(?<=[.!?])\s+", question)

    # 1. Bindungen (possess, sold) — pro Satz
    for sent in sents:
        s = sent.lower()
        for bkey in ("bought", "needs", "possess"):
            m = _match(_PATTERNS[bkey], lang, s)
            if m and "half" not in s and "halb" not in s and \
                    "than" not in s and "als" not in s:
                name = _clean_entity(m.group(3), lang)
                g.entity(name).value = Fraction(m.group(2))
                break
        if m and "half" not in s and "halb" not in s and \
                "than" not in s and "als" not in s:
            continue
        m = _match(_PATTERNS["possess"], lang, s)
        if m and "half" not in s and "halb" not in s and "than" not in s \
                and "als" not in s:
            name = _clean_entity(m.group(3), lang)
            g.entity(name).value = Fraction(m.group(2))
            continue
        m = _match(_PATTERNS["sold"], lang, s)
        if m:
            name = _clean_entity(m.group(2), lang)
            g.entity(name).value = Fraction(m.group(3))
            continue
        # "N X" am Satzanfang
        m = re.match(r"(\d+)\s+([a-zäöüß][a-zäöüß ]*?)"
                     r"(?:\s+in\s+\w+|\s+im\s+\w+|\.|,|$)", s)
        if m:
            name = _clean_entity(m.group(2), lang)
            g.entity(name).value = Fraction(m.group(1))

    # 2. Relationen
    for sent in sents:
        s = sent.lower()
        m = _match(_PATTERNS["half_less"], lang, s)
        if m:
            param = Fraction(m.group(1))
            if "less" in m.group(0) or "weniger" in m.group(0):
                param = -param
            g.relations.append(Relation(
                "half_of", _clean_entity(m.group(2), lang),
                _clean_entity(m.group(3), lang), param))
            continue
        m = _match(_PATTERNS["half_as"], lang, s)
        if m:
            g.relations.append(Relation(
                "half_of", _clean_entity(m.group(1), lang),
                _clean_entity(m.group(2), lang)))
            continue
        m = _match(_PATTERNS["half_later"], lang, s)
        if m:
            src_name = _clean_entity(m.group(1), lang)
            if src_name in g.entities:
                g.relations.append(Relation(
                    "half_of", src_name + " (spaeter)", src_name))
            continue
        # "N more/less than X" / "N mehr/weniger als X"
        m = re.search(r"(\d+)\s+(?:more|less|mehr|weniger)\s+"
                      r"(?:than|als)\s+([a-zäöüß][a-zäöüß ]*?)(?:[.!?,]|$)",
                      s)
        if m:
            src = _clean_entity(m.group(2), lang)
            param = Fraction(m.group(1))
            if "less" in m.group(0) or "weniger" in m.group(0):
                param = -param
            g.relations.append(Relation("adjust", src, src, param))
            continue
        # "X% of Y"
        m = re.search(r"(\d+)\s*(?:%|percent|prozent)\s+of\s+"
                      r"(?:von\s+)?([a-zäöüß][a-zäöüß ]*?)(?:[.!?,]|$)",
                      s)
        if m:
            src = _clean_entity(m.group(2), lang)
            g.relations.append(Relation("percent_of", src, src,
                                        Fraction(m.group(1))))

    # 3. Query
    m = _match(_PATTERNS["query_entity"], lang, question.lower())
    if m:
        g.query = _clean_entity(m.group(1), lang)
    if _match(_PATTERNS["query_total"], lang, question.lower()):
        g.query = "summe"
    return g
