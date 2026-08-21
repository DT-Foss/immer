"""
fertig.intent — die neurosymbolische Intent-Maschine.

NL-Befehl -> deterministischer Parse -> Intent-Tupel -> verifizierter Tool-Call.

Schichten (alle gewicht-frei, alle messbar):
  1. Form      : Tokenisierung + gemessene Aktions-Lexika (default + gelernt)
  2. Struktur  : Kasusrahmen-Suche — Aktion = Verb, Ziel = Entität (Jaro-Winkler
                 gegen das .causal-Vokabular)
  3. Parsing   : Kandidaten-Erzeugung + parsimonie-gewichtetes Scoring im
                 Wirkung-Stil:  score = log(fit) − 0.5·complexity
                 (Mapping auf wirkung_formulas.md: Path-Integral-Score mit
                 Parsimonie-Penalty in polnischer Notation)
  4. Ambiguität: Which-Path-Visibility — Verhältnis Top- zu Zweit-Score;
                 unter Schwelle: ehrliche Rückfrage statt Raten
  5. Bedeutung : Grounding gegen .causal — Ziel-Entität im Graphen?
  6. Intent    : Aktion + Ziel + Constraints + gemessene Konfidenz
                 (Hoffman-6-Tupel-Geist: der formale Agenten-Zustand)

Garantien: deterministisch, keine erfundenen Aktionen/Ziele, Fehler sind
sichtbar (der Parse-Baum zeigt die Stelle), Konfidenz ist gemessen.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import inference
from .pipeline import _toks, load_graph

# ---------------------------------------------------------------------------
# Gemessenes Aktions-Lexikon (Default — wächst durch fertig.learn)
# verb -> (action, gewicht)
# ---------------------------------------------------------------------------
DEFAULT_ACTIONS: Dict[str, Tuple[str, float]] = {
    # explain / erzählen
    "explain": ("explain", 0.9), "describe": ("explain", 0.9),
    "tell": ("explain", 0.8), "show": ("explain", 0.6),
    "say": ("explain", 0.7), "speak": ("explain", 0.7),
    "elaborate": ("explain", 0.8), "elaborateon": ("explain", 0.8),
    # finden / herleiten
    "find": ("find", 0.9), "list": ("find", 0.9), "get": ("find", 0.6),
    "what": ("find", 0.7), "which": ("find", 0.7), "where": ("find", 0.6),
    # verhindern / negativer Pfad
    "prevent": ("prevent", 0.9), "stop": ("prevent", 0.8),
    "avoid": ("prevent", 0.8), "block": ("prevent", 0.7),
    "reduce": ("prevent", 0.5), "quit": ("prevent", 0.7),
    # hilfe
    "help": ("help", 0.8), "assist": ("help", 0.8),
    # video erkennen
    "erkennen": ("erkennen", 0.8), "erkenne": ("erkennen", 0.8),
    "erkennt": ("erkennen", 0.7), "recognize": ("erkennen", 0.8),
    "identify": ("erkennen", 0.7), "classify": ("erkennen", 0.7),
    # consult (Index-Konsultation — die Nacht-Policy als Tool-Gate)
    "consult": ("consult", 0.8), "check": ("consult", 0.7),
    "verify": ("consult", 0.8), "confirm": ("consult", 0.7),
}

# Wörter, die niemals Ziel-Entitäten sind (Funktionswörter)
_STOP = {
    "the", "a", "an", "my", "your", "his", "her", "our", "their", "its",
    "to", "of", "in", "on", "at", "for", "with", "from", "by", "and", "or",
    "how", "what", "which", "where", "when", "why", "does", "do", "is", "are",
    "can", "could", "would", "should", "i", "you", "we", "it", "me", "us",
    "about", "me", "affect", "affects", "effect", "effects", "causes",
    "cause", "caused", "leading", "leads", "related", "regarding",
}

_AMBIGUITY_MAX = 0.50   # Which-Path: Anteil des Zweit-Parses am Erst-Parse
_JARO_TARGET = 0.85     # Mindest-Ähnlichkeit Ziel -> Graph-Vokabular

# Relations-Verben für die "how X [verb] Y"-Positions-Heuristik
_RELATION_VERBS = {"affects", "affect", "causes", "cause", "leads",
                   "lead", "reduces", "reduce", "improves", "improve",
                   "prevents", "prevent", "increases", "increase"}


# ---------------------------------------------------------------------------
# Datenstrukturen
# ---------------------------------------------------------------------------

@dataclass
class ParseCandidate:
    """Ein Parse-Kandidat: (Aktion, Ziel, Konfidenzen, Parse-Baum als Text)."""
    action: str
    action_verb: str
    action_conf: float
    target: Optional[str]
    target_conf: float
    nodes: int                       # Komplexität (Baumgröße)
    tree: str                        # sichtbarer Parse-Baum

    @property
    def fit(self) -> float:
        return self.action_conf * self.target_conf

    def parsimony_score(self) -> float:
        """Wirkung-Stil: log(fit) − 0.5·complexity (Parsimonie-gewichtetes
        Ranking, wirkung_formulas.md: Path-Integral-Score)."""
        return float(np.log(self.fit + 1e-30) - 0.5 * self.nodes)


@dataclass
class Intent:
    """Der formale Agenten-Zustand (Hoffman-6-Tupel-Geist):
    (action, target, arguments, confidence, grounded, parse, ambiguity)."""
    action: str
    target: Optional[str]
    arguments: Dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    grounded: bool = False
    ambiguity: float = 0.0            # 0 = eindeutig, 1 = Patt
    tree: str = ""
    tool: Optional[str] = None        # registrierter Tool-Call
    status: str = "ok"                # ok | ambiguous | unknown-action | unknown-target

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Intent(action={self.action!r}, target={self.target!r}, "
                f"conf={self.confidence:.3f}, grounded={self.grounded}, "
                f"amb={self.ambiguity:.2f}, tool={self.tool!r}, "
                f"status={self.status!r})")


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

def _action_candidates(tokens: List[str], lexicon=None,
                       ) -> List[Tuple[str, str, float]]:
    """Alle Token, die eine bekannte Aktion tragen (Default + gelernt)."""
    cands: List[Tuple[str, str, float]] = []
    lex = dict(DEFAULT_ACTIONS)
    if lexicon is not None:
        for verb, meta in lexicon.actions.items():
            lex.setdefault(verb, (meta["action"], float(meta["weight"])))
    for tok in tokens:
        hit = lex.get(tok)
        if hit:
            cands.append((tok, hit[0], hit[1]))
    # deduplizieren, nach Gewicht sortieren
    seen = set()
    out = []
    for verb, action, w in sorted(cands, key=lambda x: -x[2]):
        key = (verb, action)
        if key not in seen:
            seen.add(key)
            out.append((verb, action, w))
    return out[:3]


def _target_candidates(tokens: List[str], vocab: List[str],
                       lexicon=None,
                       ) -> List[Tuple[str, float]]:
    """Inhalts-Token/Phrasen gegen das Graph-Vokabular matchen (Jaro-Winkler).
    Gelernte Nomen (Lexikon) erweitern die Ziel-Abdeckung ohne Graph-Treffer."""
    content = [t for t in tokens if t not in _STOP]
    cands: List[Tuple[str, str, float]] = []
    for i in range(len(content)):
        # Einzelwort und Zweiwort-Phrasen als Kandidaten
        for phrase in (" ".join(content[i:i + 2]), content[i]):
            if not phrase:
                continue
            best, best_j = None, 0.0
            for v in vocab:
                j = inference.jaro_winkler(phrase, v)
                if j > best_j:
                    best, best_j = v, j
            if best is not None:
                cands.append((phrase, best, best_j))
    # nur Treffer über der Jaro-Schwelle sind Ziele — sonst ehrlich: keins
    cands = [(p, m, j) for p, m, j in cands if j >= _JARO_TARGET]
    # gelernte Nomen als Fallback-Ziele (nicht im Graphen, aber bekannt)
    if lexicon is not None:
        known = set(lexicon.nouns)
        for tok in content:
            if tok in known and not any(tok == p for p, _, _ in cands):
                cands.append((tok, tok, 0.5))
    # deduplizieren: beste Jaro-Ähnlichkeit pro ENTITÄT (nicht pro Phrase —
    # mehrere Phrasen, die dieselbe Entität treffen, sind KEIN Patt)
    best_per: Dict[str, Tuple[str, float]] = {}
    for phrase, match, j in cands:
        cur = best_per.get(match)
        if cur is None or j > cur[1]:
            best_per[match] = (phrase, j)
    ranked = sorted(best_per.items(), key=lambda kv: -kv[1][1])
    return [(m, j) for _, (m, j) in ranked[:2]]


def _extract_arguments(tokens: List[str]) -> Dict[str, str]:
    """Constraints: Zahlen und from/to-Pfade."""
    args: Dict[str, str] = {}
    nums = [t for t in tokens if re.fullmatch(r"\d+([.,]\d+)?", t)]
    if nums:
        args["amount"] = nums[0]
    if "from" in tokens and "to" in tokens:
        i, j = tokens.index("from"), tokens.index("to")
        if j > i + 1:
            args["path"] = f"{tokens[i + 1]} -> {tokens[j + 1] if j + 1 < len(tokens) else '?'}"
    return args


def parse_command(text: str, vocab: List[str],
                  lexicon: Optional[Dict] = None) -> Intent:
    """NL-Befehl -> Intent. Deterministisch. Ehrliche Fehlschläge sichtbar."""
    tokens = _toks(text)
    if not tokens:
        return Intent("unknown", None, status="unknown-action",
                      tree="(leer)")

    # 1. Kandidaten
    actions = _action_candidates(tokens, lexicon)
    targets = _target_candidates(tokens, vocab, lexicon)

    if not actions:
        # ehrlicher Fehlschlag: kein bekanntes Verb -> Baum zeigt den Stand
        return Intent(
            "unknown", None,
            arguments=_extract_arguments(tokens),
            confidence=0.0, tree=f"(parse: verb unbekannt in {tokens})",
            status="unknown-action")

    cands: List[ParseCandidate] = []
    for verb, action, aconf in actions:
        if targets:
            for match, j in targets:
                nodes = 2 + len(tokens) // 4
                tree = (f"(intent (action {verb}/{action} {aconf:.2f}) "
                        f"(target {match!r}~{match} {j:.2f}) "
                        f"(words {len(tokens)}))")
                cands.append(ParseCandidate(action, verb, aconf, match, j,
                                            nodes, tree))
        else:
            # kein Ziel-Treffer: ehrlicher Fehlschlag, Baum zeigt den Stand
            nodes = 1 + len(tokens) // 4
            tree = (f"(intent (action {verb}/{action} {aconf:.2f}) "
                    f"(target ?) (words {len(tokens)}))")
            cands.append(ParseCandidate(action, verb, aconf, None, 0.0,
                                        nodes, tree))

    if not cands:
        return Intent("unknown", None, status="unknown-action", tree="(leer)")

    # Positions-Heuristik: "how X [relations-verb] Y" -> X ist das Thema.
    # Ziel-Kandidaten NACH dem Verb werden abgewertet (außer es gibt keine
    # anderen) — löst "explain how smoking affects health" korrekt zu
    # smoking auf, statt ein Patt zu melden.
    if "how" in tokens and any(v in tokens for v in _RELATION_VERBS):
        verb_pos = min(tokens.index(v) for v in _RELATION_VERBS
                       if v in tokens)
        for c in cands:
            if c.target is not None:
                t_pos = tokens.index(c.target.split()[0]) \
                    if c.target.split()[0] in tokens else -1
                if t_pos > verb_pos:
                    c.action_conf *= 0.25

    # 2. Parsimonie-Scoring (Wirkung-Stil)
    scored = sorted(cands, key=lambda c: -c.parsimony_score())
    top, second = scored[0], scored[1] if len(scored) > 1 else None

    # 3. Which-Path-Visibility: Patt-Erkennung.
    # Parsimony-Scores sind stets negativ (log(fit) − 0.5·nodes) — die
    # Ambiguität nutzt die positiv verschobene Score-Relation:
    #   amb = a2/a1  (a = score − min + 1)
    # Patt (a2 == a1) -> 1.0, klarer Sieger -> gegen 0.
    ambiguity = 0.0
    if second is not None and len(scored) >= 2:
        scores = [c.parsimony_score() for c in scored]
        lo = min(scores)
        a1 = scores[0] - lo + 1.0
        a2 = scores[1] - lo + 1.0
        if a1 > 1e-12:
            ambiguity = float(min(1.0, max(0.0, a2 / a1)))

    if ambiguity > _AMBIGUITY_MAX:
        return Intent(
            top.action, top.target,
            arguments=_extract_arguments(tokens),
            confidence=float(top.fit),
            ambiguity=ambiguity,
            tree=f"(ambiguous top={top.tree} alt={second.tree if second else '?'})",
            status="ambiguous")

    # 4. Grounding gegen den Graphen
    grounded = top.target is not None and top.target in vocab
    conf = float(top.fit)
    status = "ok"
    if top.target is None:
        status = "unknown-target"
        conf *= 0.7

    return Intent(
        top.action, top.target,
        arguments=_extract_arguments(tokens),
        confidence=conf,
        grounded=grounded,
        ambiguity=ambiguity,
        tree=top.tree,
        tool=_TOOL_FOR_ACTION.get(top.action) if status == "ok" else None,
        status=status)


# Aktion -> registrierter Tool-Name (fertig.tools)
_TOOL_FOR_ACTION = {
    "explain": "speech", "find": "chain", "prevent": "prevent",
    "help": "help", "consult": "consult", "erkennen": "video",
}


def load_vocab(graph_path) -> List[str]:
    """Vokabular aus einem .causal-Graphen laden (für parse_command)."""
    vocab, stoi, adj, mech = load_graph(graph_path)
    return vocab
