"""
fertig.miner — automatischer Regel-Miner für Wortaufgaben.

GSM8K-Gold-Antworten enthalten die komplette Ableitung (<<a op b=c>>).
Das ist freies Supervisions-Signal: Der Miner lernt aus
(Zahlen-Kontext -> Operation)-Paaren, ohne handgeschriebene Muster.

  Lernen:   (Frage, Gold-Ableitung) -> (n1_Kontext, n2_Kontext) -> op
  Schluss:  neue Frage -> Zahlen-Kontexte -> gelernte Regeln -> Rechnung
  Iteration: Regeln anwenden, testen, häufige Fehler -> neue Regeln

Der Miner ist die automatische Alternative zu handgeschriebenen Templates:
die Regeln entstehen aus den Daten, sind aber als (Kontext -> op)-Paare
inspizierbar und testbar.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

_OP = {"+": "add", "*": "mul", "/": "div", "-": "sub"}


def parse_gold_ops(answer: str) -> List[Tuple[str, int, int, int]]:
    """Gold-Ableitung -> [(op, n1, n2, result), ...] aus <<...>>-Schritten."""
    out = []
    for m in re.finditer(r"<<(\d+)\s*([+*/-])\s*(\d+)\s*=\s*(\d+)>>", answer):
        out.append((_OP[m.group(2)], int(m.group(1)), int(m.group(3)),
                    int(m.group(4))))
    return out


def _num_contexts(question: str) -> List[Tuple[str, str, str]]:
    """Alle Zahlen mit Umgebung: (zahl, vorherige-Wörter, folgende-Wörter)."""
    q = question.lower()
    out = []
    for m in re.finditer(r"(\d+)", q):
        before = q[max(0, m.start() - 25):m.start()].split()[-3:]
        after = q[m.end():m.end() + 25].split()[:3]
        out.append((m.group(1), " ".join(before), " ".join(after)))
    return out


def _find_ctx(num: str, contexts: List[Tuple[str, str, str]]) -> Tuple[str, str]:
    for n, before, after in contexts:
        if n == num:
            return before, after
    return "", ""


def _between(q: str, start: int, end: int) -> str:
    """Wörter zwischen zwei Positionen (max 5)."""
    return " ".join(q[start:end].split()[:5])


# Operations-Keywords sind eine GESCHLOSSENE Klasse (linguistisch):
# per/each/times/every -> Multiplikation; more/less/total/left -> add/sub.
# Generische Wörter (to/and/has/is) sind KEINE Operations-Träger — sie
# erzeugten Rausch-Regeln mit 2% Präzision.
_KEYWORDS = ["each", "per", "times", "every", "of", "percent", "%", "more",
             "less", "total", "altogether", "plus", "minus", "left",
             "remaining", "cost", "costs", "worth"]


def _sig(between: str) -> str:
    """Schlüsselwort-Signatur: welche Operations-Wörter stehen zwischen den
    Zahlen? (each/per/times -> mul; of/percent -> prozent; ...)"""
    words = between.split()
    kws = [w for w in words if w in _KEYWORDS]
    # Ohne Operations-Keyword KEINE Signatur — sonst entstehen Rausch-
    # Regeln aus beliebigen Wörtern ("($)", "(2)", "accessories.")
    return " ".join(kws) if kws else ""


def mine_rules(questions: List[str], answers: List[str],
               min_count: int = 3) -> Dict[str, str]:
    """Aus (Frage, Gold) lernen: (Schlüsselwörter zwischen den Zahlen) -> op.
    '5 books each costing 12' und '3 classes each has 15' teilen die
    Signatur 'each' -> mul — die Generalisierung über spezifische Wörter."""
    opvotes: Dict[str, Counter] = defaultdict(Counter)
    for q, a in zip(questions, answers):
        ql = q.lower()
        nums = [(m.start(), m.end(), m.group(1))
                for m in re.finditer(r"(\d+)", ql)]
        for op, n1, n2, _ in parse_gold_ops(a):
            p1 = next((p for p in nums if p[2] == str(n1)), None)
            p2 = next((p for p in nums if p[2] == str(n2)), None)
            if p1 and p2 and p1[1] <= p2[0]:
                sig = _sig(_between(ql, p1[1], p2[0]))
                if sig:
                    opvotes[sig][op] += 1
    rules = {}
    for sig, ops in opvotes.items():
        op, cnt = ops.most_common(1)[0]
        if cnt >= min_count:
            rules[sig] = op
    return rules


def _chain_search(tokens: List[str], rules: Dict[str, str],
                   depth: int = 0) -> Optional[Fraction]:
    """Backtracking über Operations-Reihenfolgen: alle Paare probieren,
    rechnen, rekursiv fortsetzen. Erste vollständige Reduktion gewinnt.
    Verhindert Greedy-Fehler ('3 sprints 3 times 60 meters' braucht die
    richtige Reihenfolge)."""
    if depth > 12:
        return None
    nums = [(i, Fraction(t)) for i, t in enumerate(tokens)
            if re.fullmatch(r"\d+", t)]
    if len(nums) <= 1:
        return nums[0][1] if nums else None
    # alle Paare in allen Reihenfolgen probieren
    for a in range(len(nums)):
        for b in range(len(nums)):
            if a == b:
                continue
            i1, v1 = nums[a]
            i2, v2 = nums[b]
            lo, hi = (i1, i2) if i1 < i2 else (i2, i1)
            between = " ".join(tokens[lo + 1:hi])
            sig = _sig(between)
            op = rules.get(sig)
            if not op:
                for w in between.split():
                    if w in _KEYWORDS and w in rules:
                        op = rules[w]
                        break
            if not op:
                continue
            if op == "mul":
                res = v1 * v2
            elif op == "add":
                res = v1 + v2
            elif op == "sub":
                res = v1 - v2 if i1 < i2 else v2 - v1
            elif op == "div" and v2 != 0:
                res = v1 / v2
            else:
                continue
            if res.denominator != 1 and res.numerator % res.denominator:
                # nicht-ganzzahliges Zwischenergebnis: unwahrscheinlich
                continue
            nt = tokens[:]
            nt[lo] = str(res.numerator) if res.denominator == 1 \
                else str(float(res))
            del nt[lo + 1:hi + 1]
            r = _chain_search(nt, rules, depth + 1)
            if r is not None:
                return r
    return None


def apply_rules(question: str, rules: Dict[str, str]) -> Optional[str]:
    """Neue Frage: Zahlen-Kontexte -> Regeln -> Rechnung (Kette)."""
    ctxs = _num_contexts(question)
    nums = [n for n, _, _ in ctxs]
    if len(nums) < 2:
        return None
    # Backtracking-Ketten-Suche (die schwere richtige Lösung)
    ql = question.lower()
    tokens = ql.split()
    if sum(1 for t in tokens if re.fullmatch(r"\d+", t)) < 2:
        return None
    r = _chain_search(tokens, rules)
    if r is None:
        return None
    return str(r.numerator) if r.denominator == 1 else str(float(r))
