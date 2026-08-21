"""
fertig.math — neurosymbolisches Rechnen: Zahlen extrahieren, Struktur
erkennen, EXAKT rechnen (fractions — kein Float-Müll).

Die Antwort auf "wer Tool-Calls kann, kann Benchmarks" im Numerischen:
  Frage -> Zahlen + Einheiten (regulär)
  Frage -> Struktur-Template (half/percent/each/total/differenz)
  Template + Zahlen -> fraktionsgenaue Berechnung
  Ergebnis -> Antwort (pass@1 gegen den Gold-Wert)

Kein Training, keine Statistik — Struktur-Erkennung + exakte Arithmetik.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Dict, List, Optional, Tuple


def extract_numbers(question: str) -> List[Tuple[Fraction, str]]:
    """Zahlen + direkt folgendes Wort (Einheit/Kontext)."""
    out = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z%]+)?", question.lower()):
        try:
            val = Fraction(m.group(1))
        except ZeroDivisionError:
            continue
        out.append((val, m.group(2) or ""))
    return out


def _num(q: str, i: int) -> Optional[Fraction]:
    """Die i-te Zahl (0-basiert)."""
    nums = [n for n, _ in extract_numbers(q)]
    return nums[i] if i < len(nums) else None


def solve(question: str) -> Optional[str]:
    """Erst Operationsketten (v2), dann Einzeltemplates. None = ehrlich
    nicht erkannt."""
    chain = solve_chain(question)
    if chain is not None:
        return chain
    q = question.lower()

    # 1. Prozent: "X% of Y" / "X percent of Y"
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(\d+(?:\.\d+)?)",
                  q)
    if m:
        r = Fraction(m.group(1)) / 100 * Fraction(m.group(2))
        return _fmt(r)

    # 1b. "N less/more than half of X": (X/2 ∓ N)
    m = re.search(r"(\d+)\s+(?:less|more)\s+than\s+half\s+of\s+(\d+)",
                  q)
    if m:
        half = Fraction(m.group(2)) / 2
        return _fmt(half - Fraction(m.group(1))
                    if "less" in m.group(0) else half + Fraction(m.group(1)))

    # 2. Halbierung + Summe: "sold N in A, then half as many in B.
    #    How many ... in all/total?"
    if "half as many" in q:
        base = _num(q, 0)
        if base is not None:
            half = base / 2
            if re.search(r"in all|in total|altogether|combined", q):
                return _fmt(base + half)
            return _fmt(half)

    # 3. Jeder-Kauf: "N items, each costs X ... total/altogether"
    m = re.search(r"(\d+)\s+(?:of them )?each\s+(?:costs?|for|sells? for|"
                  r"worth)\s+(\d+)", q)
    if m:
        total = Fraction(m.group(1)) * Fraction(m.group(2))
        if re.search(r"total|altogether|in all|all together", q):
            return _fmt(total)
        return _fmt(total)

    # 4. Differenz: "how many more X than Y" / "N more/less than"
    m = re.search(r"how many more (\w+) than (\w+)", q)
    if m:
        # braucht Werte aus dem Kontext — meist "X has A, Y has B"
        nums = [n for n, _ in extract_numbers(q)]
        if len(nums) >= 2:
            return _fmt(abs(nums[0] - nums[1]))
    m = re.search(r"(\d+) (?:more|less) than (\d+)", q)
    if m:
        a, b = Fraction(m.group(1)), Fraction(m.group(2))
        return _fmt(a + b if "more" in m.group(0) else b - a)

    # 5. Pro-Kopf/Teilung: "N people split X equally"
    m = re.search(r"(\d+)\s+(?:people|friends|students|children|workers)"
                  r"\s+(?:split|share|divide)", q)
    if m:
        nums = [n for n, _ in extract_numbers(q)]
        if len(nums) >= 2:
            return _fmt(nums[-1] / nums[0])

    return None


def _fmt(f: Fraction) -> str:
    """Ausgabe: Ganzzahl oder Dezimal (max 4 Stellen) — ohne Float-Artefakte."""
    if f.denominator == 1:
        return str(f.numerator)
    # Dezimal mit begrenzter Präzision, exakt gerundet
    scaled = f * 10000
    if scaled.denominator == 1:
        s = str(scaled.numerator)
        return s[:-4] + "." + s[-4:] if len(s) > 4 else "0." + s.zfill(4)
    return str(float(f))  # nicht-periodisch: letzter Ausweg


def _fmt_frac(f: Fraction) -> str:
    return _fmt(f)


def solve_chain(question: str) -> Optional[str]:
    """Operationsketten v2: mehrstufige Beziehungen, exakt gerechnet.

    A) Rate-Kette:  'X per day, uses Y per day, for N days, how many
                    left/in all'  ->  X*N - Y*N  (bzw. X*N bei 'in all')
    B) Stück-Kette: 'N items, each costs X, ... total'  ->  N*X
    C) Prozent:     'X% of Y' (auch mehrstufig mit Rest)
    """
    q = question.lower()

    # A) Rate-Kette (Abstinenz verschärft): NUR mit explizitem Zeitraum
    # "X per Y, for M Y" — ohne "for M" ist die Rate keine Antwort
    m = re.search(r"(\d+)\s+(\w+)\s+per\s+(\w+)[^.]*?"
                  r"for\s+(\d+)\s+(\w+)", q)
    if m:
        rate = Fraction(m.group(1))
        u2 = m.group(3)
        m_unit = m.group(5)
        if u2 == m_unit or (u2 + "s") == m_unit or u2 == (m_unit + "s"):
            produktion = rate * Fraction(m.group(4))
            # Verbrauch: "uses/eats N per <zeit>"
            mv = re.search(r"(?:eats|uses?|consumes?)\s+(\d+)\s+"
                           r"[a-z]+\s+per\s+" + u2, q)
            if mv:
                verbrauch = Fraction(mv.group(1)) * Fraction(m.group(4))
                if re.search(r"left|remain|remaining", q):
                    return _fmt_frac(produktion - verbrauch)
                return _fmt_frac(produktion + verbrauch)
            return _fmt_frac(produktion)
        return None
    m = re.search(r"(\d+)\s+(\w+)\s+per\s+(day|hour|week|month|year)"
                  r"[^.]*?for\s+(\d+)\s+(?:\w+ ){0,2}"
                  r"(day|hour|week|month|year)s?", q)
    if m:
        return _fmt_frac(Fraction(m.group(1)) * Fraction(m.group(4)))
    if m:
        rate, unit, zeit = Fraction(m.group(1)), m.group(2), m.group(3)
        # Zeitraum: "for N days/weeks"
        mz = re.search(r"for\s+(\d+)\s+" + zeit + r"s?", q)
        mz = mz or re.search(r"(\d+)\s+" + zeit + r"s?", q)
        if not mz:
            return None
        n_zeit = Fraction(mz.group(1))
        produktion = rate * n_zeit
        # Verbrauch: "eats/uses X per day"
        mv = re.search(r"(?:eats|uses?|consumes?|needs?)\s+(\d+)\s+"
                       r"[a-z]+\s+(?:per|each)\s+" + zeit, q)
        if mv:
            verbrauch = Fraction(mv.group(1)) * n_zeit
            if re.search(r"left|remain|remaining", q):
                return _fmt_frac(produktion - verbrauch)
            return _fmt_frac(produktion + verbrauch)
        return _fmt_frac(produktion)

    # B) Stück-Kette: "N items each costs X" (Komma nach Anzahl erlaubt)
    m = re.search(r"(\d+)\s+(?:[\w,]+ ){0,3}each\s+(?:costs?|costing|"
                  r"for|worth|sells? for)\s+(\d+)", q)
    if m:
        total = Fraction(m.group(1)) * Fraction(m.group(2))
        # evtl. Rabatt: "with N% discount" -> total * (100-N)/100
        md = re.search(r"(\d+)\s*(?:%|percent)\s+(?:off|discount)", q)
        if md:
            total = total * (100 - int(md.group(1))) / 100
        if re.search(r"\b(?:total|altogether|in all|in total)\b", q):
            return _fmt_frac(total)
        # "how much does he spend/pay"
        if re.search(r"spend|pay|cost", q):
            return _fmt_frac(total)
        return _fmt_frac(total)

    # C) Prozent: "X% of Y" oder "X% of it/that/the N" (Pronomen -> erste Zahl)
    m = re.search(r"(\d+)\s*(?:%|percent)\s+of\s+(\d+)", q)
    base = None
    if m:
        base = Fraction(m.group(2))
    else:
        mp = re.search(r"(\d+)\s*(?:%|percent)\s+of\s+(?:it|that|this|"
                       r"the\s+\w+)", q)
        first = _num(q, 0)
        if mp and first is not None:
            m, base = mp, first
    if m is not None and base is not None:
        anteil = Fraction(m.group(1)) / 100 * base
        if re.search(r"left|remain|remaining", q):
            return _fmt_frac(base - anteil)
        return _fmt_frac(anteil)

    return None


def gold_answer(answer: str) -> Optional[str]:
    """Gold-Antwort: die Zahl nach '####'."""
    m = re.search(r"####\s*([\d.,]+)", answer)
    return m.group(1).replace(",", "") if m else None
