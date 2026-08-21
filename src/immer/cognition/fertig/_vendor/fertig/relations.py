"""
fertig.relations — schema-getriebene Relations-Extraktion.

Nutzung: Die Extraktion mappt Oberflächenformen IN das geschlossene
Primitiv-Schema (fertig.primitives.RELATIONS). Der Graph speichert nur
kanonische Primitiv-Namen; QA matcht auf dem Schema. Keine wachsende
Relations-Liste mehr — neue Oberflächenformen sind Lexikon-Einträge,
keine neuen Relationen.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from . import primitives
from .sources import _negated, _np, _ok_np, extract_causal


def extract_relations(text: str, base_conf: float = 0.40,
                      max_triplets: int = 40
                      ) -> List[Tuple[str, str, str, float]]:
    """Schema-getrieben: für jede Primitiv-Klasse mit Mustern werden
    Sätze gescannt; die Mechanismen sind kanonische Primitiv-Namen.

    Sonderfälle:
      created_in   : Subjekt vor dem Verb, Jahr als Objekt
      comparative  : Adjektiv + Vergleichsobjekt, Primitiv = adj_than
    Kausale Sätze laufen durch den gemeinsamen Kern (extract_causal) mit
    kanonischer Normalisierung.
    """
    out: List[Tuple[str, str, str, float]] = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if _negated(sent):
            continue
        for name, spec in primitives.RELATIONS.items():
            patterns = spec.get("patterns")
            if not patterns:
                continue
            for pat in patterns:
                m = re.search(pat, sent, flags=re.I)
                if not m:
                    continue
                if name == "created_in":
                    year = m.group(2)
                    before = sent[:m.start()].strip()
                    subj = _np(re.split(r"[,;]|(?:^|\s)(?:in|with|by) ",
                                        before)[-1])
                    if _ok_np(subj):
                        out.append((subj, "created_in", year,
                                    spec["conf"]))
                    break
                if name == "comparative":
                    adj, obj = m.group(1).lower(), m.group(2)
                    before = sent[:m.start()].strip()
                    subj = _np(re.split(r"[,;]", before)[-1])
                    if _ok_np(subj) and len(obj.split()) <= 8:
                        out.append((subj, adj + "_than", _np(obj),
                                    spec["conf"]))
                    break
                # Objekt-Muster: Subjekt bis zum Verb
                vm = re.search(r"\b(?:is|are|consists|contains|includes|"
                               r"comprises|can)\s+", sent)
                if not vm:
                    break
                before = sent[:vm.start()].strip()
                subj = _np(re.split(r"[,;]", before)[-1])
                obj = _np(m.group(1))
                if _ok_np(subj) and _ok_np(obj):
                    out.append((subj, name, obj, spec["conf"]))
                break
    # kausaler Kern (kanonisch normalisiert)
    for a, b, c, conf in extract_causal(text, base_conf=base_conf * 0.9):
        out.append((a, primitives.normalize_mechanism(b), c, conf))
    # Dedupe (max-Konfidenz)
    best: Dict[Tuple[str, str, str], float] = {}
    for a, b, c, conf in out:
        key = (a, b, c)
        best[key] = max(best.get(key, 0.0), conf)
    return [(a, b, c, conf) for (a, b, c), conf in best.items()][:max_triplets]
