"""
fertig.mined — gesprochene Form mit gemessener Muster-Bank.

Wie pipeline.verbalize, aber die Diskurs-Verknüpfer kommen aus einer aus
einem Korpus gemessenen Muster-Bank (pattern_bank) statt aus einer
handgeschriebenen Tabelle. Deterministik-Trennung: FAKTEN deterministisch
(Graph-Lookup), FORM darf zufällig sein (echter RNG, tau-kontrolliert,
mit Rezenz-Penalty für Variation).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from . import sampler
from .pattern_bank import PatternBank
from .pipeline import _NEG_VERBS, _det, load_graph, walk_chain

DEFAULT_BANK = Path(__file__).resolve().parent.parent / "data" / "faraday_bank.json"

_SAFE_PREFIX = frozenset(
    "and but so now then therefore thus however yet hence consequently "
    "moreover also indeed again still accordingly nevertheless furthermore "
    "besides".split())
# adverbiale Verknüpfer, die mit folgendem Komma besser lesen
_COMMA_AFTER = frozenset(
    "however therefore thus hence consequently now indeed moreover "
    "furthermore nevertheless accordingly again then".split())


class MinedOpeners:
    """Verknüpfer-Wahl aus der gemessenen Inventarliste. Echter RNG — die
    Form darf zufällig sein, nur die Graph-Fakten müssen deterministisch sein."""

    def __init__(self, bank: PatternBank, seed: int | None = None):
        self.pools: dict = {}
        for cls in ("cause", "contrast", "add"):
            inv = [o for o in bank.opener_inventory(polarity=cls)
                   if all(t in _SAFE_PREFIX for t in o["tokens"])]
            self.pools[cls] = inv
        # cause-Pool ist in kleinen Korpora dünn — additive Verknüpfer tragen
        # kausal-neutrale Fortsetzung, als Backoff einmischen
        self.pools["cause"] = self.pools["cause"] + self.pools["add"]
        self.recent: list = []
        self.rng = np.random.default_rng(seed)

    def pick(self, polarity: str, tau: float = 0.8) -> str:
        pool = self.pools.get(polarity) or self.pools["add"]
        if not pool:
            return ""
        logits = np.array([np.log(o["count"]) + np.log(o["conf"] + 1e-9)
                           for o in pool])
        # starke Rezenz-Penalty: ein in den letzten Sätzen benutzter
        # Verknüpfer ist praktisch blockiert — Vielfalt schlägt rohe Frequenz
        for k, idx in enumerate(self.recent[-4:]):
            if idx < len(logits):
                logits[idx] -= 4.0 * (k + 1)
        # tau -> Temperatur wie im Kontraktions-Sampler, aber mit echtem RNG
        T = max(sampler.tau_to_temperature(tau), 1e-6)
        probs = np.exp((logits - logits.max()) / T)
        probs /= probs.sum()
        choice = int(self.rng.choice(len(pool), p=probs))
        self.recent.append(choice)
        toks = pool[choice]["tokens"]
        text = " ".join(toks)
        if toks[-1] in _COMMA_AFTER:
            text += ","
        return text


def _np(entity: str) -> str:
    """Artikel nur für kurze Nominalphrasen; lange extrahierte Phrasen nackt."""
    return _det(entity) if len(entity.split()) <= 3 else entity


def _clause_tail(outcome: str) -> str:
    """Outcome an eine klausel-lange Mechanismus-Phrase anhängen, nach Form."""
    first = outcome.split()[0]
    if first.endswith("ing"):
        return f"thereby {outcome}"
    if first.endswith("s") and not first.endswith(("ss", "us", "is")):
        return f"which {outcome}"
    return f"the result is {outcome}"


def verbalize_mined(hops: List[Tuple[int, int]], vocab: List[str],
                    mech: dict, openers: MinedOpeners, tau: float = 0.8) -> str:
    """Polaritäts-propagierende Verbalisierung mit gemessenen Verknüpfern.

    Zwei Mechanismus-Formen aus echten Graphen: kurze Verben ("causes")
    nehmen das SUBJ VERB OBJ-Gerüst; klausel-lange Mechanismen
    (wissenschaftliche Extraktion, z. B. DZA) nehmen SUBJ MECH-KLAUSEL — TAIL.

    Form-tau ist bewusst locker (0.8): die Fakten sind vom Graphen gepinnt,
    die Verknüpfer-Wahl darf explorieren — dort kommt die Flüssigkeits-Variation her.
    """
    if not hops:
        return "(kein Kausalpfad von dort.)"
    sents = []
    prev_neg = False
    for i, (a, b) in enumerate(hops):
        subj, obj = vocab[a], vocab[b]
        verb = mech.get((a, b), "leads to")
        cur_neg = bool(set(verb.split()) & _NEG_VERBS)
        clause = len(verb.split()) > 2
        op = "" if i == 0 else openers.pick(
            "contrast" if (cur_neg or (prev_neg and not cur_neg)) else "cause",
            tau)
        prefix = f"{op.capitalize()} " if op else ""
        if clause:
            s = f"{prefix}{_np(subj)} {verb} — {_clause_tail(obj)}."
        elif i > 0 and prev_neg and not cur_neg:
            # Polaritäts-Umschwung (pass2-Vorzeichenregel): die Relation über
            # das Subjekt selbst aussagen, nicht implizieren, die Kette
            # verbessere das Outcome
            s = f"{prefix}{_np(subj)} is exactly what {verb} {_np(obj)}."
        else:
            s = f"{prefix}{_np(subj)} {verb} {_np(obj)}."
        if not prefix:
            s = s[0].upper() + s[1:]
        sents.append(s)
        prev_neg = cur_neg
    return " ".join(sents)
