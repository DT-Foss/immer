"""pattern_bank.py — measured language-form patterns, weight-free.

The pattern bank is the data-driven replacement for the handwritten
connective/article tables in speak_causal.py: language FORM is mined from a
corpus by deterministic counting, knowledge stays in the .causal graph.

Extraction = delexicalization: every sentence is split into closed-class
function words (kept verbatim — they ARE the form) and content-word runs
(collapsed into slots "_"). What remains is a sentence skeleton:

    "if i take a piece of platinum and apply it"
     -> ('if', 'i', '_', 'a', '_', 'of', '_', 'and', '_', 'it')

The bank stores three measured inventories:

  skeletons : full delexicalized frames with counts + example fillers
              (granularity level 1; levels 2-3 build on the same data)
  openers   : sentence-initial function-word sequences with counts,
              classified by polarity (cause / contrast / additive / neutral)
              — these replace the handwritten _CAUSE_OPENERS/_CONTRAST
  fillers   : per-skeleton sample of what real text put into each slot
              (the raw material for later slot typing)

Sparsity is handled the project's own way: weak signals are AMPLIFIED, not
discarded. A pattern seen once gets its damped frequency confidence combined
(Möbius, F30) with the confidence of its more general parent pattern (the
skeleton minus its last token). A rare pattern that extends a frequent one
keeps a usable confidence; pure noise (rare pattern with rare parent) stays
weak. This is Kneser-Ney-style backoff expressed with the F30 combinator.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from .inference import moebius_confidence

SLOT = "_"

# Closed-class English function words. This is a LEXICON (like a list of
# mathematical constants), not learned content — the measured objects are the
# sequences built from it.
FUNCTION_WORDS = frozenset("""
a an the this that these those some any no every each either neither
i you he she it we they me him her us them my your his its our their
mine yours hers ours theirs myself yourself himself herself itself
ourselves themselves
and or but nor so yet for because although though while whereas if unless
until since when whenever where wherever after before once as than whether
not never also too very quite rather just only even still already again
then there here now however therefore thus hence moreover furthermore
nevertheless meanwhile instead otherwise indeed perhaps almost always
often sometimes
in on at by with from into onto upon of off about above below under over
between among through during within without against toward towards across
behind beyond near up down out around along past
is am are was were be been being have has had do does did will would shall
should may might must can could
what which who whom whose why how
""".split())

_CONTRAST_SEEDS = frozenset(
    "but however yet though although still nevertheless whereas otherwise".split())
_CAUSE_SEEDS = frozenset(
    "so therefore thus then hence consequently accordingly because".split())
_ADD_SEEDS = frozenset(
    "and also moreover furthermore besides again further now".split())


def tokenize_sentences(text: str, min_len: int = 3) -> List[List[str]]:
    """Deterministic sentence split + word tokenization (lowercase)."""
    sentences = []
    for raw in re.split(r"[.!?;:]+", text):
        toks = re.findall(r"[a-z']+", raw.lower())
        if len(toks) >= min_len:
            sentences.append(toks)
    return sentences


def delexicalize(tokens: List[str]) -> Tuple[Tuple[str, ...], List[str]]:
    """Collapse content-word runs into SLOT markers; keep function words.

    Returns (skeleton, fillers) where fillers[i] is the content run that
    went into the i-th slot."""
    skeleton: List[str] = []
    fillers: List[str] = []
    run: List[str] = []
    for tok in tokens:
        if tok in FUNCTION_WORDS:
            if run:
                skeleton.append(SLOT)
                fillers.append(" ".join(run))
                run = []
            skeleton.append(tok)
        else:
            run.append(tok)
    if run:
        skeleton.append(SLOT)
        fillers.append(" ".join(run))
    return tuple(skeleton), fillers


def extract_opener(tokens: List[str], max_len: int = 4) -> Optional[Tuple[str, ...]]:
    """Sentence-initial function-word run (followed by content) — the measured
    discourse connective."""
    run: List[str] = []
    for tok in tokens[:max_len]:
        if tok in FUNCTION_WORDS:
            run.append(tok)
        else:
            break
    if run and len(run) < len(tokens):
        return tuple(run)
    return None


def classify_opener(opener: Tuple[str, ...]) -> str:
    """Polarity class via closed seed sets; inventory + frequencies are measured."""
    s = set(opener)
    if s & _CONTRAST_SEEDS:
        return "contrast"
    if s & _CAUSE_SEEDS:
        return "cause"
    if s & _ADD_SEEDS:
        return "add"
    return "neutral"


def _damped_conf(count: int) -> float:
    """Frequency -> confidence in [0,1): 1x=0.33, 2x=0.5, 10x=0.83."""
    return count / (count + 2.0)


class PatternBank:
    """Measured form patterns: skeletons + openers, with F30 weak-signal backoff."""

    def __init__(self) -> None:
        self.skeletons: Counter = Counter()          # skeleton -> count
        self.openers: Counter = Counter()            # opener  -> count
        self.fillers: Dict[Tuple[str, ...], List[List[str]]] = {}
        self.n_sentences = 0

    # ------------------------------------------------------------ extraction
    def extract(self, text: str, max_filler_samples: int = 8) -> None:
        for toks in tokenize_sentences(text):
            self.n_sentences += 1
            skeleton, fills = delexicalize(toks)
            if SLOT in skeleton:
                self.skeletons[skeleton] += 1
                samples = self.fillers.setdefault(skeleton, [])
                if len(samples) < max_filler_samples:
                    samples.append(fills)
            opener = extract_opener(toks)
            if opener:
                self.openers[opener] += 1

    # ------------------------------------------------------- confidence (F30)
    def confidence(self, pattern: Tuple[str, ...], table: Counter) -> float:
        """Damped own frequency, Möbius-combined with the parent pattern
        (pattern minus last token) — weak-signal amplification, not discard."""
        own = _damped_conf(table[pattern])
        parent = pattern[:-1]
        if len(parent) >= 1 and table.get(parent, 0) > 0:
            return moebius_confidence(own, 0.5 * _damped_conf(table[parent]))
        return own

    # ------------------------------------------------------------- inventories
    def opener_inventory(self, polarity: Optional[str] = None,
                         min_count: int = 1) -> List[Dict]:
        out = []
        for opener, count in self.openers.most_common():
            if count < min_count:
                continue
            cls = classify_opener(opener)
            if polarity and cls != polarity:
                continue
            out.append({
                "tokens": list(opener),
                "count": count,
                "class": cls,
                "conf": round(self.confidence(opener, self.openers), 4),
            })
        return out

    def frames(self, n_slots: int, min_count: int = 1) -> List[Dict]:
        out = []
        for skeleton, count in self.skeletons.most_common():
            if count < min_count or skeleton.count(SLOT) != n_slots:
                continue
            out.append({
                "skeleton": list(skeleton),
                "count": count,
                "conf": round(self.confidence(skeleton, self.skeletons), 4),
                "filler_samples": self.fillers.get(skeleton, [])[:3],
            })
        return out

    # ------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        payload = {
            "n_sentences": self.n_sentences,
            "skeletons": [{"skeleton": list(k), "count": v,
                           "fillers": self.fillers.get(k, [])}
                          for k, v in self.skeletons.most_common()],
            "openers": [{"tokens": list(k), "count": v,
                         "class": classify_opener(k)}
                        for k, v in self.openers.most_common()],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "PatternBank":
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        bank = cls()
        bank.n_sentences = payload.get("n_sentences", 0)
        for item in payload["skeletons"]:
            key = tuple(item["skeleton"])
            bank.skeletons[key] = item["count"]
            if item.get("fillers"):
                bank.fillers[key] = item["fillers"]
        for item in payload["openers"]:
            bank.openers[tuple(item["tokens"])] = item["count"]
        return bank
