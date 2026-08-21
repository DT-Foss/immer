"""
fertig.hsslm.bpe — leichter BPE-Tokenizer (auf dem Formen-Korpus gelernt).

Char-Level ist der Grund, warum HSSLM nach Sätzen degeneriert: ein
"Satz" ist 30-60 Zeichen, die Struktur liegt jenseits der lokalen
Fenster. BPE fasst häufige Zeichenfolgen zu Tokens — das Modell sieht
Wort-Struktur statt Buchstaben-Suppe.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


class BPETokenizer:
    """Lernbarer BPE: merges aus dem Korpus, deterministisch."""

    def __init__(self, vocab_size: int = 400):
        self.vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self._char_vocab: Dict[str, int] = {}

    # -- Lernen ------------------------------------------------------------

    def _init_chars(self, text: str):
        chars = set(text)
        self._char_vocab = {c: i for i, c in enumerate(sorted(chars))}
        self.vocab = dict(self._char_vocab)
        self.itos = {i: c for c, i in self.vocab.items()}

    def _get_stats(self, words: List[List[str]]) -> Counter:
        stats = Counter()
        for word in words:
            for pair in zip(word, word[1:]):
                stats[pair] += 1
        return stats

    def _merge(self, word: List[str], pair: Tuple[str, str]) -> List[str]:
        out, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and \
                    word[i + 1] == pair[1]:
                out.append(pair[0] + pair[1])
                i += 2
            else:
                out.append(word[i])
                i += 1
        return out

    def fit(self, text: str):
        self._init_chars(text)
        words = [list(w) for w in re.findall(r"\S+|\s+", text)]
        while len(self.vocab) < self.vocab_size:
            stats = self._get_stats(words)
            if not stats:
                break
            best = stats.most_common(1)[0][0]
            self.merges.append(best)
            self.vocab[best[0] + best[1]] = len(self.vocab)
            self.itos[len(self.vocab) - 1] = best[0] + best[1]
            words = [self._merge(w, best) for w in words]

    # -- Encode/Decode ------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        ids = []
        for word in re.findall(r"\S+|\s+", text):
            toks = list(word)
            for a, b in self.merges:
                toks = self._merge(toks, (a, b))
            for t in toks:
                ids.append(self.vocab.get(t, self._char_vocab.get(t, 0)))
        return ids

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos.get(i, "?") for i in ids)

    @property
    def VOCAB_SIZE(self) -> int:
        return len(self.vocab)
