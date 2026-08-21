#!/usr/bin/env python3
"""
MVP-4 CONTRACT MINI-INTEGRATION TEST (plain asserts, no framework).

Verifies the two functions builder_run.py depends on:
  extract_validated(text) -> list[dict]   (trigger/mechanism/outcome +
                                            trigger_key/outcome_key)
  iter_windows(...)       -> yields (tape_pos, window_text, surprise, gated)

Run: python3 src/test_curator_yield_contract.py
"""
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import portable_organism as po
from curator_yield_run import extract_validated, iter_windows, normalize_entity, HFStreamWithText


def test_extract_validated():
    text = (
        "Smoking causes lung cancer. The heavy rain leads to flooding in low areas. "
        "Chronic stress worsens cardiovascular disease."
    )
    triplets = extract_validated(text)
    assert len(triplets) == 3, f"expected 3 triplets, got {len(triplets)}: {triplets}"
    required = {"trigger", "mechanism", "outcome", "trigger_key", "outcome_key"}
    for t in triplets:
        missing = required - set(t.keys())
        assert not missing, f"missing keys {missing} in {t}"
        assert t["trigger_key"] == normalize_entity(t["trigger"])
        assert t["outcome_key"] == normalize_entity(t["outcome"])
    print(f"[PASS] extract_validated: {len(triplets)} triplets, all required keys present")
    for t in triplets:
        print(f"    {t['trigger']!r} | {t['mechanism']!r} | {t['outcome']!r}"
              f" -> keys: {t['trigger_key']!r} / {t['outcome_key']!r}")
    return True


class FakeStream(HFStreamWithText):
    """Deterministic in-memory text source for iter_windows — no network.
    block is set SMALL (not the 8192 default) so a handful of short repeated
    docs actually fill a next_block() call; otherwise next_block() returns
    empty forever once docs are exhausted and the feeder spins forever."""

    def __init__(self, source, stoi, unk, docs_texts=None, block=256, skip_docs=0):
        super().__init__(source, stoi, unk, block=block, skip_docs=skip_docs)
        self._docs_texts = list(docs_texts or [])
        self._i = 0

    def _connect(self):
        pass  # no network — docs come from self._docs_texts

    def next_block(self):
        while len(self.pending) < self.block:
            if self._i >= len(self._docs_texts):
                self._i = 0   # wrap so the fake source never starves the feeder
            t = self._docs_texts[self._i]
            self._i += 1
            self.docs += 1
            if t.strip():
                lowered_chars, orig_idx = [], []
                for i, ch in enumerate(t):
                    lc = ch.lower()
                    lowered_chars.append(lc)
                    orig_idx.extend([i] * len(lc))
                orig_idx.append(len(t))
                t_lower = "".join(lowered_chars)
                _WORD_RE = re.compile(r"[a-zA-Z]+")
                matches = list(_WORD_RE.finditer(t_lower))
                toks = [self.stoi.get(m.group(0), self.unk) for m in matches]
                segs, prev_end = [], 0
                for m in matches:
                    oe = orig_idx[m.end()]
                    segs.append(t[prev_end:oe])
                    prev_end = oe
                self.pending.extend(toks)
                self.pending_segs.extend(segs)
        out_t, self.pending = self.pending[:self.block], self.pending[self.block:]
        out_s, self.pending_segs = self.pending_segs[:self.block], self.pending_segs[self.block:]
        return out_t, out_s


def test_iter_windows():
    docs = [
        "Smoking causes lung cancer in many long-term users of tobacco products.",
        "The heavy rain leads to flooding in low areas near the river basin.",
        "Chronic stress worsens cardiovascular disease outcomes over time.",
    ]

    def make_fake(source, stoi, unk, block=8192, skip_docs=0):
        return FakeStream(source, stoi, unk, docs_texts=docs, block=256, skip_docs=skip_docs)

    rows = list(iter_windows(chunks=10, window_tokens=32, batch=2, chunk_size=16,
                              stream_cls=make_fake))
    assert len(rows) == 10, f"expected 10 rows, got {len(rows)}"
    prev_pos = -1
    for tape_pos, text, surprise, gated in rows:
        assert isinstance(tape_pos, int), f"tape_pos not int: {type(tape_pos)}"
        assert isinstance(text, str), f"window_text not str: {type(text)}"
        assert isinstance(surprise, float), f"surprise not float: {type(surprise)}"
        assert isinstance(gated, bool), f"gated not bool: {type(gated)}"
        assert tape_pos > prev_pos, "tape_pos must strictly increase in stream order"
        prev_pos = tape_pos
        assert len(text) > 0, "window text must be non-empty once streamed"
    print(f"[PASS] iter_windows: {len(rows)} rows, tape_pos strictly increasing, "
          f"all fields typed correctly")
    for tape_pos, text, surprise, gated in rows[:3]:
        print(f"    tape_pos={tape_pos} surprise={surprise:.3f} gated={gated} "
              f"text={text[:60]!r}")
    return True


if __name__ == "__main__":
    ok = True
    ok &= test_extract_validated()
    ok &= test_iter_windows()
    print("ALL TESTS PASSED" if ok else "TESTS FAILED")
    sys.exit(0 if ok else 1)
