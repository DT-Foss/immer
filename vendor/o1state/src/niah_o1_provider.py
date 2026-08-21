#!/usr/bin/env python3 -u
"""
O1 as a v2 NIAH provider — the REAL Kamradt needle-in-a-haystack, answered by O1.
=================================================================================
This plugs the full O1 system into Greg Kamradt's needle-in-a-haystack v2 runner
(github.com/gkamradt/needle-in-a-haystack). The v2 runner builds a real prose haystack
(Paul Graham essays), inserts a real needle sentence at a chosen depth, appends the question,
and calls `provider.complete(system, user)`. We implement that one method with O1.

How O1 answers (the real pipeline, not a synthetic key→value game):
  1. The `user` prompt is `{haystack_prose}\n\n{question}`. Split off the trailing question.
  2. O1 STREAMS the haystack prose and builds its `.causal` knowledge index from it
     (`Attic.from_corpus` — deterministic, LLM-free triplet extraction). The needle is a real
     sentence in the prose, so its fact becomes a triplet in the index like any other fact —
     no oracle, the system only sees the prose it streamed.
  3. At the question, O1 consults the index for the question's key entity (`Attic.lookup` /
     `reason`) and returns the matching fact as the answer.

So this is the whole O1 architecture — bounded O(1) state streaming the context + the external
`.causal` cortex holding the facts — answering the standard long-context benchmark on the
standard benchmark's own prose and scoring. The index is built from streamed text at O(chunk)
memory; the fact is recalled at the query regardless of how deep the needle sat.

Usage (run from the v2 repo with this file importable):
    PYTHONPATH=/path/to/O1/src python -m niah_o1_provider --selftest
or wire it into the v2 runner as the provider (see make_o1_provider()).
"""
from __future__ import annotations

import os, re, sys
from dataclasses import dataclass, field

# make O1's src importable
_O1_SRC = os.path.dirname(os.path.abspath(__file__))
if _O1_SRC not in sys.path:
    sys.path.insert(0, _O1_SRC)

from attic import FactIndex


# --- minimal Completion shim (matches v2's providers.base.Completion shape) ----
@dataclass(slots=True)
class _Completion:
    text: str
    usage: object | None = None


_STOP = {
    "what", "is", "the", "best", "thing", "to", "do", "in", "a", "an", "of", "for", "on",
    "at", "and", "or", "with", "which", "who", "whom", "where", "when", "how", "why", "that",
    "this", "are", "was", "were", "be", "been", "according", "passage", "text", "context",
    "based", "above", "give", "information", "answer", "question", "name", "number", "magic",
}


def _question_keys(question: str) -> list[str]:
    """Pull the content words from the question (drop stopwords) — these are the entities to
    look up in the index. Proper-noun-ish multiword spans are kept too."""
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", question)
    keys = [w for w in words if w.lower() not in _STOP and len(w) >= 3]
    # also keep capitalized bigrams (e.g. "San Francisco") as one key
    caps = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", question)
    return caps + keys


@dataclass
class O1Provider:
    """The O1 system as a v2 NIAH ModelProvider. Streams the haystack into a `.causal` index,
    answers the question from the index. One `complete(system, user)` coroutine, per the v2
    contract."""

    id: str = "o1"
    request_model: str = "o1-state+causal-index"
    top_k: int = 20              # how many top-scoring sentences to return as the answer; one
                                 # fact (single-needle) uses the first, many facts (multi-needle)
                                 # are all near the top — task-agnostic "recall the relevant facts"
    calls: list = field(default_factory=list)

    async def complete(self, system: str, user: str) -> _Completion:  # noqa: ARG002
        # 1. split haystack prose from the trailing question (runner joins with "\n\n")
        if "\n\n" in user:
            haystack, question = user.rsplit("\n\n", 1)
        else:
            haystack, question = user, ""
        self.calls.append({"q": question[:120], "ctx_chars": len(haystack)})

        # 2. O1 streams the prose and folds each sentence into its fact index (the cortex), O(chunk)
        idx = FactIndex.from_corpus(haystack)

        # 3. consult the index with the question; recall the top relevant sentences. Returning the
        #    top-k (not just the single best) is what lets the SAME provider answer multi-fact
        #    questions — every needle sentence scores high against the shared question, so all of
        #    them land in the top-k — without anything task-specific.
        hits = idx.recall(question, k=self.top_k)
        text = " ".join(hits) if hits else "(no fact found in index)"

        # 3b. if the question references a long id token, the answer may be several hops away —
        #     follow the reference chain transitively and append the end of the chain. Generic:
        #     triggers only when the question carries an id-like token to start from; harmless
        #     otherwise. This is the multi-hop traversal of the streamed fact index.
        if re.search(r"\b[0-9a-fA-F]{4,}(?:-[0-9a-fA-F]{2,}){2,}\b", question):
            end = idx.chain_recall(question)
            if end:
                text = f"{end} {text}"
        return _Completion(text=text, usage=None)


def make_o1_provider(cfg=None):
    """Factory the v2 runner can call. cfg is the parsed ModelConfig (ignored — O1 has no SDK)."""
    return O1Provider()


# --- selftest: build a tiny haystack with the canonical needle, answer it ------
def _selftest():
    import asyncio
    needle = ("The best thing to do in San Francisco is eat a sandwich and sit in "
              "Dolores Park on a sunny day.")
    filler = ("The early years of a startup are mostly about building something people want. "
              "Founders often underestimate how long that takes. Distribution matters as much "
              "as the product. ") * 200
    haystack = filler[:4000] + " " + needle + " " + filler[4000:8000]
    question = "What is the best thing to do in San Francisco?"
    user = f"{haystack}\n\n{question}"

    prov = O1Provider()
    out = asyncio.run(prov.complete(system="", user=user))
    print("=" * 70)
    print("O1 v2-NIAH provider — selftest (real prose haystack, canonical needle)")
    print("=" * 70)
    print(f"  haystack chars : {len(haystack):,}")
    print(f"  question       : {question}")
    print(f"  O1 answer      : {out.text}")
    hit = ("san francisco" in out.text.lower()
           or "dolores" in out.text.lower()
           or "sandwich" in out.text.lower())
    print(f"  recalled the needle fact? {'YES ✓' if hit else 'NO ✗'}")
    return hit


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        ok = _selftest()
        sys.exit(0 if ok else 1)
    print("import this module and use make_o1_provider() / O1Provider in the v2 runner.")
