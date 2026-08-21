#!/usr/bin/env python3 -u
"""
CLOSED LOOP — the gold edge, closed: the attic's answer flows BACK into the living stream.
==========================================================================================
The bridge proved half the loop: the GSSM O(1) state finds its gaps (high surprise) and the
attic resolves them. But the path just sat there. Here we close it:

  stream → GAP (surprise spike, attic-resolvable) → consult attic → feed the returned PATH
  through the state as tokens → THEN continue reading the original text → does the model's
  surprise on the next real tokens DROP, vs the same gap WITHOUT consultation?

If yes: the attic helped the living now *in flight* — David's "I go to the attic, look it up,
now I know it for the moment." The state did not have to learn (no gradient); it consulted an
external index and carried the answer forward in its O(1) state across the next tokens.

Honest control (the load-bearing comparison): IDENTICAL gap and IDENTICAL continuation, run
twice from the SAME pre-gap state — once WITH the path injected, once WITHOUT. The only
difference is the consultation. We own the state, so the two runs are exactly comparable.
"""
import os, sys, json, argparse, re
sys.path.insert(0, "reference"); sys.path.insert(0, "src")

import torch
import torch.nn as nn
torch.backends.mps.is_available = lambda: False
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

from streaming_train import StreamingNoPELM
from length_extrap_v2 import load_wikitext2, build_vocab, tokenize
from attic import Attic


def _clone_state(states):
    return None if states is None else [s.clone() for s in states]


@torch.no_grad()
def _read(model, token_ids, states, dev):
    """Run token_ids through the model from `states`, return (per-token surprise list, new states).
    Surprise[i] = NLL of predicting token_ids[i+1] given everything up to i."""
    if len(token_ids) < 2:
        x = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(dev)
        _, st = model(x, states)
        return [], st
    x = torch.tensor(token_ids[:-1], dtype=torch.long).unsqueeze(0).to(dev)
    y = torch.tensor(token_ids[1:], dtype=torch.long).unsqueeze(0).to(dev)
    logits, st = model(x, states)
    nll = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                      y.reshape(-1), reduction="none")
    return [float(v) for v in nll], st


@torch.no_grad()
def _advance(model, token_ids, states, dev):
    """Just push tokens through to update the state (no scoring) — used to inject the attic path."""
    if not token_ids:
        return states
    x = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(dev)
    _, st = model(x, states)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--causal", default="demo/causal_assets/wikitext.causal")
    ap.add_argument("--ckpt", default="results/streaming_train_c.pt")
    ap.add_argument("--surprise-thresh", type=float, default=9.0, help="a gap = surprise above this")
    ap.add_argument("--lookahead", type=int, default=12, help="tokens scored after the gap")
    ap.add_argument("--max-gaps", type=int, default=40)
    ap.add_argument("--rich", action="store_true", help="also consider multi-hop chains (reason)")
    ap.add_argument("--context", type=int, default=20, help="local-context window for relevance gate")
    ap.add_argument("--out", default="results/closed_loop.json")
    args = ap.parse_args()
    dev = torch.device("cpu")

    attic = Attic.from_file(args.causal)
    print(f"[attic] {attic.stats()}")

    # the trained GSSM state
    ck = torch.load(args.ckpt, weights_only=False)
    model = StreamingNoPELM(ck["vocab_size"], ck["mask_idx"], d_model=ck["d_model"],
                            n_layers=2, n_heads=4, d_head=ck["d_model"] // 4,
                            seq_len=32, dropout=0.0, causal=True).to(dev).eval()
    model.load_state_dict(ck["state_dict"])
    stoi, unk, mask = ck["stoi"], ck["unk"], ck["mask_idx"]
    print(f"[gssm] trained state {args.ckpt} (vocab {ck['vocab_size']})")

    def toks(text):
        return [stoi.get(w, unk) for w in re.findall(r"[a-zA-Z]{2,}", text.lower())]

    # the perceived stream: the attic's own domain text (where the model meets jargon = gaps)
    words = []
    for t in attic.triplets[:600]:
        words.extend(re.findall(r"[a-zA-Z]{2,}",
                     f"{t['trigger']} {t['mechanism']} {t['outcome']}".lower()))
    ids = [stoi.get(w, unk) for w in words]

    # walk the stream; at each gap, fork into WITH-consult vs WITHOUT-consult and compare the
    # surprise over the next `lookahead` real tokens.
    results = []
    states = None
    i = 0
    L = args.lookahead
    while i < len(ids) - L - 1 and len(results) < args.max_gaps:
        # score the single next token from the current carried state to detect a gap
        gap_surprise, _ = _read(model, ids[i:i + 2], _clone_state(states), dev)
        w = words[i + 1] if i + 1 < len(words) else ""
        is_gap = gap_surprise and gap_surprise[0] > args.surprise_thresh and attic.knows(w) and len(w) >= 4
        if not is_gap:
            # advance one token and continue
            states = _advance(model, ids[i:i + 1], states, dev)
            i += 1
            continue

        pre_state = _clone_state(states)            # the EXACT pre-gap state — fork point
        cont = ids[i + 1:i + 1 + L]                 # the real continuation after the gap word

        # WITHOUT consult: read the continuation straight from the pre-gap state
        s_without, _ = _read(model, [ids[i]] + cont, _clone_state(pre_state), dev)

        # WITH consult: pick the MOST RELEVANT attic path for THIS gap, then inject it. A rich attic
        # has many paths per term — most off-topic. Relevance = word-overlap with the LOCAL context
        # (the window around the gap, which is what the model is actually reading). This is the
        # convergence/relevance gate (David's IRI gap-priority), not "take the first hit".
        local_ctx = set(words[max(0, i - args.context):i + 1 + L])
        cands = []
        if args.rich:
            cands = [c["chain"] for c in attic.reason(w, k=8)]
        cands += [p["path"] for p in attic.lookup(w, k=8)]
        def relevance(txt):
            tw = set(re.findall(r"[a-z]{3,}", txt.lower()))
            return len(tw & local_ctx)
        path_text = max(cands, key=relevance) if cands else None
        path_ids = toks(path_text) if path_text else []
        st_after_inject = _advance(model, path_ids, _clone_state(pre_state), dev)
        s_with, _ = _read(model, [ids[i]] + cont, st_after_inject, dev)

        avg_without = sum(s_without) / max(1, len(s_without))
        avg_with = sum(s_with) / max(1, len(s_with))
        results.append({"gap": w, "gap_surprise": round(gap_surprise[0], 3),
                        "path": (path_text or "")[:80] if path_text else None,
                        "cont_surprise_without": round(avg_without, 4),
                        "cont_surprise_with": round(avg_with, 4),
                        "drop": round(avg_without - avg_with, 4)})
        # advance the REAL stream by one (the loop continues from the un-consulted state, as a
        # consultation is a side-trip to the attic, not part of the perceived stream)
        states = _advance(model, ids[i:i + 1], states, dev)
        i += 1

    drops = [r["drop"] for r in results]
    mean_drop = sum(drops) / max(1, len(drops))
    helped = sum(1 for d in drops if d > 0)
    out = {"causal": args.causal, "n_gaps": len(results),
           "mean_surprise_drop": round(mean_drop, 4),
           "gaps_helped": f"{helped}/{len(results)}",
           "results": results}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    print(f"\n── CLOSED LOOP: does consulting the attic lower post-gap surprise? ──")
    for r in results[:12]:
        arrow = "↓" if r["drop"] > 0 else "↑"
        print(f"  gap '{r['gap']}' (s={r['gap_surprise']:.1f}): "
              f"without {r['cont_surprise_without']:.2f} → with {r['cont_surprise_with']:.2f} "
              f"{arrow} {r['drop']:+.3f}")
    print(f"\n  MEAN surprise drop after consulting: {mean_drop:+.4f} over {len(results)} gaps "
          f"({helped}/{len(results)} helped).")
    verdict = ("THE ATTIC HELPS THE LIVING STREAM IN FLIGHT — consultation lowers post-gap surprise"
               if mean_drop > 0.02 else
               "no in-flight benefit on this setup (the path passes through but doesn't lower next-token surprise)")
    print(f"  → {verdict}")
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
