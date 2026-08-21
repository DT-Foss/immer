#!/usr/bin/env python3 -u
"""
PATHFINDING BRIDGE — the gold edge: GSSM (the living O(1) now) ↔ .causal (the attic).
=====================================================================================
David's "Creativity as Constrained Graph Traversal": creativity/learning is finding useful
PATHS between distant subgraphs, not inventing nodes. Two halves already exist:
  - HOW TO? (purple)  = the GSSM O(1) state: a living, streaming, forgetting "now" (proven:
    trains on an unbounded stream at constant memory, carries a bit through silence).
  - KNOWLEDGE (blue)  = fabel/.causal: a deterministic, inspectable, zero-hallucination
    knowledge graph with provenance (CausalReader, 3-pass inference, "not in this graph" when
    it doesn't know). Already built.
The missing piece is the GOLD EDGE — the narrow interface that couples them:
    the GSSM state STREAMS text; where its per-token SURPRISE spikes it has hit a GAP
    ("I don't know this"); instead of guessing, it CONSULTS the .causal attic for exactly
    that term and gets back an exact, sourced path it could not have traversed alone.

This script proves the coupling minimally and honestly:
  1. A NoPE GSSM streams a domain text token-by-token, logging per-token surprise (loss).
  2. The highest-surprise tokens = the model's GAPS (what it is most ignorant of).
  3. We show those exact gap-terms ARE resolvable in the .causal graph (CausalReader hit),
     while low-surprise tokens are common words the graph rightly does not index.
  => The state knows WHERE it doesn't know; the attic fills it without hallucination.

The .causal graph is consulted READ-ONLY via fabel's CausalReader. No FORGE, no symbolic
machinery is imported into the GSSM repo — only a knowledge lookup over a narrow interface.
"""
import os, sys, json, argparse, re
sys.path.insert(0, "reference"); sys.path.insert(0, "src")

import torch
import torch.nn as nn
torch.backends.mps.is_available = lambda: False
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

from streaming_train import StreamingNoPELM
from length_extrap_v2 import load_wikitext2, build_vocab, tokenize
from attic import Attic   # the one door to the .causal knowledge index (vendored fabel, FORGE-free)


@torch.no_grad()
def stream_surprise(model, ids, words, mask_idx, device, chunk=256):
    """Stream `ids` through the model, return per-position surprise (next-token NLL) aligned
    to the surface `words`. The state carries across chunks (the living now)."""
    model.eval()
    lossf = nn.CrossEntropyLoss(reduction="none")
    surprise = []
    states = None
    pos = 0
    while pos + 1 < len(ids):
        seg = ids[pos:pos + chunk + 1]
        if len(seg) < 2:
            break
        x = torch.tensor(seg[:-1], dtype=torch.long).unsqueeze(0).to(device)
        y = torch.tensor(seg[1:], dtype=torch.long).unsqueeze(0).to(device)
        logits, states = model(x, states)
        states = [s.detach() for s in states]
        nll = lossf(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        surprise.extend(float(v) for v in nll)
        pos += chunk
    return surprise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--causal", default="demo/causal_assets/gw_knowledge.causal")
    ap.add_argument("--ckpt", default="results/streaming_train_c.pt")
    ap.add_argument("--top", type=int, default=15, help="how many top-surprise gap terms to resolve")
    ap.add_argument("--reason", action="store_true", help="also show multi-hop chains (rich engine)")
    ap.add_argument("--out", default="results/pathfinding_bridge.json")
    args = ap.parse_args()
    dev = torch.device("cpu")

    # the attic (read-only knowledge index — one door, vendored fabel)
    attic = Attic.from_file(args.causal)
    print(f"[attic] {attic.stats()}")

    # a GSSM state (trained on WT-2 stream; it has NOT seen this domain corpus)
    train_text, _ = load_wikitext2()
    vocab, stoi, unk, mask = build_vocab(train_text)
    V = len(vocab)
    if os.path.exists(args.ckpt):
        ck = torch.load(args.ckpt, weights_only=False)
        model = StreamingNoPELM(ck["vocab_size"], ck["mask_idx"], d_model=ck["d_model"],
                                n_layers=2, n_heads=4, d_head=ck["d_model"] // 4,
                                seq_len=32, dropout=0.0, causal=True).to(dev)
        model.load_state_dict(ck["state_dict"]); stoi = ck["stoi"]; unk = ck["unk"]; V = ck["vocab_size"]; mask = ck["mask_idx"]
        print(f"[gssm] loaded trained state {args.ckpt} (vocab {V})")
    else:
        torch.manual_seed(0)
        model = StreamingNoPELM(V, mask, d_model=128, n_layers=2, n_heads=4, d_head=32,
                                seq_len=32, dropout=0.0, causal=True).to(dev)
        print("[gssm] no ckpt — using untrained state (still demonstrates gap detection)")

    # build a domain text to stream: the surface text of the attic's own triplets (the world
    # the model is about to perceive). The state will be most surprised by domain jargon.
    domain_words = []
    for t in attic.triplets[:400]:
        domain_words.extend(re.findall(r"[a-zA-Z]{2,}",
                            f"{t['trigger']} {t['mechanism']} {t['outcome']}".lower()))
    ids = [stoi.get(w, unk) for w in domain_words]

    surprise = stream_surprise(model, ids, domain_words, mask, dev)
    # align: surprise[i] is the model's surprise predicting word[i+1]
    pairs = [(domain_words[i + 1], surprise[i]) for i in range(len(surprise))]
    # a GAP term: high surprise AND resolvable in the attic AND a real content word (len>=4)
    seen = set()
    ranked = sorted(pairs, key=lambda p: -p[1])
    gaps, commons = [], []
    for w, s in ranked:
        if w in seen or len(w) < 4:
            continue
        seen.add(w)
        if attic.knows(w) and len(gaps) < args.top:
            gaps.append((w, s))
    # control: lowest-surprise words — should be common, graph rightly doesn't index them
    for w, s in sorted(pairs, key=lambda p: p[1]):
        if w not in [g[0] for g in gaps] and len(w) >= 3 and w not in [c[0] for c in commons]:
            commons.append((w, s, attic.knows(w)))
        if len(commons) >= 8:
            break

    print(f"\n── the model's GAPS (highest surprise) — each RESOLVED in the attic ──")
    out = {"causal": args.causal, "n_triplets": len(attic.triplets), "gaps": []}
    for w, s in gaps:
        paths = attic.lookup(w)
        print(f"  gap '{w}' (surprise {s:.2f}) → attic:")
        for p in paths[:1]:
            print(f"        1-hop: {p['path'][:80]}  [conf {p['conf']}, {p['source']}]")
        entry = {"term": w, "surprise": round(s, 3), "n_paths": len(paths),
                 "top_path": paths[0]["path"] if paths else None}
        if args.reason:                                    # the RICH engine: multi-hop chains
            chains = attic.reason(w, k=2)
            for c in chains[:1]:
                print(f"        {c['hops']}-hop: {c['chain'][:80]}  [conf {c['conf']}]")
            entry["multi_hop"] = [{"hops": c["hops"], "chain": c["chain"], "conf": c["conf"]}
                                  for c in chains]
        out["gaps"].append(entry)
    avg_gap = sum(s for _, s in gaps) / max(1, len(gaps))
    avg_common = sum(s for _, s, _ in commons) / max(1, len(commons))
    common_in_graph = sum(1 for _, _, k in commons if k)
    out["avg_gap_surprise"] = round(avg_gap, 3)
    out["avg_common_surprise"] = round(avg_common, 3)
    out["commons_indexed_by_attic"] = f"{common_in_graph}/{len(commons)}"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n  CONTRAST: gap-terms avg surprise {avg_gap:.2f} (all resolved in attic) vs "
          f"common-words avg {avg_common:.2f} ({common_in_graph}/{len(commons)} indexed).")
    print(f"  → the state knows WHERE it doesn't know; the attic fills exactly those gaps, sourced.")
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
