#!/usr/bin/env python3 -u
"""
SURPRISE IS A KNOWLEDGE FILTER, OR IT IS NOT (MS-R / P58).

The distiller's ground question with a fully deterministic instrument:
stream WT-103 through the organism, take the top-M post-ignition chunk
positions by gate surprise, expand to matched 128-token windows, and
compare their FIRST-EVER token-type and bigram rate against M
seeded-random windows from the same stream — plus a redundancy read
(share of bigrams already seen ≥5 times). If surprise ≈ random, the
filter claim dies before anything is built on it.
"""
import argparse
import json
import os
import random
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch
torch.set_num_threads(1)

import portable_organism as po
from source_swap_run import HFStream


def window_stats(tokens, start, wlen, first_tok_pos, first_bi_pos, bi_count_at):
    """Deterministic novelty/redundancy of window [start, start+wlen):
    first-ever = registry says this window position IS the first occurrence."""
    new_types = new_bis = 0
    red_bis = 0
    n_bi = 0
    for i in range(start, min(start + wlen, len(tokens))):
        t = tokens[i]
        if first_tok_pos.get(t) == i:
            new_types += 1
        if i + 1 < len(tokens):
            b = (t, tokens[i + 1])
            n_bi += 1
            if first_bi_pos.get(b) == i:
                new_bis += 1
            elif bi_count_at.get((b, i), 0) >= 5:
                red_bis += 1
    wl = min(wlen, len(tokens) - start)
    return {"new_type_rate": new_types / max(1, wl),
            "new_bigram_rate": new_bis / max(1, n_bi),
            "first_ever_rate": (new_types + new_bis) / max(1, wl + n_bi),
            "redundancy": red_bis / max(1, n_bi)}


def main():
    ap = argparse.ArgumentParser(description="P58: surprise as knowledge filter")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--chunks", type=int, default=3000)
    ap.add_argument("--top-m", type=int, default=150)
    ap.add_argument("--window-tokens", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--q", type=float, default=0.75)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--min-window", type=int, default=100)
    ap.add_argument("--ignition-chunks", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "surprise_filter.json"))
    args = ap.parse_args()
    if args.smoke:
        args.chunks, args.top_m = 400, 30

    po.D_MODEL, po.BATCH, po.CHUNK = args.d_model, args.batch, args.chunk_size
    po.GATE_Q, po.GATE_WINDOW, po.MIN_WINDOW, po.IGNITION_CHUNKS = \
        args.q, args.window, args.min_window, args.ignition_chunks
    vocab, stoi, unk, mask, val_ids = po.get_vocab()
    V = len(vocab)
    K, B = po.CHUNK, po.BATCH

    torch.manual_seed(args.seed)
    org = po.Organism("filter", V, mask, seed=args.seed)
    stream = HFStream("wt103", stoi, unk)
    feeder = po.ChunkFeeder(stream, B, K)

    # lane-0 token tape + per-chunk surprise; the tape is the stream ORDER
    tape = []
    chunk_surprise = []          # (chunk_idx, s, tape_pos_of_chunk_start)
    t0 = time.time()
    for ci in range(1, args.chunks + 1):
        x, y = feeder.next_xy()
        s, gated, nll = org.step_gated(x, y)
        chunk_surprise.append((ci, float(s), len(tape)))
        tape.extend(int(v) for v in x[0])          # lane 0 carries the tape
        if ci % 1000 == 0:
            print(f"[stream] {ci}/{args.chunks} | s {s:.3f} | tape {len(tape):,} tok "
                  f"| {time.time()-t0:.0f}s", flush=True)

    # deterministic registries over the tape (first occurrence positions,
    # and bigram prior-count at each position)
    first_tok_pos, first_bi_pos = {}, {}
    bi_count_at, bi_counter = {}, {}
    for i, t in enumerate(tape):
        if t not in first_tok_pos:
            first_tok_pos[t] = i
        if i + 1 < len(tape):
            b = (t, tape[i + 1])
            if b not in first_bi_pos:
                first_bi_pos[b] = i
            c = bi_counter.get(b, 0)
            if c >= 5:
                bi_count_at[(b, i)] = c
            bi_counter[b] = c + 1

    post = [c for c in chunk_surprise if c[0] > args.ignition_chunks]
    top = sorted(post, key=lambda c: -c[1])[:args.top_m]
    rng = random.Random(args.seed + 7)
    rand1 = rng.sample(post, min(args.top_m, len(post)))
    rng2 = random.Random(args.seed + 99)
    rand2 = rng2.sample(post, min(args.top_m, len(post)))

    def group_stats(sel):
        rows = [window_stats(tape, pos, args.window_tokens,
                             first_tok_pos, first_bi_pos, bi_count_at)
                for _, _, pos in sel]
        med = lambda k: sorted(r[k] for r in rows)[len(rows) // 2]
        return {k: round(med(k), 6) for k in rows[0]} if rows else {}

    g_sur = group_stats(top)
    g_r1 = group_stats(rand1)
    g_r2 = group_stats(rand2)

    ratio_a = g_sur["first_ever_rate"] / max(1e-9, g_r1["first_ever_rate"])
    ratio_b = g_sur["redundancy"] / max(1e-9, g_r1["redundancy"]) if g_r1["redundancy"] else None
    ratio_a2 = g_sur["first_ever_rate"] / max(1e-9, g_r2["first_ever_rate"])
    out = {"p58": True, "smoke": args.smoke,
           "cadence": {"d_model": po.D_MODEL, "batch": B, "chunk": K,
                       "q": po.GATE_Q, "window": po.GATE_WINDOW,
                       "min_window": po.MIN_WINDOW, "ignition_chunks": po.IGNITION_CHUNKS},
           "config": {"chunks": args.chunks, "top_m": args.top_m,
                      "window_tokens": args.window_tokens, "substrate": "wt103"},
           "surprise_windows": g_sur, "random_windows": g_r1, "random_seed2": g_r2,
           "p58a_first_ever_ratio": round(ratio_a, 4),
           "p58a_pass": bool(ratio_a >= 1.5),
           "p58b_redundancy_ratio": round(ratio_b, 4) if ratio_b is not None else None,
           "p58b_pass": bool(ratio_b is not None and ratio_b <= 0.8),
           "p58c_seed2_ratio": round(ratio_a2, 4),
           "p58c_pass": bool((ratio_a >= 1.5) == (ratio_a2 >= 1.5))}
    path = args.out if not args.smoke else args.out.replace(".json", "_smoke.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[p58] first-ever ratio {ratio_a:.3f} (bar 1.5) | redundancy ratio "
          f"{ratio_b if ratio_b is not None else 'n/a'} (bar 0.8) | seed2 {ratio_a2:.3f} | "
          f"a:{out['p58a_pass']} b:{out['p58b_pass']} c:{out['p58c_pass']} -> {path}", flush=True)


if __name__ == "__main__":
    main()
