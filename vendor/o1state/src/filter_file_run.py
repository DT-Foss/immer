#!/usr/bin/env python3 -u
"""
THE FILTER EARNS ITS FILE, OR IT DOES NOT (MS-Y / P64).

P58: the gate points at novelty. P55: the frozen file stores entries.
The composed question: is surprise-HARVESTED content worth more per span
once distilled into a file? One producer (identical training), two
harvest policies on the same pass: SURPRISE harvests at gated chunks
(the P47/P55 recipe), RANDOM harvests at seeded-random chunks at a
matched rate with random span centers, counts trimmed to match exactly.
Both files frozen (sha256). Three consumer twins from one init: dosed
replay of file-S, of file-R, and without. Clauses: (a) file-S diffuse
heldout gain >= 1.5x file-R's (and positive); (b) BOTH files pass keyed
recall of their own entries at the P55 bar (>= 0.05); (c) matched span
counts, same dose, both shas recorded.
"""
import argparse
import copy
import hashlib
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
from knowledge_file_run import build_c4_eval, heldout_c4
from keyed_file_run import completion_nll, make_organism


def random_spans(x, rng, span_half=32, max_per_chunk=2):
    """Random-center spans, geometry-matched to po.harvest_spans."""
    B, K = x.shape
    rows = rng.sample(range(B), min(max_per_chunk, B))
    out = []
    for b in rows:
        k = rng.randrange(K)
        lo, hi = max(0, k - span_half), min(K, k + span_half + 1)
        span = x[b, lo:hi].tolist()
        if len(span) > 1:
            out.append([int(v) for v in span])
    return out


def main():
    ap = argparse.ArgumentParser(description="P64: the filter earns its file")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--producer-chunks", type=int, default=1500)
    ap.add_argument("--consumer-chunks", type=int, default=1500)
    ap.add_argument("--replay-every", type=int, default=25)
    ap.add_argument("--n-probe", type=int, default=100)
    ap.add_argument("--r-prob", type=float, default=0.30,
                    help="random-harvest chunk probability; overshoots the "
                         "gate rate, counts are trimmed to match after")
    ap.add_argument("--producer-offset", type=int, default=0)
    ap.add_argument("--reader-offset", type=int, default=1_000_000)
    ap.add_argument("--eval-offset", type=int, default=1_200_000)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--q", type=float, default=0.75)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--min-window", type=int, default=100)
    ap.add_argument("--ignition-chunks", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "filter_file.json"))
    args = ap.parse_args()
    if args.smoke:
        args.producer_chunks, args.consumer_chunks = 120, 120
        args.replay_every, args.n_probe = 10, 20

    po.D_MODEL, po.BATCH, po.CHUNK = args.d_model, args.batch, args.chunk_size
    po.GATE_Q, po.GATE_WINDOW, po.MIN_WINDOW, po.IGNITION_CHUNKS = \
        args.q, args.window, args.min_window, args.ignition_chunks
    vocab, stoi, unk, mask, val_ids = po.get_vocab()
    V = len(vocab)
    K = po.CHUNK
    evX, evY = build_c4_eval(stoi, unk, args.eval_offset, po.EVAL_TOKENS, po.CHUNK)

    # ── one producer, two harvest policies on the same pass ────────────────
    prod = make_organism(args.seed, V, mask, "producer")
    stream = po.C4Stream(stoi, unk, skip_docs=args.producer_offset)
    feeder = po.ChunkFeeder(stream, po.BATCH, po.CHUNK)
    r_rng = random.Random(args.seed + 555)
    spans_S, spans_R = [], []
    n_gated = 0
    t0 = time.time()
    for _ in range(args.producer_chunks):
        x, y = feeder.next_xy()
        s, gated, nll = prod.step_gated(x, y)
        if gated:
            n_gated += 1
            spans_S.extend([list(map(int, sp)) for sp in po.harvest_spans(x, nll)])
        if r_rng.random() < args.r_prob:
            spans_R.extend(random_spans(x, r_rng))
    # trim the larger list (seeded subsample) to matched counts
    n = min(len(spans_S), len(spans_R))
    t_rng = random.Random(args.seed + 777)
    if len(spans_S) > n:
        spans_S = t_rng.sample(spans_S, n)
    if len(spans_R) > n:
        spans_R = t_rng.sample(spans_R, n)
    sha_S = hashlib.sha256(json.dumps(spans_S).encode()).hexdigest()
    sha_R = hashlib.sha256(json.dumps(spans_R).encode()).hexdigest()
    print(f"[producer] {args.producer_chunks} chunks | gated {n_gated} | "
          f"matched spans {n} | shaS {sha_S[:12]} shaR {sha_R[:12]} | "
          f"{time.time()-t0:.0f}s", flush=True)

    files = {"file_S": spans_S, "file_R": spans_R}

    # ── three consumer twins from one init ─────────────────────────────────
    base = make_organism(args.seed + 999, V, mask, "consumer")
    twins = {}
    for name in ("file_S", "file_R", "without"):
        org = copy.deepcopy(base)
        cstream = po.C4Stream(stoi, unk, skip_docs=args.reader_offset)
        cfeeder = po.ChunkFeeder(cstream, po.BATCH, po.CHUNK)
        spans = files.get(name)
        for ci in range(1, args.consumer_chunks + 1):
            x, y = cfeeder.next_xy()
            org.step_gated(x, y)
            if spans and ci % args.replay_every == 0:
                sp = po.SpanFeeder(po.SpanStream(spans, seed=args.seed + ci),
                                   po.BATCH, po.CHUNK)
                sx, sy = sp.next_xy()
                org.sleep_step(sx, sy)
        twins[name] = org
        print(f"[{name}] {args.consumer_chunks} chunks done", flush=True)

    # ── probes: each file's own entries, midpoint split (the P55 geometry) ─
    min_probe_len = 33
    probes = {}
    for name, spans in files.items():
        p_rng = random.Random(args.seed)
        long_spans = [sp for sp in spans if len(sp) >= min_probe_len]
        probes[name] = p_rng.sample(long_spans, min(args.n_probe, len(long_spans)))

    heldout = {name: round(heldout_c4(org.model, evX, evY), 6)
               for name, org in twins.items()}
    gain_S = heldout["without"] - heldout["file_S"]
    gain_R = heldout["without"] - heldout["file_R"]
    keyed = {}
    for name in ("file_S", "file_R"):
        with_nll = completion_nll(twins[name].model, probes[name], K)
        without_nll = completion_nll(twins["without"].model, probes[name], K)
        keyed[name] = round(without_nll - with_nll, 6)

    out = {"p64": True, "smoke": args.smoke,
           "cadence": {"d_model": po.D_MODEL, "batch": po.BATCH, "chunk": po.CHUNK,
                       "q": po.GATE_Q, "window": po.GATE_WINDOW,
                       "min_window": po.MIN_WINDOW, "ignition_chunks": po.IGNITION_CHUNKS},
           "config": {"producer_chunks": args.producer_chunks,
                      "consumer_chunks": args.consumer_chunks,
                      "replay_every": args.replay_every, "r_prob": args.r_prob,
                      "n_probe": {k: len(v) for k, v in probes.items()},
                      "min_probe_len": min_probe_len},
           "n_spans_matched": n, "n_gated_chunks": n_gated,
           "file_sha256": {"file_S": sha_S, "file_R": sha_R},
           "heldout": heldout,
           "p64a_gain_S": round(gain_S, 6), "p64a_gain_R": round(gain_R, 6),
           "p64a_pass": bool(gain_S > 0 and gain_S >= 1.5 * gain_R),
           "p64b_keyed_gain_S": keyed["file_S"],
           "p64b_keyed_gain_R": keyed["file_R"],
           "p64b_pass_S": bool(keyed["file_S"] >= 0.05),
           "p64b_pass_R": bool(keyed["file_R"] >= 0.05),
           "p64c_matched": bool(len(spans_S) == len(spans_R))}
    path = args.out if not args.smoke else args.out.replace(".json", "_smoke.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[p64] gainS {gain_S:+.4f} gainR {gain_R:+.4f} (a:{out['p64a_pass']}) | "
          f"keyed S {keyed['file_S']:+.4f} R {keyed['file_R']:+.4f} "
          f"(bS:{out['p64b_pass_S']} bR:{out['p64b_pass_R']}) -> {path}", flush=True)


if __name__ == "__main__":
    main()
