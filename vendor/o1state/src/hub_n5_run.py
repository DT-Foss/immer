#!/usr/bin/env python3 -u
"""
THE HUB AT N=5 (MS-O / P46) — five producers, one store, one reader.

Five same-init organisms stream disjoint C4 offsets and harvest surprise
spans into ONE union pool (the memory route — the layer P44 measured as the
one that composes). A separate reader then streams its own C4 budget twice,
forked from one deepcopy'd init: once WITH dosed replay from the union pool,
once without. P46(a): the collective must help the reader through the
memory interface by >= 0.02 heldout.

P46(b), the weight-route counterpart: five fresh same-init arms train
round-robin on their own streams with a weight average REDISTRIBUTED every
M chunks (bounded divergence — the regime where averaging might live, after
P44 killed the one-shot merge at full divergence). Optimizer state stays
per-arm after each redistribute; that mismatch is part of what is being
measured, not a bug. References for (b): this run's own producers double as
single-arm-at-S controls (identical protocol; harvesting reads nll and
never touches the model), and P44's same-config 5xS control (5.3586) is
cited as the cross-run reference.

Instruments shared with P47 (build_c4_eval / heldout_c4): a fixed far-offset
C4 slice. Cadence explicit and recorded per the 2026-08-05 audit rule.
"""
import argparse
import copy
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch
torch.set_num_threads(1)

import portable_organism as po
from knowledge_file_run import build_c4_eval, heldout_c4


def make_organism(seed, V, mask, name):
    torch.manual_seed(seed)
    return po.Organism(name, V, mask, seed=seed)


def producer(name, seed, skip_docs, n_chunks, stoi, unk, mask, V, evX, evY):
    org = make_organism(seed, V, mask, name)
    stream = po.C4Stream(stoi, unk, skip_docs=skip_docs)
    feeder = po.ChunkFeeder(stream, po.BATCH, po.CHUNK)
    spans = []
    t0 = time.time()
    for _ in range(n_chunks):
        x, y = feeder.next_xy()
        s, gated, nll = org.step_gated(x, y)
        if gated:
            spans.extend([list(map(int, sp)) for sp in po.harvest_spans(x, nll)])
    h = heldout_c4(org.model, evX, evY)
    print(f"[{name}] {n_chunks} chunks @doc {skip_docs:,} | heldout {h:.4f} | "
          f"{len(spans)} spans | {time.time()-t0:.0f}s", flush=True)
    return spans, h


def reader_arm(base, name, pool, n_chunks, reader_offset, stoi, unk, replay_every, seed):
    org = copy.deepcopy(base)
    stream = po.C4Stream(stoi, unk, skip_docs=reader_offset)
    feeder = po.ChunkFeeder(stream, po.BATCH, po.CHUNK)
    replays = 0
    for ci in range(1, n_chunks + 1):
        x, y = feeder.next_xy()
        org.step_gated(x, y)
        if pool and ci % replay_every == 0:
            sp = po.SpanFeeder(po.SpanStream(pool, seed=seed + ci), po.BATCH, po.CHUNK)
            sx, sy = sp.next_xy()
            org.sleep_step(sx, sy)
            replays += 1
    return org, replays


def main():
    ap = argparse.ArgumentParser(description="MS-O / P46: five producers, one store, one reader")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-arms", type=int, default=5)
    ap.add_argument("--segment-chunks", type=int, default=1500)
    ap.add_argument("--merge-every", type=int, default=100)
    ap.add_argument("--replay-every", type=int, default=25)
    ap.add_argument("--offset-docs", type=int, default=200_000)
    ap.add_argument("--reader-offset-docs", type=int, default=1_000_000)
    ap.add_argument("--eval-offset-docs", type=int, default=1_200_000)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--q", type=float, default=0.75)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--min-window", type=int, default=100)
    ap.add_argument("--ignition-chunks", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "hub_n5.json"))
    args = ap.parse_args()

    if args.smoke:
        args.segment_chunks, args.merge_every, args.replay_every = 60, 20, 20
        args.offset_docs = 20_000

    po.D_MODEL, po.BATCH, po.CHUNK = args.d_model, args.batch, args.chunk_size
    po.GATE_Q, po.GATE_WINDOW, po.MIN_WINDOW, po.IGNITION_CHUNKS = \
        args.q, args.window, args.min_window, args.ignition_chunks

    vocab, stoi, unk, mask, val_ids = po.get_vocab()
    V = len(vocab)
    evX, evY = build_c4_eval(stoi, unk, args.eval_offset_docs, po.EVAL_TOKENS, po.CHUNK)
    S, N = args.segment_chunks, args.n_arms

    # ── (a) the memory route: producers -> union pool -> reader fork ───────
    pool, prod_helds = [], []
    for k in range(N):
        spans, h = producer(f"prod{k}", args.seed, k * args.offset_docs,
                            S, stoi, unk, mask, V, evX, evY)
        pool.extend(spans)
        prod_helds.append(round(h, 6))
    print(f"[pool] union store: {len(pool)} spans from {N} producers", flush=True)

    base_reader = make_organism(args.seed, V, mask, "reader")
    r_store, n_repl = reader_arm(base_reader, "reader_store", pool, S,
                                 args.reader_offset_docs, stoi, unk,
                                 args.replay_every, args.seed)
    r_none, _ = reader_arm(base_reader, "reader_none", None, S,
                           args.reader_offset_docs, stoi, unk,
                           args.replay_every, args.seed)
    h_store = heldout_c4(r_store.model, evX, evY)
    h_none = heldout_c4(r_none.model, evX, evY)
    print(f"[reader] with store {h_store:.4f} ({n_repl} replays) | "
          f"without {h_none:.4f} | delta {h_none - h_store:+.4f}", flush=True)

    # ── (b) the weight route, bounded divergence: merge every M chunks ─────
    arms = [make_organism(args.seed, V, mask, f"im{k}") for k in range(N)]
    streams = [po.C4Stream(stoi, unk, skip_docs=k * args.offset_docs) for k in range(N)]
    feeders = [po.ChunkFeeder(st, po.BATCH, po.CHUNK) for st in streams]
    rounds = S // args.merge_every
    for r in range(rounds):
        for k in range(N):
            for _ in range(args.merge_every):
                x, y = feeders[k].next_xy()
                arms[k].step_gated(x, y)
        sds = [a.model.state_dict() for a in arms]
        avg = {}
        for key in sds[0]:
            if sds[0][key].is_floating_point():
                avg[key] = torch.stack([sd[key].to(torch.float64) for sd in sds]).mean(0).to(sds[0][key].dtype)
            else:
                avg[key] = sds[0][key]
        for a in arms:
            a.model.load_state_dict(avg)
    h_iter = heldout_c4(arms[0].model, evX, evY)
    print(f"[itermerge] {rounds} rounds x {args.merge_every} chunks | heldout {h_iter:.4f}", flush=True)

    # ── the equal-total-compute control, SAME instrument: one organism,
    #    N*S chunks. The smoke exposed that P44's 5xS number lives on the
    #    WT-2 instrument and is NOT comparable to this run's C4 slice —
    #    the embarrassment clause needs its control in-run. ─────────────────
    ctrl = make_organism(args.seed, V, mask, "ctrl_5xS")
    cstream = po.C4Stream(stoi, unk, skip_docs=0)
    cfeeder = po.ChunkFeeder(cstream, po.BATCH, po.CHUNK)
    t0 = time.time()
    for _ in range(N * S):
        x, y = cfeeder.next_xy()
        ctrl.step_gated(x, y)
    h_ctrl = heldout_c4(ctrl.model, evX, evY)
    print(f"[ctrl_5xS] {N*S} chunks | heldout {h_ctrl:.4f} | {time.time()-t0:.0f}s", flush=True)

    out = {"p46": True, "smoke": args.smoke, "n_arms": N, "S": S,
           "cadence": {"d_model": po.D_MODEL, "batch": po.BATCH, "chunk": po.CHUNK,
                       "q": po.GATE_Q, "window": po.GATE_WINDOW,
                       "min_window": po.MIN_WINDOW, "ignition_chunks": po.IGNITION_CHUNKS},
           "seed": args.seed, "union_pool_spans": len(pool),
           "producer_heldouts": prod_helds,
           "reader_with_store": round(h_store, 6),
           "reader_without_store": round(h_none, 6),
           "p46a_delta": round(h_none - h_store, 6),
           "p46a_pass": bool(h_none - h_store >= 0.02),
           "itermerge_heldout": round(h_iter, 6),
           "single_arm_reference_best": min(prod_helds),
           "p46b_itermerge_vs_best_single": round(h_iter - min(prod_helds), 6),
           "p46b_pass": bool(h_iter <= min(prod_helds)),
           "ctrl_5xS_same_instrument": round(h_ctrl, 6),
           "p44_oneshot_reference": {"brain": 5.979112, "ctrl_5xS": 5.358632,
                                     "note": "WT-2 instrument — NOT comparable to this run's C4 slice; kept for the record only"},
           "p46b_embarrassment_beats_5xS": bool(h_iter < h_ctrl)}
    path = args.out if not args.smoke else args.out.replace(".json", "_smoke.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[hub_n5] reader Δ{out['p46a_delta']:+.4f} | itermerge {h_iter:.4f} "
          f"vs best single {min(prod_helds):.4f} -> {path}", flush=True)


if __name__ == "__main__":
    main()
