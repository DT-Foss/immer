#!/usr/bin/env python3 -u
"""
THE FILTER'S FILE ON THE FILTER'S TURF (P67) — the attack P64 named.

P64: the surprise file loses average-heldout transfer 1.7x while
storing its own entries better. P67 asks the registered question: does
file-S win on NOVEL content — the class the gate selects for? The P64
producer pass is regenerated (deterministic; shas must equal P64's),
the same three consumer twins run, and the EVAL is the new instrument:
C4 heldout chunks stratified into terciles by first-ever-bigram rate
against a registry accumulated over the producer stream (the P58
instrument applied to the eval), per-tercile heldout NLL per arm.
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
from knowledge_file_run import build_c4_eval
from keyed_file_run import make_organism
from filter_file_run import random_spans

P64_SHA_S = "87c81bcc00818922c34ff207f3267e5c86a8792c2edeb915aa2e53708b4e7e34"
P64_SHA_R = "b1804f58865e79a1b70d9b904ec85bbe4d5029b24d1ba5c8879528c26e70bf8e"


def main():
    ap = argparse.ArgumentParser(description="P67: novelty-stratified file transfer")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--producer-chunks", type=int, default=1500)
    ap.add_argument("--consumer-chunks", type=int, default=1500)
    ap.add_argument("--replay-every", type=int, default=25)
    ap.add_argument("--r-prob", type=float, default=0.30)
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
    ap.add_argument("--expect-sha-s", default=P64_SHA_S)
    ap.add_argument("--expect-sha-r", default=P64_SHA_R)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "novelty_transfer.json"))
    args = ap.parse_args()
    if args.smoke:
        args.producer_chunks, args.consumer_chunks = 120, 120
        args.replay_every = 10

    po.D_MODEL, po.BATCH, po.CHUNK = args.d_model, args.batch, args.chunk_size
    po.GATE_Q, po.GATE_WINDOW, po.MIN_WINDOW, po.IGNITION_CHUNKS = \
        args.q, args.window, args.min_window, args.ignition_chunks
    vocab, stoi, unk, mask, val_ids = po.get_vocab()
    V = len(vocab)
    evX, evY = build_c4_eval(stoi, unk, args.eval_offset, po.EVAL_TOKENS, po.CHUNK)

    # ── the P64 producer pass, regenerated + registry collected ────────────
    prod = make_organism(args.seed, V, mask, "producer")
    stream = po.C4Stream(stoi, unk, skip_docs=args.producer_offset)
    feeder = po.ChunkFeeder(stream, po.BATCH, po.CHUNK)
    r_rng = random.Random(args.seed + 555)
    spans_S, spans_R = [], []
    registry = set()
    n_gated = 0
    t0 = time.time()
    for _ in range(args.producer_chunks):
        x, y = feeder.next_xy()
        for row in x.tolist():
            registry.update(zip(row, row[1:]))
        s, gated, nll = prod.step_gated(x, y)
        if gated:
            n_gated += 1
            spans_S.extend([list(map(int, sp)) for sp in po.harvest_spans(x, nll)])
        if r_rng.random() < args.r_prob:
            spans_R.extend(random_spans(x, r_rng))
    n = min(len(spans_S), len(spans_R))
    t_rng = random.Random(args.seed + 777)
    if len(spans_S) > n:
        spans_S = t_rng.sample(spans_S, n)
    if len(spans_R) > n:
        spans_R = t_rng.sample(spans_R, n)
    sha_S = hashlib.sha256(json.dumps(spans_S).encode()).hexdigest()
    sha_R = hashlib.sha256(json.dumps(spans_R).encode()).hexdigest()
    print(f"[producer] {args.producer_chunks} chunks | gated {n_gated} | matched {n} | "
          f"registry {len(registry):,} bigrams | shaS {sha_S[:12]} shaR {sha_R[:12]} | "
          f"{time.time()-t0:.0f}s", flush=True)

    files = {"file_S": spans_S, "file_R": spans_R}

    # ── the three consumer twins (the P64 protocol) ────────────────────────
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

    # ── the instrument: novelty terciles over the eval rows ────────────────
    import torch.nn.functional as F
    rows = evX.tolist()
    novelty = []
    for row in rows:
        bg = list(zip(row, row[1:]))
        new = sum(1 for b in bg if b not in registry)
        novelty.append(new / max(1, len(bg)))
    order = sorted(range(len(rows)), key=lambda i: novelty[i])
    third = len(order) // 3
    terciles = {"low": order[:third], "mid": order[third:2 * third],
                "high": order[2 * third:3 * third]}

    def ce_rows(model, idxs):
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(idxs), 64):
                sel = idxs[i:i + 64]
                X, Y = evX[sel], evY[sel]
                logits, _ = model(X, None)
                l = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), Y.reshape(-1))
                tot += float(l) * Y.numel()
                cnt += Y.numel()
        model.train()
        return tot / max(1, cnt)

    res = {}
    for name, org in twins.items():
        res[name] = {t: round(ce_rows(org.model, idxs), 6)
                     for t, idxs in terciles.items()}
        print(f"[{name}] " + " | ".join(f"{t} {res[name][t]}" for t in ("low", "mid", "high")),
              flush=True)

    gains = {t: {"S": round(res["without"][t] - res["file_S"][t], 6),
                 "R": round(res["without"][t] - res["file_R"][t], 6)}
             for t in ("low", "mid", "high")}
    delta = {t: round(gains[t]["S"] - gains[t]["R"], 6) for t in ("low", "mid", "high")}
    out = {"p67": True, "smoke": args.smoke,
           "cadence": {"d_model": po.D_MODEL, "batch": po.BATCH, "chunk": po.CHUNK,
                       "q": po.GATE_Q, "window": po.GATE_WINDOW,
                       "min_window": po.MIN_WINDOW, "ignition_chunks": po.IGNITION_CHUNKS},
           "config": {"producer_chunks": args.producer_chunks,
                      "consumer_chunks": args.consumer_chunks,
                      "replay_every": args.replay_every, "r_prob": args.r_prob,
                      "registry_bigrams": len(registry),
                      "tercile_rows": third,
                      "novelty_tercile_bounds": [round(novelty[order[third]], 5),
                                                 round(novelty[order[2 * third]], 5)]},
           "n_spans_matched": n,
           "file_sha256": {"file_S": sha_S, "file_R": sha_R},
           "sha_matches_p64": {"file_S": bool(sha_S == args.expect_sha_s),
                               "file_R": bool(sha_R == args.expect_sha_r)},
           "per_tercile_heldout": res, "per_tercile_gains": gains,
           "p67_delta_S_minus_R": delta,
           "p67a_pass": bool(gains["high"]["S"] >= gains["high"]["R"]),
           "p67b_pass": bool(delta["low"] <= delta["mid"] <= delta["high"]),
           "p67c_pass": bool(sha_S == args.expect_sha_s and sha_R == args.expect_sha_r
                             and not args.smoke)}
    path = args.out if not args.smoke else args.out.replace(".json", "_smoke.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[p67] delta(S-R) low {delta['low']:+.4f} mid {delta['mid']:+.4f} "
          f"high {delta['high']:+.4f} | a:{out['p67a_pass']} b:{out['p67b_pass']} "
          f"c:{out['p67c_pass']} -> {path}", flush=True)


if __name__ == "__main__":
    main()
