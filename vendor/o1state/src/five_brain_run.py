#!/usr/bin/env python3 -u
"""
FIVE RUNS, ONE BRAIN (MS-M / P44) — N organisms, identical init, disjoint
stream offsets; at the end their weights are averaged into one brain.

Arms and controls, all same seed/init/cadence (explicit + recorded):
  arm_k (k=0..N-1)   organism k streams S chunks from C4 at skip_docs = k*OFFSET
  brain              state_dict average of the N arms' final weights
  ctrl_S             = arm_0 (single arm at equal per-arm budget)
  ctrl_NxS           one organism, N*S chunks, offset 0 (equal total compute)

Registered expectations (P44): (a) non-collapse (brain <= arm + 0.3),
(b) brain beats the single arm at S by >= 0.02, (c) brain does NOT beat the
N*S control — with the embarrassment threshold that beating it would be the
bigger result.

Sequential in one process: deterministic, no co-load, ~minutes at d=128.
The carried states Z are NOT merged (they are stream-position-specific);
the brain is a weight-space object evaluated on the frozen WT-2 heldout.
"""
import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch
torch.set_num_threads(1)

import portable_organism as po


def stream_arm(name, seed, skip_docs, n_chunks, stoi, unk, mask, V, evX, evY):
    torch.manual_seed(seed)                       # IDENTICAL init for every arm
    org = po.Organism(name, V, mask, seed=seed)
    stream = po.C4Stream(stoi, unk, skip_docs=skip_docs)
    feeder = po.ChunkFeeder(stream, po.BATCH, po.CHUNK)
    t0 = time.time()
    for _ in range(n_chunks):
        x, y = feeder.next_xy()
        org.step_gated(x, y)
    h = float(po.heldout(org.model, evX, evY))
    print(f"[{name}] {n_chunks} chunks from doc {skip_docs:,} | heldout {h:.4f} "
          f"| grad_tokens {org.grad_tokens:,} | {time.time()-t0:.0f}s", flush=True)
    return org, h


def main():
    ap = argparse.ArgumentParser(description="MS-M / P44: five runs, one brain")
    ap.add_argument("--n-arms", type=int, default=5)
    ap.add_argument("--segment-chunks", type=int, default=1500, help="S: chunks per arm")
    ap.add_argument("--offset-docs", type=int, default=200_000,
                    help="stream offset between arms (docs); far enough apart that no arm sees another's text")
    ap.add_argument("--smoke", action="store_true", help="S=60, offset=20k")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--q", type=float, default=0.75)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--min-window", type=int, default=100)
    ap.add_argument("--ignition-chunks", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "five_brain.json"))
    args = ap.parse_args()

    S = 60 if args.smoke else args.segment_chunks
    OFF = 20_000 if args.smoke else args.offset_docs
    N = args.n_arms

    po.D_MODEL, po.BATCH, po.CHUNK = args.d_model, args.batch, args.chunk_size
    po.GATE_Q, po.GATE_WINDOW, po.MIN_WINDOW, po.IGNITION_CHUNKS = \
        args.q, args.window, args.min_window, args.ignition_chunks

    vocab, stoi, unk, mask, val_ids = po.get_vocab()
    V = len(vocab)
    evX, evY = po.build_eval_set(val_ids, po.EVAL_TOKENS, po.CHUNK)

    arms, arm_helds = [], []
    for k in range(N):
        org, h = stream_arm(f"arm{k}", args.seed, k * OFF, S, stoi, unk, mask, V, evX, evY)
        arms.append(org)
        arm_helds.append(h)

    # ── the brain: plain state_dict average of the N arms ──────────────────
    # Float tensors are averaged in float64; non-float entries (index masks,
    # integer buffers) are NOT averageable — they must be identical across
    # arms (same init, no stat-tracking layers), verified rather than assumed.
    sds = [a.model.state_dict() for a in arms]
    avg = {}
    nonfloat_mismatch = []
    for k in sds[0]:
        if sds[0][k].is_floating_point():
            avg[k] = torch.stack([sd[k].to(torch.float64) for sd in sds]).mean(0).to(sds[0][k].dtype)
        else:
            if not all(torch.equal(sd[k], sds[0][k]) for sd in sds[1:]):
                nonfloat_mismatch.append(k)
            avg[k] = sds[0][k]
    if nonfloat_mismatch:
        print(f"[brain] WARNING: non-float state entries differ across arms: {nonfloat_mismatch}", flush=True)
    torch.manual_seed(args.seed)
    brain = po.Organism("brain", V, mask, seed=args.seed)
    brain.model.load_state_dict(avg)
    brain_h = float(po.heldout(brain.model, evX, evY))
    print(f"[brain] N={N} average | heldout {brain_h:.4f}", flush=True)

    # ── equal-total-compute control: one organism, N*S chunks ──────────────
    _, ctrl_h = stream_arm("ctrl_NxS", args.seed, 0, N * S, stoi, unk, mask, V, evX, evY)

    out = {"p44": True, "smoke": args.smoke, "n_arms": N, "S": S, "offset_docs": OFF,
           "cadence": {"d_model": po.D_MODEL, "batch": po.BATCH, "chunk": po.CHUNK,
                       "q": po.GATE_Q, "window": po.GATE_WINDOW,
                       "min_window": po.MIN_WINDOW, "ignition_chunks": po.IGNITION_CHUNKS},
           "seed": args.seed,
           "arm_heldouts": [round(h, 6) for h in arm_helds],
           "arm_mean": round(sum(arm_helds) / N, 6),
           "arm_best": round(min(arm_helds), 6),
           "brain_heldout": round(brain_h, 6),
           "ctrl_NxS_heldout": round(ctrl_h, 6),
           "p44a_noncollapse": bool(brain_h <= arm_helds[0] + 0.3),
           "p44b_brain_beats_single_arm_by": round(arm_helds[0] - brain_h, 6),
           "p44b_pass": bool(arm_helds[0] - brain_h >= 0.02),
           "p44c_brain_vs_NxS": round(brain_h - ctrl_h, 6),
           "p44c_embarrassment_fired": bool(brain_h < ctrl_h)}
    path = args.out if not args.smoke else args.out.replace(".json", "_smoke.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[five_brain] arms {out['arm_heldouts']} | brain {brain_h:.4f} | "
          f"NxS ctrl {ctrl_h:.4f} -> {path}", flush=True)


if __name__ == "__main__":
    main()
