#!/usr/bin/env python3 -u
"""
THE PLASTICITY-DECAY CURVE (P66) — the follow-up P59(c) fired.

Five in-life ages of the 909M organism (four archived mid-life POS
snapshots + the final state) run the EXACT P59 veteran-side protocol
(fresh-gate rate probe + WT-103 shock, identical offsets and eval
budgets). The 0.05B young point and the 7.44B veteran point are carried
over from the P59 artifact (same protocol, same cadence). Snapshot
adapter: the POS 3-arm checkpoint's A3 (gated) arm is the organism.
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
from source_swap_run import HFStream
from aged_brain_run import heldout_from_stream, run_phase, fresh_gate


def load_pos_snapshot(path, V, mask, seed):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    po.D_MODEL = ck["config"]["d_model"]
    a3 = ck["arms"]["A3"]
    torch.manual_seed(seed)
    org = po.Organism(f"age_{ck['n_streamed']}", V, mask, seed=seed)
    org.model.load_state_dict(a3["model"])
    opt_loaded = False
    try:
        org.opt.load_state_dict(a3["opt"])
        opt_loaded = True
    except Exception as e:
        print(f"[{os.path.basename(path)}] opt not loaded "
              f"({type(e).__name__})", flush=True)
    return org, int(ck["n_streamed"]), opt_loaded


def main():
    ap = argparse.ArgumentParser(description="P66: the plasticity-decay curve")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ckpts", nargs="+", default=[
        os.path.join(REPO_ROOT, "results", "pos_snapshots", "ckpt_181995008.pt"),
        os.path.join(REPO_ROOT, "results", "pos_snapshots", "ckpt_240350208.pt"),
        os.path.join(REPO_ROOT, "results", "pos_snapshots", "ckpt_359050240.pt"),
        os.path.join(REPO_ROOT, "results", "pos_snapshots", "ckpt_744647168.pt"),
        os.path.join(REPO_ROOT, "results", "pos_ckpt.pt")])
    ap.add_argument("--rate-chunks", type=int, default=2000)
    ap.add_argument("--shock-chunks", type=int, default=1000)
    ap.add_argument("--eval-tokens", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--q", type=float, default=0.75)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--min-window", type=int, default=100)
    ap.add_argument("--ignition-chunks", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "age_ladder.json"))
    args = ap.parse_args()
    if args.smoke:
        args.rate_chunks, args.shock_chunks, args.eval_tokens = 300, 200, 30_000
        args.ckpts = args.ckpts[:1]

    vocab, stoi, unk, mask, val_ids = po.get_vocab()
    V = len(vocab)
    po.BATCH, po.CHUNK = args.batch, args.chunk_size
    po.GATE_Q, po.GATE_WINDOW, po.MIN_WINDOW, po.IGNITION_CHUNKS = \
        args.q, args.window, args.min_window, args.ignition_chunks
    K = po.CHUNK
    ev = lambda m, src, skip: heldout_from_stream(m, src, stoi, unk, skip,
                                                  args.eval_tokens, K)

    points = []
    for path in args.ckpts:
        t0 = time.time()
        org0, age, opt_loaded = load_pos_snapshot(path, V, mask, args.seed)
        # (a) rate probe: fresh gate, far C4 offset (the P59 offsets)
        org = fresh_gate(copy.deepcopy(org0))
        fired, spans = run_phase(org, HFStream("c4", stoi, unk,
                                               skip_docs=2_000_000),
                                 args.rate_chunks, harvest=True)
        rate = fired / args.rate_chunks
        # (b) shock protocol: fresh fork, identical offsets
        org = fresh_gate(copy.deepcopy(org0))
        pre_c4 = ev(org.model, "c4", 3_000_000)
        run_phase(org, HFStream("c4", stoi, unk, skip_docs=2_500_000),
                  args.shock_chunks)
        post_p1_c4 = ev(org.model, "c4", 3_000_000)
        pre_shock_wt = ev(org.model, "wt103", 0)
        run_phase(org, HFStream("wt103", stoi, unk, skip_docs=0),
                  args.shock_chunks)
        post_shock_c4 = ev(org.model, "c4", 3_000_000)
        post_shock_wt = ev(org.model, "wt103", 0)
        run_phase(org, HFStream("c4", stoi, unk, skip_docs=2_700_000),
                  args.shock_chunks)
        final_c4 = ev(org.model, "c4", 3_000_000)
        pt = {"ckpt": os.path.basename(path), "age_tokens": age,
              "opt_state_loaded": opt_loaded,
              "gate_rate": round(rate, 4), "spans": spans,
              "pre_c4": round(pre_c4, 6), "post_p1_c4": round(post_p1_c4, 6),
              "forgetting": round(post_shock_c4 - post_p1_c4, 6),
              "plasticity": round(pre_shock_wt - post_shock_wt, 6),
              "recovery_residual": round(final_c4 - post_p1_c4, 6)}
        points.append(pt)
        print(f"[{pt['ckpt']}] age {age:,} | rate {rate:.4f} | forgetting "
              f"{pt['forgetting']:+.4f} | plasticity {pt['plasticity']:+.4f} "
              f"| recovery {pt['recovery_residual']:+.4f} | "
              f"{time.time()-t0:.0f}s", flush=True)

    # anchors from the P59 artifact (same protocol, same cadence)
    anchors = {}
    try:
        ab = json.load(open(os.path.join(REPO_ROOT, "results", "aged_brain.json")))
        anchors = {
            "young": {"age_tokens": ab["young_tokens"],
                      "gate_rate": ab["a_rate_probe"]["young"]["gate_rate"],
                      **ab["b_shock"]["young"]},
            "veteran": {"age_tokens": ab["veteran_age_tokens"],
                        "gate_rate": ab["a_rate_probe"]["veteran"]["gate_rate"],
                        **ab["b_shock"]["veteran"]}}
    except Exception as e:
        print(f"[anchors] aged_brain.json not readable ({type(e).__name__})",
              flush=True)

    # one-life ladder + young point, sorted by age
    ladder = ([anchors["young"]] if anchors else []) + points
    ladder = sorted(ladder, key=lambda p: p["age_tokens"])
    plas = [p["plasticity"] for p in ladder]
    forg = [p["forgetting"] for p in ladder]
    mono_plas = all(plas[i + 1] <= plas[i] for i in range(len(plas) - 1))
    mono_forg = all(forg[i + 1] <= forg[i] for i in range(len(forg) - 1))
    far_ratio = None
    if anchors and plas:
        youngest = max(plas)
        if youngest:
            far_ratio = round(anchors["veteran"]["plasticity"] / youngest, 4)

    out = {"p66": True, "smoke": args.smoke,
           "cadence": {"d_model": po.D_MODEL, "batch": po.BATCH, "chunk": K,
                       "q": po.GATE_Q, "window": po.GATE_WINDOW,
                       "min_window": po.MIN_WINDOW,
                       "ignition_chunks": po.IGNITION_CHUNKS, "vocab": V},
           "points_new": points, "anchors_from_p59": anchors,
           "ladder_ages": [p["age_tokens"] for p in ladder],
           "ladder_plasticity": plas, "ladder_forgetting": forg,
           "p66a_plasticity_monotone_dec": bool(mono_plas),
           "p66b_far_ratio_vs_youngest": far_ratio,
           "p66b_pass": bool(far_ratio is not None and far_ratio >= 0.6),
           "p66c_forgetting_monotone_dec": bool(mono_forg)}
    path = args.out if not args.smoke else args.out.replace(".json", "_smoke.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[p66] plasticity monotone: {mono_plas} | far ratio {far_ratio} | "
          f"forgetting monotone: {mono_forg} -> {path}", flush=True)


if __name__ == "__main__":
    main()
