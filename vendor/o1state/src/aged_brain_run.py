#!/usr/bin/env python3 -u
"""
THE AGED BRAIN (MS-U / P59) — does 7.4 billion tokens of life change how
an organism gates, learns, and survives shocks?

The determinism results make the project's most expensive asset — the
single uninterrupted multi-billion-token life — forkable: its checkpoint
loads key- and shape-identical into the reference Organism class at
V=5000 (measured before this build; the living run is never touched).

Arms, both in the VETERAN'S OWN vocabulary (stoi from the checkpoint):
  veteran  the fork: weights + Adam state from the lifetime checkpoint
  young    raised fresh in-harness: same config, same vocab, 50M tokens

Protocol (fresh gate state for both — the question is whether EXPERIENCE
shapes the surprise economy, not whether window history transfers):
  (a) 2,000-chunk C4 continuation, cumulative gate rate: |Δ| > 2pp means
      age shifts the surprise economy; invariance is a law too.
  (b) shock: C4 1,000 → WT-103 1,000 → C4 1,000 (P59 amendment: WT-103
      is the shock substrate; both arms stay in the veteran's universe)
      → forgetting / plasticity / recovery per arm. Veteran LOWER
      forgetting = experience is immunity; HIGHER = ossification.
  (c) harvest behavior: spans collected during (a).
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


def heldout_from_stream(model, stream_name, stoi, unk, skip_docs, n_tokens, K):
    import torch.nn.functional as F
    s = HFStream(stream_name, stoi, unk, skip_docs=skip_docs)
    toks = []
    while len(toks) < n_tokens + 1:
        toks.extend(s.next_block())
    xs, ys = [], []
    for c in range(0, n_tokens - K, K):
        xs.append(toks[c:c + K])
        ys.append(toks[c + 1:c + K + 1])
    X = torch.tensor(xs, dtype=torch.long)
    Y = torch.tensor(ys, dtype=torch.long)
    model.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, X.shape[0], 64):
            logits, _ = model(X[i:i + 64], None)
            l = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                Y[i:i + 64].reshape(-1))
            tot += float(l) * Y[i:i + 64].numel()
            cnt += Y[i:i + 64].numel()
    model.train()
    return tot / cnt


def run_phase(org, stream, n_chunks, harvest=False):
    feeder = po.ChunkFeeder(stream, po.BATCH, po.CHUNK)
    fired = 0
    spans = 0
    for _ in range(n_chunks):
        x, y = feeder.next_xy()
        s, gated, nll = org.step_gated(x, y)
        if gated:
            fired += 1
            if harvest:
                spans += len(po.harvest_spans(x, nll))
    return fired, spans


def fresh_gate(org):
    """Reset gate statistics so both arms start with identical windows —
    the comparison targets experience in the WEIGHTS, not window history."""
    org.n_chunks = 0
    if hasattr(org, "surprise_hist"):
        try:
            org.surprise_hist.clear()
        except Exception:
            pass
    for attr in ("window", "s_window", "window_vals", "gate_window_vals"):
        if hasattr(org, attr):
            try:
                getattr(org, attr).clear()
            except Exception:
                pass
    return org


def main():
    ap = argparse.ArgumentParser(description="P59: the aged brain")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ckpt", default=os.path.join(REPO_ROOT, "results", "veteran", "veteran_8b_ckpt.pt"))
    ap.add_argument("--young-chunks", type=int, default=97656)   # 50M tokens at 8x64
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
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "aged_brain.json"))
    args = ap.parse_args()
    if args.smoke:
        args.young_chunks, args.rate_chunks, args.shock_chunks = 600, 300, 200
        args.eval_tokens = 30_000

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    V, mask = ck["vocab_size"], ck["mask_idx"]
    stoi, unk = ck["stoi"], ck["unk"]
    po.D_MODEL = ck["d_model"]
    po.BATCH, po.CHUNK = args.batch, args.chunk_size
    po.GATE_Q, po.GATE_WINDOW, po.MIN_WINDOW, po.IGNITION_CHUNKS = \
        args.q, args.window, args.min_window, args.ignition_chunks
    K = po.CHUNK

    # ── the veteran fork ───────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    veteran = po.Organism("veteran", V, mask, seed=args.seed)
    veteran.model.load_state_dict(ck["state_dict"])
    opt_loaded = False
    try:
        veteran.opt.load_state_dict(ck["opt_state"])
        opt_loaded = True
    except Exception as e:
        print(f"[veteran] opt_state not loaded ({type(e).__name__}) — fork carries weights only", flush=True)
    age = ck.get("streamed_tokens", 0)
    print(f"[veteran] forked at {age:,} lived tokens | opt_state loaded: {opt_loaded}", flush=True)

    # ── raise the young twin: same config, same vocab, 50M tokens ─────────
    torch.manual_seed(args.seed + 1)
    young = po.Organism("young", V, mask, seed=args.seed + 1)
    ystream = HFStream("c4", stoi, unk, skip_docs=0)
    t0 = time.time()
    yfeeder = po.ChunkFeeder(ystream, po.BATCH, po.CHUNK)
    for ci in range(1, args.young_chunks + 1):
        x, y = yfeeder.next_xy()
        young.step_gated(x, y)
        if ci % 10000 == 0:
            print(f"[young] {ci}/{args.young_chunks} chunks | {time.time()-t0:.0f}s", flush=True)
    print(f"[young] raised: {args.young_chunks * po.BATCH * po.CHUNK:,} tokens "
          f"| {time.time()-t0:.0f}s", flush=True)

    # ── (a) + (c): rate probe on fresh gates, far C4 offset ────────────────
    arms_a = {}
    for name, base in (("veteran", veteran), ("young", young)):
        org = fresh_gate(copy.deepcopy(base))
        stream = HFStream("c4", stoi, unk, skip_docs=2_000_000)
        fired, spans = run_phase(org, stream, args.rate_chunks, harvest=True)
        rate = fired / args.rate_chunks
        arms_a[name] = {"gate_rate": round(rate, 4), "spans": spans}
        print(f"[a:{name}] rate {rate:.4f} | spans {spans}", flush=True)
    delta_pp = abs(arms_a["veteran"]["gate_rate"] - arms_a["young"]["gate_rate"]) * 100

    # ── (b): the shock protocol, fresh forks per arm ───────────────────────
    ev = lambda m, src, skip: heldout_from_stream(m, src, stoi, unk, skip,
                                                  args.eval_tokens, K)
    arms_b = {}
    for name, base in (("veteran", veteran), ("young", young)):
        org = fresh_gate(copy.deepcopy(base))
        pre_c4 = ev(org.model, "c4", 3_000_000)
        run_phase(org, HFStream("c4", stoi, unk, skip_docs=2_500_000), args.shock_chunks)
        post_p1_c4 = ev(org.model, "c4", 3_000_000)
        pre_shock_wt = ev(org.model, "wt103", 0)
        run_phase(org, HFStream("wt103", stoi, unk, skip_docs=0), args.shock_chunks)
        post_shock_c4 = ev(org.model, "c4", 3_000_000)
        post_shock_wt = ev(org.model, "wt103", 0)
        run_phase(org, HFStream("c4", stoi, unk, skip_docs=2_700_000), args.shock_chunks)
        final_c4 = ev(org.model, "c4", 3_000_000)
        arms_b[name] = {
            "pre_c4": round(pre_c4, 6), "post_p1_c4": round(post_p1_c4, 6),
            "forgetting": round(post_shock_c4 - post_p1_c4, 6),
            "plasticity": round(pre_shock_wt - post_shock_wt, 6),
            "recovery_residual": round(final_c4 - post_p1_c4, 6)}
        print(f"[b:{name}] forgetting {arms_b[name]['forgetting']:+.4f} | "
              f"plasticity {arms_b[name]['plasticity']:+.4f} | "
              f"recovery {arms_b[name]['recovery_residual']:+.4f}", flush=True)

    vet_b, yng_b = arms_b["veteran"], arms_b["young"]
    out = {"p59": True, "smoke": args.smoke,
           "cadence": {"d_model": po.D_MODEL, "batch": po.BATCH, "chunk": K,
                       "q": po.GATE_Q, "window": po.GATE_WINDOW,
                       "min_window": po.MIN_WINDOW, "ignition_chunks": po.IGNITION_CHUNKS,
                       "vocab": V},
           "veteran_age_tokens": age, "opt_state_loaded": opt_loaded,
           "young_tokens": args.young_chunks * po.BATCH * po.CHUNK,
           "a_rate_probe": arms_a,
           "p59a_delta_pp": round(delta_pp, 3),
           "p59a_age_shifts_rate": bool(delta_pp > 2.0),
           "b_shock": arms_b,
           "p59b_veteran_forgets_less": bool(vet_b["forgetting"] < yng_b["forgetting"]),
           "p59c_plasticity_ratio": round(vet_b["plasticity"] / yng_b["plasticity"], 4)
               if yng_b["plasticity"] else None,
           "p59c_within_25pct": bool(yng_b["plasticity"] and
               abs(vet_b["plasticity"] / yng_b["plasticity"] - 1) <= 0.25)}
    path = args.out if not args.smoke else args.out.replace(".json", "_smoke.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[p59] Δrate {delta_pp:.2f}pp (bar 2.0) | vet forgets less: "
          f"{out['p59b_veteran_forgets_less']} | plasticity ratio "
          f"{out['p59c_plasticity_ratio']} -> {path}", flush=True)


if __name__ == "__main__":
    main()
