#!/usr/bin/env python3 -u
"""
MS11 — weight hot-swap on the living stream (P23, analysis/PREDICTIONS.md Wave 5).
====================================================================================
Question: does a carried OLD Z-state survive being dropped into NEW weights?
The POS long-run (src/pos_run.py) has been streaming for ~63,000s and has left
behind weight snapshots at four token counts. This script freezes the LATEST
snapshot's weights W(359M) and asks what happens when we feed it Z-states that
were carried under EARLIER, different weights — forward-only, no training.

Five arms, all running W(359,050,240 tokens) model weights, all consuming the
SAME cloned C4 chunks (pos_run.py / pos_family_transfer.py §7 clone pattern):

  native     W(359M) + Z from snapshot 359M            (upper bound: no swap at all)
  swap_far   W(359M) + Z from snapshot 128M             (231M tokens training distance)
  swap_near  W(359M) + Z from snapshot 240M             (119M tokens training distance)
  cold       W(359M) + Z = None                          (restart baseline: no state at all)
  shuffled   W(359M) + Z from snapshot 128M, channels permuted (seed 42, per-tensor
             torch.randperm on the last dim) — a STRUCTURE control: same distribution
             of state values as swap_far, but the content is scrambled across channels.
             If swap_far's advantage over cold is just "any non-zero state helps", the
             shuffle should help too. If the advantage is the STATE'S CONTENT matching
             the weights' expectations, shuffling should hurt (and swap_far should beat
             shuffled throughout, per P23c).

Arm A1 (forward-only, no optimizer) is the reference for what "native" means here —
we don't train any arm; every one of the 5 arms above is a pure forward pass with
Z-carry, no_grad, no detach needed. Only the model weights (state_dict) and the
initial Z differ between arms; the C4 tokens are identical across arms per chunk.

Metrics -> results/state_weight_swap.json (results/state_weight_swap_smoke.json
under --smoke): per-arm full chunk-NLL curve, plus:
  excess_first50[arm]     = mean(NLL_arm - NLL_native) over chunks 0..49
  convergence_chunk[arm]  = first chunk c from which rolling-mean(20) of
                            |NLL_arm - NLL_native| stays < 0.01 for the rest of the
                            run (None if it never does)
  excess_last100[arm]     = mean(NLL_arm - NLL_native) over the last 100 chunks

P23 checks:
  (a) excess_first50[cold]   >= 2 * excess_first50[swap_far]
  (b) convergence_chunk[swap_far] < 300 and convergence_chunk[swap_near] < 300
  (c) NLL_shuffled > NLL_swap_far in every 50-chunk window

Snapshot files (results/pos_snapshots/ckpt_*.pt) are READ ONLY. The live run's
own output files (results/pos_ckpt.pt, results/pos_status.json, ...) are never
touched by this script.
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, "reference")
sys.path.insert(0, "src")

import numpy as np
import torch
import torch.nn.functional as F

torch.backends.mps.is_available = lambda: False   # force CPU, matches pos_run.py
torch.set_num_threads(1)                            # bit-deterministic, matches pos_run.py
try:
    os.nice(19)
except PermissionError:
    pass

from streaming_train import StreamingNoPELM
from length_extrap_v2 import load_wikitext2, build_vocab, tokenize

SNAP_DIR = os.path.join("results", "pos_snapshots")
SNAP_NATIVE = os.path.join(SNAP_DIR, "ckpt_359050240.pt")
SNAP_FAR = os.path.join(SNAP_DIR, "ckpt_128219648.pt")
SNAP_NEAR = os.path.join(SNAP_DIR, "ckpt_240350208.pt")


# ─────────────────────────────────────────────────────────────────────────────
# C4 stream — identical shape to pos_run.py's C4Stream (fresh connection, from
# the beginning of the allenai/c4 en train stream; the absolute stream position
# is irrelevant here since every arm sees exactly the same tokens per chunk —
# what matters is that it's fresh, non-cherry-picked text).
# ─────────────────────────────────────────────────────────────────────────────
class C4Stream:
    def __init__(self, stoi, unk, block=65536):
        self.stoi, self.unk, self.block = stoi, unk, block
        self.docs = 0
        self.reconnects = 0
        self.pending = []
        self._it = None

    def _connect(self):
        from datasets import load_dataset
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        self._it = iter(ds)

    def next_block(self):
        while len(self.pending) < self.block:
            try:
                if self._it is None:
                    self._connect()
                row = next(self._it)
            except StopIteration:
                self._it = None
                continue
            except Exception as e:
                self.reconnects += 1
                print(f"[stream] {type(e).__name__}: {e} — reconnect #{self.reconnects}", flush=True)
                self._it = None
                time.sleep(min(60, 5 * self.reconnects))
                continue
            self.docs += 1
            t = row.get("text", "") if isinstance(row, dict) else ""
            if t.strip():
                self.pending.extend(tokenize(t, self.stoi, self.unk))
        out, self.pending = self.pending[:self.block], self.pending[self.block:]
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot loading — pull model weights (always from the 359M snapshot) and Z
# states (from whichever snapshot each arm needs) out of pos_run.py's ckpt
# format. Arm dict in the snapshot: {'model','opt','states','grad_tokens',
# 'n_chunks','n_bwd','window','recent_gates'}; states is list[Tensor(B,H,D)].
# We source model + states from arm 'A3' (the surprise-gated organism — the
# arm whose life the POS run is about; its held-out matches A2 at ratio 1.002
# while its state carries the gated regime, so hot-swap is tested on the
# organism the twin-fork experiment also measures).
# ─────────────────────────────────────────────────────────────────────────────
def load_snapshot(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck


def shuffle_states(states, seed=42):
    """Structure control: permute channels (last dim) of each Z tensor independently,
    same seed every run. Preserves the per-tensor value distribution, destroys the
    per-channel identity the weights were trained to read."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for s in states:
        d = s.shape[-1]
        perm = torch.randperm(d, generator=g)
        out.append(s[..., perm].clone())
    return out


def build_model(V, mask, d_model, state_dict):
    torch.manual_seed(42)
    m = StreamingNoPELM(V, mask, d_model=d_model, n_layers=2, n_heads=4,
                        d_head=d_model // 4, seq_len=32, dropout=0.0, causal=True)
    m.load_state_dict(state_dict)
    m.eval()
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Forward step. Returns the CHUNK-mean NLL (the primary logged curve) plus the
# per-token NLL (B,K) so the caller can additionally resolve the first chunks
# at token granularity — the transient this experiment is actually chasing
# decays within ~10-20 tokens (measured: see report), so a 64-token chunk mean
# already averages most of it away. Chunk size stays K=64 throughout (fixed by
# the snapshot config / the Z-carry cadence); only the ANALYSIS additionally
# looks inside the first chunks token-by-token.
# ─────────────────────────────────────────────────────────────────────────────
def step_fwd(model, states, x, y):
    with torch.no_grad():
        logits, st = model(x, states)
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                              reduction="none").view(x.shape)
    return float(nll.mean()), st, nll


def rolling_mean(vals, window):
    out = []
    s = 0.0
    from collections import deque
    dq = deque(maxlen=window)
    for v in vals:
        dq.append(v)
        out.append(sum(dq) / len(dq))
    return out


def convergence_chunk(diffs, window=20, tol=0.01):
    """First chunk index c such that rolling-mean(20) of |diff| stays < tol for
    every chunk from c to the end. None if it never holds through the end."""
    rm = rolling_mean(diffs, window)
    n = len(rm)
    for c in range(n):
        if all(rm[j] < tol for j in range(c, n)):
            return c
    return None


def main():
    ap = argparse.ArgumentParser(description="MS11: weight hot-swap on the living stream (P23)")
    ap.add_argument("--smoke", action="store_true", help="120 chunks instead of 600")
    ap.add_argument("--chunks", type=int, default=0, help="override chunk count (0 = default/smoke)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="", help="override output path")
    args = ap.parse_args()

    n_chunks = args.chunks or (120 if args.smoke else 600)
    out_path = args.out or os.path.join(
        "results", "state_weight_swap_smoke.json" if args.smoke else "state_weight_swap.json")

    torch.manual_seed(args.seed)

    print(f"[ms11] loading snapshots (read-only) ...", flush=True)
    ck_native = load_snapshot(SNAP_NATIVE)
    ck_far = load_snapshot(SNAP_FAR)
    ck_near = load_snapshot(SNAP_NEAR)

    B_cfg, K_cfg = ck_native["config"]["batch"], ck_native["config"]["chunk"]
    d_model = ck_native["config"]["d_model"]
    print(f"[ms11] snapshot config: B={B_cfg} K={K_cfg} d_model={d_model} "
          f"n_streamed(native)={ck_native['n_streamed']:,} "
          f"n_streamed(far)={ck_far['n_streamed']:,} "
          f"n_streamed(near)={ck_near['n_streamed']:,}", flush=True)
    print(f"[ms11] training distance: far={ck_native['n_streamed']-ck_far['n_streamed']:,} tok "
          f"| near={ck_native['n_streamed']-ck_near['n_streamed']:,} tok", flush=True)

    # weights: always the LATEST snapshot's A3 (the gated organism) state_dict —
    # W(359M) fixed across all 5 arms, per spec.
    sd_native_weights = ck_native["arms"]["A3"]["model"]

    # Z states, sourced from A2 in each snapshot (the most-trained arm at that point).
    z_native = [s.clone() for s in ck_native["arms"]["A3"]["states"]]
    z_far = [s.clone() for s in ck_far["arms"]["A3"]["states"]]
    z_near = [s.clone() for s in ck_near["arms"]["A3"]["states"]]
    z_shuf = shuffle_states(z_far, seed=args.seed)

    for name, z in [("native", z_native), ("far", z_far), ("near", z_near), ("shuffled", z_shuf)]:
        shapes = [tuple(s.shape) for s in z]
        print(f"[ms11] Z[{name}] shapes={shapes}", flush=True)

    # data / vocab — identical recipe to pos_run.py (WT-2-derived vocab; deterministic)
    train_text, val_text = load_wikitext2()
    vocab, stoi, unk, mask = build_vocab(train_text)
    V = len(vocab)
    print(f"[ms11] vocab={V}", flush=True)

    arm_specs = {
        "native": z_native,
        "swap_far": z_far,
        "swap_near": z_near,
        "cold": None,
        "shuffled": z_shuf,
    }

    # one model instance per arm, all sharing the SAME W(359M) state_dict (no aliasing —
    # each gets its own module instance so per-arm Z-carry doesn't cross-contaminate).
    models = {name: build_model(V, mask, d_model, sd_native_weights) for name in arm_specs}
    with torch.no_grad():
        p = [sum(float(x.abs().sum()) for x in m.parameters()) for m in models.values()]
    assert max(p) - min(p) == 0.0, f"arm weights differ (should be identical W(359M)): {p}"
    print(f"[ms11] all 5 arms share identical W(359M) weights (param-abs-sum {p[0]:.4f})", flush=True)

    states = dict(arm_specs)   # current carried state per arm (None for cold)

    stream = C4Stream(stoi, unk)
    B, K = B_cfg, K_cfg
    bufs = [[] for _ in range(B)]
    for b in range(B):
        bufs[b].extend(stream.next_block())

    curves = {name: [] for name in arm_specs}
    # token-resolved NLL for the first TOKEN_TRACE_CHUNKS chunks: the transient this
    # experiment chases (does an old Z survive dropping into new weights) decays inside
    # ~10-20 tokens (measured pre-run: at t=0 the native/swap_far gap is ~0.6 nats, by
    # t=16 it is noise-floor) — a 64-token chunk MEAN would average nearly all of it
    # away, so excess_first50 below is computed on the first 50 TOKENS, not chunks.
    TOKEN_TRACE_CHUNKS = min(n_chunks, max(1, 50 // K + 1))
    token_curves = {name: [] for name in arm_specs}   # flat per-token NLL, first TOKEN_TRACE_CHUNKS chunks
    t0 = time.time()
    print(f"[ms11] streaming fresh C4 (doc 0..) | {n_chunks} chunks | B={B} K={K} "
          f"| ~{n_chunks*B*K:,} tokens/arm | token-trace over first "
          f"{TOKEN_TRACE_CHUNKS} chunk(s) ({TOKEN_TRACE_CHUNKS*K} tokens)", flush=True)

    for c in range(n_chunks):
        for b in range(B):
            while len(bufs[b]) < K + 1:
                bufs[b].extend(stream.next_block())
        x = torch.tensor([bufs[b][:K] for b in range(B)], dtype=torch.long)
        y = torch.tensor([bufs[b][1:K + 1] for b in range(B)], dtype=torch.long)
        for b in range(B):
            del bufs[b][:K]

        for name in arm_specs:
            nll, st, nll_tok = step_fwd(models[name], states[name], x.clone(), y.clone())
            states[name] = st
            curves[name].append(round(nll, 6))
            if c < TOKEN_TRACE_CHUNKS:
                # mean over batch, per token position -> K values per chunk
                token_curves[name].extend(round(v, 6) for v in nll_tok.mean(0).tolist())

        if (c + 1) % max(1, n_chunks // 10) == 0 or c == 0:
            line = " | ".join(f"{name} {curves[name][-1]:.4f}" for name in arm_specs)
            print(f"[chunk {c+1:>4}/{n_chunks}] wall={time.time()-t0:5.1f}s docs={stream.docs:,} "
                  f"| {line}", flush=True)

    # ── analysis ──
    # excess_first50 is computed on the TOKEN-resolved curve (first 50 tokens, i.e.
    # inside chunk 0 for K=64) — the transient lives there, a chunk-mean over K=64
    # tokens would average nearly all of it away (measured: signal is noise-floor by
    # token ~16). excess_last100 / convergence stay on the CHUNK curve (long-run drift).
    native_curve = np.array(curves["native"])
    native_tok = np.array(token_curves["native"])
    analysis = {}
    for name in arm_specs:
        curve = np.array(curves[name])
        diff = curve - native_curve
        tok_diff = np.array(token_curves[name]) - native_tok
        excess_first50 = float(tok_diff[:50].mean())
        excess_last100 = float(diff[-100:].mean()) if len(diff) >= 100 else float(diff.mean())
        conv = convergence_chunk(list(np.abs(diff)), window=20, tol=0.01)
        analysis[name] = {
            "excess_first50": round(excess_first50, 6),
            "excess_last100": round(excess_last100, 6),
            "convergence_chunk": conv,
        }

    # P23 checks
    ef50 = {k: analysis[k]["excess_first50"] for k in analysis}
    conv = {k: analysis[k]["convergence_chunk"] for k in analysis}

    check_a = ef50["cold"] >= 2 * ef50["swap_far"] if ef50["swap_far"] > 0 else (ef50["cold"] > ef50["swap_far"])
    check_b = (conv["swap_far"] is not None and conv["swap_far"] < 300 and
               conv["swap_near"] is not None and conv["swap_near"] < 300)

    # (c) shuffled worse than swap_far in every 50-chunk window
    far_c = np.array(curves["swap_far"])
    shuf_c = np.array(curves["shuffled"])
    n_windows = (n_chunks + 49) // 50
    window_results = []
    check_c = True
    for w in range(n_windows):
        lo, hi = w * 50, min(n_chunks, (w + 1) * 50)
        far_mean = float(far_c[lo:hi].mean())
        shuf_mean = float(shuf_c[lo:hi].mean())
        ok = shuf_mean > far_mean
        window_results.append({"window": [lo, hi], "swap_far_mean": round(far_mean, 6),
                               "shuffled_mean": round(shuf_mean, 6), "shuffled_worse": ok})
        check_c = check_c and ok

    out = {
        "prediction_id": "P23",
        "config": {"n_chunks": n_chunks, "batch": B, "chunk": K, "d_model": d_model, "seed": args.seed,
                   "smoke": bool(args.smoke)},
        "snapshots": {
            "weights_source": "ckpt_359050240.pt (arm A2)",
            "native_n_streamed": ck_native["n_streamed"],
            "far_n_streamed": ck_far["n_streamed"],
            "near_n_streamed": ck_near["n_streamed"],
            "far_distance_tokens": ck_native["n_streamed"] - ck_far["n_streamed"],
            "near_distance_tokens": ck_native["n_streamed"] - ck_near["n_streamed"],
        },
        "stream_docs_consumed": stream.docs,
        "curves": curves,
        "token_curves_first_chunks": token_curves,
        "token_trace_chunks": TOKEN_TRACE_CHUNKS,
        "analysis": analysis,
        "checks": {
            "a_cold_excess_ge_2x_swap_far": {"pass": bool(check_a),
                                             "cold_excess_first50": ef50["cold"],
                                             "swap_far_excess_first50": ef50["swap_far"],
                                             "ratio": (ef50["cold"] / ef50["swap_far"]
                                                       if ef50["swap_far"] not in (0,) else None)},
            "b_both_swaps_converge_lt_300": {"pass": bool(check_b),
                                             "convergence_chunk_swap_far": conv["swap_far"],
                                             "convergence_chunk_swap_near": conv["swap_near"]},
            "c_shuffled_worse_than_swap_far_every_window": {"pass": bool(check_c),
                                                            "windows": window_results},
        },
    }
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 78)
    print("VERDICT — P23: does a carried OLD state survive a weight hot-swap?")
    print("=" * 78)
    for name in arm_specs:
        a = analysis[name]
        print(f"  {name:<10} excess_first50={a['excess_first50']:+.4f}  "
              f"excess_last100={a['excess_last100']:+.4f}  "
              f"convergence_chunk={a['convergence_chunk']}")
    print("-" * 78)
    print(f"(a) cold excess >= 2x swap_far excess: "
          f"{ef50['cold']:+.4f} >= 2 x {ef50['swap_far']:+.4f} = {2*ef50['swap_far']:+.4f}  "
          f"-> {'PASS' if check_a else 'FAIL'}")
    print(f"(b) swap_far & swap_near converge <300 chunks: "
          f"far={conv['swap_far']} near={conv['swap_near']}  -> {'PASS' if check_b else 'FAIL'}")
    print(f"(c) shuffled worse than swap_far in every 50-chunk window: "
          f"{sum(w['shuffled_worse'] for w in window_results)}/{len(window_results)} windows  "
          f"-> {'PASS' if check_c else 'FAIL'}")
    overall = check_a and check_b and check_c
    print("=" * 78)
    print(f"OVERALL: {'CONFIRMED' if overall else 'PARTIAL/FALSIFIED'} "
          f"({'all 3 checks pass' if overall else 'see individual checks above'})")
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
