#!/usr/bin/env python3 -u
"""
GAP LADDER — how far does the clamp+refresh prosthesis carry a binding, in
tokens, before the wall shows up?
================================================================================
MISSION (P35, analysis/PREDICTIONS.md, Wave 7): eval-only, on the M3-recipe
model with the P30 clamp+refresh prosthesis (src/holo_alpha_shut.py's
eval_gap_recall_chunked_clamprefresh: filler-span alpha clamped to 0 +
magnitude-refreshed to per-channel unit norm at every filler chunk boundary
-- a pure state op that preserves phase EXACTLY by construction, see
holo_alpha_shut.py:720-751 and analysis/HOLO_CARRIER_THEORY.md Sec.1), how
far does recall survive as the gap G climbs toward a million tokens:
{4096, 16384, 65536, 262144, 1048576}, chunked+carried throughout (the only
way G=1M is tractable at all -- O(1) state per chunk, not O(G) attention).

TWO independently-carried ladders, per the mission:

  (a) MQAR ladder (P35a, P35c): the keyed P=2 gap-recall task
      (holo_stream_recall.make_gap_mqar_batch, reused unmodified), M3-recipe
      model (src/holo_alpha_shut.py's _build_lm_alphashut +
      train_fullseq_curriculum_alphashut, lambda=0 -- byte-identical build
      path to P25/P29/P30), evaluated with the clamp+refresh arm
      (clamp_factor=0.0, refresh=True -- the SAME prosthesis P30 proved
      pushes the knee past 2048) at every rung. P35a: recall@65536 >=
      0.5 x recall@4096. Zeroed-at-gap null at every rung, eval_batch>=100
      (P36's protocol, reused via holo_alpha_shut.eval_gap_recall_chunked_
      zeroed_clamprefresh).

  (b) Beacon ladder (P35b, P35c): the 1-bit write-once-freeze carrier
      (streaming_train.py Sec.D/E via beacon_swap.py's harness --
      write_and_carry / probe_with_states, reused unmodified), trained to
      criterion recall>=0.99@256 (beacon_swap.train_beacon_to_criterion,
      reused unmodified), then walked across the SAME G-ladder chunked+
      carried, in TWO arms: WITH magnitude-refresh at filler chunk
      boundaries and WITHOUT (raw gamma-decay). The beacon carrier's state
      Z is REAL-valued (streaming_train.StreamingScanLayer: Z_t = gamma_t *
      Z_{t-1} + a_t -- no complex phase, unlike the holographic S_re/S_im),
      so "magnitude refresh" here cannot mean phase-preserving renormal-
      ization (there is no phase to preserve) -- it means compensating the
      SAME multiplicative gamma-decay the P30 refresh compensates on the
      complex accumulator, by the exact analogous operation: rescale the
      carried Z by the inverse of its decay factor over the elapsed chunk,
      channel-by-channel, a pure scalar op between chunks (never inside the
      layer forward), eps-guarded so a genuinely-dead channel (|Z|~0,
      write-once-freeze never ignited or already zeroed) is left untouched
      -- see refresh_beacon_state below, the direct structural analogue of
      holo_alpha_shut.refresh_magnitude. The trained carrier's own gamma
      (~0.9995, per HOLO_CARRIER_THEORY.md's measured tau~2000) predicts it
      dies well before G=1M WITHOUT the refresh and survives WITH it -- that
      contrast (not the raw survival number) is P35b's point: "gamma=0.9995
      alone decays at tau~2000 -- the refresh is load-bearing and that is
      the point" (mission text, verbatim). Cold control (state zeroed right
      before the probe) is the collapse-to-chance floor, run at every rung
      in both arms for reference.

CHUNK SIZE FOR LARGE G: chunked-forward correctness does not depend on chunk
size -- check_equivalence(..., chunk=C) is bit-exact (max|delta|=0.0, verified
at chunk in {16,256,1024,8192} in this file's own smoke self-test) because the
carried state is the ONLY channel between chunks; a bigger chunk changes
nothing about what gets computed, only how many Python-level forward() calls
it takes. Empirically (measured on this Mac, P_max=V_max=F=16, d_model=64,
batch=20): the eval wall-time is ~0.16-0.18 ms/token at G>=16384 and is
essentially FLAT across chunk in {16,256,1024} -- the per-token cost is
dominated by the scan's Python loop and batch/filler generation, not by the
number of chunk-boundary Python calls. So a larger chunk buys negligible
speed here (unlike, say, memory) -- CHUNK_LADDER below still scales chunk up
with G (16 -> 1024 -> 4096) for memory/call-count hygiene at G=1M (65536
Python-level forward() calls at chunk=16 vs 256 at chunk=4096), verified
against the chunk=16 baseline via the SAME equivalence gate at every chunk
size actually used, every run.

OUTPUT: results/gap_ladder.json (--smoke: results/gap_ladder_smoke.json,
G in {4096,16384} for MQAR + G in {4096} for beacon, 1 seed, reduced
training iters). Per rung: MQAR clamp+refresh accuracy + zeroed-null
accuracy (eval_batch>=100); beacon with-refresh / without-refresh / cold
bit-recall. VERDICT blocks for P35a/P35b/P35c.

CPU-only (mps disabled), torch.set_num_threads(1), os.nice(19) best-effort,
seeds fixed. Does not modify holo_alpha_shut.py, beacon_swap.py,
holo_stream_recall.py, streaming_train.py, or holo_mag_read.py -- imports
and reuses their machinery unmodified (per the mission: "you use it again").
"""
import os
import sys
import json
import time
import argparse

try:
    os.nice(19)
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference"))

import torch

torch.backends.mps.is_available = lambda: False   # force CPU (repo convention)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass
torch.set_num_threads(1)

from holo_stream_recall import _gap_vocab, check_equivalence, make_gap_mqar_batch   # noqa: E402
from holo_alpha_shut import (   # noqa: E402 -- the P29/P30 machinery, reused unmodified
    _build_lm_alphashut, train_fullseq_curriculum_alphashut,
    eval_gap_recall_chunked_clamprefresh, eval_gap_recall_chunked_zeroed_clamprefresh,
    check_equivalence_magnorm, REFRESH_EPS,
)
from streaming_train import StreamingNoPELM, _beacon_vocab, _make_beacon_batch   # noqa: E402
from beacon_swap import (   # noqa: E402 -- the MS13 harness, reused unmodified
    train_beacon_to_criterion, load_ckpt, eval_recall_at_gap,
    write_and_carry, probe_with_states, zero_states,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(REPO, "results")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Chunk-size choice per rung — bigger chunk at bigger G (call-count/memory
#    hygiene; correctness is chunk-size-independent for the MQAR/holographic
#    path, verified below). The BEACON path is DIFFERENT and gets its own
#    function (see chunk_for_beacon_gap) -- correctness there is NOT
#    chunk-size-independent, see that function's docstring for the measured
#    reason.
# ═══════════════════════════════════════════════════════════════════════════
def chunk_for_gap(G, base_chunk=16):
    """MQAR/holographic path only. Scale the streaming chunk length up with G
    so the number of Python-level forward() calls stays bounded at the top
    of the ladder (G=1048576 at chunk=16 is 65536+ calls; at chunk=4096 it
    is ~256). Equivalence is verified (not assumed) at whatever chunk this
    returns, every run, via check_equivalence(..., chunk=chunk) -- the
    holographic accumulator's chunked-forward is bit-exact at any chunk
    size (measured max|delta|=0.0 at chunk in {16,256,1024,8192} during this
    file's own build), because the only channel between chunks is the
    carried complex state itself."""
    if G >= 262144:
        return 4096
    if G >= 16384:
        return 1024
    return base_chunk


def chunk_for_beacon_gap(G, max_chunk=64):
    """BEACON path only -- deliberately NOT the same function as
    chunk_for_gap, and deliberately capped small regardless of G. Unlike the
    holographic accumulator, chunked-forward correctness for the beacon
    carrier IS chunk-size-sensitive: refresh_beacon_state needs each
    chunk's CUMULATIVE gamma product (prod_t gamma_t over the chunk) to
    stay above float32's denormal floor. Measured on this file's own
    trained model (train_gap=256, criterion 0.99, real carrier gamma per-
    token ~0.9998): the product underflows to exactly 0.0 for some
    (batch, channel) cells well before chunk=1024 -- at chunk=1024 that
    underflow (even WITH BEACON_MAX_GAIN capping the resulting 1/0-clamped
    rescale) collapsed a from-scratch-perfect refreshed arm to 0.575 at
    G=65536 and to 0.5 (chance) at G=262144; chunk=256 was already too
    coarse at G=262144 (0.5); chunk=128 gave 0.6; chunk=64 gave 1.0 at every
    G tested up to 262144. This function therefore returns a FIXED small
    chunk (default 64) independent of G -- more Python-level forward()
    calls at the top of the ladder (G=1048576 at chunk=64 is 16384 calls,
    vs 256 at chunk=4096), but that cost is affordable (measured ~100s at
    G=1048576, NB=20, well inside the mission's ~2h budget) and correctness
    is not negotiable."""
    return max_chunk


# ═══════════════════════════════════════════════════════════════════════════
# 2. (a) MQAR ladder — M3-recipe model, clamp+refresh arm (the P30 prosthesis),
#    walked across G, zeroed-at-gap null at every rung (P36 protocol).
# ═══════════════════════════════════════════════════════════════════════════
def run_mqar_ladder(args):
    P_max, V_max, F = args.p_max, args.v_max, args.f_fillers
    key_lo, val_lo, fill_lo, vocab_size = _gap_vocab(P_max, V_max, F)
    mask_idx = vocab_size
    P = args.pairs
    chance = 1.0 / V_max

    Gs = [int(g) for g in args.mqar_gaps.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    null_eval_batch = max(100, args.eval_batch)

    print("=" * 78)
    print("GAP LADDER (a) — MQAR, M3-recipe model, clamp+refresh prosthesis (P30 arm)")
    print(f"P={P}  P_max={P_max} V_max={V_max} F={F} vocab={vocab_size}  chance={chance:.4f}")
    print(f"gaps={Gs}  seeds={seeds}  eval_batch={args.eval_batch}  null_eval_batch={null_eval_batch}")
    print("=" * 78)

    out = {"config": vars(args), "chance": chance, "rungs": {}, "equivalence_by_chunk": {},
          "recipe": "M3 V2_kickstart_magnorm, lambda=0, trained ONCE per seed (identical build "
                    "path to P25/P29/P30). Eval arm: clamp_factor=0.0, refresh=True (the P30 "
                    "clamp+refresh prosthesis) via holo_alpha_shut.eval_gap_recall_chunked_"
                    "clamprefresh, reused unmodified. Zeroed-at-gap null at every rung via "
                    "eval_gap_recall_chunked_zeroed_clamprefresh, eval_batch forced >=100 (P36 "
                    "protocol)."}

    t0 = time.time()
    for seed in seeds:
        print(f"\n{'='*78}\nseed={seed}  (training M3 recipe, lambda=0, ONCE)\n{'='*78}")
        torch.manual_seed(seed)
        model = _build_lm_alphashut(vocab_size, mask_idx, args.d_model, args.n_layers,
                                    args.n_heads, args.d_head)
        curr = train_fullseq_curriculum_alphashut(
            model, P, args.g_train_max, args.iters, args.lr, seed, args.batch, 0.0,
            P_max, V_max, F, log_every=args.log_every, g_start=args.g_start,
            patience=args.patience, bar=args.curriculum_bar)
        print(f"  curriculum done: final_train_gap={curr['final_train_gap']} "
              f"final_train_acc={curr['final_train_acc']:.3f}")
        out.setdefault("curriculum", {})[f"seed{seed}"] = curr

        by_G = {}
        for G in Gs:
            chunk = chunk_for_gap(G, base_chunk=args.chunk)
            ck = str(chunk)
            if ck not in out["equivalence_by_chunk"]:
                eq_ref = check_equivalence(vocab_size, mask_idx, seed=0, T=48, chunk=chunk, use_phase=True)
                eq_mag = check_equivalence_magnorm(vocab_size, mask_idx, seed=0, T=48, chunk=chunk,
                                                   d_model=args.d_model, n_layers=args.n_layers,
                                                   n_heads=args.n_heads, d_head=args.d_head)
                eq_ok = eq_ref < 1e-5 and eq_mag < 1e-5
                out["equivalence_by_chunk"][ck] = {"reference_streaming": eq_ref, "magnorm": eq_mag,
                                                    "passed": eq_ok}
                print(f"  [equivalence @ chunk={chunk}] ref={eq_ref:.3e}  magnorm={eq_mag:.3e}  "
                      f"{'PASS' if eq_ok else 'FAIL — do not trust this chunk size'}")
                if not eq_ok:
                    out["ABORTED"] = f"equivalence gate failed at chunk={chunk}"
                    os.makedirs(RESULTS, exist_ok=True)
                    json.dump(out, open(args.out, "w"), indent=2)
                    print(f"\n→ {args.out} (ABORTED)")
                    return out

            eb = args.eval_batch
            t_g0 = time.time()
            acc = eval_gap_recall_chunked_clamprefresh(model, P, G, eb, seed + 1000 + G,
                                                        P_max, V_max, F, chunk, 0.0, True)
            wall_acc = time.time() - t_g0

            t_g1 = time.time()
            acc0 = eval_gap_recall_chunked_zeroed_clamprefresh(model, P, G, null_eval_batch,
                                                                seed + 2000 + G, P_max, V_max, F,
                                                                chunk, 0.0, True)
            wall_null = time.time() - t_g1

            by_G[G] = acc
            rk = f"seed{seed}|G{G}"
            dev_pp = round(abs(acc0 - chance) * 100, 3)
            out["rungs"][rk] = {"seed": seed, "G": G, "chunk": chunk, "eval_batch": eb,
                                "accuracy": round(acc, 4), "wall_s_accuracy_eval": round(wall_acc, 2),
                                "zeroed_null_eval_batch": null_eval_batch,
                                "zeroed_null_accuracy": round(acc0, 4),
                                "wall_s_null_eval": round(wall_null, 2),
                                "chance": round(chance, 4), "null_deviation_pp": dev_pp,
                                "null_within_3pp_of_chance": bool(dev_pp <= 3.0)}
            print(f"    G={G:>8} chunk={chunk:>5}: acc={acc:.4f} ({wall_acc:.1f}s)  "
                  f"zeroed-null={acc0:.4f} (dev {dev_pp}pp, {wall_null:.1f}s)")

        ref_G = Gs[0]
        ref_acc = by_G.get(ref_G, 0.0)
        out.setdefault("per_seed_summary", {})[f"seed{seed}"] = {
            "ref_G": ref_G, "acc_at_ref_G": round(ref_acc, 4),
            "acc_by_G": {str(g): round(v, 4) for g, v in by_G.items()},
            "ratio_to_ref": {str(g): (round(v / ref_acc, 4) if ref_acc > 1e-9 else None)
                            for g, v in by_G.items()},
        }

    out["elapsed_s_mqar"] = round(time.time() - t0, 1)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 3. (b) Beacon ladder — real-valued carrier (Z_t = gamma_t*Z_{t-1} + a_t, no
#    phase). "Magnitude refresh" here compensates the SAME gamma-decay the
#    holographic refresh compensates, by the direct structural analogue:
#    rescale the carried Z by the inverse of the decay it underwent over the
#    elapsed filler chunk. Eps-guarded (a dead/never-ignited channel is left
#    untouched, exactly refresh_magnitude's contract) so this cannot invent
#    information in a zeroed/cold state either.
#
#    LAYER SCOPE: unlike P30's holographic refresh (correctly scoped to
#    layer 0, where that repo's accumulator lives), the beacon carrier is
#    NOT guaranteed to live in layer 0 -- diagnose_carrier (beacon_swap.py)
#    measured on this file's own trained model (train_gap=256, criterion
#    0.99) that the actual write-once-freeze channel (corr=0.996,
#    gamma~0.9998, alpha_gap~0.013 -- the canonical frozen-carrier
#    signature) sits in LAYER 1, not layer 0 (layer 0's best-correlated
#    channel there measured corr=-0.154, i.e. not the carrier at all). A
#    layer-0-only refresh (the naive port of holo_alpha_shut's convention)
#    would therefore refresh the WRONG layer and leave the actual carrier to
#    decay untouched -- verified as a real bug during this file's own
#    self-smoke (with-refresh == without-refresh == cold, all identical,
#    because layer 1's true carrier was never touched). Fix: refresh EVERY
#    layer's carried state, each with ITS OWN gamma_step, at every filler
#    chunk boundary -- exactly how model.zero_states (the P30c null) already
#    zeros the ENTIRE carried state across all layers, not just layer 0.
# ═══════════════════════════════════════════════════════════════════════════
BEACON_REFRESH_EPS = 1e-6


BEACON_MAX_GAIN = 1e3
"""Ceiling on the per-chunk 1/gamma_step rescale factor. A channel's
CUMULATIVE gamma product over a chunk (prod_t gamma_t) underflows to exactly
0.0 in float32 well before the individual per-token gamma values look small
(e.g. gamma~0.9998 for every one of 1024 tokens still products down toward
denormal range for some, batch/channel-dependent) -- and 1/0.0-clamped-to-
1e-6 then applies a 1e6x gain that BLOWS UP that channel's state (measured:
|Z| reaching ~6.6e6 on this file's own trained model at chunk=1024, which
poisons the readout for every channel downstream, not just the exploded one
-- this was the actual cause of the chunk=1024 recall collapse caught during
this file's self-smoke, not a per-carrier-channel effect). The physically
correct read of prod_gamma==0 is "this channel's information was already
fully gone before this chunk ended" -- amplifying it 1e6x manufactures a
huge fake signal out of what is actually total information loss, exactly
the failure mode REFRESH_EPS/eps-guards exist to prevent elsewhere in this
repo (holo_alpha_shut.refresh_magnitude's docstring: "a refresh that
resurrects a zeroed state would violate [the null gate] by construction").
Capping the gain at 1e3 (comfortably above any single-chunk decay this
ladder's chunk sizes are expected to produce for a genuinely-surviving
carrier, per the measured chunk<=256 case where max gain stayed <<1e3) means
a channel that underflowed is refreshed as far as is defensible and no
further, rather than exploding -- the empirical fix is choosing chunk sizes
where prod_gamma does not underflow in the first place (see
chunk_for_beacon_gap below), with this cap as a hard backstop."""


def refresh_beacon_state(states, gamma_steps, eps=BEACON_REFRESH_EPS, max_gain=BEACON_MAX_GAIN):
    """Rescale EVERY layer's carried real state Z by 1/gamma_step (that
    layer's own per-channel CUMULATIVE decay over the elapsed chunk --
    gamma_steps must be prod(gamma_t) over the chunk, not mean(gamma_t)),
    channel-by-channel. states, gamma_steps: same-length lists, one (B,H,D)
    tensor per layer. This is the direct structural analogue of
    holo_alpha_shut.refresh_magnitude: that function undoes accumulated
    magnitude decay on the complex |S| by dividing by the CURRENT magnitude
    (S <- S/|S|, renormalizing to unit norm every call); the real-valued
    beacon carrier has no unit-norm target (Z is unbounded, driven by
    log(1-w) which is <=0), so the analogous "undo what this chunk's decay
    just did" operation is dividing by gamma_step instead of the magnitude
    -- both are pure scalar rescalings of the carried state between chunks,
    never touching the layer forward. TWO guards, both needed: (1) a channel
    with |Z|<eps (dead: write-once-freeze never ignited, or genuinely
    zeroed) is passed through UNCHANGED -- the same eps-guard contract as
    refresh_magnitude; (2) the rescale factor itself is capped at max_gain
    -- protects against gamma_step underflowing to exactly 0.0 (a real,
    measured failure mode at large chunk sizes, see BEACON_MAX_GAIN) turning
    a single dead-ish channel into a state-destroying outlier that poisons
    every other channel downstream through the shared FFN/LN."""
    out = []
    for z, g in zip(states, gamma_steps):
        alive = z.abs() >= eps
        inv_decay = (1.0 / g.clamp(min=1e-6)).clamp(max=max_gain)
        out.append(torch.where(alive, z * inv_decay, z))
    return out


@torch.no_grad()
def eval_beacon_ladder_chunked(model, G, NB, gen, F_, beta0, beta1, probe, chunk, refresh, dev="cpu"):
    """Chunked+carried bit-recall over a beacon episode of gap G: [beacon]
    [G fillers][probe], forwarded chunk-by-chunk (the deployment path, same
    idiom as holo_stream_recall.chunked_forward / holo_alpha_shut's clamp+
    refresh sweep). When refresh=True, EVERY layer's carried Z is rescaled
    by refresh_beacon_state at every FILLER chunk boundary (never during the
    beacon token's own chunk, never during the probe's), using EACH layer's
    OWN gamma for that chunk (captured via return_internals=True on that
    forward call -- gamma is a genuine per-forward-call quantity, not a
    fixed constant, since it is itself a function of the input x) as the
    exact decay that chunk just applied to that layer. This directly
    targets the "gamma~0.9995 alone decays at tau~2000" failure mode the
    mission names: without refresh, |Z_carrier| ~ gamma^G shrinks to noise
    well before G=1M; refresh compensates exactly that decay, chunk by
    chunk, on whichever layer actually carries the bit (empirically layer 1
    on this file's trained model -- see the module-level note above)."""
    x, k = _make_beacon_batch(NB, G, gen, F_, beta0, beta1, probe)
    x = x.to(dev)
    x_beacon, x_fill, x_probe = x[:, :1], x[:, 1:1 + G], x[:, -1:]

    model.eval()
    # beacon token: always unrefreshed (the write itself is never touched)
    logits_b, states = model(x_beacon, None)

    pos = 0
    T = x_fill.shape[1]
    while pos < T:
        hi = min(T, pos + chunk)
        xc = x_fill[:, pos:hi]
        if refresh:
            h = model.embed(xc)
            new_states = []
            gamma_steps = []
            hcur = h
            for li, layer in enumerate(model.layers):
                y, z_fin, internals = layer.scan(hcur, states[li], return_internals=True)
                # the CUMULATIVE multiplicative decay over the chunk is the PRODUCT of
                # per-token gamma, prod_t(gamma_t) -- NOT mean(gamma_t)**chunklen. Verified
                # (self-smoke bug caught here): with gamma_step per-token averaging ~1 (e.g.
                # 0.9998) the naive mean(gamma)**T estimate implies ~20% decay over a 1024-
                # token chunk, but the true product-decay measured on the actual trained
                # carrier channel was ~99.5% (ratio ~0.005) -- because gamma varies token to
                # token and a handful of low-gamma tokens dominate the PRODUCT even though
                # they barely move the ARITHMETIC MEAN. prod(dim=1) is the exact quantity
                # that inverts what stateful_scan's recurrence actually multiplied by.
                gamma_steps.append(internals["gamma"].prod(dim=1))   # (B,H,D): this layer's EXACT cumulative decay this chunk
                new_states.append(z_fin)
                hcur = layer.ln1(hcur + y)
                hcur = layer.ln2(hcur + layer.ffn(hcur))
            states = refresh_beacon_state(new_states, gamma_steps)
        else:
            _, states = model(xc, states)
        pos = hi

    # probe token: always unrefreshed (the readout itself is never touched)
    logits_p = probe_with_states(model, x_probe, states)
    acc = float((logits_p.argmax(-1).cpu() == k).float().mean())
    return acc


def run_beacon_ladder(args):
    F_, fillers, beta0, beta1, probe, V = _beacon_vocab()
    mask = V - 1
    dev = "cpu"
    Gs = [int(g) for g in args.beacon_gaps.split(",")]

    print("\n" + "=" * 78)
    print("GAP LADDER (b) — Beacon, write-once-freeze carrier, with/without magnitude-refresh")
    print(f"gaps={Gs}  criterion=recall>={args.beacon_criterion}@{args.beacon_criterion_gap}")
    print("=" * 78)

    beacon_args = argparse.Namespace(d_model=args.beacon_d_model, batch=args.beacon_batch,
                                     train_gap=args.beacon_criterion_gap,
                                     criterion_gap=args.beacon_criterion_gap,
                                     criterion=args.beacon_criterion,
                                     max_iters=args.beacon_max_iters,
                                     check_every=args.beacon_check_every)
    gen_train = torch.Generator().manual_seed(args.beacon_seed)
    t0 = time.time()
    model, t1_sd, t1_iters, t2_sd, t2_iters = train_beacon_to_criterion(
        beacon_args, V, mask, F_, beta0, beta1, probe, dev, gen_train, args.beacon_seed)
    train_wall = time.time() - t0

    gen_check = torch.Generator().manual_seed(args.beacon_seed + 555)
    recall_at_criterion = eval_recall_at_gap(model, args.beacon_criterion_gap, gen_check, F_, beta0,
                                             beta1, probe, n=200, dev=dev)
    print(f"[beacon] trained: iters={t1_iters}  recall@{args.beacon_criterion_gap}="
          f"{recall_at_criterion:.4f}  train_wall={train_wall:.1f}s")

    out = {"config_beacon": vars(beacon_args), "criterion_check": {
              "recall_at_criterion_gap": round(recall_at_criterion, 4),
              "criterion": args.beacon_criterion, "criterion_gap": args.beacon_criterion_gap,
              "iters": t1_iters, "train_wall_s": round(train_wall, 1),
              "passed": bool(recall_at_criterion >= args.beacon_criterion)}}
    if recall_at_criterion < args.beacon_criterion:
        out["ABORTED"] = "beacon model did not clear criterion recall -- ladder not run"
        return out

    rungs = {}
    for G in Gs:
        chunk = chunk_for_beacon_gap(G, max_chunk=args.beacon_chunk)
        gen = torch.Generator().manual_seed(args.beacon_seed + 9000 + G)
        NB = args.beacon_n_episodes

        t_r0 = time.time()
        acc_refresh = eval_beacon_ladder_chunked(model, G, NB, gen, F_, beta0, beta1, probe,
                                                  chunk, refresh=True, dev=dev)
        wall_refresh = time.time() - t_r0

        gen2 = torch.Generator().manual_seed(args.beacon_seed + 9000 + G)   # same episodes, no-refresh arm
        t_r1 = time.time()
        acc_norefresh = eval_beacon_ladder_chunked(model, G, NB, gen2, F_, beta0, beta1, probe,
                                                    chunk, refresh=False, dev=dev)
        wall_norefresh = time.time() - t_r1

        # cold control: state zeroed right before the probe (collapse-to-chance floor)
        gen3 = torch.Generator().manual_seed(args.beacon_seed + 9000 + G)
        x, k = _make_beacon_batch(NB, G, gen3, F_, beta0, beta1, probe)
        x_bg, x_probe = x[:, :-1], x[:, -1:]
        with torch.no_grad():
            # chunked+carried like the WITH/WITHOUT arms. The original single
            # write_and_carry(model, x_bg, None) call forwarded (NB, G) in ONE
            # pass, materializing O(G) per-token internals (logits, hidden,
            # S_re/S_im/Z_seq, drives, phases) -- ~25.6 GB at G=65536, the
            # cause of BOTH beast OOM kills of the beacon half (a single
            # sub-30s allocation burst, invisible to a 30s RSS watchdog).
            # Chunked-carry is bit-exact for this operator family (F2), so
            # the cold control's semantics are unchanged.
            states_cold = None
            posc, Tc = 0, x_bg.shape[1]
            while posc < Tc:
                hic = min(Tc, posc + chunk)
                states_cold = write_and_carry(model, x_bg[:, posc:hic], states_cold)
                posc = hic
            logits_cold = probe_with_states(model, x_probe, zero_states(states_cold))
            acc_cold = float((logits_cold.argmax(-1).cpu() == k).float().mean())

        rungs[str(G)] = {"G": G, "chunk": chunk, "n_episodes": NB,
                         "recall_with_refresh": round(acc_refresh, 4),
                         "wall_s_with_refresh": round(wall_refresh, 2),
                         "recall_without_refresh": round(acc_norefresh, 4),
                         "wall_s_without_refresh": round(wall_norefresh, 2),
                         "recall_cold": round(acc_cold, 4),
                         "refresh_advantage_pp": round((acc_refresh - acc_norefresh) * 100, 2)}
        print(f"    G={G:>8} chunk={chunk:>5}: WITH-refresh={acc_refresh:.4f} ({wall_refresh:.1f}s) | "
              f"WITHOUT-refresh={acc_norefresh:.4f} ({wall_norefresh:.1f}s) | cold={acc_cold:.4f} | "
              f"advantage={rungs[str(G)]['refresh_advantage_pp']:+.2f}pp")

    out["rungs"] = rungs
    out["elapsed_s_beacon"] = round(time.time() - t0, 1)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 4. Orchestration + P35 verdicts.
# ═══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="P35: the gap ladder to a million")
    ap.add_argument("--smoke", action="store_true",
                    help="fast sanity: MQAR G in {4096,16384} 1 seed; beacon G in {4096}; "
                         "reduced training iters")
    ap.add_argument("--full", action="store_true",
                    help="the full ladder: MQAR G in {4096,16384,65536,262144,1048576} both "
                         "seeds; beacon over the SAME G ladder")
    # ── MQAR (M3-recipe) config -- mirrors holo_alpha_shut.py's defaults ──
    ap.add_argument("--p-max", type=int, default=16)
    ap.add_argument("--v-max", type=int, default=16)
    ap.add_argument("--f-fillers", type=int, default=16)
    ap.add_argument("--pairs", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-head", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=16, help="base eval chunk; scaled up at large G, see chunk_for_gap")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-batch", type=int, default=100)
    ap.add_argument("--g-start", type=int, default=2)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--curriculum-bar", type=float, default=0.8)
    ap.add_argument("--g-train-max", type=int, default=128)
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--mqar-gaps", default="4096,16384,65536,262144,1048576")
    # ── Beacon config -- mirrors beacon_swap.py's defaults ──
    ap.add_argument("--beacon-d-model", type=int, default=64)
    ap.add_argument("--beacon-batch", type=int, default=64)
    ap.add_argument("--beacon-criterion-gap", type=int, default=256)
    ap.add_argument("--beacon-criterion", type=float, default=0.99)
    ap.add_argument("--beacon-max-iters", type=int, default=6000)
    ap.add_argument("--beacon-check-every", type=int, default=100)
    ap.add_argument("--beacon-seed", type=int, default=42)
    ap.add_argument("--beacon-n-episodes", type=int, default=200)
    ap.add_argument("--beacon-gaps", default="4096,16384,65536,262144,1048576")
    ap.add_argument("--beacon-chunk", type=int, default=64,
                    help="FIXED chunk length for the beacon ladder, independent of G -- see "
                         "chunk_for_beacon_gap's docstring for why this path cannot scale chunk "
                         "up with G the way the MQAR path does (gamma-product underflow)")
    ap.add_argument("--skip-mqar", action="store_true")
    ap.add_argument("--skip-beacon", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.out is None:
        args.out = os.path.join(RESULTS, "gap_ladder.json")

    if args.smoke:
        args.seeds = "0"
        args.mqar_gaps = "4096,16384"
        args.beacon_gaps = "4096"
        args.iters = min(args.iters, 600) if args.iters == 3000 else args.iters
        args.eval_batch = min(args.eval_batch, 40)
        args.beacon_max_iters = min(args.beacon_max_iters, 2500)
        args.beacon_check_every = min(args.beacon_check_every, 50)
        args.beacon_criterion = min(args.beacon_criterion, 0.95)
        args.beacon_criterion_gap = min(args.beacon_criterion_gap, 128)
        args.beacon_n_episodes = min(args.beacon_n_episodes, 40)
        if args.out == os.path.join(RESULTS, "gap_ladder.json"):
            root, ext = os.path.splitext(args.out)
            args.out = root + "_smoke" + ext
    elif args.full:
        args.seeds = "0,1"
        args.mqar_gaps = "4096,16384,65536,262144,1048576"
        args.beacon_gaps = "4096,16384,65536,262144,1048576"

    t0 = time.time()
    result = {"config": vars(args), "prediction_id": "P35"}

    if not args.skip_mqar:
        result["mqar"] = run_mqar_ladder(args)
    if not args.skip_beacon:
        result["beacon"] = run_beacon_ladder(args)

    # ── P35 checks ──
    verdict_bits = []

    p35a_pass = None
    if "mqar" in result and "per_seed_summary" in result["mqar"]:
        ratios_65536 = []
        for seed_summary in result["mqar"]["per_seed_summary"].values():
            r = seed_summary["ratio_to_ref"].get("65536")
            if r is not None:
                ratios_65536.append(r)
        if ratios_65536:
            p35a_pass = all(r >= 0.5 for r in ratios_65536)
            verdict_bits.append(f"P35a (MQAR recall@65536 >= 0.5x recall@4096): "
                                f"{'CONFIRMED' if p35a_pass else 'NOT MET'} (ratios={ratios_65536})")
        else:
            verdict_bits.append("P35a: G=65536 not in the measured MQAR ladder this run -- not scored")

    p35b_pass = None
    if "beacon" in result and "rungs" in result["beacon"]:
        brungs = result["beacon"]["rungs"]
        top_G = max((int(g) for g in brungs), default=None)
        if top_G is not None:
            top = brungs[str(top_G)]
            p35b_pass = top["recall_with_refresh"] >= 0.9
            verdict_bits.append(f"P35b (beacon bit survives G={top_G} at recall>=0.9 WITH refresh): "
                                f"{'CONFIRMED' if p35b_pass else 'NOT MET'} "
                                f"(with-refresh={top['recall_with_refresh']}, "
                                f"without-refresh={top['recall_without_refresh']}, "
                                f"cold={top['recall_cold']}, "
                                f"advantage={top['refresh_advantage_pp']:+.2f}pp)")
            # the mission's actual point: refresh is LOAD-BEARING (with >> without at the top rung)
            refresh_load_bearing = top["refresh_advantage_pp"] > 10.0
            verdict_bits.append(f"P35b mechanism check (refresh load-bearing: with-refresh beats "
                                f"without-refresh by >10pp at G={top_G}): "
                                f"{'CONFIRMED' if refresh_load_bearing else 'NOT MET'} "
                                f"(advantage={top['refresh_advantage_pp']:+.2f}pp)")

    p35c_pass = None
    if "mqar" in result and "rungs" in result["mqar"]:
        null_cells = [v for v in result["mqar"]["rungs"].values()]
        if null_cells:
            p35c_pass = all(c["null_within_3pp_of_chance"] for c in null_cells)
            worst = max(null_cells, key=lambda c: c["null_deviation_pp"])
            verdict_bits.append(f"P35c (zeroed-at-gap null within 3pp of chance, every MQAR rung, "
                                f"eval_batch>=100): {'CONFIRMED' if p35c_pass else 'NOT MET'} "
                                f"(worst: seed{worst['seed']} G={worst['G']} dev={worst['null_deviation_pp']}pp)")

    result["verdict"] = " | ".join(verdict_bits) if verdict_bits else "no rungs measured"
    result["elapsed_s_total"] = round(time.time() - t0, 1)

    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("\n" + "=" * 78)
    print("VERDICT")
    for b in verdict_bits:
        print("  " + b)
    print(f"\n→ {args.out}  ({result['elapsed_s_total']}s)")


if __name__ == "__main__":
    main()
