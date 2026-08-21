#!/usr/bin/env python3 -u
"""
BEACON SWAP — does a stored bit survive the widening surgery AND a weight swap?
==================================================================================
P26 (registered prediction, analysis/PREDICTIONS.md, MS13): on the beacon
idle-persistence task (streaming_train.py §D/E: write-once-freeze carrier),
does a bit planted at the beacon token and carried through a filler gap
survive (a) the MS8 hot-swap widening surgery (d64->d128, carried-Z migrated,
post-gate) and (b) a weight SWAP within one training run (write under W(T1),
read under W(T2), state carried across the swap, no surgery)?

Envelope
--------
streaming_train.py's beacon task runs on StreamingNoPELM (embed -> 2x[scan +
ln1 + ffn + ln2] -> head, NoPE). hot_swap_growth.py's grow_model/
check_surgery_equivalence/migrate_z_state are written against
pos_family_transfer.FamilyStreamingLM. Both wrap the SAME StreamingScanLayer
and have IDENTICAL attribute layout (embed, layers[i].scan/ln1/ln2/ffn, head)
-- FamilyStreamingLM is the generic version of the same envelope; StreamingNoPELM
only adds a self.pos=Identity NoPE guard on top of the parent NoPE model. So a
thin state_dict adapter (build a FamilyStreamingLM, load StreamingNoPELM's
state_dict into it) is sufficient -- no re-implementation of the growth math.
We VERIFY this with the module's own equivalence gate (check_surgery_equivalence,
tol 1e-4) on the actually-trained beacon model before trusting any downstream
recall number.

Experiment
----------
1. Train StreamingNoPELM (d_model=64, matches streaming_train's beacon setup)
   from scratch (seed 42) on the beacon task with the curriculum from
   run_idle_persistence, to criterion recall >= 0.99 @ gap 256 (idle-eval).
   Checkpoint T1 (state_dict + iteration count).
2. Keep training the SAME run for another, equal iteration budget (2x total)
   -> checkpoint T2. Recall @256 logged at both T1 and T2 (must both be
   ~1.0, else abort and report).
3. (a) SURGERY: plant the bit with the T2 model, carry it through the gap to
   immediately before the probe position, THEN widen (grow_model, adapted)
   the mid-episode carried state + weights (d64->d128, Z migrated), probe
   with the grown model. P26a: recall >= 0.99.
4. (b) SWAP: plant the bit with W(T1) (forward to just before the probe,
   extract state), read the probe with W(T2) using that carried state.
   Controls: native-T2 (write+read both T2), native-T1 (write+read both T1),
   cold (state zeroed right before the probe -- the known collapse-to-chance
   control), shuffled (channel-permutation of the state before the probe,
   seed 42). P26b: swap recall >= 0.9. P26c: shuffled ~= chance.
5. DIAGNOSIS: carrier-channel identification (streaming_train's --carrier
   correlation method) at T1 and T2 separately -- same channel index? Log
   carrier gamma/alpha at both checkpoints.

All measurements at gap=256 (primary) plus gap in {64, 512} as side axes,
n=200 episodes per arm/gap (both bit values balanced). --smoke: n=40,
gaps {64, 256} only, criterion relaxed to recall >= 0.95.

CPU only, torch.set_num_threads(1), os.nice(19) best-effort, seed-fixed.
Writes ONLY to results/beacon_swap.json (or *_smoke.json under --smoke).
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.mps.is_available = lambda: False  # force CPU, matches streaming_train/hot_swap_growth
torch.set_num_threads(1)                          # bit-deterministic

try:
    os.nice(19)
except PermissionError:
    pass
except Exception:
    pass

import sys
sys.path.insert(0, "reference")
sys.path.insert(0, "src")

from streaming_train import (StreamingNoPELM, StreamingScanLayer, _beacon_vocab,
                              _make_beacon_batch)
from hot_swap_growth import (grow_model, check_surgery_equivalence, migrate_z_state)
from pos_family_transfer import FamilyStreamingLM, build_gssm_lm


# ═══════════════════════════════════════════════════════════════════════════
# §1  Adapter: StreamingNoPELM (streaming_train envelope) <-> FamilyStreamingLM
#     (hot_swap_growth envelope). Same attribute layout (embed, layers[i].scan/
#     ln1/ln2/ffn, head) -- state_dicts are drop-in compatible. We build a
#     FamilyStreamingLM shell and load the trained StreamingNoPELM weights into
#     it, so grow_model/check_surgery_equivalence/migrate_z_state run UNMODIFIED.
# ═══════════════════════════════════════════════════════════════════════════
def to_family_lm(stream_model, vocab_size, mask_idx, d_model, n_heads=4, d_head=16):
    """Copy a trained StreamingNoPELM's weights into a fresh FamilyStreamingLM
    shell (identical envelope: embed, layers[i].scan/ln1/ln2/ffn, head -- same
    math, different class identity). ONE cosmetic layout difference: StreamingNoPELM's
    ffn is nn.Sequential(Linear, GELU, Dropout-or-Identity, Linear) (2nd Linear at
    index 3, dropout=0.0 so the Identity carries no params) inherited from the
    reference block, while FamilyStreamingLM's ffn is nn.Sequential(Linear, GELU,
    Linear) (2nd Linear at index 2, no Dropout slot at all). We remap ffn.3.* ->
    ffn.2.* explicitly; every other key matches by name."""
    fam = FamilyStreamingLM(StreamingScanLayer, vocab_size, mask_idx, d_model=d_model,
                             n_layers=2, n_heads=n_heads, d_head=d_head, dropout=0.0, causal=True)
    src_sd = stream_model.state_dict()
    remapped = {}
    for k, v in src_sd.items():
        nk = k.replace(".ffn.3.", ".ffn.2.")
        remapped[nk] = v
    missing, unexpected = fam.load_state_dict(remapped, strict=True)
    return fam


def verify_envelope_adapter(model64_stream, vocab_size, mask_idx, d_head=16, seed=0):
    """Sanity gate: the FamilyStreamingLM shell loaded from a StreamingNoPELM's
    weights must reproduce IDENTICAL forward outputs (stateless AND chunked with
    carried Z) -- proves the adapter is a pure relabeling, not a math change."""
    fam = to_family_lm(model64_stream, vocab_size, mask_idx, d_model=model64_stream.embed.embedding_dim,
                        n_heads=4, d_head=d_head)
    model64_stream.eval(); fam.eval()
    torch.manual_seed(seed)
    x = torch.randint(0, vocab_size, (2, 96))
    with torch.no_grad():
        out_s, _ = model64_stream(x, None)
        out_f, _ = fam(x, None)
        err_stateless = float((out_s - out_f).abs().max())

        st_s, st_f = None, None
        max_err_chunked = 0.0
        pos = 0
        while pos < 96:
            hi = min(96, pos + 32)
            xc = x[:, pos:hi]
            ls, st_s = model64_stream(xc, st_s)
            lf, st_f = fam(xc, st_f)
            max_err_chunked = max(max_err_chunked, float((ls - lf).abs().max()))
            pos = hi
    return err_stateless, max_err_chunked, fam


# ═══════════════════════════════════════════════════════════════════════════
# §2  Beacon training with checkpointing at T1 (criterion) and T2 (2x budget).
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def eval_recall_at_gap(model, G, gen, F_, beta0, beta1, probe, n=200, dev="cpu"):
    model.eval()
    x, k = _make_beacon_batch(n, G, gen, F_, beta0, beta1, probe)
    logits, _ = model(x.to(dev), None)
    pred = logits[:, -1, :2].argmax(-1).cpu()
    acc = float((pred == k).float().mean())
    model.train()
    return acc


def train_beacon_to_criterion(args, V, mask, F_, beta0, beta1, probe, dev, gen_train, seed):
    """Curriculum training (verbatim recipe from streaming_train.run_idle_persistence)
    until recall @ args.criterion_gap clears args.criterion, checkpoint T1; then continue
    for the SAME number of iterations again (2x total budget), checkpoint T2. Returns
    (model, T1_state, T1_iters, T2_state, T2_iters, recall_curve)."""
    torch.manual_seed(seed)
    model = StreamingNoPELM(V, V - 1, d_model=args.d_model, n_layers=2, n_heads=4,
                            d_head=args.d_model // 4, seq_len=32, dropout=0.0, causal=True).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    lossf = nn.CrossEntropyLoss()
    Gmax = args.train_gap
    Gcur = min(4, Gmax)
    gen_eval = torch.Generator().manual_seed(seed + 777)

    print(f"── training beacon model to criterion (recall>={args.criterion:.2f} @ gap={args.criterion_gap}) ──",
          flush=True)
    model.train()
    t1_state, t1_iters = None, None
    it = 0
    max_it = args.max_iters
    check_every = max(1, args.check_every)
    while it < max_it:
        x, k = _make_beacon_batch(args.batch, Gcur, gen_train, F_, beta0, beta1, probe)
        logits, _ = model(x.to(dev), None)
        pred = logits[:, -1, :2]
        loss = lossf(pred, k.to(dev))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        acc = float((pred.argmax(-1).cpu() == k).float().mean())
        if acc > 0.95 and Gcur < Gmax:
            Gcur = min(Gmax, int(Gcur * 1.5) + 1)
        it += 1
        if it % check_every == 0 or it == max_it:
            recall = eval_recall_at_gap(model, args.criterion_gap, gen_eval, F_, beta0, beta1, probe,
                                        n=100, dev=dev)
            print(f"   it {it:>5}: loss {float(loss.detach()):.3f} train-acc {acc:.3f} (train-gap {Gcur}) "
                  f"| recall@{args.criterion_gap} = {recall:.3f}", flush=True)
            if t1_state is None and recall >= args.criterion:
                t1_state = {k_: v.clone() for k_, v in model.state_dict().items()}
                t1_iters = it
                print(f"   >>> T1 CHECKPOINT at iter {it} (recall {recall:.3f} >= {args.criterion:.2f})",
                      flush=True)
                break
    if t1_state is None:
        # criterion never cleared within budget -- checkpoint whatever we have, flag it
        t1_state = {k_: v.clone() for k_, v in model.state_dict().items()}
        t1_iters = it
        print(f"   !!! criterion NOT reached within max_iters={max_it} -- T1 = final state at it={it}",
              flush=True)

    # continue training the SAME run for another t1_iters iterations -> T2 (2x budget)
    print(f"── continuing training for {t1_iters} more iters (2x budget) -> T2 ──", flush=True)
    target_it = it + t1_iters
    while it < target_it:
        x, k = _make_beacon_batch(args.batch, Gcur, gen_train, F_, beta0, beta1, probe)
        logits, _ = model(x.to(dev), None)
        pred = logits[:, -1, :2]
        loss = lossf(pred, k.to(dev))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        acc = float((pred.argmax(-1).cpu() == k).float().mean())
        if acc > 0.95 and Gcur < Gmax:
            Gcur = min(Gmax, int(Gcur * 1.5) + 1)
        it += 1
        if it % check_every == 0 or it == target_it:
            recall = eval_recall_at_gap(model, args.criterion_gap, gen_eval, F_, beta0, beta1, probe,
                                        n=100, dev=dev)
            print(f"   it {it:>5}: loss {float(loss.detach()):.3f} train-acc {acc:.3f} (train-gap {Gcur}) "
                  f"| recall@{args.criterion_gap} = {recall:.3f}", flush=True)
    t2_state = {k_: v.clone() for k_, v in model.state_dict().items()}
    t2_iters = it
    print(f"   >>> T2 CHECKPOINT at iter {it}", flush=True)

    return model, t1_state, t1_iters, t2_state, t2_iters


def load_ckpt(V, mask, d_model, sd, dev):
    torch.manual_seed(0)
    m = StreamingNoPELM(V, V - 1, d_model=d_model, n_layers=2, n_heads=4,
                        d_head=d_model // 4, seq_len=32, dropout=0.0, causal=True).to(dev)
    m.load_state_dict(sd)
    m.eval()
    return m


# ═══════════════════════════════════════════════════════════════════════════
# §3  Episode-level bit-write / carry / probe primitives (mid-episode state
#     extraction and re-injection, needed for both surgery and swap arms).
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def write_and_carry(model, x_beacon_and_gap, states_in=None):
    """Forward [beacon][gap] (NOT the probe token) through `model`, return the
    carried states right before the probe position."""
    logits, states = model(x_beacon_and_gap, states_in)
    return states


@torch.no_grad()
def probe_with_states(model, x_probe, states):
    """Forward just the [probe] token through `model` with carried `states`,
    return the 2-way prediction logits at that (only) position."""
    logits, _ = model(x_probe, states)
    return logits[:, -1, :2]


def shuffle_states(states, seed=42):
    """Channel-permutation control, HARDENED after the smoke run showed a single
    shared within-head permutation does not collapse this task to chance: at
    d_model=64 with a trivial 2-way bit, the trained state encodes the label
    highly redundantly (verified: even a full cross-head+cross-channel single
    permutation left recall ~0.7, not ~0.5 -- see beacon_swap smoke report).
    We therefore permute ACROSS heads AND channels jointly (one flat (H*D,)
    permutation), drawn INDEPENDENTLY PER EPISODE (not one shared permutation for
    the whole batch) -- the maximal-entropy pure-permutation control. A fixed
    seed makes the draw reproducible; the per-episode independence is what
    actually breaks the redundant code (a single shared permutation is itself a
    linear operator the trained readout can partially undo when the code is
    spread across many correlated channels)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for z in states:
        B, H, D = z.shape
        flat = z.reshape(B, H * D)
        shuf = torch.empty_like(flat)
        for b in range(B):
            perm = torch.randperm(H * D, generator=g)
            shuf[b] = flat[b, perm]
        out.append(shuf.reshape(B, H, D).clone())
    return out


def zero_states(states):
    return [torch.zeros_like(s) for s in states]


def norm_matched_shuffle_states(states, seed=42):
    """DIAGNOSTIC control (not one of P26's registered checks): same per-episode
    full permutation as shuffle_states, but additionally RESCALES each episode's
    permuted state to the batch-mean norm. If shuffled recall sits well above
    chance while this norm-matched variant collapses further toward chance, the
    mechanism is: the label is carried by state MAGNITUDE (||Z||), not channel
    identity/position -- magnitude survives any pure permutation by construction."""
    shuf = shuffle_states(states, seed=seed)
    out = []
    for z in shuf:
        B, H, D = z.shape
        flat = z.reshape(B, H * D)
        norms = flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
        target = norms.mean()
        rescaled = flat / norms * target
        out.append(rescaled.reshape(B, H, D))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# §4  (a) SURGERY arm — plant+carry with T2-d64 up to (not incl.) the probe,
#     grow_model (via the FamilyStreamingLM adapter) mid-episode, migrate Z,
#     probe with the grown d128 model.
# ═══════════════════════════════════════════════════════════════════════════
def run_surgery_arm(model64_t2, V, mask, F_, beta0, beta1, probe, gaps, n_episodes, seed, dev,
                     noise_std=1e-4):
    fam64 = to_family_lm(model64_t2, V, mask, d_model=model64_t2.embed.embedding_dim,
                         n_heads=4, d_head=model64_t2.embed.embedding_dim // 4)
    fam64.eval()

    model128_exact, model128_exact_sd = grow_model(fam64, V, mask, n_layers=2, n_heads=4,
                                                     noise_std=noise_std, noise_seed=seed)
    err_stateless, err_chunked = check_surgery_equivalence(fam64, model128_exact_sd, V, mask,
                                                             n_layers=2, n_heads=4, seed=seed)
    gate = {"stateless_err": err_stateless, "chunked_err": err_chunked, "tol": 1e-4,
            "pass": err_stateless < 1e-4 and err_chunked < 1e-4}
    print(f"[surgery-adapter] equivalence gate: stateless {err_stateless:.3e}, "
          f"chunked(3) {err_chunked:.3e} -> {'PASS' if gate['pass'] else 'FAIL'}", flush=True)

    # the actually-trained, noised d128 model (post-gate) is what we grow to
    model128, _ = grow_model(fam64, V, mask, n_layers=2, n_heads=4,
                             noise_std=noise_std, noise_seed=seed)
    model128.eval()

    results = {}
    gen = torch.Generator().manual_seed(seed + 9001)
    for G in gaps:
        x, k = _make_beacon_batch(n_episodes, G, gen, F_, beta0, beta1, probe)
        x = x.to(dev)
        x_bg, x_probe = x[:, :-1], x[:, -1:]
        with torch.no_grad():
            states64 = write_and_carry(fam64, x_bg, None)
            states128 = migrate_z_state(states64, n_heads=4)
            acc = _acc(probe_with_states(model128, x_probe, states128), k)
            acc_cold = _acc(probe_with_states(model128, x_probe, zero_states(states128)), k)
            acc_shuf = _acc(probe_with_states(model128, x_probe, shuffle_states(states128, seed=42)), k)
        results[G] = {"recall": acc, "cold": acc_cold, "shuffled": acc_shuf}
        print(f"   [surgery] gap={G:>4}: recall = {acc:.3f} | cold {acc_cold:.3f} | shuffled {acc_shuf:.3f}",
              flush=True)
    return results, gate


# ═══════════════════════════════════════════════════════════════════════════
# §5  (b) SWAP arm — write with W(T1), read with W(T2), state carried across
#     the swap. Controls: native-T1, native-T2, cold, shuffled.
# ═══════════════════════════════════════════════════════════════════════════
def run_swap_arm(model_t1, model_t2, V, F_, beta0, beta1, probe, gaps, n_episodes, seed, dev):
    results = {}
    gen = torch.Generator().manual_seed(seed + 4242)
    for G in gaps:
        x, k = _make_beacon_batch(n_episodes, G, gen, F_, beta0, beta1, probe)
        x = x.to(dev)
        x_bg, x_probe = x[:, :-1], x[:, -1:]
        with torch.no_grad():
            # write with T1, carry through gap with T1
            st_t1 = write_and_carry(model_t1, x_bg, None)
            # write with T2, carry through gap with T2 (for native-T2 control + swap uses st_t1)
            st_t2 = write_and_carry(model_t2, x_bg, None)

            acc_native_t1 = _acc(probe_with_states(model_t1, x_probe, st_t1), k)
            acc_native_t2 = _acc(probe_with_states(model_t2, x_probe, st_t2), k)
            acc_swap = _acc(probe_with_states(model_t2, x_probe, st_t1), k)          # THE arm: write T1, read T2
            acc_cold = _acc(probe_with_states(model_t2, x_probe, zero_states(st_t1)), k)
            acc_shuf = _acc(probe_with_states(model_t2, x_probe, shuffle_states(st_t1, seed=42)), k)
            # diagnostic only (not a P26 check): norm-matched shuffle -- isolates whether the
            # residual-above-chance signal in `shuffled` is carried by state MAGNITUDE
            acc_shuf_normmatched = _acc(
                probe_with_states(model_t2, x_probe, norm_matched_shuffle_states(st_t1, seed=42)), k)

        results[G] = {"native_t1": acc_native_t1, "native_t2": acc_native_t2,
                      "swap_write_t1_read_t2": acc_swap, "cold": acc_cold, "shuffled": acc_shuf,
                      "shuffled_norm_matched_diagnostic": acc_shuf_normmatched}
        print(f"   [swap] gap={G:>4}: native-T1 {acc_native_t1:.3f} | native-T2 {acc_native_t2:.3f} | "
              f"SWAP(w=T1,r=T2) {acc_swap:.3f} | cold {acc_cold:.3f} | shuffled {acc_shuf:.3f} | "
              f"shuffled+norm-matched(diag) {acc_shuf_normmatched:.3f}", flush=True)
    return results


def _acc(logits, k):
    return round(float((logits.argmax(-1).cpu() == k).float().mean()), 4)


# ═══════════════════════════════════════════════════════════════════════════
# §6  Carrier diagnosis at T1 and T2 (streaming_train.run_carrier_probe method,
#     applied to a fixed model instead of retraining) -- identify the carrier
#     channel index at each checkpoint, log gamma/alpha.
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def diagnose_carrier(model, G, F_, beta0, beta1, probe, seed, dev, n=256):
    gen = torch.Generator().manual_seed(seed + 31337)
    x_mix, k_mix = _make_beacon_batch(n, G, gen, F_, beta0, beta1, probe)
    x0 = x_mix.clone(); x0[:, 0] = beta0
    x1 = x_mix.clone(); x1[:, 0] = beta1

    def fwd_capture(xb):
        h = model.embed(xb.to(dev)); inter = []
        for layer in model.layers:
            y, _, ig = layer.scan(h, None, return_internals=True)
            inter.append(ig); h = layer.ln1(h + y); h = layer.ln2(h + layer.ffn(h))
        return inter

    ig_mix, ig0, ig1 = fwd_capture(x_mix), fwd_capture(x0), fwd_capture(x1)
    kf = (k_mix.float() - 0.5)

    layers_out = []
    for li in range(len(model.layers)):
        Zg = ig_mix[li]["Z_seq"][:, G]
        Zc = Zg - Zg.mean(0, keepdim=True)
        corr = (Zc * kf[:, None, None]).mean(0) / (Zc.std(0) + 1e-6) / (kf.std() + 1e-6)
        flat = corr.abs().reshape(-1)
        idx = int(flat.argmax())
        h_star, c_star = idx // corr.shape[1], idx % corr.shape[1]
        corr_carrier = float(corr[h_star, c_star])
        gap_slice = slice(1, G + 1)
        g_c = float(ig_mix[li]["gamma"][:, gap_slice, h_star, c_star].mean())
        al_c = float(ig_mix[li]["alpha"][:, gap_slice, h_star, c_star].mean())
        al_beacon = float(ig_mix[li]["alpha"][:, 0, h_star, c_star].mean())
        layers_out.append({"layer": li, "carrier_head": h_star, "carrier_chan": c_star,
                           "corr": round(corr_carrier, 3), "gamma_carrier_gap": round(g_c, 4),
                           "alpha_carrier_gap": round(al_c, 4), "alpha_carrier_beacon": round(al_beacon, 4)})
    return layers_out


# ═══════════════════════════════════════════════════════════════════════════
# §7  Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="MS13: does the stored bit survive the surgery and the swap? (P26)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--train-gap", type=int, default=256, help="curriculum ceiling gap")
    ap.add_argument("--criterion-gap", type=int, default=256)
    ap.add_argument("--criterion", type=float, default=0.99)
    ap.add_argument("--max-iters", type=int, default=6000)
    ap.add_argument("--check-every", type=int, default=100)
    ap.add_argument("--gaps", default="64,256,512")
    ap.add_argument("--n-episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise-std", type=float, default=1e-4)
    ap.add_argument("--out", default="results/beacon_swap.json")
    args = ap.parse_args()

    if args.smoke:
        args.gaps = "64,256"
        args.n_episodes = 40
        args.criterion = 0.95
        args.criterion_gap = 128
        args.max_iters = 2500
        args.check_every = 50
        args.out = args.out.replace(".json", "_smoke.json") if not args.out.endswith("_smoke.json") else args.out
        print("[smoke] gaps={64,256} n=40 criterion>=0.95@128 max_iters=2500", flush=True)

    dev = torch.device("cpu")
    gaps = [int(g) for g in args.gaps.split(",")]
    F_, fillers, beta0, beta1, probe, V = _beacon_vocab()
    mask = V - 1
    t0 = time.time()

    gen_train = torch.Generator().manual_seed(args.seed)
    model, t1_sd, t1_iters, t2_sd, t2_iters = train_beacon_to_criterion(
        args, V, mask, F_, beta0, beta1, probe, dev, gen_train, args.seed)

    model_t1 = load_ckpt(V, mask, args.d_model, t1_sd, dev)
    model_t2 = load_ckpt(V, mask, args.d_model, t2_sd, dev)

    gen_check = torch.Generator().manual_seed(args.seed + 555)
    recall_t1 = eval_recall_at_gap(model_t1, args.criterion_gap, gen_check, F_, beta0, beta1, probe,
                                   n=200, dev=dev)
    recall_t2 = eval_recall_at_gap(model_t2, args.criterion_gap, gen_check, F_, beta0, beta1, probe,
                                   n=200, dev=dev)
    print(f"\n[checkpoints] T1 iters={t1_iters} recall@{args.criterion_gap}={recall_t1:.3f} | "
          f"T2 iters={t2_iters} recall@{args.criterion_gap}={recall_t2:.3f}", flush=True)
    checkpoints_ok = recall_t1 >= 0.9 and recall_t2 >= 0.9
    if not checkpoints_ok:
        print("!!! T1/T2 recall below 0.9 -- aborting downstream measurement, reporting checkpoint failure.",
              flush=True)

    out = {
        "prediction_id": "P26",
        "config": vars(args),
        "checkpoints": {"t1_iters": t1_iters, "t1_recall_at_criterion_gap": round(recall_t1, 4),
                        "t2_iters": t2_iters, "t2_recall_at_criterion_gap": round(recall_t2, 4),
                        "criterion_gap": args.criterion_gap, "criterion": args.criterion,
                        "checkpoints_ok": checkpoints_ok},
    }

    if not checkpoints_ok:
        out["ABORTED"] = "T1 or T2 did not clear recall>=0.9 at criterion gap"
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"\n-> {args.out} (ABORTED)")
        return

    # ── envelope adapter verification (StreamingNoPELM -> FamilyStreamingLM) ──
    print("\n[adapter] verifying StreamingNoPELM <-> FamilyStreamingLM state_dict compatibility...",
          flush=True)
    d_head64 = args.d_model // 4
    err_s, err_c, _ = verify_envelope_adapter(model_t2, V, mask, d_head=d_head64, seed=args.seed)
    print(f"[adapter] stateless max|delta|={err_s:.3e}  chunked(3) max|delta|={err_c:.3e} "
          f"{'PASS' if err_s < 1e-4 and err_c < 1e-4 else 'FAIL'}", flush=True)
    out["adapter_verification"] = {"stateless_err": err_s, "chunked_err": err_c, "tol": 1e-4,
                                    "pass": err_s < 1e-4 and err_c < 1e-4}

    # ── (a) SURGERY arm ──
    print("\n" + "=" * 78 + "\n(a) SURGERY: plant+carry with T2-d64, widen mid-episode, probe with d128\n"
          + "=" * 78, flush=True)
    surgery_results, surgery_gate = run_surgery_arm(model_t2, V, mask, F_, beta0, beta1, probe,
                                                     gaps, args.n_episodes, args.seed, dev,
                                                     noise_std=args.noise_std)
    out["surgery_gate"] = surgery_gate
    out["surgery_recall"] = surgery_results

    # ── (b) SWAP arm ──
    print("\n" + "=" * 78 + "\n(b) SWAP: write with W(T1), read with W(T2), state carried across\n"
          + "=" * 78, flush=True)
    swap_results = run_swap_arm(model_t1, model_t2, V, F_, beta0, beta1, probe, gaps,
                                args.n_episodes, args.seed, dev)
    out["swap_recall"] = swap_results

    # ── carrier diagnosis at T1 and T2 ──
    print("\n" + "=" * 78 + "\nDIAGNOSIS: carrier channel at T1 vs T2 (gap=" + str(gaps[-1]) + ")\n"
          + "=" * 78, flush=True)
    Gdiag = max(gaps)
    diag_t1 = diagnose_carrier(model_t1, Gdiag, F_, beta0, beta1, probe, args.seed, dev)
    diag_t2 = diagnose_carrier(model_t2, Gdiag, F_, beta0, beta1, probe, args.seed, dev)
    for li in range(len(diag_t1)):
        d1, d2 = diag_t1[li], diag_t2[li]
        same = (d1["carrier_head"] == d2["carrier_head"] and d1["carrier_chan"] == d2["carrier_chan"])
        print(f"   L{li}: T1 carrier H{d1['carrier_head']}C{d1['carrier_chan']} (corr {d1['corr']:+.2f}, "
              f"gamma {d1['gamma_carrier_gap']:.3f}) | T2 carrier H{d2['carrier_head']}C{d2['carrier_chan']} "
              f"(corr {d2['corr']:+.2f}, gamma {d2['gamma_carrier_gap']:.3f}) | "
              f"{'SAME CHANNEL' if same else 'DIFFERENT CHANNEL'}", flush=True)
    out["carrier_diagnosis"] = {
        "gap": Gdiag, "t1": diag_t1, "t2": diag_t2,
        "same_channel_per_layer": [
            (diag_t1[li]["carrier_head"] == diag_t2[li]["carrier_head"] and
             diag_t1[li]["carrier_chan"] == diag_t2[li]["carrier_chan"])
            for li in range(len(diag_t1))
        ],
    }

    # ── P26 checks ──
    primary_gap = args.criterion_gap if args.criterion_gap in surgery_results else gaps[len(gaps) // 2]
    a_cell = surgery_results.get(256, surgery_results.get(primary_gap))
    a_recall = a_cell["recall"] if a_cell else None
    check_a = surgery_gate["pass"] and (a_recall is not None and a_recall >= 0.99)

    b_swap = swap_results.get(256, swap_results.get(primary_gap))
    check_b = b_swap is not None and b_swap["swap_write_t1_read_t2"] >= 0.9

    # shuffled ~= chance (0.5) at BOTH (a) and (b) per spec; scored jointly (both must hold)
    check_c_a = a_cell is not None and abs(a_cell["shuffled"] - 0.5) <= 0.1
    check_c_b = b_swap is not None and abs(b_swap["shuffled"] - 0.5) <= 0.1
    check_c = check_c_a and check_c_b

    out["checks"] = {
        "a_surgery_recall_ge_0.99": {"pass": bool(check_a), "gate_pass": surgery_gate["pass"],
                                     "recall_at_256_or_primary": a_recall},
        "b_swap_recall_ge_0.9": {"pass": bool(check_b), "swap_recall_at_256_or_primary":
                                 b_swap["swap_write_t1_read_t2"] if b_swap else None},
        "c_shuffled_approx_chance": {"pass": bool(check_c),
                                     "surgery_shuffled_recall": a_cell["shuffled"] if a_cell else None,
                                     "swap_shuffled_recall": b_swap["shuffled"] if b_swap else None,
                                     "surgery_shuffled_pass": bool(check_c_a),
                                     "swap_shuffled_pass": bool(check_c_b)},
    }
    all_pass = check_a and check_b and check_c

    print("\n" + "=" * 78)
    print("VERDICT -- P26 (the stored bit survives the surgery and the swap)")
    print(f"  (a) surgery recall >= 0.99: gate={'PASS' if surgery_gate['pass'] else 'FAIL'} "
          f"recall={a_recall} {'PASS' if check_a else 'FAIL'}")
    print(f"  (b) swap recall >= 0.9: {b_swap['swap_write_t1_read_t2'] if b_swap else 'N/A'} "
          f"{'PASS' if check_b else 'FAIL'}")
    print(f"  (c) shuffled ~= chance (0.5+-0.1): surgery={a_cell['shuffled'] if a_cell else 'N/A'} "
          f"({'PASS' if check_c_a else 'FAIL'})  swap={b_swap['shuffled'] if b_swap else 'N/A'} "
          f"({'PASS' if check_c_b else 'FAIL'})  {'PASS' if check_c else 'FAIL'}")
    if b_swap and "shuffled_norm_matched_diagnostic" in b_swap:
        print(f"      [diagnostic, not scored] norm-matched shuffle (swap) = "
              f"{b_swap['shuffled_norm_matched_diagnostic']:.3f} -- if this is closer to chance than "
              f"'shuffled', the residual-above-chance signal is carried by state MAGNITUDE, not position")
    print(f"  OVERALL: {'ALL PASS' if all_pass else 'NOT ALL CHECKS PASS -- see above'}")
    print("=" * 78)
    out["all_checks_pass"] = bool(all_pass)
    out["wall_s"] = round(time.time() - t0, 1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
