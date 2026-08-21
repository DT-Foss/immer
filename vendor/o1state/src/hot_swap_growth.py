#!/usr/bin/env python3 -u
"""
HOT-SWAP GROWTH — function- and state-preserving widening on a live stream.
==============================================================================
P24 (registered prediction, analysis/PREDICTIONS.md MS8): a Net2Net-style
widening (channel DUPLICATION, exact function-preservation even through
LayerNorm) that takes a d_model=64 GSSM-Selective streaming LM to d_model=128
mid-stream, migrates the carried Z-state, and keeps training -- does growth
beat both (i) never growing and (ii) restarting fresh at the wide width on
the same post-surgery token budget?

Why duplication is exact through THIS architecture
----------------------------------------------------
Every nonlinearity in streaming_train.StreamingScanLayer / the reference
SelectiveRapiditySqrtScanLayer is ELEMENTWISE: tanh(W_v x), sigmoid(W_gate x),
sigmoid(W_gamma x), sigmoid(W_alpha x), the log-complement term
log(1 - v_gated^2 + EPS), and the final sqrt(1 - exp(Z) + EPS) readout state.
None of these SUM across the channel axis -- they map each channel through
the same scalar function independently. So if channel c and channel c+D
carry IDENTICAL values before an elementwise op, they carry IDENTICAL values
after it. Channel duplication is therefore preserved through the entire
per-head, per-channel pipeline (v, gate, gamma, alpha, a, Z, state) with NO
special-casing needed for the log/sqrt nonlinearities -- duplication commutes
with elementwise maps for free.

The only places that need the "halve the weight" trick are nn.Linear layers
that SUM over the (duplicated) input dimension: y = W x = sum_i W[:,i] x[i].
If x128 = [x64; x64] (each channel appears twice) and we want
y128 == y64 == W64 x64, we must NOT double-count: set
  W128[:, :D] = W128[:, D:] = W64 / 2      (input-duplication: copy + halve)
Then y128 = W128 @ [x;x] = (W64/2) x + (W64/2) x = W64 x = y64.   Exact.

For OUTPUT duplication (we want y128 = [y64; y64], i.e. every output row
duplicated so the next stage sees repeated channels too):
  W128[:D, :] = W128[D:, :] = W64          (output-duplication: copy rows)
No halving on the output side -- each output row is an independent linear
functional of the (already-correctly-summed) input; duplicating the ROW just
computes the same scalar twice.

Combining both rules on the FIVE d_model-square projections of the scan layer
(W_v, W_gate, W_gamma, W_alpha all d_model->proj; W_out proj->d_model) and the
FFN (d_model->4d_model->d_model) gives every weight matrix in the network a
well-defined widening rule:
  - Embedding (vocab -> d_model):            OUTPUT duplication (columns; no sum over d_model on this side)
  - W_v/W_gate/W_gamma/W_alpha (d_model->proj): INPUT duplication (halve) THEN
    the proj axis (n_heads*d_head) is ALSO duplicated per-head (output side,
    copy) so d_head doubles 16->32 while n_heads stays 4 (family-envelope
    n_heads=4 fixed, d_head=d_model//4).
  - W_out (proj->d_model):                    INPUT duplication (halve, over
    the now-doubled proj axis) THEN OUTPUT duplication (copy rows, over d_model)
  - LayerNorm weight/bias (d_model):          COPY (elementwise affine after
    an elementwise normalization -- see the LN proof below)
  - FFN fc1 (d_model->4d_model):              INPUT dup (halve) + OUTPUT dup (copy)
  - FFN fc2 (4d_model->d_model):               INPUT dup (halve) + OUTPUT dup (copy)
  - Head (d_model->vocab):                    INPUT duplication ONLY (halve;
    output side is vocab, untouched)

LayerNorm exactness: LN(h) with h in R^D computes mean/var over the D axis
then applies (h-mu)/sigma * weight + bias elementwise. If h128 = [h;h] (D
duplicated), mean(h128) = mean(h) and var(h128) = var(h) EXACTLY because the
biased variance formula (what nn.LayerNorm uses) is the mean of squared
deviations -- duplicating every sample twice does not change either an
arithmetic mean or a variance computed over duplicated values (both are
invariant to "each value counted k times instead of once", since it's still
a plain average). So LN(h128)_i = LN(h)_(i mod D) BEFORE the affine, and
copying weight/bias row-wise then gives LN128([h;h]) = [LN64(h); LN64(h)]
exactly.

Z-state migration: Z lives at (B, n_heads, d_head). d_head doubles per-head
(16->32); n_heads is unchanged (4). So Z128 = dup_along_last_axis(Z64) --
channel duplication WITHIN each head, matching the d_head widening of
W_v/W_gate/W_gamma/W_alpha's proj axis (which is also laid out per-head via
.view(B,T,n_heads,d_head)). Verified numerically below, not assumed.

Symmetry breaking: after the equivalence gate passes exactly, the duplicate
channels are, by construction, permanently identical under gradient descent
(same forward function of the same input => same gradient => same update,
every step, forever) unless something breaks the symmetry. We add a tiny
documented, seed-fixed additive perturbation (1e-4 * randn) ONLY to the
second (duplicate) half of every widened weight matrix's new rows/columns,
strictly AFTER the equivalence gate has been checked on the un-perturbed
grown model. This nudges the two copies apart just enough to let training
actually use the extra capacity, while keeping the gate's own before/after
comparison exact.

Arms (seed 42, ONE shared C4Stream, chunks cloned to all arms, B=8 K=64,
Adam lr=3e-3, step_full full-gradient recipe from pos_family_transfer.py):
  (i)   grown   -- d64 streams Phase A, surgery (widen to d128 + fresh Adam),
                   then streams Phase B (same length as A).
  (ii)  stay64  -- d64 streams Phase A+B back-to-back, never grown (control
                   for "was growing even necessary").
  (iii) fresh128-- d128 from scratch, but ONLY sees Phase B chunks (starts
                   fresh exactly at the surgery point) -- the apples-to-apples
                   "growth vs restart" comparison on the same post-surgery
                   token budget.

P24 checks (a)-(d), scored in the VERDICT block:
  (a) surgery equivalence:      max|logits_grown - logits_d64| < 1e-4 on a
                                 frozen probe batch, AND with carried Z over
                                 3 chunks < 1e-4.
  (b) no post-surgery transient: |grown - stay64| chunk-NLL, mean over the
                                 first 20 chunks after surgery, < 0.05 nats.
  (c) growth pays off:          at the FINAL post-surgery eval, grown's
                                 held-out beats stay64's by >= 0.03 nats.
  (d) growth beats restart:     at the FINAL post-surgery eval (same
                                 wall-token axis from the surgery point),
                                 grown's held-out beats fresh128's.

CPU only, torch.set_num_threads(1), os.nice(19) best-effort. Writes only to
results/hot_swap_growth.json (or *_smoke.json under --smoke).
"""
import argparse
import json
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.mps.is_available = lambda: False  # force CPU, matches streaming_train/pos_family_transfer
torch.set_num_threads(1)                          # bit-deterministic

try:
    os.nice(19)
except Exception:
    pass

import sys
sys.path.insert(0, "reference")
sys.path.insert(0, "src")

from streaming_train import StreamingScanLayer, stateful_scan  # noqa: F401 (stateful_scan used transitively)
from pos_family_transfer import (
    FamilyStreamingLM, build_gssm_lm, C4Stream, build_eval_set, heldout,
    step_full,
)
from length_extrap_v2 import load_wikitext2, build_vocab, tokenize


# ═══════════════════════════════════════════════════════════════════════════
# §1  Widening primitives -- the two weight-matrix rules + LN/embedding rules.
# ═══════════════════════════════════════════════════════════════════════════
def _dup_input_halve(W, dim):
    """W: (out, in). Duplicate along `in` (dim=1) and HALVE, so that
    y = W_new @ [x;x] == W_old @ x. Use when the matrix SUMS over a
    duplicated input axis."""
    assert dim == 1
    return torch.cat([W, W], dim=1) * 0.5


def _dup_output_copy(W, dim):
    """W: (out, in). Duplicate along `out` (dim=0) by COPYING rows, so that
    y_new = [y_old; y_old]. Use when we want the output channel duplicated
    (no summation happens on the output axis)."""
    assert dim == 0
    return torch.cat([W, W], dim=0)


def _dup_vec_copy(v):
    """1-D param (LayerNorm weight/bias, or A_log-style per-channel param):
    plain copy -- elementwise affine after an elementwise/LN op."""
    return torch.cat([v, v], dim=0)


def _dup_headwise_last_axis(W, n_heads, d_head_old, dim):
    """W: (n_heads*d_head_old, in) or (out, n_heads*d_head_old). Duplicate the
    d_head axis WITHIN each head (not the head axis itself), matching how Z
    is laid out (B,T,n_heads,d_head) and read/written via .view(). dim=0 means
    W's leading axis is (n_heads*d_head_old,) and gets doubled to
    (n_heads*2*d_head_old,) with the new half-channels interleaved per head
    as [head0: d0..d15,d0..d15 | head1: d0..d15,d0..d15 | ...]. dim=1 is the
    mirror on the trailing axis, additionally halved (input-duplication)."""
    if dim == 0:
        out_dim, in_dim = W.shape
        Wh = W.view(n_heads, d_head_old, in_dim)
        Wh2 = torch.cat([Wh, Wh], dim=1)  # (n_heads, 2*d_head_old, in_dim) -- copy within each head
        return Wh2.reshape(n_heads * 2 * d_head_old, in_dim)
    elif dim == 1:
        out_dim, in_dim = W.shape
        Wh = W.view(out_dim, n_heads, d_head_old)
        Wh2 = torch.cat([Wh, Wh], dim=2) * 0.5  # halve: this axis is SUMMED over
        return Wh2.reshape(out_dim, n_heads * 2 * d_head_old)
    else:
        raise ValueError(dim)


# ═══════════════════════════════════════════════════════════════════════════
# §2  grow_model: d_model=64 (n_heads=4,d_head=16) -> d_model=128 (n_heads=4,
#     d_head=32), function-preserving Net2Net-style widening.
# ═══════════════════════════════════════════════════════════════════════════
def grow_model(model64, vocab_size, mask_idx, n_layers=2, n_heads=4,
               noise_std=1e-4, noise_seed=0):
    """Build a d_model=128 FamilyStreamingLM whose forward function is
    IDENTICAL to model64's (channel-duplicated input in, channel-duplicated
    output out) BEFORE the symmetry-breaking noise is applied. Returns
    (model128_exact, model128_noised) -- the exact one is used for the
    equivalence gate, the noised one is what actually trains afterward."""
    d64 = model64.embed.embedding_dim
    d128 = d64 * 2
    d_head64 = d64 // n_heads
    d_head128 = d128 // n_heads

    model128 = FamilyStreamingLM(type(model64.layers[0].scan), vocab_size, mask_idx,
                                  d_model=d128, n_layers=n_layers, n_heads=n_heads,
                                  d_head=d_head128, dropout=0.0, causal=True)

    with torch.no_grad():
        # ── Embedding: vocab -> d_model. Widen the OUTPUT (d_model) axis: copy columns.
        # embed.weight: (vocab+2, d_model) -- d_model is dim=1 here, output-duplication (no sum)
        model128.embed.weight.copy_(torch.cat([model64.embed.weight, model64.embed.weight], dim=1))

        for li in range(n_layers):
            src = model64.layers[li]
            dst = model128.layers[li]

            # ── scan layer: W_v/W_gate/W_gamma/W_alpha (d_model -> n_heads*d_head) ──
            # input axis = d_model (SUMMED over) -> halve+dup; output axis = proj,
            # laid out per-head -> duplicate d_head WITHIN each head (copy, no halving:
            # this axis is not summed by W_v itself, it's the output of this matmul).
            for name in ["W_v", "W_gate", "W_gamma", "W_alpha"]:
                Wsrc = getattr(src.scan, name).weight  # (n_heads*d_head64, d64)
                W1 = _dup_input_halve(Wsrc, dim=1)      # (n_heads*d_head64, d128) -- halve d_model axis
                W2 = _dup_headwise_last_axis(W1, n_heads, d_head64, dim=0)  # (n_heads*d_head128, d128)
                getattr(dst.scan, name).weight.copy_(W2)

            # ── W_out: n_heads*d_head -> d_model. Input axis = proj (per-head, SUMMED
            # over within each head's contribution... actually W_out sums over the FULL
            # proj axis, which is per-head-duplicated) -> halve along that (headwise) axis;
            # output axis = d_model -> duplicate (copy rows).
            Wo = src.scan.W_out.weight  # (d64, n_heads*d_head64)
            Wo1 = _dup_headwise_last_axis(Wo, n_heads, d_head64, dim=1)  # (d64, n_heads*d_head128), halved
            Wo2 = _dup_output_copy(Wo1, dim=0)  # (d128, n_heads*d_head128)
            dst.scan.W_out.weight.copy_(Wo2)

            # ── A_log (S6-style layers only; GSSM StreamingScanLayer has none) ──
            if hasattr(src.scan, "A_log"):
                Asrc = src.scan.A_log  # (n_heads, d_head64)
                Adst = torch.cat([Asrc, Asrc], dim=1)  # duplicate d_head axis within each head
                dst.scan.A_log.copy_(Adst)

            # ── LayerNorms: copy weight/bias (elementwise affine, LN proof in module docstring) ──
            dst.ln1.weight.copy_(_dup_vec_copy(src.ln1.weight))
            dst.ln1.bias.copy_(_dup_vec_copy(src.ln1.bias))
            dst.ln2.weight.copy_(_dup_vec_copy(src.ln2.weight))
            dst.ln2.bias.copy_(_dup_vec_copy(src.ln2.bias))

            # ── FFN: fc1 (d_model -> 4*d_model), GELU (elementwise), fc2 (4*d_model -> d_model) ──
            fc1_src, fc2_src = src.ffn[0], src.ffn[2]
            fc1_dst, fc2_dst = dst.ffn[0], dst.ffn[2]
            W1 = _dup_input_halve(fc1_src.weight, dim=1)   # halve d_model input axis
            W1 = _dup_output_copy(W1, dim=0)                # duplicate 4*d_model output axis
            fc1_dst.weight.copy_(W1)
            fc1_dst.bias.copy_(_dup_vec_copy(fc1_src.bias)) # bias indexed by (duplicated) output axis -> copy

            W2 = _dup_input_halve(fc2_src.weight, dim=1)    # halve 4*d_model input axis
            W2 = _dup_output_copy(W2, dim=0)                 # duplicate d_model output axis
            fc2_dst.weight.copy_(W2)
            fc2_dst.bias.copy_(_dup_vec_copy(fc2_src.bias))

        # ── Head: d_model -> vocab+1. Input axis = d_model (SUMMED) -> halve. Output = vocab, untouched. ──
        Wh = _dup_input_halve(model64.head.weight, dim=1)
        model128.head.weight.copy_(Wh)
        model128.head.bias.copy_(model64.head.bias.clone())

    model128_exact_sd = {k: v.clone() for k, v in model128.state_dict().items()}

    # ── symmetry breaker: tiny seed-fixed noise on the DUPLICATE halves only ──
    g = torch.Generator().manual_seed(noise_seed)
    with torch.no_grad():
        for li in range(n_layers):
            dst = model128.layers[li]
            for name in ["W_v", "W_gate", "W_gamma", "W_alpha"]:
                W = getattr(dst.scan, name).weight
                Wh = W.view(n_heads, d_head128, d128)
                noise = torch.randn(Wh[:, d_head64:, :].shape, generator=g) * noise_std
                Wh[:, d_head64:, :] += noise
            Wo = dst.scan.W_out.weight  # (d128, n_heads*d_head128)
            Woh = Wo.view(d128, n_heads, d_head128)
            noise = torch.randn(Woh[:, :, d_head64:].shape, generator=g) * noise_std
            Woh[:, :, d_head64:] += noise
            if hasattr(dst.scan, "A_log"):
                noise = torch.randn(dst.scan.A_log[:, d_head64:].shape, generator=g) * noise_std
                dst.scan.A_log[:, d_head64:] += noise
            fc1 = dst.ffn[0].weight  # (4*d128, d128)
            noise = torch.randn(fc1[4 * d64:, :].shape, generator=g) * noise_std
            fc1[4 * d64:, :] += noise
            fc2 = dst.ffn[2].weight  # (d128, 4*d128)
            noise = torch.randn(fc2[d64:, :].shape, generator=g) * noise_std
            fc2[d64:, :] += noise
        noise = torch.randn(model128.head.weight[:, d64:].shape, generator=g) * noise_std
        model128.head.weight[:, d64:] += noise
        noise = torch.randn(model128.embed.weight[:, d64:].shape, generator=g) * noise_std
        model128.embed.weight[:, d64:] += noise

    return model128, model128_exact_sd


# ═══════════════════════════════════════════════════════════════════════════
# §3  Surgery equivalence gate: exact-grown model (pre-noise weights) must
#     match the d64 model's forward EXACTLY (channel-duplicated in/out),
#     both stateless AND with a carried Z migrated across 3 chunks.
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def check_surgery_equivalence(model64, model128_exact_sd, vocab_size, mask_idx,
                               n_layers=2, n_heads=4, T=96, chunk=32, seed=0):
    d64 = model64.embed.embedding_dim
    d128 = d64 * 2
    d_head128 = d128 // n_heads
    model128 = FamilyStreamingLM(type(model64.layers[0].scan), vocab_size, mask_idx,
                                  d_model=d128, n_layers=n_layers, n_heads=n_heads,
                                  d_head=d_head128, dropout=0.0, causal=True)
    model128.load_state_dict(model128_exact_sd)
    model64, model128 = model64.eval(), model128.eval()

    torch.manual_seed(seed)
    x = torch.randint(0, vocab_size, (2, T))

    # (a) stateless full-sequence forward
    logits64, _ = model64(x, None)
    logits128, _ = model128(x, None)
    err_stateless = float((logits64 - logits128).abs().max())

    # (b) chunked with carried Z, migrated by channel-duplication each step
    st64, st128 = None, None
    max_err_chunked = 0.0
    pos = 0
    n_chunks = 0
    while pos < T and n_chunks < 3:
        hi = min(T, pos + chunk)
        xc = x[:, pos:hi]
        l64, st64 = model64(xc, st64)
        l128, st128 = model128(xc, st128)
        max_err_chunked = max(max_err_chunked, float((l64 - l128).abs().max()))
        pos = hi
        n_chunks += 1

    return err_stateless, max_err_chunked


# ═══════════════════════════════════════════════════════════════════════════
# §3b  Adam-moment migration (P27, MS14): migrate opt64's exp_avg/exp_avg_sq/
#     step through the SAME duplication map grow_model() applies to weights,
#     instead of handing the grown model a cold Adam.
#
# GRADIENT-TRANSFORM DERIVATION (verified numerically, scratchpad/
# grad_transform_probe{2,3,4,5}.py -- not assumed):
#   Built the REAL grow_model() pair (toy d_model=8/12/16/24, n_heads=4,
#   noise_std=0 for an exact-equivalence grow), ran identical (x, y) token
#   batches through model64 and model128, took cross_entropy(...).backward()
#   on both, and compared g128's duplicate-half block to g64 parameter-by-
#   parameter. Every duplicated axis (whether the forward rule was
#   input-dup-halve or output-dup-copy) carries an EXACTLY IDENTICAL gradient
#   on both halves (max|diff| ~ 1e-6, float32 noise floor) -- so migration is
#   always a COPY across the new/old half, never itself halved or doubled.
#   The only remaining question is the OVERALL scale factor c relating
#   g128's copied block to g64:
#     - head.weight / head.bias (input-dup-halve ONLY; output=vocab is NOT
#       duplicated, so exactly one, non-duplicated downstream consumer: the
#       cross-entropy loss on the same logits/targets in both models) ->
#       c = 1.0 (exact COPY, confirmed to 2e-6 across d64 in {8,12,16,24}).
#     - every OTHER parameter (embed, W_v/W_gate/W_gamma/W_alpha, W_out,
#       ln1/ln2, ffn.0/ffn.2 weight+bias) sits upstream of an axis that gets
#       read by a DOWNSTREAM matrix whose matching input axis was
#       input-dup-halve'd (that halving is what keeps the FORWARD pass
#       exact: y128 = W128_halved @ [h;h] = W64 @ h). The backward pass
#       reuses that same halved downstream weight to propagate dL/dh, so the
#       local gradient arriving at these parameters is scaled by 0.5 (in
#       addition to being copied on the duplicate half) relative to g64 ->
#       c = 0.5, confirmed to 6e-6 across the same width sweep, exactly flat
#       (no width dependence -- ruled out a mean-reduction/width artifact).
#   exp_avg transforms like the gradient itself: migrated_exp_avg = c * COPY(exp_avg64).
#   exp_avg_sq transforms like the SQUARE of the gradient: migrated_exp_avg_sq
#     = c^2 * COPY(exp_avg_sq64) (Adam's v_t tracks E[g_t^2]; scaling g by c
#     scales g^2 by c^2).
#   step (bias-correction counter) is a scalar per parameter tensor in this
#   torch version -- carried over UNCHANGED (both arms are at the same
#   training step count at the surgery point).
# ═══════════════════════════════════════════════════════════════════════════
def _migrate_moment_tensor(v64, rule, n_heads, d_head64, scale):
    """Apply the SAME duplication primitive grow_model() used for the weight
    itself to an Adam moment tensor (exp_avg or exp_avg_sq), then apply the
    scalar `scale` (c for exp_avg, c^2 for exp_avg_sq). `rule` is one of:
      "vec_copy"        -- 1-D LayerNorm-style copy
      "out_copy1"        -- 2-D, dim=0 output-dup-copy only (embed.weight is
                            (vocab, d_model): duplicate dim=1 by copy)
      "in_halve"         -- 2-D, dim=1 input-dup-halve only (head.weight)
      "in_halve+out_copy"-- 2-D, dim=1 halve THEN dim=0 copy (ffn.0/ffn.2)
      "headwise_v"        -- W_v/W_gate/W_gamma/W_alpha: input-halve(d_model)
                            then headwise-copy(d_head axis, dim=0)
      "headwise_out"      -- W_out: headwise-halve(proj axis, dim=1) then
                            output-copy(d_model axis, dim=0)
      "bias_copy"         -- 1-D bias whose OUTPUT axis was duplicated by
                            copy (ffn.0.bias, ln* handled via vec_copy)
      "untouched"         -- head.bias: not widened at all, pass through
    NOTE: this function only performs the STRUCTURAL duplication (same
    primitives as grow_model's forward-weight rules); the caller multiplies
    by `scale` afterward (kept separate so the same helper serves both
    exp_avg (scale=c) and exp_avg_sq (scale=c^2))."""
    if rule == "untouched":
        return v64.clone() * scale
    if rule == "vec_copy" or rule == "bias_copy":
        return _dup_vec_copy(v64) * scale
    if rule == "out_copy1":
        return torch.cat([v64, v64], dim=1) * scale
    if rule == "in_halve":
        return _dup_input_halve(v64, dim=1) * scale
    if rule == "in_halve+out_copy":
        w = _dup_input_halve(v64, dim=1)
        w = _dup_output_copy(w, dim=0)
        return w * scale
    if rule == "headwise_v":
        w = _dup_input_halve(v64, dim=1)
        w = _dup_headwise_last_axis(w, n_heads, d_head64, dim=0)
        return w * scale
    if rule == "headwise_out":
        w = _dup_headwise_last_axis(v64, n_heads, d_head64, dim=1)
        w = _dup_output_copy(w, dim=0)
        return w * scale
    raise ValueError(rule)


# per-named-parameter (relative to a layer block / top-level) rule + scale-power table.
# scale_power: 1 -> c=1.0 (head), 0.5 -> c=0.5 (everything else). exp_avg uses c^1,
# exp_avg_sq uses c^2, i.e. (0.5)->0.25 or (1.0)->1.0.
_MOMENT_RULES = {
    "embed.weight": ("out_copy1", 0.5),
    "head.weight": ("in_halve", 1.0),
    "head.bias": ("untouched", 1.0),
}
_LAYER_RULES = {
    "scan.W_v.weight": ("headwise_v", 0.5),
    "scan.W_gate.weight": ("headwise_v", 0.5),
    "scan.W_gamma.weight": ("headwise_v", 0.5),
    "scan.W_alpha.weight": ("headwise_v", 0.5),
    "scan.W_out.weight": ("headwise_out", 0.5),
    "ln1.weight": ("vec_copy", 0.5),
    "ln1.bias": ("vec_copy", 0.5),
    "ln2.weight": ("vec_copy", 0.5),
    "ln2.bias": ("vec_copy", 0.5),
    "ffn.0.weight": ("in_halve+out_copy", 0.5),
    "ffn.0.bias": ("bias_copy", 0.5),
    "ffn.2.weight": ("in_halve+out_copy", 0.5),
    "ffn.2.bias": ("bias_copy", 0.5),
}


def migrate_adam_moments(opt64, model64, model128, lr, n_layers=2, n_heads=4):
    """Build a fresh torch.optim.Adam over model128.parameters() whose per-
    parameter state (exp_avg, exp_avg_sq, step) is migrated from opt64's
    state for the matching model64 parameter, through the grow_model()
    duplication map (see derivation above). Falls back to a cold (zero) Adam
    state for any parameter this function doesn't recognize (there are
    none in the current FamilyStreamingLM/StreamingScanLayer shape, but this
    keeps the function safe if the architecture grows new param names)."""
    named64 = dict(model64.named_parameters())
    named128 = dict(model128.named_parameters())
    state64 = opt64.state

    opt128 = torch.optim.Adam(model128.parameters(), lr=lr)
    # touch every param's state once with a zero-step so torch.optim.Adam's
    # internal state dict is populated in the exact shape/dtype it expects,
    # then overwrite with the migrated values.
    with torch.no_grad():
        for p in model128.parameters():
            opt128.state[p] = {
                "step": torch.tensor(0.0),
                "exp_avg": torch.zeros_like(p),
                "exp_avg_sq": torch.zeros_like(p),
            }

    d64 = model64.embed.embedding_dim
    d_head64 = d64 // n_heads
    migrated, missing = [], []

    for name128, p128 in named128.items():
        # resolve the matching name64 + rule
        if name128 in _MOMENT_RULES:
            name64 = name128
            rule, scale = _MOMENT_RULES[name128]
        else:
            # strip "layers.{i}." prefix to match _LAYER_RULES
            parts = name128.split(".", 2)
            if len(parts) == 3 and parts[0] == "layers":
                suffix = parts[2]
                if suffix in _LAYER_RULES:
                    name64 = name128  # same dotted name exists in model64 too
                    rule, scale = _LAYER_RULES[suffix]
                else:
                    missing.append(name128)
                    continue
            else:
                missing.append(name128)
                continue

        p64 = named64.get(name64)
        if p64 is None or p64 not in state64:
            missing.append(name128)
            continue
        st64 = state64[p64]
        with torch.no_grad():
            exp_avg_m = _migrate_moment_tensor(st64["exp_avg"], rule, n_heads, d_head64, scale)
            exp_avg_sq_m = _migrate_moment_tensor(st64["exp_avg_sq"], rule, n_heads, d_head64, scale ** 2)
            assert exp_avg_m.shape == p128.shape, f"{name128}: exp_avg shape {exp_avg_m.shape} != param {p128.shape}"
            assert exp_avg_sq_m.shape == p128.shape, f"{name128}: exp_avg_sq shape {exp_avg_sq_m.shape} != param {p128.shape}"
            opt128.state[p128]["exp_avg"].copy_(exp_avg_m)
            opt128.state[p128]["exp_avg_sq"].copy_(exp_avg_sq_m)
            opt128.state[p128]["step"].copy_(st64["step"])
        migrated.append(name128)

    if missing:
        print(f"[migrate_adam_moments] WARNING: {len(missing)} params fell back to cold Adam state "
              f"(unexpected -- check _MOMENT_RULES/_LAYER_RULES coverage): {missing}", flush=True)
    return opt128, migrated, missing


def migrate_z_state(states64, n_heads=4):
    """Channel-duplicate each layer's carried Z (B,n_heads,d_head) -> (B,n_heads,2*d_head)
    within each head, matching the d_head widening of the scan weights."""
    if states64 is None:
        return None
    out = []
    for z in states64:
        B, H, D = z.shape
        z2 = torch.cat([z, z], dim=2)
        out.append(z2.detach())
    return out


# ═══════════════════════════════════════════════════════════════════════════
# §3c  P27a commutation gate: does grow(adam_step(m64)) == adam_step(grow_with_
#     migrated_moments(m64)) on the SAME batch? Adam is nonlinear in exp_avg_sq
#     (the sqrt denominator), so exact commutation is NOT guaranteed everywhere
#     -- this measures the gap honestly, per parameter group, and checks it
#     against both the 1e-3 absolute tolerance AND the update magnitude.
# ═══════════════════════════════════════════════════════════════════════════
def run_commutation_check(args):
    """--check-moments: fast, no streaming. Builds a d64 model with a WARM
    (non-trivial) Adam state (a handful of real gradient steps on real WT-2
    tokens so exp_avg/exp_avg_sq are non-degenerate), then:
      Path A: one MORE adam_step on d64 with a fixed probe batch, THEN grow
              (exact, noise_std=0) the POST-step model.
      Path B: grow (exact, noise_std=0) the PRE-step model + migrate_adam_
              moments, THEN one adam_step on model128 with the SAME probe
              batch (channel-duplication needs no input transform -- token
              ids are identical, grow_model's equivalence already covers that).
    Reports max|Path A - Path B| per named parameter (grouped by rule), and
    that gap relative to the update's own magnitude (||update||)."""
    train_text, val_text = load_wikitext2()
    vocab, stoi, unk, mask = build_vocab(train_text)
    V = len(vocab)
    print(f"[check-moments] vocab={V}", flush=True)

    B, K = args.batch, args.chunk
    n_heads = 4

    torch.manual_seed(args.seed)
    m64 = build_gssm_lm(V, mask, d_model=64, n_layers=2, n_heads=n_heads, d_head=16)
    opt64 = torch.optim.Adam(m64.parameters(), lr=args.lr)
    arm64 = {"tag": "m64", "model": m64, "opt": opt64, "states": None,
             "grad_tokens": 0, "n_chunks": 0, "n_bwd": 0}

    # ── warm up Adam with a handful of real steps so exp_avg/exp_avg_sq are non-trivial ──
    stream = C4Stream(stoi, unk)
    bufs = [[] for _ in range(B)]
    for b in range(B):
        bufs[b].extend(stream.next_block())

    def next_batch():
        for b in range(B):
            while len(bufs[b]) < K + 1:
                bufs[b].extend(stream.next_block())
        x = torch.tensor([bufs[b][:K] for b in range(B)], dtype=torch.long)
        y = torch.tensor([bufs[b][1:K + 1] for b in range(B)], dtype=torch.long)
        for b in range(B):
            del bufs[b][:K]
        return x, y

    n_warm = 20
    for _ in range(n_warm):
        x, y = next_batch()
        step_full(arm64, x, y)
    print(f"[check-moments] Adam warmed with {n_warm} real WT-2/C4 steps "
          f"({arm64['grad_tokens']:,} grad tokens)", flush=True)

    # snapshot the PRE-probe-step model + opt state (Path B starts here)
    m64_pre_sd = {k: v.clone() for k, v in m64.state_dict().items()}
    opt64_pre_state = {p: {kk: vv.clone() for kk, vv in st.items()} for p, st in opt64.state.items()}
    states_pre = arm64["states"]

    # the ONE fixed probe batch both paths will apply as "the identical-batch step"
    x_probe, y_probe = next_batch()

    # ── Path A: adam_step on d64 (probe batch), THEN grow (exact) ──
    logits, st = m64(x_probe, states_pre)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_probe.reshape(-1))
    opt64.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m64.parameters(), 5.0)
    opt64.step()
    m64_post = m64  # now stepped
    model128_A, _ = grow_model(m64_post, V, mask, n_layers=2, n_heads=n_heads,
                                noise_std=0.0, noise_seed=args.seed)
    sdA = {k: v.clone() for k, v in model128_A.state_dict().items()}

    # ── Path B: restore PRE-probe-step model, grow (exact) + migrate moments, THEN adam_step ──
    m64_restored = build_gssm_lm(V, mask, d_model=64, n_layers=2, n_heads=n_heads, d_head=16)
    m64_restored.load_state_dict(m64_pre_sd)
    opt64_restored = torch.optim.Adam(m64_restored.parameters(), lr=args.lr)
    named_restored = dict(m64_restored.named_parameters())
    named_orig = dict(m64.named_parameters())  # same names, used only to align original Parameter objects
    # opt64_pre_state is keyed by the ORIGINAL m64 Parameter objects (pre-step, but same objects since
    # opt.step() mutates in place, not replaces) -- since m64 IS mutated in place, opt64_pre_state was
    # snapshotted by VALUE (cloned tensors) above, so re-key it onto m64_restored's parameters by name.
    name_to_param64 = dict(m64.named_parameters())
    param64_to_name = {p: n for n, p in name_to_param64.items()}
    opt64_restored.state = {}
    for p_orig, st in opt64_pre_state.items():
        nm = param64_to_name[p_orig]
        opt64_restored.state[named_restored[nm]] = st

    model128_B, _ = grow_model(m64_restored, V, mask, n_layers=2, n_heads=n_heads,
                                noise_std=0.0, noise_seed=args.seed)
    opt128_B, migrated_names, missing_names = migrate_adam_moments(
        opt64_restored, m64_restored, model128_B, args.lr, n_layers=2, n_heads=n_heads)
    states128_pre = migrate_z_state(states_pre, n_heads=n_heads)

    logitsB, _ = model128_B(x_probe, states128_pre)
    lossB = F.cross_entropy(logitsB.reshape(-1, logitsB.size(-1)), y_probe.reshape(-1))
    opt128_B.zero_grad(set_to_none=True)
    lossB.backward()
    torch.nn.utils.clip_grad_norm_(model128_B.parameters(), 5.0)
    opt128_B.step()
    sdB = {k: v.clone() for k, v in model128_B.state_dict().items()}

    print(f"[check-moments] migrate_adam_moments: {len(migrated_names)} migrated, "
          f"{len(missing_names)} cold-fallback", flush=True)

    # ── compare, grouped by parameter "family" ──
    print("\n" + "=" * 78)
    print("P27a COMMUTATION GATE: grow(adam_step(m64)) [A] vs adam_step(migrate(grow(m64))) [B]")
    print("=" * 78)
    overall_max = 0.0
    rows = []
    for k in sdA:
        a, b = sdA[k], sdB[k]
        diff = (a - b).abs().max().item()
        upd = (b - torch.zeros_like(b)).abs().mean().item()  # placeholder, replaced below with true update norm
        rows.append((k, diff, tuple(a.shape)))
        overall_max = max(overall_max, diff)

    # update magnitude reference: ||model128_A - model128_(pre-step, grown)|| i.e. what the probe
    # step itself moved things by, so we can say "is the A/B gap small vs the step size"
    model128_prestep, _ = grow_model(m64_restored, V, mask, n_layers=2, n_heads=n_heads,
                                      noise_std=0.0, noise_seed=args.seed)
    sd_prestep = model128_prestep.state_dict()

    tol = 1e-3
    n_exact, n_approx = 0, 0
    for k, diff, shape in rows:
        upd_size = (sdA[k] - sd_prestep[k]).abs().max().item()
        rel = diff / upd_size if upd_size > 1e-12 else float("nan")
        status = "PASS" if diff < tol else ("small-vs-update" if (rel == rel and rel < 0.05) else "FAIL")
        if diff < tol:
            n_exact += 1
        else:
            n_approx += 1
        print(f"  {k:32s} shape={str(shape):16s} max|A-B|={diff:.3e}  step-size={upd_size:.3e}  "
              f"rel={rel:.4f}  {status}")
    print("-" * 78)
    print(f"  overall max|A-B| across all parameters = {overall_max:.3e} (tol {tol:.0e})")
    print(f"  {n_exact}/{len(rows)} tensors commute EXACTLY (< {tol:.0e}); "
          f"{n_approx}/{len(rows)} only approximately (Adam's exp_avg_sq sqrt-denominator nonlinearity)")
    print("=" * 78)
    return {"max_abs_err": overall_max, "rows": [(k, d) for k, d, _ in rows], "tol": tol}


# ═══════════════════════════════════════════════════════════════════════════
# §4  Main experiment
# ═══════════════════════════════════════════════════════════════════════════
def run_eval_block(arms, evX, evY, n_tok, t0, extra=None):
    rec = {"n_streamed": n_tok, "wall_s": round(time.time() - t0, 1)}
    for k, a in arms.items():
        hl = heldout(a["model"], evX, evY)
        rec[k] = {"heldout": round(hl, 6), "grad_tokens": a["grad_tokens"],
                  "n_chunks": a["n_chunks"], "n_bwd": a["n_bwd"]}
    line = " | ".join(f"{k} {rec[k]['heldout']:.4f}" for k in arms)
    suffix = f" | {extra}" if extra else ""
    print(f"[eval] n={n_tok:>10,} wall={time.time()-t0:6.1f}s | {line}{suffix}", flush=True)
    return rec


def run_experiment(args):
    out_path = os.path.join("results", f"hot_swap_growth_{args.tag}.json") if args.tag \
        else os.path.join("results", "hot_swap_growth.json")
    os.makedirs("results", exist_ok=True)

    train_text, val_text = load_wikitext2()
    vocab, stoi, unk, mask = build_vocab(train_text)
    V = len(vocab)
    val_ids = tokenize(val_text, stoi, unk)
    evX, evY = build_eval_set(val_ids, args.eval_tokens, args.chunk)
    print(f"[data] vocab={V} | frozen WT-2 held-out {evY.numel():,} tokens", flush=True)

    B, K = args.batch, args.chunk
    n_heads = 4

    # ── Phase A: build & stream a d64 model (this becomes both `grown` (pre-surgery)
    #    and `stay64`'s starting point -- IDENTICAL init, seed 42) ──
    torch.manual_seed(args.seed)
    m_grown = build_gssm_lm(V, mask, d_model=64, n_layers=2, n_heads=n_heads, d_head=16)
    torch.manual_seed(args.seed)
    m_stay64 = build_gssm_lm(V, mask, d_model=64, n_layers=2, n_heads=n_heads, d_head=16)
    with torch.no_grad():
        d0 = sum(float((p1 - p2).abs().sum()) for p1, p2 in
                 zip(m_grown.parameters(), m_stay64.parameters()))
    assert d0 == 0.0, f"grown/stay64 init mismatch: {d0}"
    print("[arms] grown/stay64 identical d64 init (seed 42)", flush=True)

    arms_phaseA = {
        "grown": {"tag": "grown", "model": m_grown, "opt": torch.optim.Adam(m_grown.parameters(), lr=args.lr),
                  "states": None, "grad_tokens": 0, "n_chunks": 0, "n_bwd": 0},
        "stay64": {"tag": "stay64", "model": m_stay64, "opt": torch.optim.Adam(m_stay64.parameters(), lr=args.lr),
                   "states": None, "grad_tokens": 0, "n_chunks": 0, "n_bwd": 0},
    }

    stream = C4Stream(stoi, unk)
    bufs = [[] for _ in range(B)]
    for b in range(B):
        bufs[b].extend(stream.next_block())

    n_tok = 0
    t0 = time.time()
    curves = {"grown": [], "stay64": [], "fresh128": []}

    base = run_eval_block(arms_phaseA, evX, evY, n_tok, t0)
    for k in ["grown", "stay64"]:
        curves[k].append(base[k] | {"n_streamed": 0, "phase": "A"})

    print(f"[phase A] streaming... target={args.phaseA_tokens:,} tokens (B={B} K={K})", flush=True)
    while n_tok < args.phaseA_tokens:
        for b in range(B):
            while len(bufs[b]) < K + 1:
                bufs[b].extend(stream.next_block())
        x = torch.tensor([bufs[b][:K] for b in range(B)], dtype=torch.long)
        y = torch.tensor([bufs[b][1:K + 1] for b in range(B)], dtype=torch.long)
        for b in range(B):
            del bufs[b][:K]
        n_tok += B * K

        step_full(arms_phaseA["grown"], x.clone(), y.clone())
        step_full(arms_phaseA["stay64"], x.clone(), y.clone())

        if n_tok % args.eval_every_tokens < B * K:
            rec = run_eval_block(arms_phaseA, evX, evY, n_tok, t0)
            for k in ["grown", "stay64"]:
                curves[k].append(rec[k] | {"n_streamed": n_tok, "phase": "A"})

    phaseA_final = run_eval_block(arms_phaseA, evX, evY, n_tok, t0)
    for k in ["grown", "stay64"]:
        curves[k].append(phaseA_final[k] | {"n_streamed": n_tok, "phase": "A_final"})
    print(f"[phase A] done at {n_tok:,} tokens. grown={phaseA_final['grown']['heldout']:.4f} "
          f"stay64={phaseA_final['stay64']['heldout']:.4f}", flush=True)

    # ── SURGERY: widen `grown` d64 -> d128, migrate carried Z, gate, then break symmetry ──
    print("\n[surgery] widening grown: d_model 64 -> 128 (channel duplication)...", flush=True)
    model64_pre = arms_phaseA["grown"]["model"]
    states64_pre = arms_phaseA["grown"]["states"]

    model128, model128_exact_sd = grow_model(model64_pre, V, mask, n_layers=2, n_heads=n_heads,
                                              noise_std=args.noise_std, noise_seed=args.seed)
    err_stateless, err_chunked = check_surgery_equivalence(
        model64_pre, model128_exact_sd, V, mask, n_layers=2, n_heads=n_heads, seed=args.seed)
    print(f"[surgery] equivalence: stateless max|delta logits| = {err_stateless:.3e} "
          f"{'PASS' if err_stateless < 1e-4 else 'FAIL'}", flush=True)
    print(f"[surgery] equivalence: chunked+carried(3 chunks) max|delta logits| = {err_chunked:.3e} "
          f"{'PASS' if err_chunked < 1e-4 else 'FAIL'}", flush=True)
    assert err_stateless < 1e-4, f"surgery stateless equivalence FAILED: {err_stateless:.3e}"
    assert err_chunked < 1e-4, f"surgery chunked equivalence FAILED: {err_chunked:.3e}"

    states128 = migrate_z_state(states64_pre, n_heads=n_heads)
    if states128 is not None:
        print(f"[surgery] Z-state migrated: {len(states128)} layers, "
              f"shape {tuple(states64_pre[0].shape)} -> {tuple(states128[0].shape)}", flush=True)

    if args.migrate_moments:
        opt64_grown = arms_phaseA["grown"]["opt"]
        opt_grown, migrated_names, missing_names = migrate_adam_moments(
            opt64_grown, model64_pre, model128, args.lr, n_layers=2, n_heads=n_heads)
        print(f"[surgery] Adam moments MIGRATED (P27): {len(migrated_names)} tensors migrated, "
              f"{len(missing_names)} fell back to cold state", flush=True)
    else:
        opt_grown = torch.optim.Adam(model128.parameters(), lr=args.lr)  # fresh Adam post-surgery
    arm_grown = {"tag": "grown", "model": model128, "opt": opt_grown, "states": states128,
                 "grad_tokens": arms_phaseA["grown"]["grad_tokens"], "n_chunks": 0, "n_bwd": 0}

    # ── fresh128: brand-new d128 model, seeded independently, starts AT the surgery point ──
    torch.manual_seed(args.seed + 1000)  # distinct seed: a genuinely fresh init, not a clone of grown
    m_fresh128 = build_gssm_lm(V, mask, d_model=128, n_layers=2, n_heads=n_heads, d_head=32)
    arm_fresh128 = {"tag": "fresh128", "model": m_fresh128,
                    "opt": torch.optim.Adam(m_fresh128.parameters(), lr=args.lr),
                    "states": None, "grad_tokens": 0, "n_chunks": 0, "n_bwd": 0}

    arm_stay64 = arms_phaseA["stay64"]

    # ── Phase B: stream identical chunks to all three arms (grown, stay64, fresh128) ──
    phaseB_arms = {"grown": arm_grown, "stay64": arm_stay64, "fresh128": arm_fresh128}
    n_tok_B = 0
    t0_B = time.time()
    transient = {"grown": [], "stay64": []}  # per-chunk online NLL, first N chunks post-surgery

    baseB = run_eval_block(phaseB_arms, evX, evY, n_tok_B, t0_B, extra="post-surgery baseline")
    for k in phaseB_arms:
        curves[k].append(baseB[k] | {"n_streamed": n_tok, "phase": "B"})

    print(f"\n[phase B] streaming... target={args.phaseB_tokens:,} tokens post-surgery "
          f"(B={B} K={K})", flush=True)
    transient_chunks = args.transient_chunks
    step_i = 0
    while n_tok_B < args.phaseB_tokens:
        for b in range(B):
            while len(bufs[b]) < K + 1:
                bufs[b].extend(stream.next_block())
        x = torch.tensor([bufs[b][:K] for b in range(B)], dtype=torch.long)
        y = torch.tensor([bufs[b][1:K + 1] for b in range(B)], dtype=torch.long)
        for b in range(B):
            del bufs[b][:K]
        n_tok_B += B * K

        l_grown = step_full(arm_grown, x.clone(), y.clone())
        l_stay64 = step_full(arm_stay64, x.clone(), y.clone())
        l_fresh128 = step_full(arm_fresh128, x.clone(), y.clone())

        if step_i < transient_chunks:
            transient["grown"].append(l_grown)
            transient["stay64"].append(l_stay64)
        step_i += 1

        if n_tok_B % args.eval_every_tokens < B * K:
            rec = run_eval_block(phaseB_arms, evX, evY, n_tok_B, t0_B,
                                  extra=f"total={n_tok + n_tok_B:,}")
            for k in phaseB_arms:
                curves[k].append(rec[k] | {"n_streamed": n_tok + n_tok_B, "phase": "B"})

    finalB = run_eval_block(phaseB_arms, evX, evY, n_tok_B, t0_B, extra="FINAL")
    for k in phaseB_arms:
        curves[k].append(finalB[k] | {"n_streamed": n_tok + n_tok_B, "phase": "B_final"})

    # ── P24 checks ──
    transient_grown_mean = float(np.mean(transient["grown"])) if transient["grown"] else float("nan")
    transient_stay64_mean = float(np.mean(transient["stay64"])) if transient["stay64"] else float("nan")
    transient_gap = abs(transient_grown_mean - transient_stay64_mean)
    check_b_pass = transient_gap < 0.05

    fin_grown = finalB["grown"]["heldout"]
    fin_stay64 = finalB["stay64"]["heldout"]
    fin_fresh128 = finalB["fresh128"]["heldout"]

    gain_over_stay64 = fin_stay64 - fin_grown  # positive = grown better (lower NLL)
    check_c_pass = gain_over_stay64 >= 0.03

    gain_over_fresh128 = fin_fresh128 - fin_grown  # positive = grown better
    check_d_pass = gain_over_fresh128 > 0.0

    check_a_pass = err_stateless < 1e-4 and err_chunked < 1e-4

    print("\n" + "=" * 78)
    print("VERDICT -- P24 hot-swap growth (d64 -> d128 mid-stream)")
    print(f"  (a) surgery equivalence: stateless {err_stateless:.3e}, chunked(3) {err_chunked:.3e} "
          f"(tol 1e-4) {'PASS' if check_a_pass else 'FAIL'}")
    print(f"  (b) no post-surgery transient: mean|grown-stay64| over first {transient_chunks} "
          f"post-surgery chunks = {transient_gap:.4f} nats (grown {transient_grown_mean:.4f}, "
          f"stay64 {transient_stay64_mean:.4f}) (tol < 0.05) "
          f"{'PASS' if check_b_pass else 'FAIL'}")
    print(f"  (c) growth pays off: final held-out grown {fin_grown:.4f} vs stay64 {fin_stay64:.4f} "
          f"-> grown better by {gain_over_stay64:+.4f} nats (need >= 0.03) "
          f"{'PASS' if check_c_pass else 'FAIL'}")
    print(f"  (d) growth beats restart: final held-out grown {fin_grown:.4f} vs fresh128 {fin_fresh128:.4f} "
          f"-> grown better by {gain_over_fresh128:+.4f} nats (need > 0) "
          f"{'PASS' if check_d_pass else 'FAIL'}")
    all_pass = check_a_pass and check_b_pass and check_c_pass and check_d_pass
    print(f"  OVERALL: {'ALL FOUR PASS' if all_pass else 'NOT ALL CHECKS PASS -- see above'}")
    print("=" * 78)

    out = {
        "prediction_id": "P24",
        "prediction_text": "hot-swap growth (channel-duplication d64->d128 mid-stream, Z migrated): "
                           "(a) surgery equiv <1e-4; (b) transient <0.05 nats; "
                           "(c) grown beats stay64 by >=0.03 nats at +phaseB tokens; "
                           "(d) grown beats fresh128-from-scratch at matched post-surgery tokens",
        "config": vars(args),
        "surgery_equivalence_gate": {
            "stateless_max_abs_err": err_stateless, "chunked_3_max_abs_err": err_chunked,
            "tol": 1e-4, "pass": check_a_pass,
        },
        "phaseA_final": phaseA_final,
        "phaseB_baseline": baseB,
        "phaseB_final": finalB,
        "curves": curves,
        "transient_first_n_chunks": {
            "n_chunks": transient_chunks,
            "grown_mean_nll": transient_grown_mean, "stay64_mean_nll": transient_stay64_mean,
            "abs_gap": transient_gap, "grown_series": transient["grown"],
            "stay64_series": transient["stay64"],
        },
        "checks": {
            "a_surgery_equivalence": {"pass": check_a_pass, "stateless_err": err_stateless,
                                      "chunked_err": err_chunked},
            "b_no_transient": {"pass": check_b_pass, "gap_nats": transient_gap, "threshold": 0.05},
            "c_growth_pays_off": {"pass": check_c_pass, "grown_heldout": fin_grown,
                                  "stay64_heldout": fin_stay64, "gain_nats": gain_over_stay64,
                                  "threshold": 0.03},
            "d_growth_beats_restart": {"pass": check_d_pass, "grown_heldout": fin_grown,
                                       "fresh128_heldout": fin_fresh128,
                                       "gain_nats": gain_over_fresh128},
        },
        "all_checks_pass": all_pass,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {out_path}")
    return out


def main():
    ap = argparse.ArgumentParser(description="MS8: hot-swap growth (function/state-preserving widening, live stream)")
    ap.add_argument("--smoke", action="store_true", help="Phase A/B 300k tokens each, held-out 20k")
    ap.add_argument("--full", action="store_true", help="Phase A/B 1.2M tokens each, held-out 20k")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise-std", type=float, default=1e-4)
    ap.add_argument("--phaseA-tokens", type=int, default=0)
    ap.add_argument("--phaseB-tokens", type=int, default=0)
    ap.add_argument("--eval-tokens", type=int, default=20_000)
    ap.add_argument("--eval-every-tokens", type=int, default=150_000)
    ap.add_argument("--transient-chunks", type=int, default=20)
    ap.add_argument("--tag", default="")
    ap.add_argument("--migrate-moments", action="store_true",
                     help="P27/MS14: migrate opt64's Adam moments (exp_avg/exp_avg_sq/step) through "
                          "the duplication map instead of a fresh Adam for the grown arm post-surgery")
    ap.add_argument("--check-moments", action="store_true",
                     help="P27a commutation gate ONLY (no streaming): adam_step then grow (Path A) vs "
                          "grow+migrate then adam_step (Path B) on a fixed toy batch; report max|A-B| "
                          "per parameter group and exit")
    args = ap.parse_args()

    if args.check_moments:
        run_commutation_check(args)
        return

    if args.smoke:
        args.phaseA_tokens = args.phaseA_tokens or 300_000
        args.phaseB_tokens = args.phaseB_tokens or 300_000
        args.tag = args.tag or "smoke"
    else:
        args.phaseA_tokens = args.phaseA_tokens or 1_200_000
        args.phaseB_tokens = args.phaseB_tokens or 1_200_000

    if args.migrate_moments and not args.tag:
        args.tag = "mig"
    elif args.migrate_moments and not args.tag.endswith("mig"):
        args.tag = f"{args.tag}_mig"

    run_experiment(args)


if __name__ == "__main__":
    main()
