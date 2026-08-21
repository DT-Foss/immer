#!/usr/bin/env python3 -u
"""
HOLO-ALPHA-SHUT (MS12) — testing whether F3's filler phi-drift is CAUSAL, by
regularizing it away at train time and watching whether the recall knee moves.
======================================================================================
MISSION (P25, analysis/PREDICTIONS.md): src/phi_drift_probe.py (F3) measured
that the BEST known knee recipe (M3: src/holo_mag_read.py's V2_kickstart_magnorm
-- GammaKickstartScanLayer's fixed gamma->1 offset on head 0, composed with
MagNormReadMixin's per-channel-unit-magnitude read, trained via
holo_gap_knee.train_fullseq_curriculum with g_train_max=128, bar=0.8) is not
actually phase-clean in deployment: real filler tokens drive a_t > 0 (alpha_t
is NOT shut on fillers the way the idle-persistence beacon's alpha shuts), so
arg(S) drifts 1.52 rad by G=512 (mag_ratio up to 35x) even though the NULL-
INPUT control (a forced to exactly 0) proved arg(S) is EXACTLY invariant when
the drive vanishes (HOLO_CARRIER_THEORY.md Sec.1, LAW LOCKED in phi_drift.json).
The gap between "phase invariant when a=0" (proven) and "phase drifts a lot in
practice" (measured) is filler alpha not shutting -- an OBSERVATIONAL finding.
This file asks whether it is CAUSAL: does training the model to keep alpha_t
small ON FILLER POSITIONS SPECIFICALLY (not on the write positions, where a
large drive is exactly the desired write) reduce the measured drift, and does
reducing the drift move the recall knee.

REGULARIZER: loss = task_loss + lambda * mean(alpha_t over FILLER positions).
alpha_t = torch.sigmoid(self.W_alpha(x)) is the INPUT gate computed inside
_drive_and_gamma (src/holographic_gssm.py:194) -- the "read-time write
suppressor" the idle-persistence beacon (streaming_train.py Pillar E) already
proved CAN be driven near 0 on fillers for a 1-bit task; this regularizer
tests whether the SAME mechanism can be induced for the keyed P=2 write under
an explicit loss term rather than relying on it to emerge unshaped (M3's
V2 never asked for it, and F3 measured that it mostly didn't happen).
alpha lives at (B,T,n_heads,d_head); mean() reduces across heads/channels AND
across the batch/filler-position set actually included in the loss term (see
AlphaCaptureMixin below for exactly how it's captured without touching
holographic_gssm.py or holo_stream_recall.py).

FILLER MASK: the task's sequence layout (holo_stream_recall.make_gap_mqar_batch,
reused unmodified) is fixed and deterministic: [k_1,v_1,...,k_P,v_P,
fill_1,...,fill_G, query_key], length 2P+G+1. Filler positions are therefore
EXACTLY the index range [2P, 2P+G) (0-indexed) for every trial in a batch --
no content-dependent masking needed, no ambiguity with key/value/query
positions. This mask is recomputed per-Gcur inside the training loop (Gcur
changes as the curriculum grows).

ALPHA CAPTURE (no changes to holographic_gssm.py / holo_stream_recall.py /
holo_gap_knee.py / holo_mag_read.py): AlphaCaptureMixin overrides
_drive_and_gamma to call the class's own (possibly kickstart-offset) gamma
computation via super()._drive_and_gamma(x) for the RETURN VALUE (a, gamma)
unchanged -- byte-identical drive/gamma to the parent chain -- and SEPARATELY
recomputes alpha = torch.sigmoid(self.W_alpha(x)).view(B,T,H,D) as a pure
function of x and the layer's own W_alpha (the identical formula every parent
in the MRO already uses internally to build `a`, just not returned). This is
cheap, exact, and non-invasive: one extra Linear+sigmoid per forward call,
cached on self._last_alpha for the training loop to read immediately after
each forward. Composes with GammaKickstartScanLayer (kickstart offset only
touches gamma, never alpha) and MagNormReadMixin (only touches the post-write
READ, never the drive/gate computation) via the same multiple-inheritance
pattern holo_mag_read.py already established for V2.

ARMS: lambda in {0 (CONTROL = M3's exact V2_kickstart_magnorm recipe, byte-
for-byte -- AlphaCaptureMixin adds instrumentation only, the loss is
task_loss alone when lambda=0), 3e-3, 1e-2, 3e-2}. P=2, seeds {0,1}. Training:
full-sequence (unchunked) curriculum, g_train_max=128, bar=0.8, patience=25,
identical hyperparameters to M3. Eval: chunked+carried (chunk=16, the
deployment path), G in {128,256,512,1024,2048}, zeroed-at-gap null at every G.

DRIFT DIAGNOSIS per trained model, G in {128,512,2048}: SAME methodology as
phi_drift_probe.py -- SNR-gated (SNR_FLOOR=1e-4 on |S|_t) weighted mean|Delta-
phi| and mag_ratio over the filler span, measured on REAL fillers (condition
(a) in that file's terms; the null-input control is not re-measured here
since phi_drift_probe.py already LAW-LOCKED it and this file is not testing
that claim again). Causal bridge: does the drift number at G=512 fall as
lambda grows, and does the knee rise as lambda grows.

OUTPUT: results/holo_alpha_shut.json (--smoke: results/holo_alpha_shut_smoke.json,
lambda in {0, 1e-2}, seed 0 only, G in {128,512}, reduced iters). Per (lambda,
seed): curriculum summary, mean alpha_filler during the LAST training window
(the regularizer's own target, logged so lambda=0 vs lambda>0 is directly
checkable before even looking at eval), recall curve over G, knee (last G with
acc >= 0.5*acc@G=128 -- ref_G=128 here, not 32, because this file's Gs_eval
starts at 128, matching the smaller sweep the mission specifies), drift
diagnosis at G in {128,512,2048}. P25 checks: (a) drift@512 < 0.5 rad for the
best lambda>0 arm; (b) knee >= 1024 for the best lambda>0 arm; (c) dose-
monotonicity across the 4 lambda points (Spearman sign check: drift@512
decreasing in lambda, knee non-decreasing in lambda); trade-off warning if
recall@128 drops > 10pp between lambda=0 and any lambda>0 arm.

CPU-only (mps disabled), torch.set_num_threads(1), os.nice(19) best-effort,
seed-fixed. Does NOT write to results/pos_* or overwrite any existing
holo_*.json (own namespace: results/holo_alpha_shut*.json only). Does not
modify holo_mag_read.py, holo_gap_knee.py, holo_stream_recall.py,
holographic_gssm.py, or phi_drift_probe.py.
"""
import os
import sys
import json
import math
import time
import argparse

try:
    os.nice(19)
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference"))

import torch
import torch.nn as nn

torch.backends.mps.is_available = lambda: False   # force CPU (repo convention)
torch.set_num_threads(1)

from holo_stream_recall import (   # noqa: E402 -- reuse the proven streaming layer/LM/task, unmodified
    StreamingHolographicScanLayer, StreamingHolographicLM, _build_lm,
    check_equivalence, _gap_vocab, make_gap_mqar_batch, chunked_forward,
    stateful_linear_scan,
)
from holo_gap_knee import (   # noqa: E402 -- reuse the kickstart + curriculum machinery, unmodified
    GammaKickstartScanLayer, KICKSTART_LOGIT, KICKSTART_HEAD, estimate_knee,
)
from holo_mag_read import (   # noqa: E402 -- reuse the M3 magnitude-normalized read, unmodified
    MagNormReadMixin, MagNormKickstartScanLayer, check_equivalence_magnorm,
    eval_gap_recall_chunked,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(REPO, "results")


# ═══════════════════════════════════════════════════════════════════════════
# 1. AlphaCaptureMixin — non-invasive: recomputes alpha (the input gate) as a
#    pure function of x and the layer's own W_alpha, caches it on
#    self._last_alpha, and returns (a, gamma) UNCHANGED from whatever the rest
#    of the MRO chain (GammaKickstartScanLayer's offset gamma, or the ordinary
#    parent) would have produced. Composes with MagNormReadMixin (which only
#    overrides forward's post-write READ, never _drive_and_gamma) and with
#    GammaKickstartScanLayer (which only offsets gamma, never alpha).
# ═══════════════════════════════════════════════════════════════════════════
class AlphaCaptureMixin:
    """Mixin placed FIRST in MRO so its _drive_and_gamma runs, captures alpha,
    then delegates to super()._drive_and_gamma(x) for the actual (a, gamma)
    the rest of the chain would have computed -- byte-identical drive/gamma,
    zero behavioral change. self._last_alpha is (B,T,n_heads,d_head), the
    SAME alpha = sigmoid(W_alpha(x)) every _drive_and_gamma implementation in
    this repo already computes internally; here it is simply also kept."""

    _alpha_clamp_factor = None   # None = inactive; float in [0,1] = eval-time filler clamp

    def _drive_and_gamma(self, x):
        a, gamma = super()._drive_and_gamma(x)
        B, T, _ = x.shape
        alpha = torch.sigmoid(self.W_alpha(x)).view(B, T, self.n_heads, self.d_head)
        self._last_alpha = alpha
        if self._alpha_clamp_factor is not None:
            # P29 (MS12b): EVAL-ONLY inference-time clamp. `a` is already
            # alpha * z_in (holographic_gssm.py:206) -- since z_in does not
            # depend on alpha, rescaling `a` by the SAME factor as rescaling
            # alpha is algebraically exact: (f*alpha)*z_in == f*(alpha*z_in).
            # The caller (see EvalClampContext below) sets this factor ONLY
            # for the duration of chunks it knows are pure filler (the KV
            # write phase and the query position are NEVER clamped -- the
            # write itself and the final readout are untouched, exactly the
            # ZeroDriveScanLayer pattern in phi_drift_probe.py but continuous
            # (0..1) instead of a hard layer swap, and gradient-free (no_grad
            # eval context) so this never touches training in any way).
            a = a * self._alpha_clamp_factor
        return a, gamma


class AlphaShutKickstartMagNormScanLayer(AlphaCaptureMixin, MagNormReadMixin, GammaKickstartScanLayer):
    """The M3-best recipe (Kickstart gamma-offset + MagNorm read), plus alpha
    capture for the training-time filler regularizer. MRO: AlphaCaptureMixin
    -> MagNormReadMixin -> GammaKickstartScanLayer -> StreamingHolographicScanLayer.
    forward() resolves to MagNormReadMixin.forward (mag_norm_read=True set in
    __init__ below), which internally calls self._drive_and_gamma(x) -- that
    call is what AlphaCaptureMixin intercepts to populate self._last_alpha
    BEFORE MagNormReadMixin's read-normalization logic ever runs. gamma still
    carries GammaKickstartScanLayer's fixed offset (via super() chain);
    nothing about the write or read math differs from M3's V2 at lambda=0."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mag_norm_read = True
        self._last_alpha = None


def _build_lm_alphashut(vocab_size, mask_idx, d_model=64, n_layers=2, n_heads=4, d_head=16):
    """Byte-for-byte the M3 V2_kickstart_magnorm construction path
    (holo_mag_read._build_lm_variant), but swapping in
    AlphaShutKickstartMagNormScanLayer so layer 0 additionally exposes
    _last_alpha. Weights are loaded from the SAME ordinary StreamingHolographicLM
    init (same xavier init given the same seed) as every other variant in this
    repo's M3/T-lineage, so lambda=0 reproduces M3 exactly modulo the
    (inert) alpha-capture instrumentation."""
    model = StreamingHolographicLM(
        vocab_size, mask_idx, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        d_head=d_head, seq_len=32, dropout=0.0, causal=True,
        phase_scale=math.pi, use_phase=True, readout="rms",
        separate_qk=False, n_slots=1)
    old = model.layers[0].scan
    swapped = AlphaShutKickstartMagNormScanLayer(
        old.d_model, d_head=old.d_head, n_heads=old.n_heads, causal=old.causal,
        dropout=0.0, phase_scale=old.phase_scale, use_phase=old.use_phase,
        readout=old.readout, separate_qk=old.separate_qk, n_slots=old.n_slots)
    swapped.load_state_dict(old.state_dict(), strict=False)
    model.layers[0].scan = swapped
    return model


# ═══════════════════════════════════════════════════════════════════════════
# 2. Training: full-sequence (unchunked) curriculum, IDENTICAL to
#    holo_gap_knee.train_fullseq_curriculum, plus the alpha-shut regularizer
#    added to the loss at every step. Reproduced locally (not imported) only
#    because the regularizer term needs the filler-position mask and
#    layer 0's _last_alpha, neither of which the imported function's signature
#    has any hook for -- everything else (curriculum growth logic, optimizer,
#    grad-clip, bar/patience) is copied verbatim.
# ═══════════════════════════════════════════════════════════════════════════
def train_fullseq_curriculum_alphashut(model, P, Gmax, iters, lr, seed, batch, lam,
                                       P_max, V_max, F, log_every=0, g_start=2,
                                       patience=25, bar=0.8, alpha_log_window=100):
    """Unchunked curriculum training (model(x, None), one autograd graph over
    the whole sequence -- the T2/M3 trick that keeps the query-loss gradient
    alive back through the write, at any gap length). Adds
    lambda * mean(alpha_t over filler positions) to the loss every step.
    Filler positions for a given Gcur are exactly indices [2P, 2P+Gcur) of the
    (B, 2P+Gcur+1) sequence x (make_gap_mqar_batch's fixed layout) -- recomputed
    each time Gcur changes. When Gcur==0 there are no filler positions and the
    regularizer term is simply 0 (no fillers exist yet at the start of the
    curriculum)."""
    key_lo, val_lo, fill_lo, _ = _gap_vocab(P_max, V_max, F)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    model.train()
    scan0 = model.layers[0].scan
    Gcur = min(g_start, Gmax) if Gmax > 0 else 0
    acc = 0.0
    good = 0
    task_loss_last = float("nan")
    alpha_reg_last = float("nan")
    alpha_filler_window = []   # rolling log of mean alpha-on-filler, last alpha_log_window steps with Gcur>0

    for it in range(iters):
        x, y = make_gap_mqar_batch(batch, P, Gcur, gen, P_max, V_max, F, key_lo, val_lo, fill_lo)
        logits, _ = model(x, None)                 # full-sequence graph, no chunking
        pred = logits[:, -1, :V_max]
        task_loss = lossf(pred, y)

        alpha_reg = torch.zeros((), dtype=task_loss.dtype)
        alpha_mean_this_step = None
        if Gcur > 0 and lam > 0.0:
            fpos_lo, fpos_hi = 2 * P, 2 * P + Gcur     # filler index range, x's fixed layout
            a_layer0 = scan0._last_alpha               # (B, T, n_heads, d_head), populated by this forward
            alpha_fillers = a_layer0[:, fpos_lo:fpos_hi]
            alpha_reg = alpha_fillers.mean()
            alpha_mean_this_step = float(alpha_reg.detach())
        elif Gcur > 0:
            # lambda==0 CONTROL: still log the (unregularized) filler alpha mean
            # for the honest lambda=0-vs-lambda>0 comparison, but do not add it
            # to the loss (task_loss alone -- exactly the M3 V2 recipe).
            a_layer0 = scan0._last_alpha
            fpos_lo, fpos_hi = 2 * P, 2 * P + Gcur
            with torch.no_grad():
                alpha_mean_this_step = float(a_layer0[:, fpos_lo:fpos_hi].mean())

        loss = task_loss + lam * alpha_reg

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        acc = float((pred.argmax(-1) == y).float().mean())
        task_loss_last = float(task_loss.detach())
        alpha_reg_last = float(alpha_reg.detach()) if isinstance(alpha_reg, torch.Tensor) else alpha_reg
        if alpha_mean_this_step is not None:
            alpha_filler_window.append(alpha_mean_this_step)
            if len(alpha_filler_window) > alpha_log_window:
                alpha_filler_window.pop(0)

        good = good + 1 if acc > bar else 0
        if good >= patience and Gcur < Gmax:
            Gcur = min(Gmax, int(Gcur * 1.5) + 1)
            good = 0
        if log_every and (it + 1) % log_every == 0:
            am = alpha_filler_window[-1] if alpha_filler_window else float("nan")
            print(f"    it {it+1:>4}/{iters}: loss {float(loss):.3f} (task {task_loss_last:.3f} "
                  f"+ lam*alpha {alpha_reg_last:.4f}) acc {acc:.3f} (train-gap {Gcur}) alpha_filler~{am:.4f}")

    mean_alpha_filler_last_window = (sum(alpha_filler_window) / len(alpha_filler_window)
                                     if alpha_filler_window else None)
    return {"final_train_gap": Gcur, "final_train_acc": round(acc, 4),
           "final_task_loss": round(task_loss_last, 4),
           "final_alpha_reg": round(alpha_reg_last, 6) if not math.isnan(alpha_reg_last) else None,
           "mean_alpha_filler_last_window": (round(mean_alpha_filler_last_window, 6)
                                             if mean_alpha_filler_last_window is not None else None),
           "alpha_log_window": alpha_log_window}


# ═══════════════════════════════════════════════════════════════════════════
# 3. Drift diagnosis — SAME methodology as phi_drift_probe.py: SNR-gated
#    (SNR_FLOOR=1e-4 on |S|_t) weighted mean|Delta-phi| and mag_ratio, measured
#    on REAL filler tokens (condition (a) there). Reproduced locally (not
#    imported) because phi_drift_probe.py's run_trials/summarize_condition are
#    written around a two-model (normal vs zero-drive) comparison this file
#    does not need -- only condition (a)'s machinery, on our own trained model.
# ═══════════════════════════════════════════════════════════════════════════
SNR_FLOOR = 1e-4


def wrapped_delta(phase_t, phase_0):
    d = phase_t - phase_0
    return torch.atan2(torch.sin(d), torch.cos(d))


@torch.no_grad()
def measure_filler_drift(model, P, G, NB, seed, P_max, V_max, F, chunk):
    """Write P pairs (chunked+carried), snapshot layer-0 (S_re,S_im) at the
    gap's start, continue through G real filler tokens (chunked+carried,
    return_internals), measure SNR-gated weighted mean|Delta-phi| and
    mag_ratio at the LAST filler position (offset G) -- the same quantity
    phi_drift_probe.py reports at each offset, evaluated here at the single
    offset G (the gap's full length) since that's what the knee causally
    depends on."""
    key_lo, val_lo, fill_lo, _ = _gap_vocab(P_max, V_max, F)
    gen = torch.Generator().manual_seed(seed)
    keys = torch.stack([torch.randperm(P_max, generator=gen)[:P] for _ in range(NB)]) + key_lo
    vals = torch.randint(0, V_max, (NB, P), generator=gen) + val_lo
    fillers = torch.randint(0, F, (NB, G), generator=gen) + fill_lo
    kv = torch.stack([keys, vals], dim=2).reshape(NB, 2 * P)

    model.eval()
    logits_kv, states = chunked_forward(model, kv, chunk)
    s0_re = states[0]["S_re"].clone()
    s0_im = states[0]["S_im"].clone()
    mag0 = torch.sqrt(s0_re ** 2 + s0_im ** 2 + 1e-12)
    phase0 = torch.atan2(s0_im, s0_re)

    st = states
    pos = 0
    s_re_last, s_im_last = s0_re, s0_im
    while pos < G:
        hi = min(G, pos + chunk)
        xc = fillers[:, pos:hi]
        h = model.embed(xc)
        y0, st_new, internals = model.layers[0].scan(h, st[0], return_internals=True)
        s_re_last = internals["S_re"][:, -1].clone()
        s_im_last = internals["S_im"][:, -1].clone()
        h1 = model.layers[0].ln1(h + y0)
        h1 = model.layers[0].ln2(h1 + model.layers[0].ffn(h1))
        new_states = [st_new]
        hcur = h1
        for li in range(1, len(model.layers)):
            y, sto = model.layers[li].scan(hcur, st[li])
            hcur = model.layers[li].ln1(hcur + y)
            hcur = model.layers[li].ln2(hcur + model.layers[li].ffn(hcur))
            new_states.append(sto)
        st = [{k: v.detach() for k, v in s.items()} for s in new_states]
        pos = hi

    mag_t = torch.sqrt(s_re_last ** 2 + s_im_last ** 2 + 1e-12)
    phase_t = torch.atan2(s_im_last, s_re_last)
    dphi = wrapped_delta(phase_t, phase0).abs()

    alive = mag_t >= SNR_FLOOR
    n_alive = int(alive.sum())
    n_total = int(alive.numel())
    if n_alive > 0:
        w_snr = mag_t[alive] / (mag_t[alive].sum() + 1e-12)
        dphi_snr_weighted = float((dphi[alive] * w_snr).sum())
    else:
        dphi_snr_weighted = float("nan")
    mag_ratio_mean = float((mag_t / (mag0 + 1e-12)).mean())

    return {"G": G, "dphi_snr_gated_weighted": round(dphi_snr_weighted, 6) if n_alive else None,
           "mag_ratio_mean": round(mag_ratio_mean, 6),
           "snr_alive_frac": round(n_alive / n_total, 4)}


# ═══════════════════════════════════════════════════════════════════════════
# 3b. P29 (MS12b) — EVAL-TIME alpha clamp on filler positions. No training
#     change anywhere: this operates on an already-trained (lambda=0, M3
#     recipe, gap-2-trained, knee-512) model, at eval time only, via the
#     _alpha_clamp_factor hook AlphaCaptureMixin._drive_and_gamma checks.
#     scan0._alpha_clamp_factor is set to `factor` for the duration of a
#     filler CHUNK's forward call and reset to None immediately after --
#     the KV write phase and the query position are ALWAYS run with the
#     hook inactive (factor=None), so the write itself and the final
#     readout are byte-identical to the unclamped model; only the filler
#     span's contribution to `a` (and therefore to the drive_re/drive_im fed
#     into the complex accumulator during the gap) is rescaled.
# ═══════════════════════════════════════════════════════════════════════════
class EvalClampContext:
    """Context manager: sets scan0._alpha_clamp_factor = factor on entry,
    restores the PREVIOUS value (always None in this file's usage, but
    restoring rather than hardcoding None is the safe pattern for a
    stateful flag mutated outside normal control flow) on exit. factor=None
    is a no-op context (explicitly supported so callers can request the
    "unclamped" arm through the same code path as the clamped ones)."""
    def __init__(self, scan0, factor):
        self.scan0 = scan0
        self.factor = factor
        self._prev = None

    def __enter__(self):
        self._prev = getattr(self.scan0, "_alpha_clamp_factor", None)
        self.scan0._alpha_clamp_factor = self.factor
        return self

    def __exit__(self, *exc):
        self.scan0._alpha_clamp_factor = self._prev
        return False


@torch.no_grad()
def eval_gap_recall_chunked_alphaclamp(model, P, G, NB, seed, P_max, V_max, F, chunk, clamp_factor):
    """Same task/machinery as holo_mag_read.eval_gap_recall_chunked (chunked+
    carried, the deployment path), but the FILLER span specifically is run
    with layer 0's alpha clamped by `clamp_factor` (None = unclamped control,
    0.0 = full P29 clamp, 0.5 = dose midpoint). The KV write phase (chunk-by-
    chunk over the first 2P tokens) and the query position (the final token)
    are ALWAYS unclamped -- only whichever chunk positions fall strictly
    inside [2P, 2P+G) are affected, matching the task's fixed sequence layout
    (make_gap_mqar_batch: [k1,v1,...,kP,vP, fill_1..fill_G, query_key])."""
    key_lo, val_lo, fill_lo, _ = _gap_vocab(P_max, V_max, F)
    gen = torch.Generator().manual_seed(seed)
    x, y = make_gap_mqar_batch(NB, P, G, gen, P_max, V_max, F, key_lo, val_lo, fill_lo)
    model.eval()
    scan0 = model.layers[0].scan
    kv_len = 2 * P
    T = x.shape[1]

    states = None
    last_logits = None
    pos = 0
    while pos < T:
        hi = min(T, pos + chunk)
        # a chunk is "pure filler" iff it lies entirely inside [kv_len, kv_len+G)
        # -- with chunk=16 and kv_len=2P=4, every chunk boundary after the first
        # falls cleanly inside or outside the filler span except possibly the
        # FIRST chunk (which mixes the tail of the KV phase with the start of
        # the filler span) and the LAST chunk (which may mix filler with the
        # query token). Mixed chunks are conservatively left UNCLAMPED (the
        # clamp only engages for chunks with pos >= kv_len and hi <= kv_len+G,
        # i.e. no touching of KV or query content) -- this slightly
        # under-clamps at chunk edges rather than ever touching the write or
        # the query, matching the "no training/write touch" mission constraint.
        is_pure_filler = (clamp_factor is not None) and (pos >= kv_len) and (hi <= kv_len + G)
        with EvalClampContext(scan0, clamp_factor if is_pure_filler else None):
            logits_c, states = model(x[:, pos:hi], states)
        last_logits = logits_c   # only the final position is ever read below;
        pos = hi                 # storing every chunk's logits is O(G) memory
    logits = last_logits         # (measured: 20 GB dead logits -> OOM at G=1M)

    pred = logits[:, -1, :V_max]
    acc = float((pred.argmax(-1) == y).float().mean())
    return acc


@torch.no_grad()
def measure_filler_drift_alphaclamp(model, P, G, NB, seed, P_max, V_max, F, chunk, clamp_factor):
    """Same as measure_filler_drift, but every filler chunk is run with layer
    0's alpha clamped by clamp_factor (None = unclamped). The KV write phase
    is always unclamped. Structurally identical to measure_filler_drift --
    duplicated (not parameterized in place) so the unclamped drift numbers
    already reported under P25 remain reproducible from the untouched
    function, and this P29 variant is the only one that reads
    _alpha_clamp_factor."""
    key_lo, val_lo, fill_lo, _ = _gap_vocab(P_max, V_max, F)
    gen = torch.Generator().manual_seed(seed)
    keys = torch.stack([torch.randperm(P_max, generator=gen)[:P] for _ in range(NB)]) + key_lo
    vals = torch.randint(0, V_max, (NB, P), generator=gen) + val_lo
    fillers = torch.randint(0, F, (NB, G), generator=gen) + fill_lo
    kv = torch.stack([keys, vals], dim=2).reshape(NB, 2 * P)

    model.eval()
    scan0 = model.layers[0].scan
    logits_kv, states = chunked_forward(model, kv, chunk)   # KV phase: always unclamped
    s0_re = states[0]["S_re"].clone()
    s0_im = states[0]["S_im"].clone()
    mag0 = torch.sqrt(s0_re ** 2 + s0_im ** 2 + 1e-12)
    phase0 = torch.atan2(s0_im, s0_re)

    st = states
    pos = 0
    s_re_last, s_im_last = s0_re, s0_im
    while pos < G:
        hi = min(G, pos + chunk)
        xc = fillers[:, pos:hi]
        h = model.embed(xc)
        with EvalClampContext(scan0, clamp_factor):   # entire filler span is clamped when requested
            y0, st_new, internals = model.layers[0].scan(h, st[0], return_internals=True)
        s_re_last = internals["S_re"][:, -1].clone()
        s_im_last = internals["S_im"][:, -1].clone()
        h1 = model.layers[0].ln1(h + y0)
        h1 = model.layers[0].ln2(h1 + model.layers[0].ffn(h1))
        new_states = [st_new]
        hcur = h1
        for li in range(1, len(model.layers)):
            y, sto = model.layers[li].scan(hcur, st[li])
            hcur = model.layers[li].ln1(hcur + y)
            hcur = model.layers[li].ln2(hcur + model.layers[li].ffn(hcur))
            new_states.append(sto)
        st = [{k: v.detach() for k, v in s.items()} for s in new_states]
        pos = hi

    mag_t = torch.sqrt(s_re_last ** 2 + s_im_last ** 2 + 1e-12)
    phase_t = torch.atan2(s_im_last, s_re_last)
    dphi = wrapped_delta(phase_t, phase0).abs()

    alive = mag_t >= SNR_FLOOR
    n_alive = int(alive.sum())
    n_total = int(alive.numel())
    if n_alive > 0:
        w_snr = mag_t[alive] / (mag_t[alive].sum() + 1e-12)
        dphi_snr_weighted = float((dphi[alive] * w_snr).sum())
    else:
        dphi_snr_weighted = float("nan")
    mag_ratio_mean = float((mag_t / (mag0 + 1e-12)).mean())

    return {"G": G, "dphi_snr_gated_weighted": round(dphi_snr_weighted, 6) if n_alive else None,
           "mag_ratio_mean": round(mag_ratio_mean, 6),
           "snr_alive_frac": round(n_alive / n_total, 4)}


def run_eval_alpha_clamp(args):
    """P29 orchestration: train ONE model per seed with the M3 recipe
    (lambda=0, train_fullseq_curriculum_alphashut with lam=0.0 -- identical
    code path to the P25 control arm, so this reuses the exact same
    training function rather than a new one), then run the eval sweep and
    drift diagnosis THREE times per seed: unclamped (factor=None), full
    clamp (factor=0.0), and half-dose (factor=0.5). No training-time change
    at all -- clamping is purely a forward-pass hook active only during eval
    calls in this function."""
    P_max, V_max, F = args.p_max, args.v_max, args.f_fillers
    key_lo, val_lo, fill_lo, vocab_size = _gap_vocab(P_max, V_max, F)
    mask_idx = vocab_size
    P = args.pairs
    chance = 1.0 / V_max

    Gs_eval = [int(g) for g in args.gaps.split(",")]
    Gs_drift = [int(g) for g in args.drift_gaps.split(",") if g]
    seeds = [int(s) for s in args.seeds.split(",")]
    arms = [("unclamped", None), ("clamp_full", 0.0), ("clamp_half", 0.5)]

    print("=" * 78)
    print("HOLO-ALPHA-SHUT — P29/MS12b: inference-time alpha-clamp on filler positions")
    print(f"P={P}  P_max={P_max} V_max={V_max} F={F} vocab={vocab_size}  chance={chance:.4f}")
    print(f"eval gaps(G)={Gs_eval}  drift gaps={Gs_drift}  seeds={seeds}  arms={[a for a,_ in arms]}")
    print("=" * 78)

    print("\n── equivalence gates (reused, un-normalized + magnorm) ──")
    eq_ref = check_equivalence(vocab_size, mask_idx, seed=0, T=48, chunk=16, use_phase=True)
    eq_magnorm = check_equivalence_magnorm(vocab_size, mask_idx, seed=0, T=48, chunk=args.chunk,
                                           d_model=args.d_model, n_layers=args.n_layers,
                                           n_heads=args.n_heads, d_head=args.d_head)
    print(f"   reference_streaming max|Δ| = {eq_ref:.3e}")
    print(f"   magnorm             max|Δ| = {eq_magnorm:.3e}")
    eq_ok = eq_ref < 1e-5 and eq_magnorm < 1e-5
    print(f"   {'PASS' if eq_ok else 'FAIL — do not trust anything below'}")
    equivalence = {"reference_streaming": eq_ref, "magnorm": eq_magnorm, "passed": eq_ok}
    if not eq_ok:
        out = {"config": vars(args), "equivalence": equivalence,
              "verdict": "VOID — equivalence check failed"}
        os.makedirs(RESULTS, exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"\n→ {args.out}")
        return

    t0 = time.time()
    results = {"config": vars(args), "equivalence": equivalence, "chance": chance,
              "recipe": "M3 V2_kickstart_magnorm, lambda=0 (identical to P25's control arm / "
                        "lam0check), trained ONCE per seed. Eval sweep + drift diagnosis run "
                        "THREE times per seed against the SAME trained model: unclamped, "
                        "alpha-clamped-to-0 on filler positions (eval-only), and "
                        "alpha-clamped-to-0.5x (dose midpoint). No training-time change.",
              "kickstart_logit": KICKSTART_LOGIT, "kickstart_head": KICKSTART_HEAD,
              "curriculum": {}, "sweep": {}, "knees": {}, "drift": {}}

    for seed in seeds:
        print(f"\n{'='*78}\nseed={seed}  (training M3 recipe, lambda=0, ONCE)\n{'='*78}")
        torch.manual_seed(seed)
        model = _build_lm_alphashut(vocab_size, mask_idx, args.d_model, args.n_layers,
                                    args.n_heads, args.d_head)
        curr = train_fullseq_curriculum_alphashut(
            model, P, args.g_train_max, args.iters, args.lr, seed, args.batch, 0.0,
            P_max, V_max, F, log_every=args.log_every, g_start=args.g_start,
            patience=args.patience, bar=args.curriculum_bar)
        results["curriculum"][f"seed{seed}"] = curr
        print(f"  curriculum done: final_train_gap={curr['final_train_gap']} "
              f"final_train_acc={curr['final_train_acc']:.3f}")

        for arm_name, factor in arms:
            key = f"seed{seed}_{arm_name}"
            print(f"  ── arm={arm_name} (clamp_factor={factor}) ──")
            by_G = {}
            for G in Gs_eval:
                if G >= 2048:
                    eb = max(10, args.eval_batch // 4)
                elif G >= 1024:
                    eb = max(10, args.eval_batch // 2)
                else:
                    eb = args.eval_batch
                acc = eval_gap_recall_chunked_alphaclamp(model, P, G, eb, seed + 1000 + G,
                                                          P_max, V_max, F, args.chunk, factor)
                acc0 = eval_gap_recall_chunked(model, P, G, eb, seed + 1000 + G, P_max, V_max, F,
                                               args.chunk, zero_at_gap=True)
                by_G[G] = acc
                sk = f"{arm_name}|seed{seed}|G{G}"
                results["sweep"][sk] = {"seed": seed, "arm": arm_name, "clamp_factor": factor,
                                        "P": P, "G": G, "accuracy": round(acc, 4),
                                        "chance": round(chance, 4),
                                        "beats_chance_3x": bool(acc > 3 * chance),
                                        "zeroed_null_accuracy": round(acc0, 4)}
                print(f"      G={G:>5}: acc={acc:.4f} (chance {chance:.4f})  zeroed-null={acc0:.4f}")

            ref_G = Gs_eval[0]
            knee = estimate_knee(by_G, Gs_eval, ref_G=ref_G)
            results["knees"][key] = {"knee_G": knee, "ref_G": ref_G,
                                     "acc_at_ref": round(by_G.get(ref_G, float("nan")), 4)}
            print(f"      knee estimate (last G with acc >= 0.5*acc@{ref_G}): {knee}")

            drift_by_G = {}
            for G in Gs_drift:
                nb_drift = max(20, min(60, args.eval_batch // 2))
                d = measure_filler_drift_alphaclamp(model, P, G, nb_drift, seed + 3000 + G,
                                                    P_max, V_max, F, args.chunk, factor)
                drift_by_G[str(G)] = d
                print(f"      drift@G={G:>5}: Δφ_snr={d['dphi_snr_gated_weighted']}  "
                      f"mag_ratio={d['mag_ratio_mean']}  alive={d['snr_alive_frac']}")
            results["drift"][key] = drift_by_G

    # ── aggregate per-arm (mean over seeds): knee, drift@512, acc@ref_G ──
    ref_G0 = Gs_eval[0]
    per_arm = {}
    for arm_name, factor in arms:
        knee_vals = [v["knee_G"] for k, v in results["knees"].items()
                    if k.endswith(f"_{arm_name}") and v["knee_G"] is not None]
        mean_knee = sum(knee_vals) / len(knee_vals) if knee_vals else None

        drift512_vals = [v["512"]["dphi_snr_gated_weighted"] for k, v in results["drift"].items()
                         if k.endswith(f"_{arm_name}") and "512" in v
                         and v["512"]["dphi_snr_gated_weighted"] is not None]
        mean_drift512 = sum(drift512_vals) / len(drift512_vals) if drift512_vals else None

        acc_ref_vals = [v["accuracy"] for k, v in results["sweep"].items()
                        if v["arm"] == arm_name and v["G"] == ref_G0]
        mean_acc_ref = sum(acc_ref_vals) / len(acc_ref_vals) if acc_ref_vals else None

        per_arm[arm_name] = {"clamp_factor": factor, "mean_knee_G": mean_knee,
                             "mean_drift_at_512": mean_drift512,
                             f"mean_acc_at_G{ref_G0}": mean_acc_ref}
    results["per_arm_summary"] = per_arm

    # ── P29 checks ──
    unclamped = per_arm["unclamped"]
    clamped = per_arm["clamp_full"]

    p29a_pass = (clamped["mean_drift_at_512"] is not None and clamped["mean_drift_at_512"] < 0.3)
    p29b_pass = (clamped["mean_knee_G"] is not None and clamped["mean_knee_G"] >= 1024)
    acc_u = unclamped.get(f"mean_acc_at_G{ref_G0}")
    acc_c = clamped.get(f"mean_acc_at_G{ref_G0}")
    p29c_pass = (acc_u is not None and acc_c is not None and abs(acc_u - acc_c) <= 0.05)

    verdict_bits = []
    verdict_bits.append(f"P29a (clamped drift@512 < 0.3 rad): {'CONFIRMED' if p29a_pass else 'NOT MET'} "
                        f"(clamped={clamped['mean_drift_at_512']}, unclamped={unclamped['mean_drift_at_512']})")
    verdict_bits.append(f"P29b (clamped knee >= 1024): {'CONFIRMED' if p29b_pass else 'NOT MET'} "
                        f"(clamped={clamped['mean_knee_G']}, unclamped={unclamped['mean_knee_G']})")
    verdict_bits.append(f"P29c (recall@G{ref_G0} unchanged within 5pp): "
                        f"{'CONFIRMED' if p29c_pass else 'NOT MET'} (unclamped={acc_u}, clamped={acc_c})")
    if p29a_pass and not p29b_pass:
        verdict_bits.append("HONEST READING: pollution is real (drift suppressed) but NOT the binding "
                            "constraint at 512+ -- the knee is capped by something else (magnitude floor? "
                            "phase SNR?) even once filler alpha-write is eliminated at eval time. That "
                            "becomes the next measurement, per the pre-registered fallback in P29's text.")
    half = per_arm["clamp_half"]
    verdict_bits.append(f"dose midpoint (clamp_half=0.5x): drift@512={half['mean_drift_at_512']}, "
                        f"knee={half['mean_knee_G']}, acc@G{ref_G0}={half.get(f'mean_acc_at_G{ref_G0}')}")

    results["verdict"] = " | ".join(verdict_bits)
    results["elapsed_s"] = round(time.time() - t0, 1)

    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "=" * 78)
    print("VERDICT")
    for b in verdict_bits:
        print("  " + b)
    print(f"\n→ {args.out}  ({results['elapsed_s']}s)")


# ═══════════════════════════════════════════════════════════════════════════
# 3c. P30 (MS12c) — the 2x2 that disentangles the two axes. P29 found the
#     filler write is a DOUBLE AGENT: it pollutes the phase (drift) but also
#     REFRESHES the magnitude (mag_ratio 30-80x unclamped vs ~0.0007 under
#     full clamp @G=2048) -- the knee past 512 is magnitude-bound, not
#     pollution-bound. This crosses clamp (kills the pollution, also kills
#     the feed) with an EXPLICIT magnitude refresh (feeds the state WITHOUT
#     reintroducing pollution, since renormalizing |S|->1 touches ONLY the
#     magnitude -- arg(S) = atan2(S_im,S_re) is invariant under a positive
#     rescaling of both components by the same factor, so phase is untouched
#     BY CONSTRUCTION regardless of what the phase currently is).
#
#     REFRESH IS A STATE OP BETWEEN CHUNKS, NOT A LAYER-FORWARD CHANGE: it
#     mutates the carried (S_re, S_im) dict returned by chunked_forward /
#     model(...) between chunk calls -- exactly the shape/call-site
#     zero_states already uses for the zeroed-at-gap null (holo_stream_recall.
#     StreamingHolographicScanLayer... consumed via model.zero_states(states)
#     between two chunked_forward calls). refresh_magnitude below is the
#     same idiom: read states[0]["S_re"/"S_im"], renormalize, hand back a new
#     states list. No layer class is touched, no forward() is overridden for
#     this purpose (clamp still uses its own existing forward hook, refresh
#     does not add another one).
# ═══════════════════════════════════════════════════════════════════════════
REFRESH_EPS = 1e-8
"""Below this |S|, the channel is treated as dead (write-once-frozen-at-zero
or never written): renormalizing it would manufacture a unit-magnitude
vector out of what is essentially float noise, inventing information the
mission's P30c gate explicitly forbids (zeroed-at-gap null must stay at
chance in every arm, refresh included -- a refresh that resurrects a
zeroed state would violate that by construction). Channels with
|S| < REFRESH_EPS are left EXACTLY as they are (both S_re and S_im
untouched), so a genuinely zeroed state stays genuinely zero under refresh."""


def refresh_magnitude(states, eps=REFRESH_EPS):
    """Renormalize layer 0's carried complex state to per-channel unit
    magnitude: S <- S / |S|, applied identically to S_re and S_im so the
    RATIO S_im/S_re (and therefore phase = atan2(S_im, S_re)) is exactly
    preserved -- a positive scalar rescaling of a 2D vector never changes
    its angle. Channels with |S| < eps are passed through UNCHANGED (the
    eps-guard from the mission spec: "a zeroed state stays dead"). Returns a
    NEW states list (does not mutate the input in place), mirroring
    model.zero_states's return-a-new-list convention. Only layer 0 is
    touched -- deeper layers' states are untouched, matching P29's clamp
    scope (layer 0 is where the holographic accumulator under test lives)."""
    st0 = states[0]
    S_re, S_im = st0["S_re"], st0["S_im"]
    mag = torch.sqrt(S_re * S_re + S_im * S_im + 1e-12)
    alive = mag >= eps
    S_re_new = torch.where(alive, S_re / mag.clamp(min=eps), S_re)
    S_im_new = torch.where(alive, S_im / mag.clamp(min=eps), S_im)
    new_st0 = dict(st0)
    new_st0["S_re"] = S_re_new
    new_st0["S_im"] = S_im_new
    return [new_st0] + list(states[1:])


@torch.no_grad()
def eval_gap_recall_chunked_clamprefresh(model, P, G, NB, seed, P_max, V_max, F, chunk,
                                         clamp_factor, refresh):
    """The P30 2x2 eval sweep: chunked+carried (the deployment path), with
    the FILLER span optionally alpha-clamped (clamp_factor, same semantics
    as P29's eval_gap_recall_chunked_alphaclamp -- None=off) AND/OR
    magnitude-refreshed at every chunk boundary strictly inside the filler
    span (refresh: bool). Refresh is applied to the RETURNED states AFTER
    each filler chunk's forward call, before the next chunk consumes them --
    a pure state-op between chunks, never inside the layer's forward. The KV
    write phase and the query position are NEVER clamped and NEVER
    refreshed (matching P29's write-untouched constraint, extended here to
    the new refresh axis for the same reason: the write itself is not what
    either intervention is meant to test)."""
    key_lo, val_lo, fill_lo, _ = _gap_vocab(P_max, V_max, F)
    gen = torch.Generator().manual_seed(seed)
    x, y = make_gap_mqar_batch(NB, P, G, gen, P_max, V_max, F, key_lo, val_lo, fill_lo)
    model.eval()
    scan0 = model.layers[0].scan
    kv_len = 2 * P
    T = x.shape[1]

    states = None
    last_logits = None
    pos = 0
    while pos < T:
        hi = min(T, pos + chunk)
        is_pure_filler = (pos >= kv_len) and (hi <= kv_len + G)
        clamp_here = clamp_factor if (is_pure_filler and clamp_factor is not None) else None
        with EvalClampContext(scan0, clamp_here):
            logits_c, states = model(x[:, pos:hi], states)
        if is_pure_filler and refresh:
            states = refresh_magnitude(states)   # STATE OP between chunks, not in forward
        last_logits = logits_c   # only the final position is ever read below;
        pos = hi                 # storing every chunk's logits is O(G) memory
    logits = last_logits         # (measured: 20 GB dead logits -> OOM at G=1M)

    pred = logits[:, -1, :V_max]
    acc = float((pred.argmax(-1) == y).float().mean())
    return acc


@torch.no_grad()
def eval_gap_recall_chunked_zeroed_clamprefresh(model, P, G, NB, seed, P_max, V_max, F, chunk,
                                                clamp_factor, refresh):
    """P30c control: zeroed-at-gap null UNDER each of the four arms -- the
    KV phase runs normally, then model.zero_states zeros the ENTIRE carried
    state (all layers, exactly holo_stream_recall.zero_states), then the
    filler span runs with clamp/refresh exactly as in the non-null sweep
    (refresh's eps-guard means a zeroed state, |S|=0 < REFRESH_EPS, is left
    at zero -- refresh must not manufacture information from nothing, the
    literal P30c requirement). Accuracy here should sit at chance in EVERY
    arm; a refresh arm that beats chance here would mean refresh invented a
    signal, which the eps-guard is specifically built to prevent."""
    key_lo, val_lo, fill_lo, _ = _gap_vocab(P_max, V_max, F)
    gen = torch.Generator().manual_seed(seed)
    x, y = make_gap_mqar_batch(NB, P, G, gen, P_max, V_max, F, key_lo, val_lo, fill_lo)
    model.eval()
    scan0 = model.layers[0].scan
    kv_len = 2 * P

    logits_kv, states = chunked_forward(model, x[:, :kv_len], chunk)
    states = model.zero_states(states)   # decisive null: zero EVERY carried tensor, every layer

    last_logits = logits_kv
    pos = kv_len
    T = x.shape[1]
    while pos < T:
        hi = min(T, pos + chunk)
        is_pure_filler = (hi <= kv_len + G)
        clamp_here = clamp_factor if (is_pure_filler and clamp_factor is not None) else None
        with EvalClampContext(scan0, clamp_here):
            logits_c, states = model(x[:, pos:hi], states)
        if is_pure_filler and refresh:
            states = refresh_magnitude(states)   # eps-guard: zeroed |S|=0 stays untouched
        last_logits = logits_c   # only the final position is ever read below;
        pos = hi                 # storing every chunk's logits is O(G) memory
    logits = last_logits         # (measured: 20 GB dead logits -> OOM at G=1M)

    pred = logits[:, -1, :V_max]
    acc = float((pred.argmax(-1) == y).float().mean())
    return acc


@torch.no_grad()
def measure_filler_drift_clamprefresh(model, P, G, NB, seed, P_max, V_max, F, chunk,
                                      clamp_factor, refresh):
    """Drift + mag_ratio diagnosis under the 2x2: same machinery as
    measure_filler_drift_alphaclamp, plus refresh applied to layer 0's
    (S_re,S_im) at every filler chunk boundary (after the chunk's forward,
    before the next chunk starts) when refresh=True. Phase is measured
    exactly as before (atan2 on the possibly-refreshed S_re/S_im) -- since
    refresh preserves phase EXACTLY by construction, any measured drift under
    refresh=True is not an artifact of the renormalization; it can only come
    from the alpha-clamp axis (or its absence)."""
    key_lo, val_lo, fill_lo, _ = _gap_vocab(P_max, V_max, F)
    gen = torch.Generator().manual_seed(seed)
    keys = torch.stack([torch.randperm(P_max, generator=gen)[:P] for _ in range(NB)]) + key_lo
    vals = torch.randint(0, V_max, (NB, P), generator=gen) + val_lo
    fillers = torch.randint(0, F, (NB, G), generator=gen) + fill_lo
    kv = torch.stack([keys, vals], dim=2).reshape(NB, 2 * P)

    model.eval()
    scan0 = model.layers[0].scan
    logits_kv, states = chunked_forward(model, kv, chunk)   # KV phase: always unclamped, unrefreshed
    s0_re = states[0]["S_re"].clone()
    s0_im = states[0]["S_im"].clone()
    mag0 = torch.sqrt(s0_re ** 2 + s0_im ** 2 + 1e-12)
    phase0 = torch.atan2(s0_im, s0_re)

    st = states
    pos = 0
    s_re_last, s_im_last = s0_re, s0_im
    while pos < G:
        hi = min(G, pos + chunk)
        xc = fillers[:, pos:hi]
        h = model.embed(xc)
        with EvalClampContext(scan0, clamp_factor):
            y0, st_new, internals = model.layers[0].scan(h, st[0], return_internals=True)
        s_re_last = internals["S_re"][:, -1].clone()
        s_im_last = internals["S_im"][:, -1].clone()
        h1 = model.layers[0].ln1(h + y0)
        h1 = model.layers[0].ln2(h1 + model.layers[0].ffn(h1))
        new_states = [st_new]
        hcur = h1
        for li in range(1, len(model.layers)):
            y, sto = model.layers[li].scan(hcur, st[li])
            hcur = model.layers[li].ln1(hcur + y)
            hcur = model.layers[li].ln2(hcur + model.layers[li].ffn(hcur))
            new_states.append(sto)
        st = [{k: v.detach() for k, v in s.items()} for s in new_states]
        if refresh:
            st = refresh_magnitude(st)
            s_re_last, s_im_last = st[0]["S_re"], st[0]["S_im"]
        pos = hi

    mag_t = torch.sqrt(s_re_last ** 2 + s_im_last ** 2 + 1e-12)
    phase_t = torch.atan2(s_im_last, s_re_last)
    dphi = wrapped_delta(phase_t, phase0).abs()

    alive = mag_t >= SNR_FLOOR
    n_alive = int(alive.sum())
    n_total = int(alive.numel())
    if n_alive > 0:
        w_snr = mag_t[alive] / (mag_t[alive].sum() + 1e-12)
        dphi_snr_weighted = float((dphi[alive] * w_snr).sum())
    else:
        dphi_snr_weighted = float("nan")
    mag_ratio_mean = float((mag_t / (mag0 + 1e-12)).mean())

    return {"G": G, "dphi_snr_gated_weighted": round(dphi_snr_weighted, 6) if n_alive else None,
           "mag_ratio_mean": round(mag_ratio_mean, 6),
           "snr_alive_frac": round(n_alive / n_total, 4)}


def run_eval_clamp_refresh(args):
    """P30 orchestration: train ONE lambda=0 M3-recipe model per seed
    (identical training path to P25's control / P29's build), then run the
    2x2 {clamp, no-clamp} x {refresh, no-refresh} eval sweep + drift
    diagnosis + zeroed-at-gap null, all four arms against the SAME trained
    model per seed. No training-time change anywhere -- both axes are
    eval-only hooks."""
    P_max, V_max, F = args.p_max, args.v_max, args.f_fillers
    key_lo, val_lo, fill_lo, vocab_size = _gap_vocab(P_max, V_max, F)
    mask_idx = vocab_size
    P = args.pairs
    chance = 1.0 / V_max

    Gs_eval = [int(g) for g in args.gaps.split(",")]
    Gs_drift = [int(g) for g in args.drift_gaps.split(",") if g]
    seeds = [int(s) for s in args.seeds.split(",")]
    # arm name -> (clamp_factor, refresh)
    arms = [("unclamped_norefresh", (None, False)),
            ("clamp_norefresh", (0.0, False)),
            ("unclamped_refresh", (None, True)),
            ("clamp_refresh", (0.0, True))]

    print("=" * 78)
    print("HOLO-ALPHA-SHUT — P30/MS12c: the 2x2 that disentangles phase-pollution from magnitude-feed")
    print(f"P={P}  P_max={P_max} V_max={V_max} F={F} vocab={vocab_size}  chance={chance:.4f}")
    print(f"eval gaps(G)={Gs_eval}  drift gaps={Gs_drift}  seeds={seeds}  arms={[a for a,_ in arms]}")
    print("=" * 78)

    print("\n── equivalence gates (reused, un-normalized + magnorm) ──")
    eq_ref = check_equivalence(vocab_size, mask_idx, seed=0, T=48, chunk=16, use_phase=True)
    eq_magnorm = check_equivalence_magnorm(vocab_size, mask_idx, seed=0, T=48, chunk=args.chunk,
                                           d_model=args.d_model, n_layers=args.n_layers,
                                           n_heads=args.n_heads, d_head=args.d_head)
    print(f"   reference_streaming max|Δ| = {eq_ref:.3e}")
    print(f"   magnorm             max|Δ| = {eq_magnorm:.3e}")
    eq_ok = eq_ref < 1e-5 and eq_magnorm < 1e-5
    print(f"   {'PASS' if eq_ok else 'FAIL — do not trust anything below'}")
    equivalence = {"reference_streaming": eq_ref, "magnorm": eq_magnorm, "passed": eq_ok}
    if not eq_ok:
        out = {"config": vars(args), "equivalence": equivalence,
              "verdict": "VOID — equivalence check failed"}
        os.makedirs(RESULTS, exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"\n→ {args.out}")
        return

    t0 = time.time()
    results = {"config": vars(args), "equivalence": equivalence, "chance": chance,
              "recipe": "M3 V2_kickstart_magnorm, lambda=0, trained ONCE per seed (identical to "
                        "P25's control / P29's build). Eval-only 2x2: alpha-clamp on filler "
                        "positions (P29 mechanism) crossed with in-state magnitude refresh at "
                        "filler chunk boundaries (S <- S/|S| per channel, eps-guarded, phase "
                        "exactly preserved by construction). No training-time change.",
              "refresh_eps": REFRESH_EPS,
              "kickstart_logit": KICKSTART_LOGIT, "kickstart_head": KICKSTART_HEAD,
              "curriculum": {}, "sweep": {}, "knees": {}, "drift": {}, "zeroed_null": {}}

    for seed in seeds:
        print(f"\n{'='*78}\nseed={seed}  (training M3 recipe, lambda=0, ONCE)\n{'='*78}")
        torch.manual_seed(seed)
        model = _build_lm_alphashut(vocab_size, mask_idx, args.d_model, args.n_layers,
                                    args.n_heads, args.d_head)
        curr = train_fullseq_curriculum_alphashut(
            model, P, args.g_train_max, args.iters, args.lr, seed, args.batch, 0.0,
            P_max, V_max, F, log_every=args.log_every, g_start=args.g_start,
            patience=args.patience, bar=args.curriculum_bar)
        results["curriculum"][f"seed{seed}"] = curr
        print(f"  curriculum done: final_train_gap={curr['final_train_gap']} "
              f"final_train_acc={curr['final_train_acc']:.3f}")

        for arm_name, (factor, refresh) in arms:
            key = f"seed{seed}_{arm_name}"
            print(f"  ── arm={arm_name} (clamp_factor={factor}, refresh={refresh}) ──")
            by_G = {}
            for G in Gs_eval:
                if G >= 4096:
                    eb = max(10, args.eval_batch // 8)
                elif G >= 2048:
                    eb = max(10, args.eval_batch // 4)
                elif G >= 1024:
                    eb = max(10, args.eval_batch // 2)
                else:
                    eb = args.eval_batch
                acc = eval_gap_recall_chunked_clamprefresh(model, P, G, eb, seed + 1000 + G,
                                                            P_max, V_max, F, args.chunk,
                                                            factor, refresh)
                acc0 = eval_gap_recall_chunked_zeroed_clamprefresh(model, P, G, eb, seed + 1000 + G,
                                                                   P_max, V_max, F, args.chunk,
                                                                   factor, refresh)
                by_G[G] = acc
                sk = f"{arm_name}|seed{seed}|G{G}"
                results["sweep"][sk] = {"seed": seed, "arm": arm_name, "clamp_factor": factor,
                                        "refresh": refresh, "P": P, "G": G,
                                        "accuracy": round(acc, 4), "chance": round(chance, 4),
                                        "beats_chance_3x": bool(acc > 3 * chance)}
                results["zeroed_null"][sk] = {"seed": seed, "arm": arm_name, "G": G,
                                              "accuracy": round(acc0, 4), "chance": round(chance, 4),
                                              "within_5pp_of_chance": bool(abs(acc0 - chance) <= 0.05)}
                print(f"      G={G:>5}: acc={acc:.4f} (chance {chance:.4f})  "
                      f"zeroed-null={acc0:.4f}")

            ref_G = Gs_eval[0]
            knee = estimate_knee(by_G, Gs_eval, ref_G=ref_G)
            results["knees"][key] = {"knee_G": knee, "ref_G": ref_G,
                                     "acc_at_ref": round(by_G.get(ref_G, float("nan")), 4)}
            print(f"      knee estimate (last G with acc >= 0.5*acc@{ref_G}): {knee}")

            drift_by_G = {}
            for G in Gs_drift:
                nb_drift = max(20, min(60, args.eval_batch // 2))
                d = measure_filler_drift_clamprefresh(model, P, G, nb_drift, seed + 3000 + G,
                                                      P_max, V_max, F, args.chunk, factor, refresh)
                drift_by_G[str(G)] = d
                print(f"      drift@G={G:>5}: Δφ_snr={d['dphi_snr_gated_weighted']}  "
                      f"mag_ratio={d['mag_ratio_mean']}  alive={d['snr_alive_frac']}")
            results["drift"][key] = drift_by_G

    # ── aggregate per-arm (mean over seeds): knee, drift@512, acc@ref_G, zeroed-null pass ──
    ref_G0 = Gs_eval[0]
    per_arm = {}
    for arm_name, (factor, refresh) in arms:
        knee_vals = [v["knee_G"] for k, v in results["knees"].items()
                    if k.endswith(f"_{arm_name}") and v["knee_G"] is not None]
        mean_knee = sum(knee_vals) / len(knee_vals) if knee_vals else None

        drift512_vals = [v["512"]["dphi_snr_gated_weighted"] for k, v in results["drift"].items()
                         if k.endswith(f"_{arm_name}") and "512" in v
                         and v["512"]["dphi_snr_gated_weighted"] is not None]
        mean_drift512 = sum(drift512_vals) / len(drift512_vals) if drift512_vals else None

        acc_ref_vals = [v["accuracy"] for k, v in results["sweep"].items()
                        if v["arm"] == arm_name and v["G"] == ref_G0]
        mean_acc_ref = sum(acc_ref_vals) / len(acc_ref_vals) if acc_ref_vals else None

        null_ok_vals = [v["within_5pp_of_chance"] for k, v in results["zeroed_null"].items()
                        if v["arm"] == arm_name]
        all_null_ok = all(null_ok_vals) if null_ok_vals else None

        per_arm[arm_name] = {"clamp_factor": factor, "refresh": refresh, "mean_knee_G": mean_knee,
                             "mean_drift_at_512": mean_drift512,
                             f"mean_acc_at_G{ref_G0}": mean_acc_ref,
                             "zeroed_null_all_within_5pp_of_chance": all_null_ok}
    results["per_arm_summary"] = per_arm

    # ── P30 checks ──
    cr = per_arm["clamp_refresh"]
    rr = per_arm["unclamped_refresh"]
    uu = per_arm["unclamped_norefresh"]
    cc = per_arm["clamp_norefresh"]

    p30a_pass = (cr["mean_knee_G"] is not None and cr["mean_knee_G"] >= 2048)

    def _k(arm):
        v = per_arm[arm]["mean_knee_G"]
        return v if v is not None else -1
    ordering = ["clamp_refresh", "unclamped_refresh", "unclamped_norefresh", "clamp_norefresh"]
    knee_by_arm = {a: _k(a) for a in ordering}
    p30b_pass = (knee_by_arm["clamp_refresh"] >= knee_by_arm["unclamped_refresh"] >=
                knee_by_arm["unclamped_norefresh"] >= knee_by_arm["clamp_norefresh"])

    p30c_pass = all(per_arm[a]["zeroed_null_all_within_5pp_of_chance"] for a, _ in arms)

    acc_u = uu.get(f"mean_acc_at_G{ref_G0}")
    p30d_pass = True
    p30d_detail = []
    for arm_name, _ in arms:
        a = per_arm[arm_name].get(f"mean_acc_at_G{ref_G0}")
        if acc_u is not None and a is not None:
            ok = abs(acc_u - a) <= 0.05
            p30d_pass = p30d_pass and ok
            p30d_detail.append(f"{arm_name}={a}")

    verdict_bits = []
    verdict_bits.append(f"P30a (clamp+refresh knee >= 2048): {'CONFIRMED' if p30a_pass else 'NOT MET'} "
                        f"(measured {cr['mean_knee_G']})")
    verdict_bits.append(f"P30b (ordering clamp+refresh >= refresh-only >= unclamped >= clamp-only): "
                        f"{'CONFIRMED' if p30b_pass else 'NOT MET'} (knees: {knee_by_arm})")
    verdict_bits.append(f"P30c (zeroed-at-gap null within 5pp of chance, every arm): "
                        f"{'CONFIRMED' if p30c_pass else 'NOT MET'}")
    verdict_bits.append(f"P30d (in-range recall @G{ref_G0} within 5pp of unclamped/no-refresh baseline, "
                        f"every arm): {'CONFIRMED' if p30d_pass else 'NOT MET'} ({', '.join(p30d_detail)})")

    results["verdict"] = " | ".join(verdict_bits)
    results["elapsed_s"] = round(time.time() - t0, 1)

    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "=" * 78)
    print("VERDICT")
    for b in verdict_bits:
        print("  " + b)
    print(f"\n→ {args.out}  ({results['elapsed_s']}s)")


# ═══════════════════════════════════════════════════════════════════════════
# 3d. P36 (MS12d) — null hardening. F6 flagged that P30c's zeroed-at-gap null
#     (the decisive "no information is being invented" control) was measured
#     at whatever eval_batch the sweep happened to use, which at small G can
#     be as few as ~25-100 episodes -- not enough to rule out sampling noise
#     being the reason a cell reads a point or two off chance. This mode
#     trains the SAME lambda=0 M3-recipe model per seed (identical path to
#     P29/P30's build -- reused, not re-derived) and RE-MEASURES the P30c
#     zeroed-at-gap null under the same four {clamp,no-clamp} x {refresh,
#     no-refresh} arms at G in {1024, 2048} ONLY, with eval_batch forced to
#     >= 100 (no G-dependent shrinkage the P30 sweep applies at 1024+ -- this
#     mode exists specifically to remove that shrinkage). Gate: every cell
#     within 3pp of chance (tighter than P30c's 5pp -- the hardened bar F6
#     asked for).
# ═══════════════════════════════════════════════════════════════════════════
def run_null_hardening(args):
    """P36 orchestration: train ONE lambda=0 M3-recipe model per seed
    (train_fullseq_curriculum_alphashut with lam=0.0, byte-identical build
    path to P29/P30), then re-measure ONLY the zeroed-at-gap null
    (eval_gap_recall_chunked_zeroed_clamprefresh) across the four P30 arms
    at G in {1024, 2048}, with eval_batch >= 100 forced regardless of what
    --eval-batch requests (max() floor below) -- large-batch is the entire
    point of this hardening pass."""
    P_max, V_max, F = args.p_max, args.v_max, args.f_fillers
    key_lo, val_lo, fill_lo, vocab_size = _gap_vocab(P_max, V_max, F)
    mask_idx = vocab_size
    P = args.pairs
    chance = 1.0 / V_max

    Gs_null = [int(g) for g in args.null_gaps.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    eval_batch_hardened = max(100, args.eval_batch)
    arms = [("unclamped_norefresh", (None, False)),
            ("clamp_norefresh", (0.0, False)),
            ("unclamped_refresh", (None, True)),
            ("clamp_refresh", (0.0, True))]

    print("=" * 78)
    print("HOLO-ALPHA-SHUT — P36/MS12d: null hardening (P30c zeroed-at-gap null, eval_batch>=100)")
    print(f"P={P}  P_max={P_max} V_max={V_max} F={F} vocab={vocab_size}  chance={chance:.4f}")
    print(f"null gaps(G)={Gs_null}  seeds={seeds}  eval_batch={eval_batch_hardened}  arms={[a for a,_ in arms]}")
    print("=" * 78)

    print("\n── equivalence gates (reused, un-normalized + magnorm) ──")
    eq_ref = check_equivalence(vocab_size, mask_idx, seed=0, T=48, chunk=16, use_phase=True)
    eq_magnorm = check_equivalence_magnorm(vocab_size, mask_idx, seed=0, T=48, chunk=args.chunk,
                                           d_model=args.d_model, n_layers=args.n_layers,
                                           n_heads=args.n_heads, d_head=args.d_head)
    print(f"   reference_streaming max|Δ| = {eq_ref:.3e}")
    print(f"   magnorm             max|Δ| = {eq_magnorm:.3e}")
    eq_ok = eq_ref < 1e-5 and eq_magnorm < 1e-5
    print(f"   {'PASS' if eq_ok else 'FAIL — do not trust anything below'}")
    equivalence = {"reference_streaming": eq_ref, "magnorm": eq_magnorm, "passed": eq_ok}
    if not eq_ok:
        out = {"config": vars(args), "equivalence": equivalence,
              "verdict": "VOID — equivalence check failed"}
        os.makedirs(RESULTS, exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"\n→ {args.out}")
        return

    t0 = time.time()
    results = {"config": vars(args), "equivalence": equivalence, "chance": chance,
              "eval_batch_hardened": eval_batch_hardened,
              "recipe": "M3 V2_kickstart_magnorm, lambda=0, trained ONCE per seed (identical build "
                        "path to P29/P30). Re-measures ONLY the P30c zeroed-at-gap null across the "
                        "same four clamp/refresh arms, at G in {1024,2048}, with eval_batch forced "
                        ">= 100 (no shrinkage) -- the F6-flagged sampling-noise gap on the decisive "
                        "no-information-invented control.",
              "refresh_eps": REFRESH_EPS, "gate_pp": 3.0,
              "curriculum": {}, "null": {}}

    for seed in seeds:
        print(f"\n{'='*78}\nseed={seed}  (training M3 recipe, lambda=0, ONCE)\n{'='*78}")
        torch.manual_seed(seed)
        model = _build_lm_alphashut(vocab_size, mask_idx, args.d_model, args.n_layers,
                                    args.n_heads, args.d_head)
        curr = train_fullseq_curriculum_alphashut(
            model, P, args.g_train_max, args.iters, args.lr, seed, args.batch, 0.0,
            P_max, V_max, F, log_every=args.log_every, g_start=args.g_start,
            patience=args.patience, bar=args.curriculum_bar)
        results["curriculum"][f"seed{seed}"] = curr
        print(f"  curriculum done: final_train_gap={curr['final_train_gap']} "
              f"final_train_acc={curr['final_train_acc']:.3f}")

        for arm_name, (factor, refresh) in arms:
            print(f"  ── arm={arm_name} (clamp_factor={factor}, refresh={refresh}) ──")
            for G in Gs_null:
                acc0 = eval_gap_recall_chunked_zeroed_clamprefresh(
                    model, P, G, eval_batch_hardened, seed + 1000 + G,
                    P_max, V_max, F, args.chunk, factor, refresh)
                dev_pp = round(abs(acc0 - chance) * 100, 3)
                sk = f"{arm_name}|seed{seed}|G{G}"
                results["null"][sk] = {"seed": seed, "arm": arm_name, "clamp_factor": factor,
                                       "refresh": refresh, "G": G, "eval_batch": eval_batch_hardened,
                                       "accuracy": round(acc0, 4), "chance": round(chance, 4),
                                       "deviation_pp": dev_pp, "within_3pp_of_chance": bool(dev_pp <= 3.0)}
                print(f"      G={G:>5}: zeroed-null acc={acc0:.4f} (chance {chance:.4f})  "
                      f"deviation={dev_pp}pp  {'PASS' if dev_pp <= 3.0 else 'FAIL'}")

    all_cells = list(results["null"].values())
    all_within_3pp = all(c["within_3pp_of_chance"] for c in all_cells) if all_cells else None
    worst = max(all_cells, key=lambda c: c["deviation_pp"]) if all_cells else None

    per_G = {}
    for G in Gs_null:
        cells_G = [c for c in all_cells if c["G"] == G]
        per_G[str(G)] = {"n_cells": len(cells_G),
                         "all_within_3pp": all(c["within_3pp_of_chance"] for c in cells_G) if cells_G else None,
                         "max_deviation_pp": max((c["deviation_pp"] for c in cells_G), default=None)}
    results["per_G_summary"] = per_G

    verdict_bits = []
    verdict_bits.append(f"P36 (zeroed-at-gap null within 3pp of chance, every cell, eval_batch>={eval_batch_hardened}, "
                        f"G in {Gs_null}, {len(seeds)} seed(s) x 4 arms = {len(all_cells)} cells): "
                        f"{'CONFIRMED' if all_within_3pp else 'NOT MET'}")
    if worst is not None:
        verdict_bits.append(f"worst cell: {worst['arm']} seed{worst['seed']} G={worst['G']} "
                            f"acc={worst['accuracy']} (chance {worst['chance']}) "
                            f"deviation={worst['deviation_pp']}pp")

    results["all_checks_pass"] = bool(all_within_3pp) if all_within_3pp is not None else False
    results["verdict"] = " | ".join(verdict_bits)
    results["elapsed_s"] = round(time.time() - t0, 1)

    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "=" * 78)
    print("VERDICT")
    for b in verdict_bits:
        print("  " + b)
    print(f"\n→ {args.out}  ({results['elapsed_s']}s)")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Orchestration: build lambda x seed grid, train, eval sweep, drift
#    diagnosis, dose-monotonicity + P25 checks, write JSON.
# ═══════════════════════════════════════════════════════════════════════════
def run(args):
    P_max, V_max, F = args.p_max, args.v_max, args.f_fillers
    key_lo, val_lo, fill_lo, vocab_size = _gap_vocab(P_max, V_max, F)
    mask_idx = vocab_size
    P = args.pairs
    chance = 1.0 / V_max

    Gs_eval = [int(g) for g in args.gaps.split(",")]
    Gs_drift = [int(g) for g in args.drift_gaps.split(",") if g]
    seeds = [int(s) for s in args.seeds.split(",")]
    lambdas = [float(v) for v in args.lam_list.split(",")]

    print("=" * 78)
    print("HOLO-ALPHA-SHUT (MS12) — does regularizing filler alpha causally reduce drift and move the knee?")
    print(f"P={P}  P_max={P_max} V_max={V_max} F={F} vocab={vocab_size}  chance={chance:.4f}")
    print(f"eval gaps(G)={Gs_eval}  drift gaps={Gs_drift}  seeds={seeds}  lambdas={lambdas}")
    print("=" * 78)

    print("\n── equivalence gates (reused, un-normalized + magnorm) ──")
    eq_ref = check_equivalence(vocab_size, mask_idx, seed=0, T=48, chunk=16, use_phase=True)
    eq_magnorm = check_equivalence_magnorm(vocab_size, mask_idx, seed=0, T=48, chunk=args.chunk,
                                           d_model=args.d_model, n_layers=args.n_layers,
                                           n_heads=args.n_heads, d_head=args.d_head)
    print(f"   reference_streaming max|Δ| = {eq_ref:.3e}")
    print(f"   magnorm             max|Δ| = {eq_magnorm:.3e}")
    eq_ok = eq_ref < 1e-5 and eq_magnorm < 1e-5
    print(f"   {'PASS' if eq_ok else 'FAIL — do not trust anything below'}")
    equivalence = {"reference_streaming": eq_ref, "magnorm": eq_magnorm, "passed": eq_ok}
    if not eq_ok:
        out = {"config": vars(args), "equivalence": equivalence,
              "verdict": "VOID — equivalence check failed"}
        os.makedirs(RESULTS, exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"\n→ {args.out}")
        return

    t0 = time.time()
    results = {"config": vars(args), "equivalence": equivalence, "chance": chance,
              "recipe": "M3 V2_kickstart_magnorm (GammaKickstartScanLayer + MagNormReadMixin), "
                        "trained via unchunked full-sequence gap curriculum "
                        "(g_train_max=128, bar=0.8, patience=25), eval chunked+carried (chunk=16). "
                        "lambda=0 arm is this recipe exactly (byte-identical loss), lambda>0 arms add "
                        "lambda*mean(alpha_t over filler positions) to the training loss.",
              "kickstart_logit": KICKSTART_LOGIT, "kickstart_head": KICKSTART_HEAD,
              "curriculum": {}, "sweep": {}, "knees": {}, "drift": {}}

    for seed in seeds:
        for lam in lambdas:
            lam_key = f"lam{lam:g}"
            key = f"seed{seed}_{lam_key}"
            print(f"\n{'='*78}\nseed={seed}  lambda={lam:g}\n{'='*78}")
            torch.manual_seed(seed)
            model = _build_lm_alphashut(vocab_size, mask_idx, args.d_model, args.n_layers,
                                        args.n_heads, args.d_head)

            curr = train_fullseq_curriculum_alphashut(
                model, P, args.g_train_max, args.iters, args.lr, seed, args.batch, lam,
                P_max, V_max, F, log_every=args.log_every, g_start=args.g_start,
                patience=args.patience, bar=args.curriculum_bar)
            results["curriculum"][key] = curr
            print(f"  curriculum done: final_train_gap={curr['final_train_gap']} "
                  f"final_train_acc={curr['final_train_acc']:.3f}  "
                  f"mean_alpha_filler(last window)={curr['mean_alpha_filler_last_window']}")

            by_G = {}
            for G in Gs_eval:
                if G >= 2048:
                    eb = max(10, args.eval_batch // 4)
                elif G >= 1024:
                    eb = max(10, args.eval_batch // 2)
                else:
                    eb = args.eval_batch
                acc = eval_gap_recall_chunked(model, P, G, eb, seed + 1000 + G, P_max, V_max, F,
                                              args.chunk, zero_at_gap=False)
                acc0 = eval_gap_recall_chunked(model, P, G, eb, seed + 1000 + G, P_max, V_max, F,
                                               args.chunk, zero_at_gap=True)
                by_G[G] = acc
                sk = f"{lam_key}|seed{seed}|G{G}"
                results["sweep"][sk] = {"seed": seed, "lambda": lam, "P": P, "G": G,
                                        "accuracy": round(acc, 4), "chance": round(chance, 4),
                                        "beats_chance_3x": bool(acc > 3 * chance),
                                        "zeroed_null_accuracy": round(acc0, 4)}
                print(f"    G={G:>5}: acc={acc:.4f} (chance {chance:.4f})  zeroed-null={acc0:.4f}")

            ref_G = Gs_eval[0]
            knee = estimate_knee(by_G, Gs_eval, ref_G=ref_G)
            results["knees"][key] = {"knee_G": knee, "ref_G": ref_G,
                                     "acc_at_ref": round(by_G.get(ref_G, float("nan")), 4)}
            print(f"    knee estimate (last G with acc >= 0.5*acc@{ref_G}): {knee}")

            drift_by_G = {}
            for G in Gs_drift:
                nb_drift = max(20, min(60, args.eval_batch // 2))
                d = measure_filler_drift(model, P, G, nb_drift, seed + 3000 + G, P_max, V_max, F, args.chunk)
                drift_by_G[str(G)] = d
                print(f"    drift@G={G:>5}: Δφ_snr={d['dphi_snr_gated_weighted']}  "
                      f"mag_ratio={d['mag_ratio_mean']}  alive={d['snr_alive_frac']}")
            results["drift"][key] = drift_by_G

    # ── aggregate per-lambda (mean over seeds): knee, drift@512, acc@ref_G ──
    def _agg(field_fn, lam):
        vals = [field_fn(k, v) for k, v in results["knees"].items()
                if k.endswith(f"_lam{lam:g}") and field_fn(k, v) is not None]
        return sum(vals) / len(vals) if vals else None

    ref_G0 = Gs_eval[0]
    per_lambda = {}
    for lam in lambdas:
        lam_key = f"lam{lam:g}"
        knee_vals = [v["knee_G"] for k, v in results["knees"].items()
                    if k.endswith(f"_{lam_key}") and v["knee_G"] is not None]
        mean_knee = sum(knee_vals) / len(knee_vals) if knee_vals else None

        drift512_vals = [v["512"]["dphi_snr_gated_weighted"] for k, v in results["drift"].items()
                         if k.endswith(f"_{lam_key}") and "512" in v
                         and v["512"]["dphi_snr_gated_weighted"] is not None]
        mean_drift512 = sum(drift512_vals) / len(drift512_vals) if drift512_vals else None

        acc_ref_vals = [v["accuracy"] for k, v in results["sweep"].items()
                        if v["lambda"] == lam and v["G"] == ref_G0]
        mean_acc_ref = sum(acc_ref_vals) / len(acc_ref_vals) if acc_ref_vals else None

        alpha_vals = [v["mean_alpha_filler_last_window"] for k, v in results["curriculum"].items()
                     if k.endswith(f"_{lam_key}") and v["mean_alpha_filler_last_window"] is not None]
        mean_alpha_filler = sum(alpha_vals) / len(alpha_vals) if alpha_vals else None

        per_lambda[lam] = {"mean_knee_G": mean_knee, "mean_drift_at_512": mean_drift512,
                           f"mean_acc_at_G{ref_G0}": mean_acc_ref,
                           "mean_alpha_filler_trainwindow": mean_alpha_filler}
    results["per_lambda_summary"] = {f"lam{lam:g}": v for lam, v in per_lambda.items()}

    # ── P25 checks ──
    lam0 = 0.0
    lam_pos = [l for l in lambdas if l > 0.0]
    best_lam_pos = None
    if lam_pos:
        # "best lambda>0 arm" for the drift/knee checks: the one with the
        # LOWEST mean_drift_at_512 among lambda>0 arms that have a measurement.
        scored = [(l, per_lambda[l]["mean_drift_at_512"]) for l in lam_pos
                 if per_lambda[l]["mean_drift_at_512"] is not None]
        if scored:
            best_lam_pos = min(scored, key=lambda t: t[1])[0]

    p25a_pass = (best_lam_pos is not None and per_lambda[best_lam_pos]["mean_drift_at_512"] is not None
                and per_lambda[best_lam_pos]["mean_drift_at_512"] < 0.5)
    p25b_pass = (best_lam_pos is not None and per_lambda[best_lam_pos]["mean_knee_G"] is not None
                and per_lambda[best_lam_pos]["mean_knee_G"] >= 1024)

    # dose-monotonicity: Spearman-style DIRECTION check across the (up to 4)
    # lambda points actually run -- drift@512 should be non-increasing in
    # lambda, knee should be non-decreasing in lambda. With only 4 points a
    # strict rank check is used (ties broken as non-violations).
    lam_sorted = sorted(lambdas)
    drift_seq = [per_lambda[l]["mean_drift_at_512"] for l in lam_sorted]
    knee_seq = [per_lambda[l]["mean_knee_G"] for l in lam_sorted]

    def _monotone_nonincreasing(seq):
        pairs = [(a, b) for a, b in zip(seq, seq[1:]) if a is not None and b is not None]
        if not pairs:
            return None
        return all(b <= a + 1e-9 for a, b in pairs)

    def _monotone_nondecreasing(seq):
        pairs = [(a, b) for a, b in zip(seq, seq[1:]) if a is not None and b is not None]
        if not pairs:
            return None
        return all(b >= a - 1e-9 for a, b in pairs)

    drift_monotone = _monotone_nonincreasing(drift_seq)
    knee_monotone = _monotone_nondecreasing(knee_seq)
    p25c_pass = bool(drift_monotone) and bool(knee_monotone)

    # trade-off warning: does recall@ref_G drop >10pp from lambda=0 to any lambda>0 arm?
    acc0 = per_lambda.get(lam0, {}).get(f"mean_acc_at_G{ref_G0}")
    tradeoff_flags = []
    if acc0 is not None:
        for l in lam_pos:
            accl = per_lambda[l][f"mean_acc_at_G{ref_G0}"]
            if accl is not None and acc0 - accl > 0.10:
                tradeoff_flags.append(f"lambda={l:g}: acc@G{ref_G0} drops {round((acc0-accl)*100,1)}pp "
                                      f"below lambda=0 ({accl:.3f} vs {acc0:.3f})")

    verdict_bits = []
    verdict_bits.append(f"P25a (best lambda>0 drift@512 < 0.5 rad): "
                        f"{'CONFIRMED' if p25a_pass else 'NOT MET'} "
                        f"(best_lambda={best_lam_pos}, drift={per_lambda.get(best_lam_pos, {}).get('mean_drift_at_512') if best_lam_pos is not None else None}, "
                        f"lambda=0 reference drift={per_lambda.get(lam0, {}).get('mean_drift_at_512')})")
    verdict_bits.append(f"P25b (best lambda>0 knee >= 1024): "
                        f"{'CONFIRMED' if p25b_pass else 'NOT MET'} "
                        f"(best_lambda={best_lam_pos}, knee={per_lambda.get(best_lam_pos, {}).get('mean_knee_G') if best_lam_pos is not None else None}, "
                        f"lambda=0 reference knee={per_lambda.get(lam0, {}).get('mean_knee_G')})")
    verdict_bits.append(f"P25c (dose-monotonicity: drift@512 non-increasing AND knee non-decreasing in lambda): "
                        f"{'CONFIRMED' if p25c_pass else 'NOT MET'} "
                        f"(drift sequence over lambda={lam_sorted}: {drift_seq}, knee sequence: {knee_seq})")
    if tradeoff_flags:
        verdict_bits.append("TRADE-OFF WARNING: " + "; ".join(tradeoff_flags) +
                            " — alpha-shut regularization measurably strangles near-field recall at this lambda.")
    else:
        verdict_bits.append("No trade-off flag: recall@ref_G stays within 10pp of lambda=0 across all lambda>0 arms.")

    results["verdict"] = " | ".join(verdict_bits)
    results["elapsed_s"] = round(time.time() - t0, 1)

    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "=" * 78)
    print("VERDICT")
    for b in verdict_bits:
        print("  " + b)
    print(f"\n→ {args.out}  ({results['elapsed_s']}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="fast sanity: seed 0 only, lambda in {0,1e-2}, G in {128,512}, reduced iters")
    ap.add_argument("--full", action="store_true",
                    help="the full sweep: seeds {0,1}, lambda in {0,3e-3,1e-2,3e-2}, G in {128..2048}")
    ap.add_argument("--eval-alpha-clamp", action="store_true",
                    help="P29/MS12b mode: train ONE lambda=0 (M3 recipe) model per seed, then run "
                         "the eval sweep + drift diagnosis three times (unclamped, clamp_full=0.0, "
                         "clamp_half=0.5) against filler positions ONLY, at eval time. Combine with "
                         "--smoke for a fast 1-seed/G-in-{128,512} sanity pass.")
    ap.add_argument("--eval-clamp-refresh", action="store_true",
                    help="P30/MS12c mode: train ONE lambda=0 (M3 recipe) model per seed, then run "
                         "the 2x2 {clamp,no-clamp} x {magnitude-refresh,no-refresh} eval sweep + "
                         "drift diagnosis + zeroed-at-gap null, four arms, at eval time. --full "
                         "extends gaps to include 4096. Combine with --smoke for a fast sanity pass.")
    ap.add_argument("--null-hardening", action="store_true",
                    help="P36/MS12d mode: train ONE lambda=0 (M3 recipe) model per seed, then "
                         "RE-MEASURE ONLY the P30c zeroed-at-gap null (four clamp/refresh arms) at "
                         "G in --null-gaps (default 1024,2048) with eval_batch forced >= 100 (no "
                         "G-dependent shrinkage). Gate: every cell within 3pp of chance. Combine "
                         "with --smoke for a fast (smaller-batch, single-seed) sanity pass.")
    ap.add_argument("--null-gaps", default="1024,2048",
                    help="comma list of gaps for --null-hardening's zeroed-at-gap re-measurement")
    ap.add_argument("--p-max", type=int, default=16)
    ap.add_argument("--v-max", type=int, default=16)
    ap.add_argument("--f-fillers", type=int, default=16)
    ap.add_argument("--pairs", type=int, default=2, help="n_pairs P for the gap-MQAR task")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-head", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=16, help="EVAL streaming chunk length (deployment path)")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-batch", type=int, default=100)
    ap.add_argument("--g-start", type=int, default=2)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--curriculum-bar", type=float, default=0.8)
    ap.add_argument("--g-train-max", type=int, default=128,
                    help="max TRAINING gap for the full-sequence curriculum (M3 recipe: 128)")
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--gaps", default="128,256,512,1024,2048", help="comma list of EVAL gaps")
    ap.add_argument("--drift-gaps", default="128,512,2048", help="comma list of gaps for the phi-drift diagnosis")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--lam-list", default="0,3e-3,1e-2,3e-2", help="comma list of lambda values (alpha-shut weight)")
    ap.add_argument("--out", default=None, help="defaults to results/holo_alpha_shut.json ("
                    "or holo_alpha_clamp.json under --eval-alpha-clamp, holo_clamp_refresh.json "
                    "under --eval-clamp-refresh, holo_null_hardening.json under --null-hardening), "
                    "_smoke suffixed under --smoke")
    args = ap.parse_args()

    if args.null_hardening:
        default_out = os.path.join(RESULTS, "holo_null_hardening.json")
    elif args.eval_clamp_refresh:
        default_out = os.path.join(RESULTS, "holo_clamp_refresh.json")
    elif args.eval_alpha_clamp:
        default_out = os.path.join(RESULTS, "holo_alpha_clamp.json")
    else:
        default_out = os.path.join(RESULTS, "holo_alpha_shut.json")
    if args.out is None:
        args.out = default_out

    if args.smoke:
        args.seeds = "0"
        args.lam_list = "0,1e-2"
        args.iters = min(args.iters, 600) if args.iters == 3000 else args.iters
        args.gaps = "128,512"
        args.drift_gaps = "128,512"
        args.g_train_max = min(args.g_train_max, 128)
        args.eval_batch = min(args.eval_batch, 40)
        if args.null_hardening:
            # smoke here means "fast build/gate sanity", NOT a relaxed null gate --
            # P36 exists specifically to force eval_batch>=100, so smoke only
            # shrinks --null-gaps to a single rung and shortens training iters;
            # eval_batch stays hardened at >=100 inside run_null_hardening.
            args.null_gaps = "1024"
        if args.out == default_out:
            root, ext = os.path.splitext(default_out)
            args.out = root + "_smoke" + ext
    elif args.full:
        args.seeds = "0,1"
        args.lam_list = "0,3e-3,1e-2,3e-2"
        args.iters = max(args.iters, 3000)
        if args.eval_clamp_refresh:
            args.gaps = "128,256,512,1024,2048,4096"
        else:
            args.gaps = "128,256,512,1024,2048"
        args.drift_gaps = "128,512,2048"

    if args.null_hardening:
        run_null_hardening(args)
    elif args.eval_clamp_refresh:
        run_eval_clamp_refresh(args)
    elif args.eval_alpha_clamp:
        run_eval_alpha_clamp(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
