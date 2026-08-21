#!/usr/bin/env python3 -u
"""
HORIZON POS — the future-trained organism (P37, MS15).
========================================================================
F1's multi-horizon extension (FOUNDATIONS.md): a prediction head DEPOSITS a
forecast of a summary functional of chunk t+H at chunk t, holds it until t+H
arrives, scores it against the realized stream, and feeds the per-horizon
error back as (i) an early-warning signal, (ii) a training signal for the
depositing head itself, (iii) — under test here — a gating co-signal. This
is the only setting in which the loop closes at deployment: a batch-trained
model never experiences its own future.

HEAD DESIGN (documented per mission instructions):
  Input:  mean-pooled hidden state of chunk t (pre-head, post-final-layer,
          averaged over the (B, K) chunk -> (B, d_model); pooling over the
          batch too collapses to one summary vector per chunk since the
          target functional below is itself a whole-chunk statistic).
  Head:   d_model -> 256 -> 256 MLP (ReLU), output = 256 logits.
  Target: the BAG-OF-TOKEN HISTOGRAM of chunk t+H over the top-256 vocabulary
          buckets by corpus frequency (WT-2 train, computed once, deterministic)
          plus one overflow bucket folded into the nearest-frequency kept
          bucket's mass is NOT modeled separately -- tokens outside the
          top-256 are simply excluded from both numerator and denominator, so
          the target is the top-256 buckets' RELATIVE mass among themselves
          (renormalized to sum 1). This keeps the target dense enough at
          chunk-sized token counts (B*K, e.g. 8*64=512 tokens/chunk) that the
          histogram is not degenerate.
  Loss:   cross-entropy of the head's softmax against the normalized target
          histogram (a soft-label / distributional cross-entropy, i.e.
          -sum_i target_i * log_softmax(logits)_i). CHOSEN over cosine
          because it is scale-consistent with the per-token NLL the rest of
          the system already gates on (both are negative-log-likelihood-
          family scores), so horizon_surprise and 1-step-NLL sit in
          comparable units for the z-mixture in arm (ii) below. Cosine
          distance was considered but discarded: it is invariant to the
          histogram's entropy (a peaked and a flat target at the same angle
          score identically), which would wash out exactly the
          distribution-shift signal P37 is testing for.
  Training: the head has ITS OWN small Adam optimizer, separate from the
          base model's. It is trained on EVERY chunk once its deposited
          prediction is scored (H chunks after deposit) -- ungated, always
          on -- because it is "the layer that learns about the future" per
          the mission brief, independent of whether the base model's own
          gate fires that chunk.

QUEUE / TIME-AXIS (documented per mission instructions, load-bearing for a
fair P37a comparison):
  At chunk t, the head deposits pred(t) = f(pool(h_t)) into a FIFO queue of
  length H, tagged with target arrival index t+H. It also enqueues pool(h_t)
  itself is NOT needed again -- only pred(t) and the tag are kept (O(H)
  memory, each entry one (256,) logit vector).
  When chunk t+H's tokens arrive (i.e. we are AT chunk index t+H processing
  its own (x, y)), we now hold both pred(t) (dequeued, since queue length H
  means it was deposited exactly H steps ago) and the REALIZED histogram of
  chunk t+H's tokens. horizon_surprise[t+H] := CE(pred(t) -> realized hist of
  chunk t+H). By construction horizon_surprise[t+H] becomes AVAILABLE at the
  same wall-chunk index t+H as 1-step-NLL[t+H] (both require chunk t+H's
  tokens to exist). THAT is the fair axis for the P37a detector race: both
  detectors are compared as functions of the chunk index at which their
  value becomes computable, not at the (earlier) index t at which the
  underlying hidden state was formed. A win for horizon_surprise in this
  framing is not "it saw the future before it happened" (impossible) but
  "the H-step-old hidden state already encoded enough about the impending
  distribution that scoring it against the NEW chunk spikes sooner / harder
  around a domain boundary than the NEW chunk's own 1-step NLL does" -- a
  real and useful early-warning property if it holds, honestly scoped if it
  does not.
  Boundary handling: horizon_surprise has no defined value for the first H
  chunks of a run/phase-cursor (queue not yet full) -- those indices are
  left unscored (None in the raw curve; the detector simply has fewer
  samples to work with there, matching what a real deployment would face).

ARMS (Schock-Protokoll: C4 -> code -> C4, phase_chunks as pos_domain_shock.py,
matched budgets, same three-phase feeders, byte-identical stream order per
regime via independent-but-identically-parameterized source instantiation):
  base_gate     R2's exact recipe (pos_run.py step_gated mechanics): backward
                iff 1-step chunk-mean NLL > rolling-q75(window=200) of PRIOR
                1-step NLLs. No horizon head influence on the gate (the head
                still runs and trains in the background for logging parity,
                but its output never touches base_gate's threshold decision).
  horizon_gate  gate signal = 0.5*z(1-step NLL) + 0.5*z(horizon_surprise of
                the most-recently-SCORED prediction, i.e. horizon_surprise at
                the current chunk index if available this chunk, else the
                last available value held constant / z-scored against its
                own rolling window) -- z-normalized against EACH signal's own
                rolling window (window=200, matching base_gate's window),
                then mixed and thresholded at the mixed score's OWN rolling
                q75 (same q, same window length, same min_window gate as
                base_gate) -- i.e. identical q75 MECHANICS, applied to the
                mixed z-score instead of the raw 1-step NLL.
  shuffled      identical to horizon_gate, EXCEPT the deposited predictions
                are permuted WITHIN the queue (seed 42) immediately before
                being dequeued/scored -- pred(t) that would have scored
                against chunk t+H instead scores against a different chunk
                lag from the same H-deep window, destroying the temporal
                binding between a specific deposit and its true target while
                preserving the marginal statistics of "some prediction from
                H-ish chunks ago, scored now" (P37c falsifier).

MEASUREMENTS:
  (a) EARLY WARNING: at the phase1->2 and phase2->3 boundaries, the chunk
      index (relative to the boundary) at which a rolling-mean/2-sigma
      CUSUM-style detector fires on horizon_surprise vs on 1-step-NLL. Rule:
      reference mean/std computed from the last `ref_window` chunks of the
      PRIOR phase (phase 1 for the 1->2 boundary, phase 2 for the 2->3
      boundary); detector fires at the first chunk index >= boundary where a
      rolling mean (window=`fire_window`) of the signal exceeds
      ref_mean + 2*ref_std. P37a: horizon fires >= 5 chunks earlier than
      1-step at both boundaries.
  (b) forgetting / plasticity / recovery, computed exactly as
      pos_domain_shock.py, for all three arms. P37b: horizon_gate forgetting
      <= base_gate forgetting, at horizon_gate plasticity >= 0.9x base_gate
      plasticity.
  (c) shuffled must destroy both (a) (no reliable early lead over 1-step)
      and (b) (no forgetting advantage over base_gate).

TECHNIQUE: threads=1 + interop try/except, CPU, os.nice try/except, seed 42,
argparse --smoke/--full/--horizon/--out. Read-only w.r.t. the live run's
checkpoint (loads results/pos_snapshots/ckpt_359050240.pt A3, same convention
as pos_domain_shock.py / pos_sleep.py).

Usage:
  python src/horizon_pos.py --smoke                 # phase_chunks=40
  python src/horizon_pos.py --full                  # phase_chunks=150
  python src/horizon_pos.py --full --horizon 12 --out results/horizon_pos_h12.json
"""
import os
import sys
import json
import copy
import argparse
import tempfile
from collections import deque, Counter

sys.path.insert(0, "reference")
sys.path.insert(0, "src")

try:
    os.nice(19)
except (PermissionError, AttributeError):
    pass  # already niced by the launcher (macOS EPERM on re-nice) / no os.nice on this platform

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.mps.is_available = lambda: False          # force CPU (same as the live run)
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass  # must be set before any parallel work has run; harmless if already set

from streaming_train import StreamingNoPELM
from length_extrap_v2 import load_wikitext2, build_vocab, tokenize
from pos_run import build_eval_set, heldout                    # safe: pos_run's main() only runs under __main__
from pos_sleep import ChunkFeeder, C4ValStream, load_snapshot, _real_vocab
from pos_domain_shock import CodeStream, make_phase1_source, make_phase2_source, grad_step, nograd_step


# ───────────────────────────────────────────────────────────────────────────
#  Top-256 vocabulary buckets (by WT-2 train frequency) — the horizon head's
#  fixed target space. Computed once, deterministic given the WT-2 corpus.
# ───────────────────────────────────────────────────────────────────────────
def build_top_buckets(stoi, unk, n_buckets=256):
    train_text, _ = load_wikitext2()
    ids = tokenize(train_text, stoi, unk)
    counts = Counter(ids)
    counts.pop(unk, None)                                  # exclude unk from the target space
    top = [tok for tok, _ in counts.most_common(n_buckets)]
    if len(top) < n_buckets:
        # pad with arbitrary distinct ids not already chosen (defensive; WT-2's
        # vocab is far larger than 256 so this path is not expected to trigger)
        rest = [t for t in stoi.values() if t not in set(top) and t != unk]
        top.extend(rest[: n_buckets - len(top)])
    bucket_of = {tok: i for i, tok in enumerate(top)}       # token id -> bucket index [0,256)
    return top, bucket_of


def chunk_histogram(y, bucket_of, n_buckets=256):
    """y: (B, K) target token ids for one chunk. Returns a (n_buckets,) float32
    tensor, the normalized (sum=1, or all-zero if no in-vocab tokens landed)
    relative mass of y's tokens among the top-256 buckets."""
    hist = torch.zeros(n_buckets, dtype=torch.float32)
    flat = y.reshape(-1).tolist()
    for tok in flat:
        b = bucket_of.get(tok)
        if b is not None:
            hist[b] += 1.0
    s = hist.sum()
    if s > 0:
        hist = hist / s
    return hist


# ───────────────────────────────────────────────────────────────────────────
#  Horizon head: d_model -> 256 -> 256 MLP, own optimizer, own loss.
# ───────────────────────────────────────────────────────────────────────────
class HorizonHead(nn.Module):
    def __init__(self, d_model, n_buckets=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, n_buckets), nn.ReLU(),
            nn.Linear(n_buckets, n_buckets),
        )

    def forward(self, pooled):
        return self.net(pooled)                            # (B_or_1, n_buckets) logits


def horizon_loss(logits, target_hist):
    """Soft-label cross-entropy: -sum_i target_i * log_softmax(logits)_i.
    logits: (n_buckets,); target_hist: (n_buckets,), sums to 1 (or all-zero,
    in which case the loss is defined as 0 — no signal to learn from)."""
    if float(target_hist.sum()) == 0.0:
        return torch.zeros((), dtype=logits.dtype)
    logp = F.log_softmax(logits, dim=-1)
    return -(target_hist * logp).sum()


def mean_pool_hidden(model, x, states):
    """Runs the SAME per-layer scan math as StreamingNoPELM.forward (the frozen
    reference class is never edited — this mirrors the established repo
    pattern of driving model.embed / layer.scan directly, see gap_ladder.py /
    holo_alpha_shut.py), returning (new_states, pooled_h) where pooled_h is
    the final layer's hidden state mean-pooled over (batch, position) into a
    single (d_model,) vector — one summary per chunk, matching the
    chunk-level target histogram."""
    h = model.embed(x)
    si = states if states is not None else [None] * len(model.layers)
    new_states = []
    for layer, z in zip(model.layers, si):
        y, z_out = layer.scan(h, z)
        h = layer.ln1(h + y)
        h = layer.ln2(h + layer.ffn(h))
        new_states.append(z_out)
    pooled = h.mean(dim=(0, 1))                             # (d_model,)
    return new_states, pooled, h


def model_forward_from_h(model, h, x, y):
    """Given the hidden state already computed by mean_pool_hidden (avoids a
    second scan pass), finishes the forward (head projection) and returns the
    per-token NLL — used so nograd/grad steps and the horizon pooling share
    ONE forward, not two."""
    logits = model.head(h)
    nll_flat = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
    return nll_flat.view(x.shape)


# ───────────────────────────────────────────────────────────────────────────
#  z-normalized rolling gate (shared mechanics: q75 threshold over a window,
#  min_window gate before the threshold is trusted — same shape as pos_run.py
#  step_gated / pos_domain_shock.py run_r2_chunk, generalized to accept a
#  MIXED scalar score instead of raw 1-step NLL).
# ───────────────────────────────────────────────────────────────────────────
def rolling_z(value, window):
    """z-score of `value` against the (pre-update) contents of `window`
    (a deque). Returns 0.0 if the window has fewer than 2 samples (not
    enough spread to normalize) — the caller appends AFTER computing z."""
    if len(window) < 2:
        return 0.0
    arr = np.fromiter(window, dtype=np.float64)
    mu, sd = arr.mean(), arr.std()
    if sd < 1e-12:
        return 0.0
    return float((value - mu) / sd)


def gate_decision(score, window, gate_q, min_window):
    """Same q75-over-rolling-window mechanics as pos_run.py step_gated /
    pos_domain_shock.py run_r2_chunk, generalized to any scalar score."""
    if len(window) >= min_window:
        thresh = float(np.quantile(np.fromiter(window, dtype=np.float64), gate_q))
        gated = score > thresh
    else:
        thresh = None
        gated = True                                        # not enough window yet: learn
    return gated, thresh


# ───────────────────────────────────────────────────────────────────────────
#  Horizon prediction queue — FIFO of length H, (deposit_idx, pred_logits).
# ───────────────────────────────────────────────────────────────────────────
class HorizonQueue:
    """Holds up to H pending (deposit_chunk_idx, pred_logits) entries. `shuffle`
    permutes the CURRENT contents of the queue (seeded) before a dequeue is
    served, breaking the deposit<->target binding without changing queue
    depth or timing (P37c falsifier)."""

    def __init__(self, horizon, shuffle=False, seed=42):
        self.H = horizon
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.buf = deque()                                  # list of (idx, pred) in deposit order

    def deposit(self, idx, pred):
        self.buf.append((idx, pred))

    def ready(self):
        return len(self.buf) >= self.H

    def pop_for_scoring(self):
        """Pops the oldest entry (deposited exactly H chunks ago under normal
        operation, since we only ever hold <= H entries — see push order in
        the main loop). If `shuffle`, permutes buf's order first so the
        popped prediction is a random one of the currently-queued predictions
        rather than deterministically the oldest — same queue depth, broken
        temporal binding."""
        if self.shuffle and len(self.buf) > 1:
            items = list(self.buf)
            order = self.rng.permutation(len(items))
            self.buf = deque(items[i] for i in order)
        return self.buf.popleft()


# ───────────────────────────────────────────────────────────────────────────
#  One horizon-aware chunk step: forward (no_grad or grad per gate decision),
#  horizon head deposit + score + train, returns everything the caller needs.
# ───────────────────────────────────────────────────────────────────────────
def horizon_chunk_step(model, opt, head, head_opt, x, y, states, global_idx,
                       queue, bucket_of, n_buckets, gate_fn, gate_state):
    """gate_fn(score_1step, horizon_surprise_or_None, gate_state) -> (gated: bool,
    mixed_score: float, thresh: float|None) decides whether THIS chunk's base-
    model backward fires. Returns a dict of everything logged per chunk."""
    # ---- 1-step forward (always; no_grad first so the gate can see the score
    #      before deciding whether to redo it with a graph) ----
    with torch.no_grad():
        states_ng, pooled_ng, h_ng = mean_pool_hidden(model, x, states)
        nll_ng = model_forward_from_h(model, h_ng, x, y)
    s1 = float(nll_ng.mean())

    # ---- score any prediction whose target has just arrived (chunk global_idx
    #      IS the target of a prediction deposited H chunks ago) ----
    horizon_surprise = None
    if queue.ready():
        # The queue stores the DETACHED pooled context of the deposit chunk;
        # the prediction is computed FRESH here with the CURRENT head. Keeping
        # a graph-bearing prediction across H chunks breaks autograd (the head
        # updates between deposit and scoring — inplace-version error, found
        # live) and is also the wrong semantics: the head should be scored as
        # it stands when the future arrives.
        dep_idx, pooled_stored = queue.pop_for_scoring()
        pred_logits = head(pooled_stored)
        target_hist = chunk_histogram(y, bucket_of, n_buckets)
        hloss = horizon_loss(pred_logits, target_hist)
        horizon_surprise = float(hloss.detach())
        # train the head on this realized (pred, target) pair — always on,
        # regardless of the base model's gate this chunk (mission: "the layer
        # that learns about the future" is never gated).
        head_opt.zero_grad(set_to_none=True)
        hloss.backward()
        head_opt.step()

    # ---- gate decision (base_gate ignores horizon_surprise; horizon_gate /
    #      shuffled mix it in — gate_fn encapsulates the difference). The
    #      current global chunk index is stamped into gate_state so
    #      make_noise_gate can compute its lag-eligibility window; the other
    #      gate_fns (base_gate, mixed_gate) simply never read this key, so
    #      this stamp is a no-op for them. ----
    if isinstance(gate_state, dict):
        gate_state["idx"] = global_idx
    gated, mixed_score, thresh = gate_fn(s1, horizon_surprise, gate_state)

    if gated:
        states_new, gt, loss, _, nll = grad_step(model, opt, x, y, states)
        # deposit AFTER the possible weight update, from the POST-update
        # hidden state — the head predicts what the model, as it now stands
        # (having just possibly learned from this very chunk), thinks H
        # chunks ahead looks like. Recompute pooled_h under no_grad from the
        # (possibly) updated weights so the deposit reflects live weights
        # without needing to keep the training graph alive.
        with torch.no_grad():
            _, pooled_dep, _ = mean_pool_hidden(model, x, states)
    else:
        states_new, gt = states_ng, 0
        nll = nll_ng
        pooled_dep = pooled_ng

    # ---- deposit this chunk's pooled context; the prediction over chunk
    #      (global_idx + H) is computed at scoring time with the then-current
    #      head (see scoring block above) ----
    queue.deposit(global_idx, pooled_dep.detach())

    return {
        "states": states_new, "grad_tokens": gt, "gated": gated,
        "s1": s1, "horizon_surprise": horizon_surprise,
        "mixed_score": mixed_score, "thresh": thresh,
        "x": x.detach(), "nll": nll.detach(),
    }


# ───────────────────────────────────────────────────────────────────────────
#  Gate functions for the three arms.
# ───────────────────────────────────────────────────────────────────────────
def make_base_gate(gate_q, min_window):
    window = deque(maxlen=200)

    def gate_fn(s1, hsurp, _state):
        gated, thresh = gate_decision(s1, window, gate_q, min_window)
        window.append(s1)
        return gated, s1, thresh
    return gate_fn, {"window": window}


def make_mixed_gate(gate_q, min_window):
    """horizon_gate / shuffled share this mechanics: z-mix of 1-step NLL and
    the most-recently-SCORED horizon_surprise, q75-gated on the mixed score's
    own rolling window. When no horizon_surprise has been scored yet this
    chunk (queue not ready, e.g. the first H chunks of a phase-cursor), the
    LAST available horizon_surprise value is held constant for the z-mix
    (falls back to pure 1-step-z, i.e. weight collapses to the 1-step term,
    if none has ever been scored) — documented in the module docstring."""
    win_1step = deque(maxlen=200)
    win_horizon = deque(maxlen=200)
    win_mixed = deque(maxlen=200)
    last_horizon = {"val": None}

    def gate_fn(s1, hsurp, _state):
        if hsurp is not None:
            last_horizon["val"] = hsurp
        z1 = rolling_z(s1, win_1step)
        if last_horizon["val"] is not None:
            zh = rolling_z(last_horizon["val"], win_horizon)
            mixed = 0.5 * z1 + 0.5 * zh
        else:
            mixed = z1                                       # no horizon signal yet: pure 1-step
        gated, thresh = gate_decision(mixed, win_mixed, gate_q, min_window)
        win_1step.append(s1)
        if last_horizon["val"] is not None:
            win_horizon.append(last_horizon["val"])
        win_mixed.append(mixed)
        return gated, mixed, thresh
    return gate_fn, {"win_1step": win_1step, "win_horizon": win_horizon, "win_mixed": win_mixed}


def make_noise_gate(gate_q, min_window, min_lag, seed):
    """P40's decorrelated-control arm: mechanically identical to
    make_mixed_gate (same 1-step/horizon z-mix, same q75-on-rolling-window
    gate), EXCEPT the second stream's value fed into the mix is NOT the
    most-recently-scored horizon_surprise but the horizon_surprise of a
    uniformly-random EARLIER chunk, drawn with lag >= min_lag from the
    current chunk (min_lag defaults to args.ref_window, NOT the full
    args.gate_window: decorrelation needs distance from the rolling
    statistics' own autocorrelation time -- ref_window is well past
    fire_window and gives room for the rolling z/quantile windows to have
    turned over -- but does not need the FULL 200-chunk gate-window depth,
    which collides with the run's phase geometry (150 chunks/phase in
    --full, 40 in --smoke) and would make most or all lag draws structurally
    ineligible, degenerating the arm toward the cold-start fallback i.e.
    toward horizon_gate itself. P40's registered text only specifies "large
    random lag"; the concrete depth is a dimensioning choice, corrected here
    after review to fit the run geometry rather than collide with it). This
    is a SEPARATE implementation from make_mixed_gate (not a thin wrapper)
    so that horizon_gate's code path is untouched by construction -- the
    only way to guarantee zero regression on the existing regime is to not
    reuse/branch its function body.

    History bookkeeping: every SCORED horizon_surprise value is appended to
    `history` (a plain list, append-only, one entry per scored chunk, in
    scoring order) together with the chunk index it was scored at. At each
    gate decision, if len(history) has at least one entry with index <=
    (current_index - min_lag), a uniformly random one of those eligible
    entries is drawn (own seeded RNG, not the global torch RNG, not the
    HorizonQueue's shuffle RNG) and its value is used as the z-mix's second
    stream this chunk. Below that lag depth (cold-start: early chunks of a
    phase-cursor, or the first ~min_lag scored chunks of the whole run), the
    CURRENT chunk's own most-recently-scored horizon_surprise is used
    instead (identical fallback semantics to make_mixed_gate's
    `last_horizon["val"]`), and the fallback is counted so its rate can be
    reported and checked for contamination."""
    win_1step = deque(maxlen=200)
    win_noise = deque(maxlen=200)
    win_mixed = deque(maxlen=200)
    last_horizon = {"val": None}
    history = []                                             # [(scored_at_idx, value), ...] append-only
    rng = np.random.default_rng(seed)
    counters = {"n_decisions": 0, "n_fallback": 0}

    def gate_fn(s1, hsurp, state):
        current_idx = state["idx"]
        if hsurp is not None:
            last_horizon["val"] = hsurp
            history.append((current_idx, hsurp))
        eligible_hi = current_idx - min_lag                  # inclusive upper bound on eligible idx
        # binary-search-free linear scan is fine here (history grows by <=1
        # per chunk and phase_chunks is O(100s)); eligibility is idx <= eligible_hi
        eligible = [v for (i, v) in history if i <= eligible_hi]
        counters["n_decisions"] += 1
        if eligible:
            noise_val = eligible[int(rng.integers(0, len(eligible)))]
        elif last_horizon["val"] is not None:
            noise_val = last_horizon["val"]                  # cold-start fallback
            counters["n_fallback"] += 1
        else:
            noise_val = None                                 # no horizon signal at all yet
        z1 = rolling_z(s1, win_1step)
        if noise_val is not None:
            zn = rolling_z(noise_val, win_noise)
            mixed = 0.5 * z1 + 0.5 * zn
        else:
            mixed = z1
        gated, thresh = gate_decision(mixed, win_mixed, gate_q, min_window)
        win_1step.append(s1)
        if noise_val is not None:
            win_noise.append(noise_val)
        win_mixed.append(mixed)
        return gated, mixed, thresh

    gate_state = {"idx": 0, "counters": counters,
                  "win_1step": win_1step, "win_noise": win_noise, "win_mixed": win_mixed}
    return gate_fn, gate_state


# ───────────────────────────────────────────────────────────────────────────
#  Early-warning detector: rolling-mean(window=fire_window) > ref_mean +
#  2*ref_std, first-fire chunk index relative to a boundary.
# ───────────────────────────────────────────────────────────────────────────
def first_fire_index(signal, boundary_idx, ref_window, fire_window, indices=None):
    """signal: list of float (may contain None for unscored chunks — skipped).
    indices: parallel list of the signal's global chunk indices (defaults to
    0..len-1). Reference mean/std computed from the ref_window values
    immediately BEFORE boundary_idx (the prior phase's own statistics — a
    detector calibrated on what "normal" looked like in the phase that is
    about to end). Returns the first global chunk index >= boundary_idx at
    which a rolling mean of the last `fire_window` valid signal values
    exceeds ref_mean + 2*ref_std, or None if it never fires within the
    provided signal."""
    if indices is None:
        indices = list(range(len(signal)))
    pairs = [(i, v) for i, v in zip(indices, signal) if v is not None]
    if not pairs:
        return None, None, None
    ref_vals = [v for i, v in pairs if i < boundary_idx][-ref_window:]
    if len(ref_vals) < max(5, ref_window // 4):
        return None, None, None                              # not enough reference material
    ref_mean, ref_std = float(np.mean(ref_vals)), float(np.std(ref_vals))
    if ref_std < 1e-12:
        ref_std = 1e-12
    fire_thresh = ref_mean + 2 * ref_std
    roll = deque(maxlen=fire_window)
    for i, v in pairs:
        roll.append(v)
        if i < boundary_idx:
            continue
        if len(roll) >= max(2, fire_window // 2) and float(np.mean(roll)) > fire_thresh:
            return i, ref_mean, ref_std
    return None, ref_mean, ref_std


# ───────────────────────────────────────────────────────────────────────────
#  Main measurement
# ───────────────────────────────────────────────────────────────────────────
def run_horizon(args, eval_wt2_fn, vocab_fn=_real_vocab):
    ck, cfg, base_model, opt_sd, stoi, unk, mask, V = load_snapshot(args.ckpt, vocab_fn)
    B, K = cfg["batch"], cfg["chunk"]
    lr = cfg["lr"]
    d_model = cfg["d_model"]
    H = args.horizon
    n_buckets = args.n_buckets

    phase_chunks = args.phase_chunks
    eval_every = args.eval_every

    print(f"[horizon] ckpt n_streamed={ck['n_streamed']:,} | phase_chunks={phase_chunks} "
          f"eval_every={eval_every} | H={H} n_buckets={n_buckets} | B={B} K={K} lr={lr}", flush=True)

    top_buckets, bucket_of = build_top_buckets(stoi, unk, n_buckets)
    print(f"[horizon] top-{n_buckets} vocab buckets built (by WT-2 train frequency)", flush=True)

    code_val_src = CodeStream(stoi, unk)
    code_val_ids = code_val_src.next_block(args.code_val_tokens)
    code_val_unk_rate = code_val_src.unk_rate()
    cvX, cvY = build_eval_set(code_val_ids, len(code_val_ids) - 1, K)
    print(f"[horizon] code-val slice: {cvY.numel():,} tokens, unk_rate={code_val_unk_rate:.4f}", flush=True)

    def eval_code_fn(model):
        return heldout(model, cvX, cvY)

    def fresh_model_opt_head():
        m = copy.deepcopy(base_model)
        o = torch.optim.Adam(m.parameters(), lr=lr)
        if opt_sd is not None:
            o.load_state_dict(opt_sd)
            for g in o.param_groups:
                g["lr"] = lr
        head = HorizonHead(d_model, n_buckets)
        head_opt = torch.optim.Adam(head.parameters(), lr=args.head_lr)
        return m, o, head, head_opt

    base_wt2 = eval_wt2_fn(base_model)
    print(f"[horizon] base_heldout_wt2={base_wt2:.6f}", flush=True)

    out = {
        "ckpt_n_streamed": ck["n_streamed"],
        "horizon": H, "n_buckets": n_buckets,
        "head_design": "d_model->256->256 MLP, mean-pooled(batch,pos) final-layer hidden state -> "
                       "soft-label cross-entropy against normalized top-256-bucket bag-of-token "
                       "histogram of chunk t+H; head has its own Adam optimizer, trained on every "
                       "scored prediction regardless of the base model's gate decision that chunk",
        "time_axis_note": "horizon_surprise[t] scores the prediction deposited at chunk t-H against "
                           "chunk t's realized tokens; it becomes AVAILABLE at chunk index t, the same "
                           "index at which 1-step-NLL[t] is available -- the P37a detector race compares "
                           "both signals as functions of that shared availability index, not of the "
                           "(earlier) index at which the underlying hidden state was formed",
        "budget": {"phase_chunks": phase_chunks, "eval_every": eval_every,
                  "gate_q": args.gate_q, "gate_window": args.gate_window,
                  "min_window": args.min_window, "ref_window": args.ref_window,
                  "fire_window": args.fire_window, "head_lr": args.head_lr},
        "code_val_tokens": cvY.numel(),
        "code_val_unk_rate": round(code_val_unk_rate, 6),
        "base_heldout_wt2": round(base_wt2, 6),
        "regimes": {},
    }

    regime_tags = [t.strip() for t in args.regimes.split(",") if t.strip()]
    # min_lag default = ref_window (50), not gate_window (200): decorrelation
    # needs distance from the rolling stats' autocorrelation time, not the
    # full gate-window depth, which collides with phase geometry (150/40
    # chunks per phase) and would starve the eligible-draw pool. Corrected
    # post-review; see make_noise_gate's docstring for the full reasoning.
    noise_min_lag = args.noise_min_lag if args.noise_min_lag is not None else args.ref_window

    regime_results = {}
    for tag in regime_tags:
        print(f"\n[horizon] ===== regime {tag} =====", flush=True)
        # pin the torch RNG per regime: the base model comes from the ckpt
        # (deterministic) but the HorizonHead init used to draw from the
        # UNSEEDED global generator -- the only nondeterministic element in
        # the whole run (found by the P40 deterministic A/B: every model
        # curve bit-identical, only the head-dependent horizon curves moved).
        # Seeding HERE gives every arm the identical head init on top of the
        # identical base model -- a strictly cleaner matched comparison.
        torch.manual_seed(args.seed)
        model, opt, head, head_opt = fresh_model_opt_head()

        phase13_src = make_phase1_source(stoi, unk)
        phase2_src = make_phase2_source(stoi, unk)
        feeder13 = ChunkFeeder(phase13_src, B, K)
        feeder2 = ChunkFeeder(phase2_src, B, K)

        states = None
        queue = HorizonQueue(H, shuffle=(tag == "shuffled"), seed=args.seed)
        if tag == "base_gate":
            gate_fn, gate_state = make_base_gate(args.gate_q, args.min_window)
        elif tag == "noise_gate":
            # own seed namespace (seed+4000), independent of HorizonQueue's
            # shuffle RNG (plain args.seed) and of the global torch RNG, so
            # the noise-lag draw stream cannot diverge/collide with any
            # other arm's randomness (P40 requirement 3)
            gate_fn, gate_state = make_noise_gate(args.gate_q, args.min_window,
                                                   noise_min_lag, args.seed + 4000)
        else:
            gate_fn, gate_state = make_mixed_gate(args.gate_q, args.min_window)

        curve_wt2 = []
        curve_code = []
        curve_s1 = []                                          # [(idx, phase, 1-step NLL)]
        curve_horizon = []                                      # [(idx, phase, horizon_surprise or None)]
        grad_tokens_total = 0
        gate_frac_log = []
        phase_gate_log = {"phase1": [], "phase2": [], "phase3": []}
        boundary_idx = {}

        def record(global_idx, phase):
            hl_wt2 = eval_wt2_fn(model)
            hl_code = eval_code_fn(model)
            curve_wt2.append([global_idx, phase, round(hl_wt2, 6)])
            curve_code.append([global_idx, phase, round(hl_code, 6)])
            print(f"[horizon][{tag}] phase={phase:<7} chunk={global_idx:>4} "
                  f"wt2={hl_wt2:.6f} code={hl_code:.6f}", flush=True)
            return hl_wt2, hl_code

        record(0, "base")
        global_idx = 0

        def run_wake_block(feeder, n_chunks, phase_name):
            nonlocal states, global_idx, grad_tokens_total
            done = 0
            while done < n_chunks:
                step_n = min(eval_every, n_chunks - done)
                for _ in range(step_n):
                    x, y = feeder.next_xy()
                    res = horizon_chunk_step(model, opt, head, head_opt, x, y, states,
                                             global_idx, queue, bucket_of, n_buckets,
                                             gate_fn, gate_state)
                    states = res["states"]
                    grad_tokens_total += res["grad_tokens"]
                    gate_frac_log.append(1 if res["gated"] else 0)
                    phase_gate_log[phase_name].append(1 if res["gated"] else 0)
                    curve_s1.append([global_idx, phase_name, round(res["s1"], 6)])
                    curve_horizon.append([global_idx, phase_name,
                                          None if res["horizon_surprise"] is None
                                          else round(res["horizon_surprise"], 6)])
                    done += 1
                    global_idx += 1
                record(global_idx, phase_name)
            return done

        run_wake_block(feeder13, phase_chunks, "phase1")
        pre_phase2_wt2 = curve_wt2[-1][2]
        pre_phase2_code = curve_code[-1][2]
        boundary_idx["phase1_to_2"] = global_idx

        run_wake_block(feeder2, phase_chunks, "phase2")
        boundary_idx["phase2_to_3"] = global_idx

        run_wake_block(feeder13, phase_chunks, "phase3")

        wt2_values_phase2 = [v for idx, ph, v in curve_wt2 if ph == "phase2"]
        code_values_phase2 = [v for idx, ph, v in curve_code if ph == "phase2"]
        post_phase3_wt2 = curve_wt2[-1][2]

        forgetting = round(max(wt2_values_phase2 + [pre_phase2_wt2]) - pre_phase2_wt2, 6)
        plasticity = round(pre_phase2_code - min(code_values_phase2 + [pre_phase2_code]), 6)
        recovery = round(post_phase3_wt2 - pre_phase2_wt2, 6)

        gate_frac = sum(gate_frac_log) / max(1, len(gate_frac_log))

        # ---- early-warning detector at both boundaries, on 1-step vs horizon ----
        s1_idx = [r[0] for r in curve_s1]
        s1_vals = [r[2] for r in curve_s1]
        h_idx = [r[0] for r in curve_horizon]
        h_vals = [r[2] for r in curve_horizon]

        early_warning = {}
        for bname, bidx in boundary_idx.items():
            fire_1step, ref_mu_1, ref_sd_1 = first_fire_index(
                s1_vals, bidx, args.ref_window, args.fire_window, indices=s1_idx)
            fire_horizon, ref_mu_h, ref_sd_h = first_fire_index(
                h_vals, bidx, args.ref_window, args.fire_window, indices=h_idx)
            lead = None
            if fire_1step is not None and fire_horizon is not None:
                lead = fire_1step - fire_horizon                # positive = horizon fired earlier
            early_warning[bname] = {
                "boundary_idx": bidx,
                "fire_1step": fire_1step, "fire_horizon": fire_horizon,
                "lead_chunks_horizon_earlier": lead,
                "ref_mean_1step": ref_mu_1, "ref_std_1step": ref_sd_1,
                "ref_mean_horizon": ref_mu_h, "ref_std_horizon": ref_sd_h,
            }

        # ---- per-phase realized gate ("second-stream") firing rate, P40 req. 4 ----
        phase_gate_rate = {
            ph: (round(sum(v) / len(v), 4) if v else None) for ph, v in phase_gate_log.items()
        }

        # ---- noise_gate's cold-start fallback fraction (P40 req. 2): must be
        #      small, else the arm is contaminated by current-chunk leakage ----
        noise_fallback = None
        if tag == "noise_gate":
            c = gate_state["counters"]
            noise_fallback = {
                "n_decisions": c["n_decisions"], "n_fallback": c["n_fallback"],
                "fallback_frac": round(c["n_fallback"] / max(1, c["n_decisions"]), 4),
            }

        regime_results[tag] = {
            "curve_wt2": curve_wt2, "curve_code": curve_code,
            "curve_s1": curve_s1, "curve_horizon": curve_horizon,
            "grad_tokens_total": grad_tokens_total,
            "n_chunks_gated": sum(gate_frac_log), "n_chunks_seen": len(gate_frac_log),
            "gate_frac": round(gate_frac, 4),
            "gate_frac_by_phase": phase_gate_rate,
            "noise_fallback": noise_fallback,
            "pre_phase2_wt2": round(pre_phase2_wt2, 6), "pre_phase2_code": round(pre_phase2_code, 6),
            "post_phase3_wt2": round(post_phase3_wt2, 6),
            "forgetting": forgetting, "plasticity": plasticity, "recovery": recovery,
            "early_warning": early_warning,
        }
        print(f"[horizon][{tag}] forgetting={forgetting:+.6f} plasticity={plasticity:+.6f} "
              f"recovery={recovery:+.6f} gate_frac={gate_frac:.4f} "
              f"by_phase={phase_gate_rate} "
              f"grad_tokens={grad_tokens_total:,}", flush=True)
        if noise_fallback is not None:
            print(f"[horizon][{tag}] noise_fallback: {noise_fallback}", flush=True)
        for bname, ew in early_warning.items():
            print(f"[horizon][{tag}] {bname}: fire_1step={ew['fire_1step']} "
                  f"fire_horizon={ew['fire_horizon']} lead={ew['lead_chunks_horizon_earlier']}", flush=True)

    out["regimes"] = regime_results

    present = set(regime_results.keys())
    n_visited = {t: regime_results[t]["n_chunks_seen"] for t in present}
    out["chunk_budget_check"] = {
        "n_chunks_visited": n_visited,
        "equal": len(set(n_visited.values())) <= 1,
    }

    def leads(regime):
        vals = [ew["lead_chunks_horizon_earlier"] for ew in regime["early_warning"].values()
                if ew["lead_chunks_horizon_earlier"] is not None]
        return vals

    verdict_parts = []

    # ---- P37 scoring: only meaningful/computable if its three original arms
    #      (base_gate, horizon_gate, shuffled) were all part of THIS run ----
    if {"base_gate", "horizon_gate", "shuffled"} <= present:
        bg = regime_results["base_gate"]
        hg = regime_results["horizon_gate"]
        sh = regime_results["shuffled"]

        hg_leads = leads(hg)
        sh_leads = leads(sh)
        p37a = bool(hg_leads) and all(v >= 5 for v in hg_leads)

        plasticity_ok = (bg["plasticity"] == 0 and hg["plasticity"] == 0) or (
            bg["plasticity"] != 0 and hg["plasticity"] >= 0.9 * bg["plasticity"])
        p37b = (hg["forgetting"] <= bg["forgetting"]) and plasticity_ok

        # (c) shuffled must destroy (a) and (b): no reliable early lead, no forgetting edge
        shuffled_destroys_a = not (bool(sh_leads) and all(v >= 5 for v in sh_leads))
        shuffled_destroys_b = not (sh["forgetting"] <= bg["forgetting"])
        p37c = shuffled_destroys_a and shuffled_destroys_b

        out["p37_scoring"] = {
            "a_early_warning_geq_5_chunks": {
                "pass": p37a, "horizon_gate_leads": hg["early_warning"],
            },
            "b_forgetting_leq_base_at_plasticity_geq_0.9x": {
                "pass": bool(p37b), "horizon_gate_forgetting": hg["forgetting"],
                "base_gate_forgetting": bg["forgetting"],
                "horizon_gate_plasticity": hg["plasticity"], "base_gate_plasticity": bg["plasticity"],
                "plasticity_ok": bool(plasticity_ok),
            },
            "c_shuffled_destroys_a_and_b": {
                "pass": bool(p37c), "shuffled_destroys_early_warning": bool(shuffled_destroys_a),
                "shuffled_destroys_forgetting_edge": bool(shuffled_destroys_b),
                "shuffled_leads": sh["early_warning"],
            },
        }
        if p37a and not p37b:
            out["p37_scoring"]["note"] = ("(a) holds but (b) does not: the horizon signal is a "
                                          "detector, not yet a teacher -- reported as such per "
                                          "PREDICTIONS.md P37.")

        all_pass = p37a and p37b and p37c
        verdict_parts.append(
            f"forgetting base={bg['forgetting']:+.6f} horizon={hg['forgetting']:+.6f} "
            f"shuffled={sh['forgetting']:+.6f} | "
            f"plasticity base={bg['plasticity']:+.6f} horizon={hg['plasticity']:+.6f} "
            f"(ok: {plasticity_ok}) | "
            f"early-warning leads (horizon-earlier, chunks): {hg_leads} | "
            f"shuffled leads: {sh_leads} | "
            f"P37: {'PASS' if all_pass else 'PARTIAL/FAIL'} "
            f"(a={p37a}, b={p37b}, c={p37c})"
        )

    # ---- P40 scoring: gate DIVERSITY vs. prediction CONTENT attribution,
    #      only computable if base_gate/horizon_gate/noise_gate were all run
    #      together (within-run consistency, matched gradient tokens) ----
    if {"base_gate", "horizon_gate", "noise_gate"} <= present:
        bg = regime_results["base_gate"]
        hg = regime_results["horizon_gate"]
        ng = regime_results["noise_gate"]

        forg_base, forg_horizon, forg_noise = bg["forgetting"], hg["forgetting"], ng["forgetting"]
        denom = forg_base - forg_horizon
        if abs(denom) < 1e-12:
            retained_fraction = None                          # horizon_gate had no forgetting edge to attribute
        else:
            retained_fraction = (forg_base - forg_noise) / denom
        retained_pass = retained_fraction is not None and retained_fraction >= 0.70

        # noise plasticity vs horizon plasticity (P40: "plasticity comparison",
        # same >=0.9x-of-reference convention used by P37b, reference=horizon here)
        noise_plasticity_ok = (hg["plasticity"] == 0 and ng["plasticity"] == 0) or (
            hg["plasticity"] != 0 and ng["plasticity"] >= 0.9 * hg["plasticity"])

        # counter-clause: horizon beats noise by > 0.03 nats forgetting at >= noise's plasticity
        content_matters = ((forg_horizon - forg_noise) > 0.03) and (hg["plasticity"] >= ng["plasticity"])

        rate_noise = ng["gate_frac"]
        rate_horizon = hg["gate_frac"]
        rate_diff_pp = abs(rate_noise - rate_horizon) * 100.0
        rate_match_pass = rate_diff_pp <= 2.0

        out["p40_scoring"] = {
            "a_retained_fraction": {
                "pass": bool(retained_pass), "retained_fraction": (
                    round(retained_fraction, 4) if retained_fraction is not None else None),
                "forgetting_base": forg_base, "forgetting_horizon": forg_horizon,
                "forgetting_noise": forg_noise,
                "threshold": 0.70,
                "interpretation": "retained_fraction >= 0.70 => the P37-(b) win is gate DIVERSITY, "
                                   "not prediction content (registered point call)",
            },
            "a_counter_content_matters": {
                "pass": bool(content_matters),
                "horizon_minus_noise_forgetting": round(forg_horizon - forg_noise, 6),
                "threshold_nats": 0.03,
                "noise_plasticity_ok_for_counter": bool(hg["plasticity"] >= ng["plasticity"]),
                "interpretation": "horizon beats noise by >0.03 nats forgetting at >= noise's "
                                   "plasticity => CONTENT matters, P37's shuffle control was too weak",
            },
            "plasticity_comparison": {
                "horizon_gate_plasticity": hg["plasticity"], "noise_gate_plasticity": ng["plasticity"],
                "noise_plasticity_geq_0.9x_horizon": bool(noise_plasticity_ok),
            },
            "b_rate_match": {
                "pass": bool(rate_match_pass), "rate_noise": rate_noise, "rate_horizon": rate_horizon,
                "diff_pp": round(rate_diff_pp, 4), "threshold_pp": 2.0,
                "rate_noise_by_phase": ng["gate_frac_by_phase"],
                "rate_horizon_by_phase": hg["gate_frac_by_phase"],
            },
            "noise_fallback": ng["noise_fallback"],
            "noise_min_lag": noise_min_lag,
        }
        verdict_parts.append(
            f"P40: retained_fraction={out['p40_scoring']['a_retained_fraction']['retained_fraction']} "
            f"(pass={retained_pass}) | content_matters_counter={content_matters} | "
            f"rate_match diff_pp={round(rate_diff_pp, 4)} (pass={rate_match_pass}) | "
            f"noise_fallback_frac={ng['noise_fallback']['fallback_frac'] if ng['noise_fallback'] else None}"
        )

    out["verdict"] = " || ".join(verdict_parts) if verdict_parts else (
        f"regimes run: {sorted(present)} (neither P37's nor P40's full arm-set was present -- "
        f"no scoring block computed)")
    print(f"\n[horizon] {out['verdict']}", flush=True)

    d = os.path.dirname(args.out) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_")
    with os.fdopen(fd, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, args.out)
    print(f"[horizon] -> {args.out}", flush=True)
    return out


# ───────────────────────────────────────────────────────────────────────────
#  CLI
# ───────────────────────────────────────────────────────────────────────────
def build_argparser():
    ap = argparse.ArgumentParser(
        description="HORIZON POS: H-step prediction head as early-warning + gating co-signal, C4->code->C4 (P37/MS15)")
    ap.add_argument("--ckpt", default="results/pos_snapshots/ckpt_359050240.pt")
    ap.add_argument("--phase-chunks", type=int, default=150, help="chunks per phase (1/2/3)")
    ap.add_argument("--eval-every", type=int, default=25, help="WT-2 + code-val eval cadence, in chunks")
    ap.add_argument("--code-val-tokens", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon", type=int, default=8, help="H, chunks ahead the head predicts")
    ap.add_argument("--n-buckets", type=int, default=256, help="top-N vocab buckets for the histogram target")
    ap.add_argument("--head-lr", type=float, default=3e-3, help="horizon head's own Adam lr")
    # gate mechanics (matches pos_run.py / pos_domain_shock.py R2 defaults)
    ap.add_argument("--gate-q", type=float, default=0.75, help="rolling gate quantile (matches live run's q)")
    ap.add_argument("--gate-window", type=int, default=200, help="rolling gate window length, per mission spec")
    ap.add_argument("--min-window", type=int, default=50, help="min window fill before gating kicks in")
    # early-warning detector
    ap.add_argument("--ref-window", type=int, default=50, help="chunks of prior-phase reference for the CUSUM-ish detector")
    ap.add_argument("--fire-window", type=int, default=5, help="rolling-mean window the detector fires on")
    ap.add_argument("--smoke", action="store_true", help="phase-chunks=40, eval-every=10, out=*_smoke.json")
    ap.add_argument("--full", action="store_true", help="phase-chunks=150, eval-every=25 (explicit; also the default)")
    # P40: noise_gate decorrelated-control arm (min-lag draw from horizon_surprise history)
    ap.add_argument("--noise-min-lag", type=int, default=None,
                     help="min lag (chunks) for noise_gate's random-earlier-chunk draw; "
                          "defaults to --ref-window (decorrelation distance, sized to the "
                          "run's phase geometry -- NOT --gate-window, which is too deep "
                          "relative to phase length and starves eligible draws)")
    ap.add_argument("--regimes", default="base_gate,horizon_gate,shuffled",
                     help="comma-separated regime list to run this invocation. Default is the "
                          "ORIGINAL P37 triple, unchanged, so every pre-P40 invocation (no "
                          "--regimes flag) reproduces bit-identical behavior. Pass "
                          "'base_gate,horizon_gate,noise_gate' for the P40 three-arm run, or a "
                          "single tag (e.g. 'horizon_gate') to smoke-test one regime in isolation "
                          "for a regression check.")
    ap.add_argument("--out", default="results/horizon_pos.json")
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    if args.smoke:
        args.phase_chunks = 40
        args.eval_every = 10
        args.ref_window = 15
        if args.out == "results/horizon_pos.json":
            args.out = "results/horizon_pos_smoke.json"
    elif args.full:
        args.phase_chunks = 150
        args.eval_every = 25

    def eval_wt2_fn(model):
        train_text, val_text = load_wikitext2()
        _, stoi, unk, mask = build_vocab(train_text)
        val_ids = tokenize(val_text, stoi, unk)
        cfg = torch.load(args.ckpt, weights_only=False)["config"]
        evX, evY = build_eval_set(val_ids, cfg["eval_tokens"], cfg["chunk"])
        return heldout(model, evX, evY)

    run_horizon(args, eval_wt2_fn)


if __name__ == "__main__":
    main()
