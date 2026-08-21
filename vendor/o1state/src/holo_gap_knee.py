#!/usr/bin/env python3 -u
"""
HOLO-GAP-KNEE — moving the gamma-knee by growing a gamma->1 carrier under a
full-sequence gap curriculum, then testing the CHUNKED deployment path.
======================================================================================
MISSION: v3 (analysis/HOLO_STREAM_VERDICT.md) holds P=2 keyed recall perfectly
to G=32 and dies at G=128 -- a gamma-knee. analysis/HOLO_CARRIER_THEORY.md Sec.1
derives why: during a gap the complex accumulator degenerates to a pure real
decay S_t = gamma_t * S_{t-1} (the phase never rotates in the gap -- only |S|
shrinks, by Gamma(G) = prod_gap gamma_t). The knee sits at

    G* ~= ln(|S0|/m_min) / (1 - gamma)

so gamma->1 moves the knee by ORDERS OF MAGNITUDE. streaming_train.py's
run_idle_persistence already proved gamma->1 (0.9999, tau~1000) is learnable
for a 1-bit 2-way beacon under a full-sequence gap curriculum. Question here:
does the SAME mechanism, applied to KEYED P=2 recall, move the knee from
G*~64 (v3) toward G>=512, ideally 2048 -- "keyed memory over three orders of
magnitude of silence at O(1) state"?

THE CENTRAL TRICK (read off the v1/v2/v3 diagnosis in HOLO_STREAM_VERDICT.md):
Multi-chunk CHUNKED training fails structurally: truncated BPTT detaches the
state at each chunk boundary, so once the gap crosses a chunk edge the query
loss has ZERO gradient path back to the write phase (v1/v2's collapse -- ~1900
iters of full-momentum Adam optimizing a signal that cannot teach writing,
which catastrophically UNDOES a consolidated write). v3's fix was to cap
TRAINING gap at one chunk and let large-G be pure eval extrapolation -- but
that caps how large a gamma the curriculum ever has reason to grow, because a
single-chunk gap is already "cheap" for a middling gamma.

The equivalence proof this file leans on (check_equivalence below, ported
verbatim from holo_stream_recall.py Sec.3: full-sequence forward with
state=None is the SAME operator as chunked+carried forward, delta<1.2e-6)
licenses a way out that keeps the gradient alive at ANY gap length: train
UNCHUNKED. model(x, None) builds one full autograd graph over the whole
sequence, so the query-position loss gradient reaches the write step through
the ENTIRE gap, however long. This is exactly streaming_train.py's
run_idle_persistence recipe (full-sequence forward during training, curriculum
grows G, no chunk boundary exists until deployment) -- which is the run that
grew the gamma=0.9999 / tau~1000 carrier for the 1-bit beacon (Pillar E). We
apply the identical training recipe to the keyed P=2 write, then evaluate on
the CHUNKED+carried path (chunk=16) because THAT is the deployment regime
holo_stream_recall.py established and that chunking is a strictly separate
axis from how the gradient was computed at train time (the equivalence proof
is exactly the license to decouple them).

THREE TRAINING VARIANTS (same init per seed, use_phase=True throughout):
  T1 baseline-v3   : CHUNKED training (chunk=16), train-gap capped at one
                      chunk (=12 for P=2) -- the v3 reference, reproduced here
                      as the control this file's claims are measured against.
  T2 fullseq-curric: UNCHUNKED training (model(x, None), full graph), gap
                      curriculum with patience=25 growing G_train up to 128.
  T3 fullseq+kick  : identical to T2, plus a GAMMA-KICKSTART: head 0 of layer
                      0's forget gate is initialized to start at gamma~=0.995
                      instead of the ordinary xavier-small init (which starts
                      gamma~=0.5, sigmoid(0)). The kickstart is implemented as
                      GammaKickstartScanLayer, a subclass of
                      StreamingHolographicScanLayer that overrides
                      _drive_and_gamma to add a FIXED constant logit offset
                      c=5.3 (sigmoid(5.3)=0.99503) to head 0's W_gamma output
                      before the sigmoid, via a registered buffer -- so it is
                      an ADDITIVE bias on top of the existing bias-free
                      nn.Linear, not a mutation of the reference module (the
                      reference holographic_gssm.py is never touched; only
                      this file's local streaming subclass is extended).  The
                      offset is a fixed constant (not a learned parameter): it
                      just relocates the SIGMOID OPERATING POINT so gradient
                      descent starts already near the gamma->1 regime instead
                      of having to climb there from gamma~=0.5 against the
                      pull of every other loss term across 3000 iterations.

EVAL (identical across all three variants -- the deployment path):
  CHUNKED+carried forward (chunk=16), G in {32,128,256,512,1024,2048},
  eval_batch=100 (50 for G>=1024 -- sequences are G+5 tokens long and the
  chunked python-loop scan is O(T) per trial), zeroed-at-gap null at G in
  {128,512} as the decisive control (exactly run_idle_persistence's
  "zeroed_at_gap" / holo_stream_recall's "holo_zeroed_at_gap" arm), 2 seeds.

PER-VARIANT gamma-SPECTRUM: mean per-head gamma on a pure-filler batch (the
gamma_spectrum idiom from streaming_train.py Sec.C, adapted to read
W_gamma directly off the trained StreamingHolographicScanLayer) -- links the
measured knee position to the measured carrier gamma, so the theory curve
G*(gamma) = ln-margin/(1-gamma) is directly checkable per variant, not just
asserted.

PREDICTIONS (registered before running anything below):
  P1: T2 knee >= 256 (full-sequence gradient alone, no kickstart, should grow
      gamma well past v3's single-chunk-capped carrier).
  P2: T3 knee >= 1024 (the kickstart removes the climb from gamma~=0.5, so the
      curriculum only has to WIDEN an already-large gamma, not discover one).
  P3: across variants, knee position correlates with the largest per-head
      filler-gamma measured in that variant's gamma-spectrum (the theory's
      G* ~ 1/(1-gamma) law, checked as an ordering, not a fitted constant).

CPU-only (mps disabled), torch.set_num_threads(1), os.nice(19) at script
start. Results -> results/holo_knee_*.json only. Does not modify
holo_stream_recall.py, holographic_gssm.py, or streaming_train.py.
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

from holographic_gssm import sequential_linear_scan  # noqa: E402  (unused directly but keeps import parity)
from holo_stream_recall import (   # noqa: E402  -- reuse the proven streaming layer/LM/task/eval, unmodified
    StreamingHolographicScanLayer, StreamingHolographicLM, _build_lm,
    check_equivalence, _gap_vocab, make_gap_mqar_batch, chunked_forward,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(REPO, "results")


# ═══════════════════════════════════════════════════════════════════════════
# 1. GAMMA-KICKSTART — a local subclass, reference module untouched.
#    Relocates the sigmoid operating point for ONE head of ONE layer so
#    gradient descent starts near gamma~=0.995 instead of gamma~=0.5.
# ═══════════════════════════════════════════════════════════════════════════
KICKSTART_LOGIT = 5.3   # sigmoid(5.3) = 0.99503... -> tau = 1/(1-gamma) ~= 201 tokens AT INIT
KICKSTART_HEAD = 0      # which head (within n_heads) receives the offset


class GammaKickstartScanLayer(StreamingHolographicScanLayer):
    """StreamingHolographicScanLayer, plus a FIXED additive logit offset on
    W_gamma's output for one head, applied inside _drive_and_gamma (the single
    choke point both the magnitude scan and the complex write read gamma
    from). The offset is a non-trainable buffer -- gradient still flows
    through W_gamma normally; the constant only shifts where the sigmoid
    starts. This is intentionally NOT a learned parameter: the point is to
    remove the climb from gamma~=0.5 to gamma~=1, not to hand-pick the final
    gamma (the curriculum still has to widen/hold it via ordinary training)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        offset = torch.zeros(self.n_heads, self.d_head)
        offset[KICKSTART_HEAD, :] = KICKSTART_LOGIT
        self.register_buffer("_gamma_kick", offset.view(1, 1, self.n_heads, self.d_head))

    def _drive_and_gamma(self, x):
        B, T, _ = x.shape
        v = torch.tanh(self.W_v(x))
        gate = torch.sigmoid(self.W_gate(x))
        gamma_logit = self.W_gamma(x).view(B, T, self.n_heads, self.d_head) + self._gamma_kick
        gamma = torch.sigmoid(gamma_logit)
        alpha = torch.sigmoid(self.W_alpha(x))

        v_gated = v * gate
        if self.dropout is not None:
            v_gated = self.dropout(v_gated)

        v_gated = v_gated.view(B, T, self.n_heads, self.d_head)
        alpha = alpha.view(B, T, self.n_heads, self.d_head)

        from holographic_gssm import LOG_COMPLEMENT_CLAMP, EPS
        w = torch.clamp(v_gated * v_gated, max=LOG_COMPLEMENT_CLAMP)
        z_in = torch.log(1.0 - w + EPS)
        a = alpha * z_in
        return a, gamma


def _build_lm_kickstart(vocab_size, mask_idx, d_model=64, n_layers=2, n_heads=4,
                        d_head=16, seq_len=32):
    """Same construction as holo_stream_recall._build_lm(use_phase=True), but
    layer 0's scan is swapped for GammaKickstartScanLayer AFTER the ordinary
    StreamingHolographicLM init (so W_key/W_gamma/etc. keep the SAME xavier
    init the other two variants get from the same seed -- only the forward's
    gamma computation gains the fixed offset)."""
    model = StreamingHolographicLM(
        vocab_size, mask_idx, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        d_head=d_head, seq_len=seq_len, dropout=0.0, causal=True,
        phase_scale=math.pi, use_phase=True, readout="rms",
        separate_qk=False, n_slots=1)
    old = model.layers[0].scan
    kicked = GammaKickstartScanLayer(
        old.d_model, d_head=old.d_head, n_heads=old.n_heads, causal=old.causal,
        dropout=0.0, phase_scale=old.phase_scale, use_phase=old.use_phase,
        readout=old.readout, separate_qk=old.separate_qk, n_slots=old.n_slots)
    # strict=False: the kickstart buffer _gamma_kick exists only on the new layer
    # (and must keep its constructed value, not come from the donor state_dict)
    kicked.load_state_dict(old.state_dict(), strict=False)
    model.layers[0].scan = kicked
    return model


# ═══════════════════════════════════════════════════════════════════════════
# 2. Full-sequence (unchunked) training -- the central trick. Gap curriculum
#    identical in spirit to train_gap_curriculum (holo_stream_recall.py) and
#    run_idle_persistence (streaming_train.py), but the forward pass is
#    model(x, None) -- ONE graph over the whole sequence, no detach anywhere.
# ═══════════════════════════════════════════════════════════════════════════
def train_fullseq_curriculum(model, P, Gmax, iters, lr, seed, batch,
                             P_max, V_max, F, log_every=0, g_start=2, patience=25, bar=0.9):
    """Unchunked analogue of holo_stream_recall.train_gap_curriculum. The only
    difference that matters: logits, _ = model(x, None) instead of
    chunked_forward(model, x, chunk) -- so the query-loss gradient reaches the
    write phase through the FULL gap, at any G, at the cost of O(T) autograd
    graph depth (T = 2P + Gcur + 1)."""
    key_lo, val_lo, fill_lo, _ = _gap_vocab(P_max, V_max, F)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    model.train()
    Gcur = min(g_start, Gmax) if Gmax > 0 else 0
    acc = 0.0
    good = 0
    for it in range(iters):
        x, y = make_gap_mqar_batch(batch, P, Gcur, gen, P_max, V_max, F, key_lo, val_lo, fill_lo)
        logits, _ = model(x, None)                 # <<< full-sequence graph, no chunking
        pred = logits[:, -1, :V_max]
        loss = lossf(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        acc = float((pred.argmax(-1) == y).float().mean())
        good = good + 1 if acc > bar else 0
        if good >= patience and Gcur < Gmax:
            Gcur = min(Gmax, int(Gcur * 1.5) + 1)
            good = 0
        if log_every and (it + 1) % log_every == 0:
            print(f"    it {it+1:>4}/{iters}: loss {float(loss):.3f} acc {acc:.3f} (train-gap {Gcur})")
    return {"final_train_gap": Gcur, "final_train_acc": round(acc, 4)}


def train_chunked_v3(model, P, Gmax_train, iters, lr, seed, batch, chunk,
                     P_max, V_max, F, log_every=0, g_start=2, patience=25, bar=0.9):
    """T1 baseline: identical to holo_stream_recall.train_gap_curriculum
    (chunked training, gap capped at one chunk). Reproduced locally (rather
    than imported) only because the import would also need chunked_forward,
    which we already import; kept as a thin wrapper for a single call site
    that mirrors train_fullseq_curriculum's signature exactly."""
    key_lo, val_lo, fill_lo, _ = _gap_vocab(P_max, V_max, F)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    model.train()
    Gcur = min(g_start, Gmax_train) if Gmax_train > 0 else 0
    acc = 0.0
    good = 0
    for it in range(iters):
        x, y = make_gap_mqar_batch(batch, P, Gcur, gen, P_max, V_max, F, key_lo, val_lo, fill_lo)
        logits, _ = chunked_forward(model, x, chunk)
        pred = logits[:, -1, :V_max]
        loss = lossf(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        acc = float((pred.argmax(-1) == y).float().mean())
        good = good + 1 if acc > bar else 0
        if good >= patience and Gcur < Gmax_train:
            Gcur = min(Gmax_train, int(Gcur * 1.5) + 1)
            good = 0
        if log_every and (it + 1) % log_every == 0:
            print(f"    it {it+1:>4}/{iters}: loss {float(loss):.3f} acc {acc:.3f} (train-gap {Gcur})")
    return {"final_train_gap": Gcur, "final_train_acc": round(acc, 4)}


# ═══════════════════════════════════════════════════════════════════════════
# 3. Eval (CHUNKED+carried, the deployment path) -- identical machinery to
#    holo_stream_recall.eval_gap_recall, imported via chunked_forward.
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def eval_gap_recall_chunked(model, P, G, NB, seed, P_max, V_max, F, chunk, zero_at_gap=False):
    key_lo, val_lo, fill_lo, _ = _gap_vocab(P_max, V_max, F)
    gen = torch.Generator().manual_seed(seed)
    x, y = make_gap_mqar_batch(NB, P, G, gen, P_max, V_max, F, key_lo, val_lo, fill_lo)
    model.eval()
    kv_len = 2 * P

    if not zero_at_gap:
        logits, _ = chunked_forward(model, x, chunk)
    else:
        logits_kv, states = chunked_forward(model, x[:, :kv_len], chunk)
        states = model.zero_states(states)
        logits_rest, _ = chunked_forward(model, x[:, kv_len:], chunk)
        logits = torch.cat([logits_kv, logits_rest], dim=1)

    pred = logits[:, -1, :V_max]
    acc = float((pred.argmax(-1) == y).float().mean())
    return acc


# ═══════════════════════════════════════════════════════════════════════════
# 4. gamma-spectrum on FILLER tokens -- per-head mean gamma, per variant.
#    Idiom ported from streaming_train.gamma_spectrum, adapted to read
#    W_gamma off StreamingHolographicScanLayer / GammaKickstartScanLayer.
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def gamma_spectrum_fillers(model, P_max, V_max, F, seed=777, n_tok=2000):
    """Mean per-(layer,head) gamma over a batch of PURE FILLER tokens (the
    tokens that actually populate a gap) -- the quantity the theory's
    Gamma(G)=prod_gap gamma_t is built from. Returns list[layer][head]->float
    plus, for convenience, the max gamma seen at layer 0 (where the kickstart
    lives in T3) and the implied tau=1/(1-gamma)."""
    key_lo, val_lo, fill_lo, _ = _gap_vocab(P_max, V_max, F)
    gen = torch.Generator().manual_seed(seed)
    fillers = torch.randint(0, F, (1, n_tok), generator=gen) + fill_lo
    model.eval()
    h = model.embed(fillers)
    spec = []
    chan_max = 0.0
    for layer in model.layers:
        sc = layer.scan
        _, gamma = sc._drive_and_gamma(h)          # (1,T,H,D)
        g_head = gamma.mean(dim=(0, 1, 3))          # per-head mean gamma
        spec.append([round(float(v), 6) for v in g_head])
        # Pillar-E lesson ("the average hides the carrier"): the theory's Gamma(G)
        # is set by the single BEST channel, so also track the per-CHANNEL max
        # (mean over batch/time, max over head x channel), not just head means.
        chan_max = max(chan_max, float(gamma.mean(dim=(0, 1)).max()))
        y, _ = sc(h, None)
        h = layer.ln1(h + y)
        h = layer.ln2(h + layer.ffn(h))
    flat_max = max(v for layer_spec in spec for v in layer_spec)
    return {"per_layer_head": spec, "max_gamma": round(flat_max, 6),
            "tau_at_max_gamma": round(1.0 / max(1e-6, 1.0 - flat_max), 1),
            "max_channel_gamma": round(chan_max, 6),
            "tau_at_max_channel": round(1.0 / max(1e-6, 1.0 - chan_max), 1)}


# ═══════════════════════════════════════════════════════════════════════════
# 5. Orchestration: build 3 variants x seeds, train, sweep eval G, gamma-spectrum,
#    estimate knee, theory-check, write JSON.
# ═══════════════════════════════════════════════════════════════════════════
VARIANTS = ["T1_baseline_v3", "T2_fullseq_curriculum", "T3_fullseq_kickstart"]


def _build_variant(name, vocab_size, mask_idx, d_model, n_layers, n_heads, d_head):
    if name == "T3_fullseq_kickstart":
        return _build_lm_kickstart(vocab_size, mask_idx, d_model=d_model, n_layers=n_layers,
                                   n_heads=n_heads, d_head=d_head)
    return _build_lm(vocab_size, mask_idx, use_phase=True, d_model=d_model, n_layers=n_layers,
                     n_heads=n_heads, d_head=d_head)


def estimate_knee(sweep_by_G, Gs, ref_G=32):
    """Last G with acc >= 0.5 * acc@ref_G (per the mission spec). If ref_G is
    missing or its acc is ~0, knee is undefined (None)."""
    if ref_G not in sweep_by_G:
        return None
    ref_acc = sweep_by_G[ref_G]
    if ref_acc <= 1e-6:
        return None
    knee = None
    for G in sorted(Gs):
        if G in sweep_by_G and sweep_by_G[G] >= 0.5 * ref_acc:
            knee = G
    return knee


def run(args):
    P_max, V_max, F = args.p_max, args.v_max, args.f_fillers
    key_lo, val_lo, fill_lo, vocab_size = _gap_vocab(P_max, V_max, F)
    mask_idx = vocab_size
    P = args.pairs
    chance = 1.0 / V_max

    Gs_eval = [int(g) for g in args.gaps.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    zero_null_Gs = [int(g) for g in args.zero_null_gaps.split(",") if g]

    print("=" * 78)
    print("HOLO-GAP-KNEE — moving the gamma-knee via full-sequence gap-curriculum training")
    print(f"P={P}  P_max={P_max} V_max={V_max} F={F} vocab={vocab_size}  chance={chance:.4f}")
    print(f"eval gaps(G)={Gs_eval}  seeds={seeds}  chunk={args.chunk}  variants={VARIANTS}")
    print("=" * 78)

    print("\n── equivalence: full-sequence forward == chunked+carried forward (gate) ──")
    eq_on = check_equivalence(vocab_size, mask_idx, seed=0, T=48, chunk=16, use_phase=True)
    print(f"   use_phase=True   max|Δ| = {eq_on:.3e}")
    eq_ok = eq_on < 1e-5
    print(f"   {'PASS' if eq_ok else 'FAIL — do not trust anything below'}")
    if not eq_ok:
        out = {"config": vars(args), "equivalence": eq_on, "verdict": "VOID — equivalence check failed"}
        os.makedirs(RESULTS, exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"\n→ {args.out}")
        return

    t0 = time.time()
    results = {"config": vars(args), "equivalence": eq_on, "chance": chance,
              "kickstart_logit": KICKSTART_LOGIT, "kickstart_head": KICKSTART_HEAD,
              "curriculum": {}, "sweep": {}, "gamma_spectrum": {}, "knees": {}}

    for seed in seeds:
        for variant in args.variants_list:
            print(f"\n{'='*78}\nseed={seed}  variant={variant}\n{'='*78}")
            torch.manual_seed(seed)
            model = _build_variant(variant, vocab_size, mask_idx, args.d_model,
                                   args.n_layers, args.n_heads, args.d_head)

            if variant == "T1_baseline_v3":
                cap = max(2, args.chunk - 2 * P - 2)
                curr = train_chunked_v3(model, P, cap, args.iters, args.lr, seed, args.batch,
                                        args.chunk, P_max, V_max, F, log_every=args.log_every,
                                        g_start=args.g_start, patience=args.patience,
                                        bar=args.curriculum_bar)
                curr["train_mode"] = "chunked"
                curr["train_gap_cap"] = cap
            else:
                curr = train_fullseq_curriculum(model, P, args.g_train_max, args.iters, args.lr,
                                                seed, args.batch, P_max, V_max, F,
                                                log_every=args.log_every, g_start=args.g_start,
                                                patience=args.patience, bar=args.curriculum_bar)
                curr["train_mode"] = "fullseq"
                curr["train_gap_cap"] = args.g_train_max

            key = f"seed{seed}_{variant}"
            results["curriculum"][key] = curr
            print(f"  [{variant:24s}] curriculum done: final_train_gap={curr['final_train_gap']} "
                  f"final_train_acc={curr['final_train_acc']:.3f} ({curr['train_mode']})")

            spec = gamma_spectrum_fillers(model, P_max, V_max, F, seed=seed + 5000)
            results["gamma_spectrum"][key] = spec
            print(f"    filler γ-spectrum: max_gamma={spec['max_gamma']:.4f} "
                  f"(tau≈{spec['tau_at_max_gamma']:.1f} tok)  per_layer_head={spec['per_layer_head']}")

            by_G = {}
            for G in Gs_eval:
                eb = args.eval_batch if G < 1024 else max(10, args.eval_batch // 2)
                acc = eval_gap_recall_chunked(model, P, G, eb, seed + 1000 + G, P_max, V_max, F,
                                              args.chunk, zero_at_gap=False)
                by_G[G] = acc
                sk = f"{variant}|seed{seed}|G{G}"
                results["sweep"][sk] = {"seed": seed, "variant": variant, "P": P, "G": G,
                                        "accuracy": round(acc, 4), "chance": round(chance, 4),
                                        "beats_chance_3x": bool(acc > 3 * chance)}
                line = f"    G={G:>5}: acc={acc:.4f} (chance {chance:.4f})"
                if G in zero_null_Gs:
                    acc0 = eval_gap_recall_chunked(model, P, G, eb, seed + 1000 + G, P_max, V_max, F,
                                                   args.chunk, zero_at_gap=True)
                    sk0 = f"{variant}|seed{seed}|G{G}|zeroed"
                    results["sweep"][sk0] = {"seed": seed, "variant": variant, "P": P, "G": G,
                                             "accuracy": round(acc0, 4), "arm": "zeroed_at_gap"}
                    line += f"   zeroed-null={acc0:.4f}"
                print(line)

            ref_G = 32 if 32 in Gs_eval else min(Gs_eval)
            knee = estimate_knee(by_G, Gs_eval, ref_G=ref_G)
            results["knees"][key] = {"knee_G": knee, "ref_G": ref_G,
                                     "acc_at_ref": round(by_G.get(ref_G, float("nan")), 4)}
            print(f"    knee estimate (last G with acc >= 0.5*acc@{ref_G}): {knee}")

    # ── theory check: knee vs 1/(1-gamma) of the best filler-gamma head, per variant ──
    def _mean_knee(variant):
        ks = [v["knee_G"] for k, v in results["knees"].items() if k.split("_", 1)[1] == variant
              and v["knee_G"] is not None]
        return sum(ks) / len(ks) if ks else None

    def _mean_max_gamma(variant):
        gs = [v["max_gamma"] for k, v in results["gamma_spectrum"].items()
              if k.split("_", 1)[1] == variant]
        return sum(gs) / len(gs) if gs else None

    theory_lines = []
    variant_knees = {}
    for variant in args.variants_list:
        mk = _mean_knee(variant)
        mg = _mean_max_gamma(variant)
        variant_knees[variant] = mk
        if mg is not None:
            tau = 1.0 / max(1e-6, 1.0 - mg)
            theory_lines.append(f"{variant}: mean_knee_G={mk}  max_filler_gamma={mg:.4f}  "
                                f"tau=1/(1-gamma)={tau:.1f}")
        else:
            theory_lines.append(f"{variant}: mean_knee_G={mk}  max_filler_gamma=n/a")

    results["theory_check"] = theory_lines

    p1 = variant_knees.get("T2_fullseq_curriculum")
    p2 = variant_knees.get("T3_fullseq_kickstart")
    p1_pass = p1 is not None and p1 >= 256
    p2_pass = p2 is not None and p2 >= 1024
    knee_order = [v for v in args.variants_list if variant_knees.get(v) is not None]
    knee_order_sorted = sorted(knee_order, key=lambda v: variant_knees[v])
    gamma_order_sorted = sorted(knee_order, key=lambda v: _mean_max_gamma(v) or 0.0)
    p3_pass = knee_order_sorted == gamma_order_sorted and len(knee_order_sorted) >= 2

    verdict_bits = []
    verdict_bits.append(f"P1 (T2 knee>=256): {'CONFIRMED' if p1_pass else 'NOT MET'} (measured {p1})")
    verdict_bits.append(f"P2 (T3 knee>=1024): {'CONFIRMED' if p2_pass else 'NOT MET'} (measured {p2})")
    verdict_bits.append(f"P3 (knee order tracks filler-gamma order): "
                        f"{'CONFIRMED' if p3_pass else 'NOT MET'} "
                        f"(knee order {knee_order_sorted}, gamma order {gamma_order_sorted})")
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
                    help="fast sanity: 1 seed, T1+T2 only, 800 iters, eval G up to 256")
    ap.add_argument("--full", action="store_true",
                    help="the full sweep: 2 seeds, T1+T2+T3, 3000 iters, eval G up to 2048")
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
    ap.add_argument("--curriculum-bar", type=float, default=0.9,
                    help="acc bar the curriculum must sustain before growing the gap (0.9 never fires for P=2, which consolidates ~0.85 — the v-full T1==T2 diagnosis)")
    ap.add_argument("--g-train-max", type=int, default=128,
                    help="max TRAINING gap for T2/T3 full-sequence curriculum")
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--gaps", default="32,128,256,512,1024,2048", help="comma list of EVAL gaps")
    ap.add_argument("--zero-null-gaps", default="128,512",
                    help="comma list of gaps to also run the zeroed-at-gap decisive null")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--variants", default="T1_baseline_v3,T2_fullseq_curriculum,T3_fullseq_kickstart")
    ap.add_argument("--out", default=os.path.join(RESULTS, "holo_knee.json"))
    args = ap.parse_args()

    if args.smoke:
        args.seeds = "0"
        args.variants = "T1_baseline_v3,T2_fullseq_curriculum"
        args.iters = min(args.iters, 800) if args.iters == 3000 else args.iters
        args.gaps = "32,128,256"
        args.g_train_max = min(args.g_train_max, 128)
        args.eval_batch = min(args.eval_batch, 60)
        if args.out == os.path.join(RESULTS, "holo_knee.json"):
            args.out = os.path.join(RESULTS, "holo_knee_smoke.json")
    elif args.full:
        args.seeds = "0,1"
        args.variants = "T1_baseline_v3,T2_fullseq_curriculum,T3_fullseq_kickstart"
        args.iters = max(args.iters, 3000)
        args.gaps = "32,128,256,512,1024,2048"

    args.variants_list = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in args.variants_list:
        assert v in VARIANTS, f"unknown variant {v}"

    run(args)


if __name__ == "__main__":
    main()
