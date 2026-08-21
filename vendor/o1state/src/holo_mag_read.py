#!/usr/bin/env python3 -u
"""
HOLO-MAG-READ — the magnitude-normalized read: pushing the gamma-knee past
the rms-readout's ~30% magnitude-loss tolerance, not by growing gamma further.
======================================================================================
MISSION (P14, registered): analysis/HOLO_STREAM_VERDICT.md's M2 section closes
with a relocated blocker. holo_gap_knee.py already proved carrier channels at
tau ~= 160-700 tokens exist in every trained variant (the per-CHANNEL gamma
metric, "the average hides the carrier"), yet the measured knee sat at G=256 --
an order of magnitude below what those tau values license. The diagnosis: the
rms READOUT (StreamingHolographicScanLayer.forward, readout="rms") normalizes
the DE-ROTATED read by its own rms, so in principle a read that has decayed to
30% of its write-time magnitude still produces a full-scale unit-rms output --
except the SIGN/structure of that read is dominated by whichever noise floor
sits under the signal once |S| has shrunk enough that the signal-to-floor
ratio crosses some critical margin. G* = tau * ln(S0/S_min) with a "thin
margin" (per HOLO_STREAM_VERDICT.md) is exactly a statement that the rms-read
pipeline, AS WRITTEN, only tolerates ~30% magnitude loss before the argmax
flips -- nowhere near the full tau-scale decay.

THE FIX THIS FILE TESTS: analysis/HOLO_CARRIER_THEORY.md Sec.1 guarantees the
PHASE of the complex accumulator is exact after ANY gap (nothing rotates a
pure-decay filler drive) -- only the MAGNITUDE |S| shrinks by Gamma(G) =
prod_gap gamma_t. If we renormalize |S| to unit magnitude PER CHANNEL before
de-rotating, the de-rotated read should carry the SAME angular information at
G=4096 that it carried at G=32, because normalizing a decayed-but-not-rotated
vector to unit length recovers the direction exactly (up to whatever noise
was added ON TOP of the decayed signal by the filler drive itself -- a
separate, additive source of degradation the theory does NOT promise away,
and which this file measures rather than assumes).

WHERE THE NORMALIZATION GOES (read off StreamingHolographicScanLayer.forward,
holo_stream_recall.py lines ~141-205): the ordinary path computes
  read_re = S_re*cos(phi_r) + S_im*sin(phi_r)
  read_im = S_im*cos(phi_r) - S_re*cos(phi_r)          <- de-rotation FIRST
  [rms-normalize read_re, read_im]                      <- THEN rms-readout
i.e. the existing rms-readout normalizes AFTER de-rotation, over the
d_head axis (jointly across the channel dimension) -- not per-channel, and not
before the rotation. MagNorm changes what happens BEFORE de-rotation instead:
  mag = sqrt(S_re**2 + S_im**2 + eps)     <- PER CHANNEL, PER (B,T,H,D) element
  S_re_n = S_re / mag ; S_im_n = S_im / mag
  read_re = S_re_n*cos(phi_r) + S_im_n*sin(phi_r)        <- de-rotate the NORMALIZED state
  read_im = S_im_n*cos(phi_r) - S_re_n*sin(phi_r)
  [existing rms-readout, unchanged, applied to read_re/read_im as before]
Only 3 lines change relative to the parent's use_phase=True/n_slots==1 branch
(see MagNormScanLayer.forward docstring for the literal diff). CRITICALLY:
the CARRIED state (Sre_fin, Sim_fin threaded to state_out, and consumed by
stateful_linear_scan on the NEXT chunk's write) is NEVER touched by this
normalization -- only the local READ variables inside this forward call are
renormalized. The accumulation recurrence S_t = gamma_t*S_{t-1} + drive_t
still needs the real (unnormalized) magnitude of S_{t-1} as the interference
weight when a new write lands on an already-occupied channel; normalizing the
CARRY would silently discard exactly the amplitude information the write-phase
superposition relies on to combine multiple keys. This file's docstring
promise: state_out is bit-identical to the un-normalized parent at every call
site; only the four local read_re/read_im/S_re_n/S_im_n temporaries differ.

THREE VARIANTS (P=2, same recipe as holo_gap_knee.py -- fullseq training,
patience curriculum, bar=0.8, g_train_max=128):
  V1 = T3 reference from holo_gap_knee.py (Kickstart + ordinary rms-read),
       REPRODUCED here as the control (same build path: GammaKickstartScanLayer
       composed with the ordinary StreamingHolographicScanLayer.forward).
  V2 = Kickstart + mag_norm_read=True (the fix under test: does normalizing
       the read BEFORE de-rotation move the knee out toward the tau-scale?).
  V3 = mag_norm_read=True WITHOUT the kickstart (isolates whether the read
       fix alone -- no gamma-shaping help -- already buys most of the gain,
       or whether it needs a grown carrier gamma to have anything to read).

EQUIVALENCE (mandatory gate, ported from holo_stream_recall.check_equivalence
/ holo_gap_knee's re-use of it): full-sequence forward(x, None) must equal
chunked+carried forward (chunk=16, detach at boundaries) for MagNorm too --
the normalization is a pointwise function of the SAME per-step complex state
either forward path produces, so the equality is expected to hold at the same
tolerance (<1e-5) as the un-normalized layer; this file re-derives it rather
than assuming it transfers.

EVAL (chunked+carried, the deployment path, identical machinery to
holo_gap_knee.py): G in {32,128,256,512,1024,2048,4096}, zeroed-at-gap null at
G in {128,1024}, 2 seeds, eval_batch 100 (50 for G>=1024, 25 for G=4096).
Per-channel gamma-spectrum (imported from holo_gap_knee.gamma_spectrum_fillers)
so each variant's measured knee can be checked against its OWN carrier tau,
not just the fixed constant from the M2 write-up.

PREDICTIONS (P14, registered before running anything below):
  P14a: V2 knee >= 1024 (mag-norm read unlocks the tau~700 channel's full
        decay budget; the M2 blocker -- readout margin, not gamma -- is real
        and fixable by this specific mechanism).
  P14b: V3 knee also improves materially over V1 (>=512) even WITHOUT the
        kickstart -- i.e. the read fix is not merely additive with a grown
        gamma; a modest carrier already benefits once the read stops wasting
        most of its margin on magnitude decay.
  P14c: V3's short-gap (G=32) accuracy is NOT meaningfully below V1's --
        i.e. normalizing the read does not destroy near-field recall by
        discarding a magnitude cue the un-normalized rms-read was secretly
        using. If V3 @ G=32 drops well below V1 @ G=32, that is reported
        honestly as the cost side of this fix, not hidden.

CPU-only (mps disabled), torch.set_num_threads(1), os.nice(19) at script
start. Results -> results/holo_magread_*.json only. Does not modify
holo_stream_recall.py, holo_gap_knee.py, or holographic_gssm.py.
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
from holo_gap_knee import (   # noqa: E402 -- reuse the kickstart + fullseq-curriculum machinery, unmodified
    GammaKickstartScanLayer, KICKSTART_LOGIT, KICKSTART_HEAD,
    train_fullseq_curriculum, gamma_spectrum_fillers, estimate_knee,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(REPO, "results")


# ═══════════════════════════════════════════════════════════════════════════
# 1. MagNormScanLayer — per-channel magnitude-normalized READ, state untouched.
#    Mixed in via multiple inheritance so it composes with GammaKickstartScanLayer
#    (V2 = kickstart + mag-norm) as well as standing alone (V3 = mag-norm only).
# ═══════════════════════════════════════════════════════════════════════════
class MagNormReadMixin:
    """Overrides forward's use_phase=True / n_slots==1 branch to renormalize
    the complex accumulator to PER-CHANNEL unit magnitude before de-rotating,
    i.e. before computing read_re/read_im. Everything else (drive/gamma
    computation, the stateful_linear_scan write recurrence, state_out, the
    n_slots>1 branch, the tanh_m/rms/layernorm readout AFTER the read) is a
    byte-for-byte copy of StreamingHolographicScanLayer.forward.

    THE EXACT 3 CHANGED LINES relative to the parent (n_slots==1 branch):
      parent:
        read_re = S_re * torch.cos(phi_r) + S_im * torch.sin(phi_r)
        read_im = S_im * torch.cos(phi_r) - S_re * torch.sin(phi_r)
      here (mag_norm_read=True):
        mag = torch.sqrt(S_re * S_re + S_im * S_im + self._eps())      # NEW line
        S_re_n, S_im_n = S_re / mag, S_im / mag                        # NEW line
        read_re = S_re_n * torch.cos(phi_r) + S_im_n * torch.sin(phi_r)  # S_re->S_re_n, S_im->S_im_n
        read_im = S_im_n * torch.cos(phi_r) - S_re_n * torch.sin(phi_r)  # S_im->S_im_n, S_re->S_re_n
    i.e. 2 new lines (mag, and the S_re_n/S_im_n normalization) plus the
    substitution of S_re/S_im -> S_re_n/S_im_n inside the existing 2
    de-rotation lines. The CARRIED state_out = {"S_re": Sre_fin, "S_im":
    Sim_fin} is built from Sre_fin/Sim_fin exactly as in the parent -- those
    come out of stateful_linear_scan BEFORE this method ever computes mag/
    S_re_n/S_im_n, so normalization cannot leak into what crosses a chunk
    boundary. n_slots>1 is NOT covered by mag_norm_read (this mission is
    P=2, n_slots=1 throughout, matching holo_gap_knee.py); if mag_norm_read
    is set on an n_slots>1 layer this raises, rather than silently reading
    un-normalized.
    """
    mag_norm_read = False   # class default; True on the instances built below

    def forward(self, x: torch.Tensor, state_in=None, return_internals: bool = False):
        if not self.mag_norm_read or not self.use_phase:
            return super().forward(x, state_in=state_in, return_internals=return_internals)

        B, T, _ = x.shape
        state_in = state_in or {}
        if self.n_slots > 1:
            raise NotImplementedError("mag_norm_read is only implemented for n_slots==1 "
                                       "(this mission's P=2 recipe never uses slots>1)")

        a, gamma = self._drive_and_gamma(x)
        phi_w = self.phase_scale * torch.tanh(self.W_key(x))
        phi_w = phi_w.view(B, T, self.n_heads, self.d_head)
        if self.separate_qk:
            phi_r = self.phase_scale * torch.tanh(self.W_read_key(x))
            phi_r = phi_r.view(B, T, self.n_heads, self.d_head)
        else:
            phi_r = phi_w

        drive_re = a * torch.cos(phi_w)
        drive_im = a * torch.sin(phi_w)

        Sre_all, Sre_fin = stateful_linear_scan(drive_re, gamma, state_in.get("S_re"))
        Sim_all, Sim_fin = stateful_linear_scan(drive_im, gamma, state_in.get("S_im"))
        S_re, S_im = Sre_all, Sim_all   # UNNORMALIZED — this is what gets carried

        # ─── the 3 changed lines (see docstring): per-channel magnitude-normalized
        # read, computed on LOCAL temporaries only, never on Sre_fin/Sim_fin/state_out ───
        mag = torch.sqrt(S_re * S_re + S_im * S_im + self._eps())
        S_re_n, S_im_n = S_re / mag, S_im / mag
        read_re = S_re_n * torch.cos(phi_r) + S_im_n * torch.sin(phi_r)
        read_im = S_im_n * torch.cos(phi_r) - S_re_n * torch.sin(phi_r)
        # ─── end changed block; everything below is the parent's unmodified tail ───

        state_out = {"S_re": Sre_fin, "S_im": Sim_fin}

        if self.readout == "tanh_m":
            Z_seq, Z_fin = stateful_linear_scan(a, gamma, state_in.get("Z_mag"))
            s_sq = torch.clamp(1.0 - torch.exp(Z_seq), min=0.0)
            m = torch.sqrt(s_sq + self._eps())
            read_re = m * torch.tanh(read_re)
            read_im = m * torch.tanh(read_im)
            state_out["Z_mag"] = Z_fin
        elif self.readout == "rms":
            rms_re = read_re.pow(2).mean(dim=-1, keepdim=True).add(self._eps()).sqrt()
            rms_im = read_im.pow(2).mean(dim=-1, keepdim=True).add(self._eps()).sqrt()
            read_re = read_re / rms_re
            read_im = read_im / rms_im
        # "layernorm": pass raw (normalized-read) through, no extra state.

        read_re = read_re.view(B, T, self.n_heads * self.d_head)
        read_im = read_im.view(B, T, self.n_heads * self.d_head)
        out = self.W_out(read_re) + self.W_im(read_im)

        if return_internals:
            internals = {"S_re": S_re, "S_im": S_im, "phi_w": phi_w, "phi_r": phi_r,
                         "S_re_n": S_re_n, "S_im_n": S_im_n}
            return out, state_out, internals
        return out, state_out


class MagNormScanLayer(MagNormReadMixin, StreamingHolographicScanLayer):
    """StreamingHolographicScanLayer + magnitude-normalized read. Used standalone
    for V3 (mag-norm, no kickstart)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mag_norm_read = True


class MagNormKickstartScanLayer(MagNormReadMixin, GammaKickstartScanLayer):
    """GammaKickstartScanLayer + magnitude-normalized read. MRO puts
    MagNormReadMixin.forward first (uses super() to fall through to
    GammaKickstartScanLayer's inherited _drive_and_gamma/forward chain when
    mag_norm_read is False or use_phase is False), so head 0's fixed gamma
    offset and the mag-norm read compose rather than compete. Used for V2."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mag_norm_read = True


def _build_lm_variant(name, vocab_size, mask_idx, d_model, n_layers, n_heads, d_head):
    """Constructs V1/V2/V3. V1 is byte-for-byte holo_gap_knee._build_lm_kickstart
    (imported logic reproduced here only because that function is module-level
    there and hardcodes GammaKickstartScanLayer; we need the SAME construction
    but swapping in MagNormKickstartScanLayer for V2). All three share seeding
    discipline: torch.manual_seed(seed) is called by the caller BEFORE this
    function runs, exactly as holo_gap_knee.py's run() does, so the donor
    model's xavier init (before the scan-layer swap) is identical across
    variants for a given seed."""
    if name == "V1_kickstart_rmsread":
        model = StreamingHolographicLM(
            vocab_size, mask_idx, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            d_head=d_head, seq_len=32, dropout=0.0, causal=True,
            phase_scale=math.pi, use_phase=True, readout="rms",
            separate_qk=False, n_slots=1)
        old = model.layers[0].scan
        swapped = GammaKickstartScanLayer(
            old.d_model, d_head=old.d_head, n_heads=old.n_heads, causal=old.causal,
            dropout=0.0, phase_scale=old.phase_scale, use_phase=old.use_phase,
            readout=old.readout, separate_qk=old.separate_qk, n_slots=old.n_slots)
        swapped.load_state_dict(old.state_dict(), strict=False)
        model.layers[0].scan = swapped
        return model

    if name == "V2_kickstart_magnorm":
        model = StreamingHolographicLM(
            vocab_size, mask_idx, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            d_head=d_head, seq_len=32, dropout=0.0, causal=True,
            phase_scale=math.pi, use_phase=True, readout="rms",
            separate_qk=False, n_slots=1)
        old = model.layers[0].scan
        swapped = MagNormKickstartScanLayer(
            old.d_model, d_head=old.d_head, n_heads=old.n_heads, causal=old.causal,
            dropout=0.0, phase_scale=old.phase_scale, use_phase=old.use_phase,
            readout=old.readout, separate_qk=old.separate_qk, n_slots=old.n_slots)
        swapped.load_state_dict(old.state_dict(), strict=False)
        model.layers[0].scan = swapped
        return model

    if name == "V3_magnorm_only":
        model = _build_lm(vocab_size, mask_idx, use_phase=True, d_model=d_model,
                          n_layers=n_layers, n_heads=n_heads, d_head=d_head)
        old = model.layers[0].scan
        swapped = MagNormScanLayer(
            old.d_model, d_head=old.d_head, n_heads=old.n_heads, causal=old.causal,
            dropout=0.0, phase_scale=old.phase_scale, use_phase=old.use_phase,
            readout=old.readout, separate_qk=old.separate_qk, n_slots=old.n_slots)
        swapped.load_state_dict(old.state_dict(), strict=False)
        model.layers[0].scan = swapped
        return model

    raise ValueError(f"unknown variant {name}")


VARIANTS = ["V1_kickstart_rmsread", "V2_kickstart_magnorm", "V3_magnorm_only"]


# ═══════════════════════════════════════════════════════════════════════════
# 2. Equivalence check for MagNorm — full-sequence forward(x, None) must equal
#    chunked+carried forward at the same <1e-5 tolerance the un-normalized
#    layer holds. Ported from holo_stream_recall.check_equivalence, but builds
#    a MagNormScanLayer-swapped model instead of calling _build_lm directly.
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def check_equivalence_magnorm(vocab_size, mask_idx, seed=0, T=48, chunk=16, d_model=64,
                              n_layers=2, n_heads=4, d_head=16):
    torch.manual_seed(seed)
    model = _build_lm_variant("V3_magnorm_only", vocab_size, mask_idx, d_model, n_layers,
                              n_heads, d_head).eval()
    x = torch.randint(0, vocab_size, (2, T))

    out_full, _ = model(x, None)

    states = None
    outs = []
    pos = 0
    while pos < T:
        hi = min(T, pos + chunk)
        xc = x[:, pos:hi]
        logits_c, states = model(xc, states)
        states = [{k: v.detach() for k, v in st.items()} for st in states]
        outs.append(logits_c)
        pos = hi
    out_chunked = torch.cat(outs, dim=1)

    delta = float((out_full - out_chunked).abs().max())
    return delta


# ═══════════════════════════════════════════════════════════════════════════
# 3. Eval (CHUNKED+carried, the deployment path) — identical machinery to
#    holo_gap_knee.eval_gap_recall_chunked, reproduced here for a local import
#    surface (no change to the underlying chunked_forward/model.zero_states).
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
# 4. Orchestration: build 3 variants x seeds, train (fullseq curriculum, same
#    recipe as holo_gap_knee.py's T3), sweep eval G, gamma-spectrum, knee,
#    theory-check (knee vs tau_max_channel per variant), write JSON.
# ═══════════════════════════════════════════════════════════════════════════
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
    print("HOLO-MAG-READ — per-channel magnitude-normalized read, pushing the knee toward tau")
    print(f"P={P}  P_max={P_max} V_max={V_max} F={F} vocab={vocab_size}  chance={chance:.4f}")
    print(f"eval gaps(G)={Gs_eval}  seeds={seeds}  chunk={args.chunk}  variants={args.variants_list}")
    print("=" * 78)

    print("\n── equivalence (un-normalized reference, gate) ──")
    eq_ref = check_equivalence(vocab_size, mask_idx, seed=0, T=48, chunk=16, use_phase=True)
    print(f"   StreamingHolographicScanLayer  max|Δ| = {eq_ref:.3e}")

    print("── equivalence (MagNorm read, the new gate) ──")
    eq_magnorm = check_equivalence_magnorm(vocab_size, mask_idx, seed=0, T=48, chunk=args.chunk,
                                           d_model=args.d_model, n_layers=args.n_layers,
                                           n_heads=args.n_heads, d_head=args.d_head)
    print(f"   MagNormScanLayer               max|Δ| = {eq_magnorm:.3e}")

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
              "kickstart_logit": KICKSTART_LOGIT, "kickstart_head": KICKSTART_HEAD,
              "curriculum": {}, "sweep": {}, "gamma_spectrum": {}, "knees": {}}

    for seed in seeds:
        for variant in args.variants_list:
            print(f"\n{'='*78}\nseed={seed}  variant={variant}\n{'='*78}")
            torch.manual_seed(seed)
            model = _build_lm_variant(variant, vocab_size, mask_idx, args.d_model,
                                      args.n_layers, args.n_heads, args.d_head)

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
                  f"(tau≈{spec['tau_at_max_gamma']:.1f} tok)  max_channel_gamma={spec['max_channel_gamma']:.4f} "
                  f"(tau_channel≈{spec['tau_at_max_channel']:.1f} tok)")

            by_G = {}
            for G in Gs_eval:
                if G >= 4096:
                    eb = max(10, args.eval_batch // 4)
                elif G >= 1024:
                    eb = max(10, args.eval_batch // 2)
                else:
                    eb = args.eval_batch
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

    # ── theory check: knee vs tau=1/(1-gamma) of the MAX-CHANNEL filler-gamma, per variant ──
    def _mean_knee(variant):
        ks = [v["knee_G"] for k, v in results["knees"].items() if k.split("_", 1)[1] == variant
              and v["knee_G"] is not None]
        return sum(ks) / len(ks) if ks else None

    def _mean_tau_channel(variant):
        ts = [v["tau_at_max_channel"] for k, v in results["gamma_spectrum"].items()
              if k.split("_", 1)[1] == variant]
        return sum(ts) / len(ts) if ts else None

    def _acc_at(variant, G):
        vals = [v["accuracy"] for k, v in results["sweep"].items()
                if v["variant"] == variant and v["G"] == G and "zeroed" not in k]
        return sum(vals) / len(vals) if vals else None

    theory_lines = []
    variant_knees = {}
    for variant in args.variants_list:
        mk = _mean_knee(variant)
        mt = _mean_tau_channel(variant)
        variant_knees[variant] = mk
        theory_lines.append(f"{variant}: mean_knee_G={mk}  mean_tau_max_channel={mt}  "
                            f"knee/tau_ratio={round(mk / mt, 3) if (mk and mt) else 'n/a'}")

    results["theory_check"] = theory_lines

    v1_knee = variant_knees.get("V1_kickstart_rmsread")
    v2_knee = variant_knees.get("V2_kickstart_magnorm")
    v3_knee = variant_knees.get("V3_magnorm_only")
    v1_g32 = _acc_at("V1_kickstart_rmsread", 32)
    v3_g32 = _acc_at("V3_magnorm_only", 32)

    p14a_pass = v2_knee is not None and v2_knee >= 1024
    p14b_pass = v3_knee is not None and v3_knee >= 512
    p14c_pass = (v1_g32 is not None and v3_g32 is not None and
                v3_g32 >= v1_g32 - 0.10)   # "not meaningfully below": within 10pp

    verdict_bits = []
    verdict_bits.append(f"P14a (V2 knee>=1024): {'CONFIRMED' if p14a_pass else 'NOT MET'} "
                        f"(measured {v2_knee}, V1 reference knee {v1_knee})")
    verdict_bits.append(f"P14b (V3 knee>=512, no kickstart needed): "
                        f"{'CONFIRMED' if p14b_pass else 'NOT MET'} (measured {v3_knee})")
    verdict_bits.append(f"P14c (V3 @G=32 within 10pp of V1 @G=32, i.e. mag-norm doesn't cost "
                        f"near-field recall): {'CONFIRMED' if p14c_pass else 'NOT MET'} "
                        f"(V1={v1_g32}, V3={v3_g32})")
    if v3_g32 is not None and v1_g32 is not None and v3_g32 < v1_g32 - 0.10:
        verdict_bits.append(f"HONEST FLAG: mag-norm read costs {round((v1_g32 - v3_g32) * 100, 1)}pp "
                            f"of accuracy at G=32 relative to the un-normalized rms-read reference "
                            f"— the write-time magnitude cue the un-normalized read exploits is "
                            f"partially lost once the read is forced to unit-magnitude-per-channel.")
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
                    help="fast sanity: 1 seed, V2 only, 800 iters, eval G up to 512")
    ap.add_argument("--full", action="store_true",
                    help="the full sweep: 2 seeds, V1+V2+V3, 3000 iters, eval G up to 4096")
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
    ap.add_argument("--curriculum-bar", type=float, default=0.8,
                    help="acc bar the curriculum must sustain before growing the gap (0.8 -- "
                         "the bar holo_gap_knee.py found necessary to make the full-seq mechanism "
                         "ignite for P=2; 0.9 never fires)")
    ap.add_argument("--g-train-max", type=int, default=128,
                    help="max TRAINING gap for the full-sequence curriculum")
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--gaps", default="32,128,256,512,1024,2048,4096", help="comma list of EVAL gaps")
    ap.add_argument("--zero-null-gaps", default="128,1024",
                    help="comma list of gaps to also run the zeroed-at-gap decisive null")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--variants", default="V1_kickstart_rmsread,V2_kickstart_magnorm,V3_magnorm_only")
    ap.add_argument("--out", default=os.path.join(RESULTS, "holo_magread.json"))
    args = ap.parse_args()

    if args.smoke:
        args.seeds = "0"
        args.variants = "V2_kickstart_magnorm"
        args.iters = min(args.iters, 800) if args.iters == 3000 else args.iters
        args.gaps = "32,128,256,512"
        args.g_train_max = min(args.g_train_max, 128)
        args.eval_batch = min(args.eval_batch, 60)
        args.zero_null_gaps = "128"
        if args.out == os.path.join(RESULTS, "holo_magread.json"):
            args.out = os.path.join(RESULTS, "holo_magread_smoke.json")
    elif args.full:
        args.seeds = "0,1"
        args.variants = "V1_kickstart_rmsread,V2_kickstart_magnorm,V3_magnorm_only"
        args.iters = max(args.iters, 3000)
        args.gaps = "32,128,256,512,1024,2048,4096"

    args.variants_list = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in args.variants_list:
        assert v in VARIANTS, f"unknown variant {v}"

    run(args)


if __name__ == "__main__":
    main()
