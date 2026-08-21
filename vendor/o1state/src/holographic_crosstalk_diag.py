#!/usr/bin/env python3 -u
"""
Holographic crosstalk diagnostic — is interference the recall cap?  — by Opus 4.8
=================================================================================

CONTEXT (verified upstream).  The key-conditioned holographic write
(src/holographic_gssm.py, readout='tanh_m', shared-QK) broke the 14% scalar-recall
wall and plateaus at ~7-9% MQAR recall (5-seed: 8.89%±1.86%).  Adding channels
(d_head 32->64->96) is FLAT; separate-QK is flat/worse.  The standing HYPOTHESIS is
that the cap is the classic HRR/VSA holographic-memory CROSSTALK limit:

    read at query q  =  Re( S · e^{-iφ_q} )  =  Σ_k γ_{k→t} u_k cos(φ_k − φ_q)

The matched key (φ_k ≈ φ_q) gives cos ≈ 1 → coherent SIGNAL.  The N−1 mismatched
keys give cos(φ_j − φ_q) which does NOT average to exactly zero for finite N — the
residual INTERFERENCE grows with the number of superposed pairs.  More channels do
not fix it (each channel carries the same superposition); FEWER superposed pairs do.

THIS SCRIPT measures two things and writes JSON:

  (1) RECALL vs n_pairs ∈ {1,2,3,4,6,8,12}, multi-seed.  Train a fresh tanh_m
      holographic model at each load, measure MQAR recall.  SMOKING GUN: recall is
      HIGH at n_pairs=1-2 and DECAYS toward chance (1/n_values) as n_pairs grows ⇒
      crosstalk IS the cap.  (If recall were flat in n_pairs, the cap would be
      something else — a binding/readout failure, not superposition interference.)

  (2) SIGNAL-to-INTERFERENCE ratio.  On a trained model, at every query position,
      decompose the actual holographic read into:
          signal_q       = γ u_{k*} cos(φ_{k*} − φ_q)      (the matched key k*)
          interference_q = Σ_{j≠k*} γ u_j cos(φ_j − φ_q)   (all mismatched keys)
      We recover the per-source-token contributions exactly by re-running the leaky
      complex scan with a one-hot source mask (no approximation).  We then report
      S/I = mean |signal| / mean |interference| per channel/head, aggregated.
      If interference ≈ signal, that magnitude IS the wall.

CPU-deterministic.  Multi-seed MANDATORY (single seed = noise on a chance-flat loss).
Offline, self-contained.  Reference: Foss 2026.
"""

import os
import sys
import json
import math
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "reference"))
sys.path.insert(0, HERE)

from holographic_gssm import HolographicLM, sequential_linear_scan  # noqa: E402
from mqar import make_mqar_batch, mqar_accuracy, mqar_train          # noqa: E402

DEVICE = torch.device("cpu")          # determinism on chance-flat loss

# ── fixed MQAR vocabulary (matches the upstream 7-9% runs) ──
N_KEYS = N_VALUES = 64
VOCAB = N_KEYS + N_VALUES + 1
MASK_IDX = VOCAB
CHANCE = 1.0 / N_VALUES               # 1.5625% — recall floor

# ── load sweep ──
N_PAIRS_GRID = [1, 2, 3, 4, 6, 8, 12]
SEEDS = [0, 1, 2]                     # >=3 seeds (chance-flat loss → single seed is noise)
TRAIN_LEN = 64
TRAIN_STEPS = 3000                    # same budget as the upstream full runs
LR = 3e-3
BATCH = 32


# ===========================================================================
# Part 1 — RECALL vs n_pairs
# ===========================================================================

def make_model():
    """Fresh working holographic model: tanh_m readout, shared-QK (the 7-9% config)."""
    return HolographicLM(
        VOCAB, MASK_IDX, seq_len=TRAIN_LEN,
        readout="tanh_m", use_phase=True, separate_qk=False,
    )


def train_and_eval(n_pairs, seed):
    """Train one fresh holographic model at this load, return train-len recall."""
    torch.manual_seed(seed)
    cfg = dict(batch_size=BATCH, seq_len=TRAIN_LEN, n_pairs=n_pairs,
               n_queries=n_pairs, n_keys=N_KEYS, n_values=N_VALUES)
    model = make_model().to(DEVICE)
    mqar_train(model, cfg, steps=TRAIN_STEPS, lr=LR, seed=seed,
               device=DEVICE, log_every=0)
    overall, _, _ = mqar_accuracy(model, cfg, n_batches=8, seed=seed + 100,
                                  device=DEVICE)
    return overall, model, cfg


def recall_vs_npairs():
    print("=" * 74)
    print("PART 1 — recall vs n_pairs (fresh tanh_m holographic model per load)")
    print(f"   chance = {CHANCE*100:.3f}%   seeds = {SEEDS}   steps = {TRAIN_STEPS}")
    print("=" * 74)
    curve = {}
    trained = {}        # keep one trained model per n_pairs for Part 2
    for n_pairs in N_PAIRS_GRID:
        accs = []
        t0 = time.time()
        for seed in SEEDS:
            acc, model, cfg = train_and_eval(n_pairs, seed)
            accs.append(acc)
            if seed == SEEDS[0]:
                trained[n_pairs] = (model, cfg)
            print(f"   n_pairs={n_pairs:2d} seed={seed}  recall={acc*100:6.2f}%")
        accs_t = torch.tensor(accs)
        mean = accs_t.mean().item()
        std = accs_t.std(unbiased=False).item()
        lift = (mean - CHANCE) * 100
        curve[n_pairs] = {
            "mean_recall_pct": round(mean * 100, 3),
            "std_recall_pct": round(std * 100, 3),
            "lift_over_chance_pp": round(lift, 3),
            "per_seed_pct": [round(a * 100, 3) for a in accs],
            "n_seeds": len(SEEDS),
        }
        dt = time.time() - t0
        print(f"   -> n_pairs={n_pairs:2d}  {mean*100:6.2f}% ± {std*100:5.2f}%  "
              f"(+{lift:5.2f}pp over chance)   [{dt:5.1f}s]\n")
    return curve, trained


# ===========================================================================
# Part 2 — SIGNAL / INTERFERENCE decomposition
# ===========================================================================

@torch.no_grad()
def extract_scan_tensors(scan, x):
    """Re-derive the exact holographic intermediates for one HolographicScanLayer.

    Returns, all (B,T,H,D):
        a       : the bounded value drive u_t (=a_t ≤ 0)
        gamma   : forget γ_t
        phi_w   : write key angle φ_write_t
        phi_r   : read/query angle φ_read_t (== phi_w for shared-QK)
        S_re,S_im : the full accumulated complex state (causal leaky scan)
    Mirrors HolographicScanLayer.forward exactly (causal branch).
    """
    B, T, _ = x.shape
    a, gamma = scan._drive_and_gamma(x)
    phi_w = scan.phase_scale * torch.tanh(scan.W_key(x))
    phi_w = phi_w.view(B, T, scan.n_heads, scan.d_head)
    if scan.separate_qk:
        phi_r = scan.phase_scale * torch.tanh(scan.W_read_key(x))
        phi_r = phi_r.view(B, T, scan.n_heads, scan.d_head)
    else:
        phi_r = phi_w
    drive_re = a * torch.cos(phi_w)
    drive_im = a * torch.sin(phi_w)
    S_re = sequential_linear_scan(drive_re, gamma)
    S_im = sequential_linear_scan(drive_im, gamma)
    return a, gamma, phi_w, phi_r, S_re, S_im


@torch.no_grad()
def signal_interference(scan, tokens, targets, mask, n_keys=N_KEYS):
    """Decompose the holographic read at each query position into matched-key
    SIGNAL vs summed mismatched INTERFERENCE, exactly.

    The state at query position t is  S_t = Σ_{s≤t} γ_{s→t} u_s e^{iφ_w(s)}.
    The read is  read_t = Re(S_t e^{-iφ_r(t)}) = Σ_{s≤t} γ_{s→t} u_s cos(φ_w(s)−φ_r(t)).

    A "matched" source s is a KV-write token whose KEY equals the query key at t
    (i.e. the token that bound the value being recalled). All other writes are
    interference. We split the per-source sum into matched vs mismatched using the
    token-id identity of the write positions (this is the ground-truth binding the
    task defines), without any phase-threshold heuristic.

    Returns per-channel signal/interference magnitudes aggregated over query slots.
    """
    x = scan_input_embed(scan, tokens)        # embed → the x the scan layer sees
    B, T, _ = x.shape
    H, D = scan.n_heads, scan.d_head
    a, gamma, phi_w, phi_r, _, _ = extract_scan_tensors(scan, x)

    # Per-source contribution to the read at query t:
    #   contrib(s→t) = γ_{s→t} · u_s · cos(φ_w(s) − φ_r(t))
    # We build it directly. γ_{s→t} = prod_{r=s+1..t} γ_r (per channel).
    # log γ cumulative for a numerically clean prefix-ratio.
    log_gamma = torch.log(gamma.clamp_min(1e-12))            # (B,T,H,D)
    cum = torch.cumsum(log_gamma, dim=1)                     # G_t = Σ_{r≤t} logγ_r
    # γ_{s→t} = exp(G_t − G_s) with the convention that the write at s is applied
    # AFTER γ_s (state update: S_s = γ_s S_{s-1} + u_s e^{iφ}), so the surviving
    # factor from s to t is exp(G_t − G_s).  (s==t → factor 1.)

    # Identify, per (batch, query position), the matched source position.
    # The query token id == its key id; the matched write is the KEY token with the
    # same id in the KV block (positions where the same id was written as a key).
    # We score every query slot in `mask`.
    results = {
        "signal_abs": [], "interf_abs": [], "n_active_interf": [],
    }
    SEP_ID = n_keys + N_VALUES

    for b in range(B):
        q_positions = torch.nonzero(mask[b], as_tuple=False).flatten().tolist()
        if not q_positions:
            continue
        toks_b = tokens[b]
        # KV-key positions: even indices in the front KV block carry key ids (< n_keys).
        # A token is a "key write" if its id < n_keys AND it is not itself a query slot.
        is_query = mask[b]
        for qp in q_positions:
            qkey = int(toks_b[qp])                       # query key id
            # matched source = the KV key-write position with this id (the earliest
            # non-query occurrence of qkey before qp).
            cand = (toks_b[:qp] == qkey) & (~is_query[:qp])
            matched_pos = torch.nonzero(cand, as_tuple=False).flatten()
            if matched_pos.numel() == 0:
                continue
            ms = int(matched_pos[0])
            # all OTHER key-write positions before qp (the interferers).
            other_writes = ((toks_b[:qp] < n_keys) & (~is_query[:qp]))
            other_writes[ms] = False
            ow = torch.nonzero(other_writes, as_tuple=False).flatten()

            Gt = cum[b, qp]                              # (H,D)
            phir_t = phi_r[b, qp]                        # (H,D)

            # matched contribution (H,D): γ_{ms→qp} u_ms cos(φ_w(ms) − φ_r(qp))
            decay_m = torch.exp(Gt - cum[b, ms])         # (H,D)
            sig = decay_m * a[b, ms] * torch.cos(phi_w[b, ms] - phir_t)

            # summed mismatched contribution (H,D)
            if ow.numel() > 0:
                decay_o = torch.exp(Gt.unsqueeze(0) - cum[b, ow])      # (M,H,D)
                cos_o = torch.cos(phi_w[b, ow] - phir_t.unsqueeze(0))  # (M,H,D)
                interf = (decay_o * a[b, ow] * cos_o).sum(dim=0)       # (H,D)
            else:
                interf = torch.zeros_like(sig)

            results["signal_abs"].append(sig.abs().mean().item())
            results["interf_abs"].append(interf.abs().mean().item())
            results["n_active_interf"].append(int(ow.numel()))

    sa = torch.tensor(results["signal_abs"])
    ia = torch.tensor(results["interf_abs"])
    n_q = sa.numel()
    if n_q == 0:
        return None
    mean_sig = sa.mean().item()
    mean_interf = ia.mean().item()
    # S/I ratio: aggregate (mean over query slots), and per-slot then averaged.
    si_aggregate = mean_sig / (mean_interf + 1e-12)
    per_slot = (sa / (ia + 1e-12))
    si_per_slot_median = per_slot.median().item()
    return {
        "n_query_slots": int(n_q),
        "mean_signal_abs": mean_sig,
        "mean_interf_abs": mean_interf,
        "SI_ratio_aggregate": si_aggregate,
        "SI_ratio_per_slot_median": si_per_slot_median,
        "mean_n_interferers": float(torch.tensor(results["n_active_interf"]).float().mean()),
    }


def scan_input_embed(scan, tokens):
    """Reconstruct the residual-stream input x that the FIRST scan layer sees.

    The scan layer operates on the post-embedding, post-pos-encoding hidden state.
    We rebuild exactly that (embed + positional) so the extracted intermediates are
    the real ones the trained model used.  We attach the parent LM via scan._parent.
    """
    lm = scan._parent
    return lm.pos(lm.embed(tokens))


@torch.no_grad()
def measure_si(trained):
    print("=" * 74)
    print("PART 2 — signal / interference decomposition (trained models)")
    print("=" * 74)
    out = {}
    for n_pairs, (model, cfg) in trained.items():
        model.eval()
        # attach parent so scan_input_embed can rebuild the layer-0 input
        first_scan = model.layers[0].scan
        first_scan._parent = model
        gen = torch.Generator().manual_seed(777)
        tokens, targets, mask, _ = make_mqar_batch(
            batch_size=BATCH, seq_len=TRAIN_LEN, n_pairs=n_pairs,
            n_queries=n_pairs, n_keys=N_KEYS, n_values=N_VALUES,
            device=DEVICE, generator=gen)
        si = signal_interference(first_scan, tokens, targets, mask)
        out[n_pairs] = si
        if si is not None:
            print(f"   n_pairs={n_pairs:2d}  S/I(agg)={si['SI_ratio_aggregate']:6.3f}  "
                  f"S/I(med)={si['SI_ratio_per_slot_median']:6.3f}  "
                  f"|sig|={si['mean_signal_abs']:.4e}  "
                  f"|interf|={si['mean_interf_abs']:.4e}  "
                  f"#interf={si['mean_n_interferers']:.1f}")
        else:
            print(f"   n_pairs={n_pairs:2d}  (no valid query slots)")
    return out


# ===========================================================================
# main
# ===========================================================================

def main():
    t_start = time.time()
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    curve, trained = recall_vs_npairs()
    si = measure_si(trained)

    # verdict logic
    r1 = curve[N_PAIRS_GRID[0]]["mean_recall_pct"]
    r_last = curve[N_PAIRS_GRID[-1]]["mean_recall_pct"]
    decays = r1 > r_last + 5.0          # >5pp drop from smallest to largest load
    high_at_low = r1 > 30.0             # clearly above chance at n_pairs=1

    result = {
        "meta": {
            "chance_pct": round(CHANCE * 100, 4),
            "n_keys": N_KEYS, "n_values": N_VALUES,
            "train_len": TRAIN_LEN, "train_steps": TRAIN_STEPS,
            "lr": LR, "batch": BATCH, "seeds": SEEDS,
            "readout": "tanh_m", "separate_qk": False,
            "device": str(DEVICE),
            "wall_clock_sec": round(time.time() - t_start, 1),
        },
        "recall_vs_npairs": curve,
        "signal_interference": si,
        "verdict": {
            "recall_decays_with_load": decays,
            "high_recall_at_npairs_1": high_at_low,
            "recall_npairs_1_pct": r1,
            "recall_npairs_max_pct": r_last,
            "crosstalk_is_the_cap": bool(decays and high_at_low),
        },
    }
    out_path = os.path.join(HERE, "holographic_crosstalk_diag.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print("  recall vs n_pairs (mean ± std, lift over chance):")
    for n_pairs in N_PAIRS_GRID:
        c = curve[n_pairs]
        print(f"    n_pairs={n_pairs:2d}  {c['mean_recall_pct']:6.2f}% "
              f"± {c['std_recall_pct']:5.2f}%  (+{c['lift_over_chance_pp']:5.2f}pp)")
    print("  signal/interference:")
    for n_pairs in N_PAIRS_GRID:
        s = si.get(n_pairs)
        if s:
            print(f"    n_pairs={n_pairs:2d}  S/I(agg)={s['SI_ratio_aggregate']:6.3f}  "
                  f"#interf={s['mean_n_interferers']:.1f}")
    print(f"\n  recall(n=1)={r1:.2f}%  recall(n={N_PAIRS_GRID[-1]})={r_last:.2f}%  "
          f"chance={CHANCE*100:.2f}%")
    print(f"  VERDICT crosstalk_is_the_cap = {result['verdict']['crosstalk_is_the_cap']}")
    print(f"  wrote {out_path}")
    print(f"  total wall clock: {result['meta']['wall_clock_sec']}s")
    return result


if __name__ == "__main__":
    main()
