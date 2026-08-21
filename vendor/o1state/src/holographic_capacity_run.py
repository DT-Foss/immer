"""
Holographic-GSSM FRONT 1 — PHASE-COLLISION CAPACITY sweep — by Opus 4.8
======================================================================

The key-conditioned complex write (src/holographic_gssm.py, readout="tanh_m") clears the
scalar recall wall: +5.57pp over Selective (7.18% vs 1.61%), attn validity gate 0.997.
But it PLATEAUS at ~7% — far from attention's ~100%. FRONT 1 tests ONE plateau cause:

  PHASE-COLLISION CAPACITY.  Each token's key gets a phase φ_t = π·tanh(W_key x_t) per
  complex channel.  With 8 (key,value) pairs superposed in d_head complex channels, two
  DISTINCT keys can land near the same phase → cos(φ_k − φ_q) ≈ 1 for the wrong key →
  crosstalk → recall caps.  If collision is the cap, giving the phase channel more room
  must raise recall.  Three independent levers, each isolating the collision hypothesis:

  (a) d_head   32 → 64 → 96   : more independent complex channels = more phase room.
  (b) phase_scale  π → 2π → 3π: keys spread over more of the circle (less angular density).
  (c) n_pairs  8 → 4 → 2      : DIAGNOSTIC. Fewer pairs = fewer superposed keys = less
                                collision. If recall SHOOTS UP at n_pairs=2, collision IS
                                the cap. If it stays flat, the cap is elsewhere (mechanism).

DECISIVE DIAGNOSTIC (committed before reading results):
  - Does recall scale MONOTONICALLY with d_head and/or phase_scale?  → collision contributes.
  - Does recall JUMP at n_pairs=2 vs n_pairs=8?  → collision IS the dominant cap.
  - If all three are FLAT (within seed noise), phase-collision is NOT the plateau cause;
    the next front (write/read key sharing, or the read nonlinearity) owns the cap.

CONTROLS, every config:
  - readout="tanh_m" — the WORKING, load-bearing readout (rms/layernorm FAILED, +0.7/+0.4pp).
    m is the learned RELEVANCE GATE that triggers WHEN to read; do not touch it.
  - holo_off (use_phase=False == Selective) is the FLOOR, recomputed per n_pairs.
  - attn (TinyCausalTransformerLM) is the VALIDITY GATE, recomputed per n_pairs; if attn
    < 0.90 at some n_pairs, the GSSM numbers at that n_pairs are VOID.
  - CPU-deterministic, multi-seed (>=3), mean ± std. A single seed is noise on a chance-flat
    loss — that is how the original phase-GSSM positive was a false positive.

Offline, self-contained, python3.
"""
import os
import sys
import math
import json
import time
import argparse

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "reference"))
sys.path.insert(0, HERE)

from mqar import make_mqar_batch, mqar_accuracy, TinyCausalTransformerLM  # noqa: E402
from holographic_gssm import HolographicLM  # noqa: E402


# ── builders ─────────────────────────────────────────────────────────────────
def build_attn(vocab, dm, nl, nh, sl):
    return TinyCausalTransformerLM(vocab, d_model=dm, n_layers=nl, n_heads=nh,
                                   max_len=max(sl, 1024))


def build_holo_off(vocab, mask_idx, dm, nl, nh, dh, sl):
    return HolographicLM(vocab, mask_idx, d_model=dm, n_layers=nl, n_heads=nh,
                         d_head=dh, seq_len=sl, use_phase=False, readout="tanh_m")


def build_holo_on(vocab, mask_idx, dm, nl, nh, dh, sl, phase_scale):
    # readout="tanh_m" is the load-bearing working readout (m = learned relevance gate).
    return HolographicLM(vocab, mask_idx, d_model=dm, n_layers=nl, n_heads=nh,
                         d_head=dh, seq_len=sl, use_phase=True, readout="tanh_m",
                         phase_scale=phase_scale)


def train(model, cfg, steps, lr, seed, device):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(steps):
        tok, tgt, mask, _ = make_mqar_batch(generator=gen, device=device, **cfg)
        logits = model(tok)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               tgt.reshape(-1), reduction="none")
        loss = (loss * mask.reshape(-1).float()).sum() / (mask.sum() + 1e-6)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    return model


def mean_std(xs):
    mu = sum(xs) / len(xs)
    return mu, (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5


def run_config(label, builder, cfg, steps, lr, seeds, device, eval_batches=8):
    """Train+eval `builder()` over seeds at MQAR `cfg`. Returns dict with per-seed + agg."""
    accs = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = builder()
        train(model, cfg, steps, lr, seed, device)
        model.eval()
        ov, _, _ = mqar_accuracy(model, cfg, eval_batches, seed + 1, device)
        accs.append(ov)
        print(f"    {label:24s} seed {seed:5d}  recall {ov:.4f}", flush=True)
    mu, sd = mean_std(accs)
    print(f"    {label:24s} MEAN        {mu:.4f} ± {sd:.4f}", flush=True)
    return {"mean": mu, "std": sd, "per_seed": accs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", default="1,7,42")
    ap.add_argument("--train-len", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    # baseline (== the 7% working point)
    ap.add_argument("--base-d-head", type=int, default=32)
    ap.add_argument("--base-n-pairs", type=int, default=8)
    # sweep knobs
    ap.add_argument("--d-heads", default="32,64,96")
    ap.add_argument("--phase-scales", default="1,2,3")  # multiples of pi
    ap.add_argument("--n-pairs-list", default="8,4,2")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "holographic_capacity.json"))
    args = ap.parse_args()

    device = torch.device("cpu")
    nk = nv = 64
    vocab = nk + nv + 1
    mask_idx = vocab
    chance = 1.0 / nv
    seeds = [int(s) for s in args.seeds.split(",")]
    dm, nl, nh, sl, lr = args.d_model, args.n_layers, args.n_heads, args.train_len, args.lr
    steps = args.steps

    d_heads = [int(x) for x in args.d_heads.split(",")]
    phase_mults = [float(x) for x in args.phase_scales.split(",")]
    n_pairs_list = [int(x) for x in args.n_pairs_list.split(",")]

    def cfg_for(n_pairs):
        return dict(batch_size=32, seq_len=sl, n_pairs=n_pairs, n_queries=n_pairs,
                    n_keys=nk, n_values=nv)

    print("=" * 80)
    print("FRONT 1 — PHASE-COLLISION CAPACITY sweep (key-conditioned holographic write)")
    print(f"device=cpu  steps={steps}  seeds={seeds}  train_len={sl}  d_model={dm}")
    print(f"base d_head={args.base_d_head}  base n_pairs={args.base_n_pairs}  "
          f"readout=tanh_m (working)  chance=1/{nv}={chance:.4f}")
    print("=" * 80)

    t0 = time.time()
    results = {
        "config": {
            "steps": steps, "seeds": seeds, "train_len": sl, "d_model": dm,
            "n_layers": nl, "n_heads": nh, "lr": lr, "readout": "tanh_m",
            "base_d_head": args.base_d_head, "base_n_pairs": args.base_n_pairs,
            "n_keys": nk, "n_values": nv, "chance": chance, "device": "cpu",
        },
        "baseline": {}, "sweep_d_head": {}, "sweep_phase_scale": {},
        "sweep_n_pairs": {}, "gates": {},
    }

    base_cfg = cfg_for(args.base_n_pairs)

    # ── 0. baseline gates at base n_pairs: attn (validity) + holo_off (floor) ──
    print("\n[gates @ base n_pairs] attn validity + holo_off floor")
    results["gates"][f"attn_np{args.base_n_pairs}"] = run_config(
        f"attn np{args.base_n_pairs}", lambda: build_attn(vocab, dm, nl, nh, sl),
        base_cfg, steps, lr, seeds, device)
    results["gates"][f"holo_off_np{args.base_n_pairs}"] = run_config(
        f"holo_off np{args.base_n_pairs}",
        lambda: build_holo_off(vocab, mask_idx, dm, nl, nh, args.base_d_head, sl),
        base_cfg, steps, lr, seeds, device)

    # ── baseline holo_on (d_head=base, phase=pi, n_pairs=base) — the 7% point ──
    print("\n[baseline] holo_on  d_head=%d  phase=pi  n_pairs=%d" %
          (args.base_d_head, args.base_n_pairs))
    base_holo = run_config(
        "holo_on BASE",
        lambda: build_holo_on(vocab, mask_idx, dm, nl, nh, args.base_d_head, sl, math.pi),
        base_cfg, steps, lr, seeds, device)
    results["baseline"]["holo_on"] = base_holo

    # ── (a) d_head sweep — more independent complex channels = more phase room ──
    print("\n[(a) d_head sweep]  phase=pi, n_pairs=%d" % args.base_n_pairs)
    for dh in d_heads:
        if dh == args.base_d_head:
            results["sweep_d_head"][str(dh)] = base_holo  # reuse baseline
            print(f"    d_head={dh:3d}  (== baseline, reused)  "
                  f"{base_holo['mean']:.4f} ± {base_holo['std']:.4f}", flush=True)
            continue
        results["sweep_d_head"][str(dh)] = run_config(
            f"holo_on d_head={dh}",
            lambda dh=dh: build_holo_on(vocab, mask_idx, dm, nl, nh, dh, sl, math.pi),
            base_cfg, steps, lr, seeds, device)

    # ── (b) phase_scale sweep — keys spread over more of the circle ──
    print("\n[(b) phase_scale sweep]  d_head=%d, n_pairs=%d" %
          (args.base_d_head, args.base_n_pairs))
    for pm in phase_mults:
        ps = pm * math.pi
        if abs(pm - 1.0) < 1e-9:
            results["sweep_phase_scale"][f"{pm:g}pi"] = base_holo  # reuse baseline
            print(f"    phase={pm:g}pi  (== baseline, reused)  "
                  f"{base_holo['mean']:.4f} ± {base_holo['std']:.4f}", flush=True)
            continue
        results["sweep_phase_scale"][f"{pm:g}pi"] = run_config(
            f"holo_on phase={pm:g}pi",
            lambda ps=ps: build_holo_on(vocab, mask_idx, dm, nl, nh, args.base_d_head, sl, ps),
            base_cfg, steps, lr, seeds, device)

    # ── (c) n_pairs sweep — DIAGNOSTIC. Fewer superposed keys = less collision. ──
    #     Each n_pairs gets its OWN attn gate + holo_off floor (recall scale changes).
    print("\n[(c) n_pairs DIAGNOSTIC]  d_head=%d, phase=pi" % args.base_d_head)
    for npairs in n_pairs_list:
        cfg = cfg_for(npairs)
        print(f"  -- n_pairs={npairs} --")
        on = (base_holo if npairs == args.base_n_pairs else run_config(
            f"holo_on n_pairs={npairs}",
            lambda: build_holo_on(vocab, mask_idx, dm, nl, nh, args.base_d_head, sl, math.pi),
            cfg, steps, lr, seeds, device))
        if npairs == args.base_n_pairs:
            print(f"    holo_on n_pairs={npairs} (== baseline, reused)  "
                  f"{on['mean']:.4f} ± {on['std']:.4f}", flush=True)
        # gate + floor for this n_pairs (reuse base if already computed)
        if npairs == args.base_n_pairs:
            attn_g = results["gates"][f"attn_np{args.base_n_pairs}"]
            off_f = results["gates"][f"holo_off_np{args.base_n_pairs}"]
        else:
            attn_g = run_config(f"attn n_pairs={npairs}",
                                lambda: build_attn(vocab, dm, nl, nh, sl),
                                cfg, steps, lr, seeds, device)
            off_f = run_config(
                f"holo_off n_pairs={npairs}",
                lambda: build_holo_off(vocab, mask_idx, dm, nl, nh, args.base_d_head, sl),
                cfg, steps, lr, seeds, device)
        results["sweep_n_pairs"][str(npairs)] = {
            "holo_on": on, "attn_gate": attn_g, "holo_off_floor": off_f,
            "contribution_pp": round(100 * (on["mean"] - off_f["mean"]), 2),
        }

    # ── verdict / diagnostic readout ──
    def trend(d, keys):
        vals = [d[k]["mean"] for k in keys]
        return vals, (vals[-1] - vals[0])

    print("\n" + "=" * 80)
    print("DIAGNOSTIC")
    dh_keys = [str(x) for x in d_heads]
    dh_vals, dh_delta = trend(results["sweep_d_head"], dh_keys)
    print(f"  (a) d_head {dh_keys}  recall {[round(v,4) for v in dh_vals]}  "
          f"Δ(last-first)={100*dh_delta:+.2f}pp")
    ps_keys = list(results["sweep_phase_scale"].keys())
    ps_vals, ps_delta = trend(results["sweep_phase_scale"], ps_keys)
    print(f"  (b) phase  {ps_keys}  recall {[round(v,4) for v in ps_vals]}  "
          f"Δ(last-first)={100*ps_delta:+.2f}pp")
    np_keys = [str(x) for x in n_pairs_list]
    np_on = [results["sweep_n_pairs"][k]["holo_on"]["mean"] for k in np_keys]
    np_contrib = [results["sweep_n_pairs"][k]["contribution_pp"] for k in np_keys]
    print(f"  (c) n_pairs {np_keys}  holo_on {[round(v,4) for v in np_on]}  "
          f"contrib(pp) {np_contrib}")
    # decisive: does recall jump as pairs drop?  (compare smallest vs largest n_pairs)
    np_sorted = sorted(n_pairs_list)
    lo, hi = str(np_sorted[0]), str(np_sorted[-1])
    np_jump = (results["sweep_n_pairs"][lo]["holo_on"]["mean"]
               - results["sweep_n_pairs"][hi]["holo_on"]["mean"])

    attn_base = results["gates"][f"attn_np{args.base_n_pairs}"]["mean"]
    validity = attn_base >= 0.90
    all_holo_stds = ([base_holo["std"]] +
                     [results["sweep_d_head"][k]["std"] for k in dh_keys] +
                     [results["sweep_phase_scale"][k]["std"] for k in ps_keys] +
                     [results["sweep_n_pairs"][k]["holo_on"]["std"] for k in np_keys])
    band = 2 * max(max(all_holo_stds), 1e-6)
    collision_scales = (dh_delta > band) or (ps_delta > band)
    collision_dominant = np_jump > band

    verdict = {
        "validity_gate_attn_base": round(attn_base, 4),
        "validity_passed": bool(validity),
        "d_head_recall": [round(v, 4) for v in dh_vals],
        "d_head_delta_pp": round(100 * dh_delta, 2),
        "phase_scale_recall": [round(v, 4) for v in ps_vals],
        "phase_scale_delta_pp": round(100 * ps_delta, 2),
        "n_pairs_holo_on": {k: round(results["sweep_n_pairs"][k]["holo_on"]["mean"], 4)
                            for k in np_keys},
        "n_pairs_jump_lo_minus_hi_pp": round(100 * np_jump, 2),
        "noise_band_2sigma_pp": round(100 * band, 2),
        "collision_scales_with_resolution": bool(collision_scales),
        "collision_is_dominant_cap": bool(collision_dominant),
        "interpretation": (
            "VOID — attn validity gate failed at base n_pairs" if not validity else
            ("PHASE-COLLISION IS THE CAP — recall jumps as pairs drop and/or scales with "
             "phase resolution; next lever = more phase room (bigger d_head / phase_scale "
             "/ orthogonalized keys)" if (collision_scales or collision_dominant) else
             "PHASE-COLLISION IS NOT THE CAP — recall flat across d_head, phase_scale AND "
             "n_pairs within seed noise; the plateau lives elsewhere (read nonlinearity or "
             "write/read key sharing) — that is the next front")),
    }
    results["verdict"] = verdict
    results["elapsed_s"] = round(time.time() - t0, 1)

    print("\nVERDICT")
    print(f"  validity (attn@base ≥0.90)      : {attn_base:.4f}  "
          f"{'PASS' if validity else 'FAIL → numbers VOID'}")
    print(f"  (a) d_head Δ                     : {100*dh_delta:+.2f}pp")
    print(f"  (b) phase_scale Δ                : {100*ps_delta:+.2f}pp")
    print(f"  (c) n_pairs jump (lo−hi)         : {100*np_jump:+.2f}pp")
    print(f"  noise band (2σ)                  : {100*band:.2f}pp")
    print(f"  >>> {verdict['interpretation']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.out}  ({results['elapsed_s']}s)")


if __name__ == "__main__":
    main()
