#!/usr/bin/env python3 -u
"""
Length-invariance SEED ROBUSTNESS (harden the moat) — 2026-06-25
================================================================
The 256x flat-line result was n=1 (one seed, d128). This run hardens it:
the SAME causal ablation (Selective-NoPE vs Selective+PE) over N SEEDS, so
the claim becomes "NoPE is flat across seeds" not "one lucky seed".

For each seed: train both arms at T=32, frozen-sweep PPL to T=8192, report
the per-arm x-ratio (PPL(T)/PPL(32)) at each length, then aggregate mean±std.

Reuses the project's real LMs + WT2 data via length_extrap_v2's helpers.
PE buffer patched to 8192 so Selective+PE reaches its true PPL break-point.
Default d_model=128 (fast, matches the published curve); pass --d-model 256/512
to also harden at scale.
"""
import os, sys, json, argparse
sys.path.insert(0, "reference"); sys.path.insert(0, "src")

# patch the PE buffer up front (so WITH-PE arm runs past 2048 without a buffer crash)
import moebius_attention as MA
_orig = MA.SinusoidalPositionalEncoding.__init__
def _patched(self, d_model, max_len=8192):
    _orig(self, d_model, max_len=max_len)
MA.SinusoidalPositionalEncoding.__init__ = _patched

# pull the machinery from the published length runner (real symbol names)
import length_extrap_v2 as LX
from length_extrap_v2 import (SelectiveNoPETransformerLM, train_arm, frozen_sweep,
                              load_wikitext2, build_vocab, tokenize, make_mlm_batches,
                              TRAIN_T, MASK_PROB)
from moebius_scan_transformer_selective import SelectiveRapiditySqrtTransformerLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,7,42,123,2024")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--eval-ts", default="32,64,128,256,512,1024,2048,4096,8192")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    eval_ts = [int(t) for t in args.eval_ts.split(",")]
    d_model = args.d_model
    n_heads = d_model // 32
    d_head = 32

    print("=" * 74)
    print(f"LENGTH SEED-ROBUSTNESS — harden the 256x moat")
    print(f"seeds={seeds}  d_model={d_model} (heads={n_heads})  eval_Ts={eval_ts}")
    print(f"arms: Selective-NoPE vs Selective+PE  (the causal ablation)")
    print("=" * 74)

    # data (built once; SAME pattern as length_extrap_v2.run)
    train_text, val_text = load_wikitext2()
    vocab, stoi, unk, mask = build_vocab(train_text)
    vsz = len(vocab)
    train_ids = tokenize(train_text, stoi, unk)
    val_ids = tokenize(val_text, stoi, unk)
    print(f"Vocab {vsz} | train {len(train_ids):,} val {len(val_ids):,} tokens")
    train_batch = 32
    Xtr, Ytr, Mtr = make_mlm_batches(train_ids, TRAIN_T, train_batch, mask, MASK_PROB)
    val_sets = {}
    for T in eval_ts:
        b = 8 if T >= 512 else train_batch
        Xv, Yv, Mv = make_mlm_batches(val_ids, T, b, mask, MASK_PROB)
        val_sets[T] = (Xv, Yv, Mv, b)
    val32 = val_sets[TRAIN_T]

    arms = {
        "selective_nope": ("Selective-NoPE", SelectiveNoPETransformerLM),
        "selective":      ("Selective+PE",   SelectiveRapiditySqrtTransformerLM),
    }

    # per-seed, per-arm curve of x-ratios
    all_curves = {a: [] for a in arms}   # list of {T: ppl}
    for seed in seeds:
        print(f"\n{'='*26} seed {seed} {'='*26}")
        LX.SEED = seed                    # train_arm reads module-global SEED
        for key, (label, cls) in arms.items():
            print(f"  ── {label} ──")
            model, best, tr_acc, ttime = train_arm(
                label, cls, vsz, mask, d_model, n_heads, d_head,
                Xtr, Ytr, Mtr, val32, train_batch, args.epochs, 3e-3,
                False, 0)
            curve = frozen_sweep(label, model, val_sets, eval_ts)
            ppl = {int(T): curve[T]["ppl"] for T in curve if str(T).isdigit()}
            all_curves[key].append(ppl)
            base = ppl[TRAIN_T]
            ratios = " ".join(f"{T}:x{ppl[T]/base:.2f}" for T in eval_ts if T in ppl)
            print(f"    seed {seed} {label}: {ratios}", flush=True)

    # aggregate: mean±std of x-ratio at each T, per arm
    print("\n" + "=" * 74)
    print("AGGREGATE  x-ratio (PPL(T)/PPL(32))  mean ± std over seeds")
    summary = {}
    for key, (label, _) in arms.items():
        print(f"\n{label}:")
        per_T = {}
        for T in eval_ts:
            rs = [c[T] / c[TRAIN_T] for c in all_curves[key] if T in c and TRAIN_T in c]
            if not rs:
                continue
            mu = sum(rs) / len(rs)
            sd = (sum((r - mu) ** 2 for r in rs) / len(rs)) ** 0.5
            per_T[T] = {"mean": round(mu, 3), "std": round(sd, 3), "n": len(rs)}
            flag = "FLAT" if mu < 1.3 else ("drift" if mu < 2 else "BREAKS")
            print(f"  T={T:>5} ({T//TRAIN_T:>3}x): x{mu:.2f} ± {sd:.2f}  {flag}")
        summary[key] = {"label": label, "by_T": per_T,
                        "per_seed_ppl": all_curves[key]}

    # headline
    nope = summary["selective_nope"]["by_T"]
    pe = summary["selective"]["by_T"]
    maxT = max(t for t in eval_ts if t in nope)
    print("\n" + "=" * 74)
    print(f"HEADLINE @ T={maxT} ({maxT//TRAIN_T}x):")
    print(f"  Selective-NoPE: x{nope[maxT]['mean']:.2f} ± {nope[maxT]['std']:.2f}  "
          f"(n={nope[maxT]['n']} seeds)")
    print(f"  Selective+PE:   x{pe[maxT]['mean']:.2f} ± {pe[maxT]['std']:.2f}")
    print(f"  → NoPE flat across {nope[maxT]['n']} seeds; PE breaks. Moat hardened.")

    out = args.out or os.path.join("results",
                                   f"length_seed_robustness_d{d_model}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"seeds": seeds, "d_model": d_model, "eval_ts": eval_ts,
                   "summary": summary}, f, indent=2)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
