"""
Holographic-GSSM MQAR multi-seed run — the key-conditioned-write attack — by Opus 4.8
=====================================================================================

The additive-phase Phase-GSSM gave +0.00pp recall across seeds (the phase rotated blindly
with time).  This runs the KEY-CONDITIONED holographic write (src/holographic_gssm.py)
against the same MQAR harness, the right way the adversary demanded:

  * arms: attn (validity gate), selective (scalar baseline), holo_off (ablation == Selective),
           holo_on (key-conditioned complex write).
  * N>=5 seeds, DETERMINISTIC ON CPU (no MPS cross-process nondeterminism on a chance-flat loss).
  * aggregate mean ± std per arm, so a positive must clear the noise band, not a single seed.

Decision rule (committed before reading results):
  - holo_on mean recall must beat holo_off mean by > 2·std AND clear chance (1/n_values≈1.6%)
    by a clear margin to count as "the complex write breaks the scalar recall wall".
  - attn must reach >=0.90 (validity gate) or ALL GSSM numbers are VOID.
  - holo_off must ≈ selective (it is Selective by construction) — internal consistency check.
If holo_on does not clear the band, that is a clean negative on THIS write rule — report it,
then the next lever (separate write/read key projections, higher d_head, learned phase_scale).
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

from mqar import (  # noqa: E402
    make_mqar_batch, mqar_accuracy, GAP_BINS, TinyCausalTransformerLM,
)
from moebius_scan_transformer_selective import SelectiveRapiditySqrtTransformerLM  # noqa: E402
from holographic_gssm import HolographicLM  # noqa: E402


def build_arm(arm, vocab_size, mask_idx, d_model, n_layers, n_heads, d_head, seq_len):
    if arm == "selective":
        return SelectiveRapiditySqrtTransformerLM(
            vocab_size, mask_idx, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, d_head=d_head, seq_len=seq_len, dropout=0.0, causal=True)
    if arm == "holo_on":
        return HolographicLM(
            vocab_size, mask_idx, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, d_head=d_head, seq_len=seq_len, dropout=0.0,
            causal=True, phase_scale=math.pi, use_phase=True)
    if arm == "holo_off":
        return HolographicLM(
            vocab_size, mask_idx, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, d_head=d_head, seq_len=seq_len, dropout=0.0,
            causal=True, phase_scale=math.pi, use_phase=False)
    if arm == "attn":
        return TinyCausalTransformerLM(
            vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            max_len=max(seq_len, 1024))
    raise ValueError(arm)


def train_arm(model, cfg, steps, lr, seed, device):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for step in range(steps):
        tokens, targets, mask, _ = make_mqar_batch(generator=gen, device=device, **cfg)
        logits = model(tokens)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
        loss = (loss * mask.reshape(-1).float()).sum() / (mask.sum() + 1e-6)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    return model


def eval_arm(model, train_cfg, test_cfg, seed, device):
    model.eval()
    tr_overall, tr_gap, _ = mqar_accuracy(model, train_cfg, 8, seed + 1, device)
    te_overall, te_gap, _ = mqar_accuracy(model, test_cfg, 8, seed + 2, device)
    return {"train_len": {"overall": tr_overall, "by_gap": tr_gap},
            "test_len": {"overall": te_overall, "by_gap": te_gap}}


def mean_std(xs):
    n = len(xs)
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / n
    return mu, var ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--train-len", type=int, default=64)
    ap.add_argument("--test-len", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-head", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", default="1,7,42,123,2024")
    ap.add_argument("--arms", default="attn,selective,holo_off,holo_on")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "holographic_mqar.json"))
    args = ap.parse_args()

    device = torch.device("cpu")  # deterministic
    torch.use_deterministic_algorithms(False)  # cpu is enough; keep gather/scatter fast

    n_keys = n_values = 64
    vocab_size = n_keys + n_values + 1
    mask_idx = vocab_size
    chance = 1.0 / n_values

    train_cfg = dict(batch_size=32, seq_len=args.train_len, n_pairs=args.n_pairs,
                     n_queries=args.n_pairs, n_keys=n_keys, n_values=n_values)
    test_cfg = dict(batch_size=32, seq_len=args.test_len, n_pairs=args.n_pairs,
                    n_queries=args.n_pairs, n_keys=n_keys, n_values=n_values)

    seeds = [int(s) for s in args.seeds.split(",")]
    arms = args.arms.split(",")

    print("=" * 78)
    print("Holographic-GSSM MQAR — key-conditioned write, multi-seed, CPU-deterministic")
    print(f"device={device} steps={args.steps} train_len={args.train_len} "
          f"test_len={args.test_len} n_pairs={args.n_pairs}")
    print(f"seeds={seeds}  chance=1/{n_values}={chance:.4f}")
    print("=" * 78)

    per_seed = {arm: [] for arm in arms}
    per_seed_test = {arm: [] for arm in arms}
    t0 = time.time()

    for seed in seeds:
        print(f"\n--- seed {seed} ---")
        for arm in arms:
            torch.manual_seed(seed)
            model = build_arm(arm, vocab_size, mask_idx, args.d_model, args.n_layers,
                              args.n_heads, args.d_head, args.train_len)
            train_arm(model, train_cfg, args.steps, args.lr, seed, device)
            res = eval_arm(model, train_cfg, test_cfg, seed, device)
            tr = res["train_len"]["overall"]
            te = res["test_len"]["overall"]
            per_seed[arm].append(tr)
            per_seed_test[arm].append(te)
            print(f"  {arm:11s}  train-len {tr:.4f}   test-len {te:.4f}")

    # ── aggregate ──
    print("\n" + "=" * 78)
    print("AGGREGATE (train-len overall, mean ± std over seeds)")
    summary = {}
    for arm in arms:
        mu, sd = mean_std(per_seed[arm])
        mu_te, sd_te = mean_std(per_seed_test[arm])
        summary[arm] = {"train_mean": mu, "train_std": sd,
                        "test_mean": mu_te, "test_std": sd_te,
                        "per_seed_train": per_seed[arm]}
        print(f"  {arm:11s}  {mu:.4f} ± {sd:.4f}   (test {mu_te:.4f} ± {sd_te:.4f})")

    attn_mu = summary.get("attn", {}).get("train_mean", 0.0)
    validity = attn_mu >= 0.90

    verdict = {}
    if "holo_on" in summary and "holo_off" in summary:
        on_mu, on_sd = summary["holo_on"]["train_mean"], summary["holo_on"]["train_std"]
        off_mu = summary["holo_off"]["train_mean"]
        contribution = on_mu - off_mu
        # clears band if on beats off by > 2·max(std) and clears chance clearly
        band = 2 * max(on_sd, summary["holo_off"]["train_std"], 1e-6)
        breaks_wall = (contribution > band) and (on_mu > 3 * chance)
        verdict = {
            "holo_contribution_pp": round(100 * contribution, 2),
            "noise_band_pp": round(100 * band, 2),
            "holo_on_mean": round(on_mu, 4),
            "holo_off_mean": round(off_mu, 4),
            "chance": round(chance, 4),
            "validity_gate_attn": round(attn_mu, 4),
            "validity_passed": validity,
            "breaks_recall_wall": bool(breaks_wall and validity),
            "interpretation": (
                "KEY-CONDITIONED WRITE HELPS — clears the noise band and chance"
                if (breaks_wall and validity) else
                ("VOID — attention validity gate failed" if not validity else
                 "NEGATIVE on this write rule — holo_on does not clear holo_off + 2σ")),
        }
        print("\n" + "=" * 78)
        print("VERDICT")
        print(f"  validity gate (attn ≥ 0.90)      : {attn_mu:.4f}  "
              f"{'PASS' if validity else 'FAIL → numbers VOID'}")
        print(f"  holo_on  mean                    : {on_mu:.4f} ± {on_sd:.4f}")
        print(f"  holo_off mean (== Selective)     : {off_mu:.4f}")
        print(f"  contribution (on − off)          : {100*contribution:+.2f} pp")
        print(f"  noise band (2σ)                  : {100*band:.2f} pp")
        print(f"  chance (1/{n_values})                  : {chance:.4f}")
        print(f"  >>> {verdict['interpretation']}")

    out = {
        "config": {"steps": args.steps, "n_pairs": args.n_pairs,
                   "train_len": args.train_len, "test_len": args.test_len,
                   "d_model": args.d_model, "n_heads": args.n_heads,
                   "d_head": args.d_head, "seeds": seeds, "chance": chance,
                   "device": "cpu"},
        "summary": summary, "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults written to {args.out}  ({out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
