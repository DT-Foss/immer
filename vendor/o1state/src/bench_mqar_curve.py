#!/usr/bin/env python3 -u
"""
MQAR recall-vs-#pairs curve — the holographic 1/sqrt(N) law, clean — by Opus 4.8
================================================================================

WHAT THIS MEASURES.  A clean MQAR associative-recall curve for the key-conditioned
holographic write (src/holographic_gssm.py): exact-match recall of a SINGLE bounded
complex channel as a function of how many (key,value) pairs are held in superposition
at once, swept over n_pairs ∈ {1,2,4,8,16}.

The holographic read is

    read = Σ_k γ_{k→t} u_k cos(φ_k − φ_q)

The matched key contributes cos(0)=1; the other N−1 keys contribute cos(φ_k−φ_q),
a random walk of magnitude ~sqrt(N−1) on the phase circle. So signal-to-interference
falls like 1/sqrt(N) — the classic HRR/VSA holographic-memory capacity law. The README
cites the anchor point of this curve: 25.8% recall at 2 pairs. This benchmark traces the
whole curve and fits recall_above_floor ≈ C/sqrt(n_pairs), reporting the coefficient of
variation of above·sqrt(N) (≈const ⇒ the law holds).

ARMS per n_pairs:
  * holo_on   : key-conditioned holographic write (the curve of interest).
  * holo_off  : use_phase=False == GSSM-Selective floor (the recall wall the write must clear).
  * attn      : tiny causal Transformer — the VALIDITY GATE (must reach ≥0.90, else
                the harness mislabels recall and ALL GSSM numbers are void).
Multi-seed, CPU-deterministic (a chance-flat recall must not ride MPS nondeterminism).

OUTPUTS (single script):
  * results/bench_mqar_curve.json   — per-n_pairs mean±std + the 1/sqrt(N) law fit.
  * plots/mqar_curve.png            — recall(holo_on) vs n_pairs with the C/sqrt(N) overlay.

Reuses src/holographic_gssm.py (the model) and src/mqar.py (the harness: make_mqar_batch,
mqar_accuracy). Same harness/style as src/longcontext_run.py and src/holographic_mqar_run.py
(sys.path.insert for reference+src, argparse, writes results/<name>.json, has __main__).
psutil memory watchdog SIGKILLs above ~10 GB (the pattern from src/percolation_hard.py).
Offline. Reference: Foss 2026, "From Markov Chains to Minkowski Space".
"""

import os
import sys
import math
import json
import time
import signal
import argparse
import threading

sys.path.insert(0, "reference")
sys.path.insert(0, "src")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "reference"))
sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F

from mqar import make_mqar_batch, mqar_accuracy, TinyCausalTransformerLM  # noqa: E402
from holographic_gssm import HolographicLM  # noqa: E402
from moebius_scan_transformer_selective import SelectiveRapiditySqrtTransformerLM  # noqa: E402

# ── psutil memory watchdog: SIGKILL above hard cap (pattern from percolation_hard.py) ──
try:
    import psutil
    _P = psutil.Process(os.getpid())
    def _rss():
        return _P.memory_info().rss / 1e9
except ImportError:
    def _rss():
        return 0.0


def _watchdog(hard_gb=10.0):
    def w():
        while True:
            if _rss() > hard_gb:
                os.kill(os.getpid(), signal.SIGKILL)
            time.sleep(0.5)
    threading.Thread(target=w, daemon=True).start()


# ── train / eval (same recipe as holographic_mqar_run.py) ──────────────────────
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


def build_arm(arm, vocab, mask_idx, d_model, n_layers, n_heads, d_head, seq_len, readout):
    if arm == "holo_on":
        return HolographicLM(vocab, mask_idx, d_model=d_model, n_layers=n_layers,
                             n_heads=n_heads, d_head=d_head, seq_len=seq_len,
                             use_phase=True, readout=readout)
    if arm == "holo_off":
        return SelectiveRapiditySqrtTransformerLM(vocab, mask_idx, d_model=d_model,
                                                  n_layers=n_layers, n_heads=n_heads,
                                                  d_head=d_head, seq_len=seq_len,
                                                  dropout=0.0, causal=True)
    if arm == "attn":
        return TinyCausalTransformerLM(vocab, d_model=d_model, n_layers=n_layers,
                                       n_heads=n_heads, max_len=max(seq_len, 1024))
    raise ValueError(arm)


def mean_std(xs):
    mu = sum(xs) / len(xs)
    return mu, (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5


# ── plot (self-contained so one script makes JSON + plot) ──────────────────────
def make_plot(rows, chance, sqrtN_const_mean, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    npairs = [r["n_pairs"] for r in rows]
    on = [100 * r["holo_on"] for r in rows]
    on_sd = [100 * r["holo_on_std"] for r in rows]
    floor = [100 * r["holo_off"] for r in rows]

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    # the 1/sqrt(N) law overlay: recall ≈ floor + C/sqrt(N)
    xs = [n / 8 for n in range(8, 8 * (max(npairs) + 1))]
    floor_mu = sum(floor) / len(floor)
    ys = [floor_mu + 100 * sqrtN_const_mean / math.sqrt(x) for x in xs]
    ax.plot(xs, ys, "--", color="#999999", lw=1.8, zorder=2,
            label=r"holographic law  floor + $C/\sqrt{N}$")

    ax.errorbar(npairs, on, yerr=on_sd, fmt="-o", lw=2.6, ms=8, capsize=4,
                color="#1b9e77", zorder=5, label="holo_on (key-conditioned write)")
    ax.plot(npairs, floor, "-s", lw=1.8, ms=6, color="#d95f02", zorder=4,
            label="holo_off floor (GSSM-Selective)")
    ax.axhline(100 * chance, ls=":", color="#666666", lw=1.4, zorder=1,
               label=f"chance (1/{int(round(1/chance))} = {100*chance:.1f}%)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(npairs)
    ax.set_xticklabels([str(n) for n in npairs])
    ax.set_xlabel("number of (key,value) pairs in superposition  (N)")
    ax.set_ylabel("MQAR exact-match recall  (%)")
    ax.set_title("Holographic key-conditioned write: recall decays ~1/√N "
                 "(interference-bound, not capacity-bound)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", default="1,2,4,8,16",
                    help="comma-separated #pairs in superposition to sweep")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--train-len", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-head", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--readout", default="tanh_m",
                    help="holo readout (tanh_m matches the README 25.8%%@2 anchor)")
    ap.add_argument("--seeds", default="1,7,42")
    ap.add_argument("--arms", default="attn,holo_off,holo_on")
    ap.add_argument("--mem-cap-gb", type=float, default=10.0)
    ap.add_argument("--out", default=os.path.join(REPO, "results", "bench_mqar_curve.json"))
    ap.add_argument("--plot", default=os.path.join(REPO, "plots", "mqar_curve.png"))
    args = ap.parse_args()

    _watchdog(hard_gb=args.mem_cap_gb)

    device = torch.device("cpu")  # deterministic; recall is near chance, no MPS noise
    nk = nv = 64
    vocab = nk + nv + 1
    mask_idx = vocab
    chance = 1.0 / nv

    n_pairs_list = [int(x) for x in args.n_pairs.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    arms = args.arms.split(",")
    # the query region needs 2*n_pairs + n_queries <= seq_len; auto-grow train_len if too small.
    train_len = max(args.train_len, max(3 * np_ + 4 for np_ in n_pairs_list))

    print("=" * 78)
    print("MQAR RECALL-vs-#PAIRS CURVE — holographic 1/sqrt(N) law")
    print(f"n_pairs={n_pairs_list}  seeds={seeds}  steps={args.steps}  train_len={train_len}")
    print(f"d_model={args.d_model} n_heads={args.n_heads} d_head={args.d_head} "
          f"readout={args.readout}  chance=1/{nv}={chance:.4f}")
    print(f"arms={arms}  device=cpu  mem_cap={args.mem_cap_gb}GB  rss={_rss():.2f}GB")
    print("=" * 78)

    rows = []
    t0 = time.time()
    for npairs in n_pairs_list:
        cfg = dict(batch_size=args.batch_size, seq_len=train_len, n_pairs=npairs,
                   n_queries=npairs, n_keys=nk, n_values=nv)
        per_arm = {a: [] for a in arms}
        for seed in seeds:
            for arm in arms:
                torch.manual_seed(seed)
                model = build_arm(arm, vocab, mask_idx, args.d_model, args.n_layers,
                                  args.n_heads, args.d_head, train_len, args.readout)
                train(model, cfg, args.steps, args.lr, seed, device)
                model.eval()
                acc = mqar_accuracy(model, cfg, 8, seed + 1, device)[0]
                per_arm[arm].append(acc)
        row = {"n_pairs": npairs}
        for arm in arms:
            mu, sd = mean_std(per_arm[arm])
            row[arm] = mu
            row[f"{arm}_std"] = sd
        on_mu = row.get("holo_on", 0.0)
        off_mu = row.get("holo_off", 0.0)
        above = on_mu - off_mu
        row["above_floor"] = above
        row["above_x_sqrtN"] = above * math.sqrt(npairs)
        rows.append(row)
        print(f"  N={npairs:2d}  holo_on {row.get('holo_on',0):.4f}"
              f"±{row.get('holo_on_std',0):.4f}  floor {off_mu:.4f}  "
              f"attn {row.get('attn',0):.4f}  above {above:+.4f}  "
              f"above·sqrt(N) {row['above_x_sqrtN']:.4f}  rss {_rss():.2f}GB", flush=True)

    # ── 1/sqrt(N) law test: above_floor·sqrt(N) ≈ const ⇒ crosstalk-limited ──
    consts = [r["above_x_sqrtN"] for r in rows if r["above_floor"] > 0]
    cmu, csd = mean_std(consts) if consts else (0.0, 0.0)
    cv = (csd / cmu) if cmu else float("inf")

    decays = len(rows) >= 2 and rows[0]["above_floor"] > rows[-1]["above_floor"]
    if cv < 0.35 and decays:
        verdict = "1/sqrt(N) HOLDS — recall_above_floor·sqrt(N) ~ const (interference-bound)"
    elif decays and rows[0]["above_floor"] > 2 * max(rows[-1]["above_floor"], 1e-9):
        verdict = "DECAYS with N but not cleanly 1/sqrt(N)"
    else:
        verdict = "NOT 1/sqrt(N) — recall ~flat in n_pairs"

    # validity gate (attention must solve MQAR or recall labels are void)
    attn_min = min((r.get("attn", 0.0) for r in rows), default=0.0)
    validity = attn_min >= 0.90

    out = {
        "config": {"n_pairs_list": n_pairs_list, "seeds": seeds, "steps": args.steps,
                   "train_len": train_len, "d_model": args.d_model, "n_heads": args.n_heads,
                   "d_head": args.d_head, "readout": args.readout, "lr": args.lr,
                   "batch_size": args.batch_size, "chance": chance, "device": "cpu"},
        "rows": rows,
        "sqrtN_const_mean": cmu, "sqrtN_const_std": csd, "sqrtN_const_cv": cv,
        "law_verdict": verdict,
        "validity_gate": {"attn_min": attn_min, "passed": bool(validity)},
        "elapsed_s": round(time.time() - t0, 1),
        "peak_rss_gb": round(_rss(), 2),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # plot (best-effort; the JSON is the source of truth)
    plotted = None
    try:
        make_plot(rows, chance, cmu, args.plot)
        plotted = args.plot
    except Exception as e:  # noqa: BLE001
        print(f"[plot] skipped: {type(e).__name__}: {str(e)[:80]}")

    print("\n" + "=" * 78)
    print("1/sqrt(N) LAW")
    print(f"  above_floor·sqrt(N): mean {cmu:.4f}  std {csd:.4f}  CV {cv:.3f}")
    print(f"  validity gate (attn ≥ 0.90, min over N): {attn_min:.4f}  "
          f"{'PASS' if validity else 'FAIL → recall labels VOID'}")
    print(f"  >>> {verdict}")
    print(f"\n→ {args.out}" + (f"\n→ {plotted}" if plotted else "")
          + f"   ({out['elapsed_s']}s, peak rss {out['peak_rss_gb']:.2f}GB)")


if __name__ == "__main__":
    main()
