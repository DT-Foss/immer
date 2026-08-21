#!/usr/bin/env python3 -u
"""
Capability-boundary figure: flip-flop state tracking, NoPE-GSSM vs Transformer.
NoPE-GSSM holds 100% to 128x; the Transformer degrades then its forward pass crashes.
Data: results/longcontext_flipflop.json
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

with open(os.path.join(REPO, "results", "longcontext_flipflop.json")) as f:
    d = json.load(f)

TRAIN_T = d["train_len"]
Ts = d["eval_lens"]
nope = [d["arms"]["nope_gssm"][str(T)]["acc"] for T in Ts]
tf_acc, tf_crash = [], []
for T in Ts:
    e = d["arms"]["transformer"][str(T)]
    if e.get("acc") is not None:
        tf_acc.append((T, e["acc"]))
    else:
        tf_crash.append(T)

fig, ax = plt.subplots(figsize=(9, 5.4))

# NoPE — perfect flat 100%
ax.plot(Ts, [a * 100 for a in nope], "-o", lw=3.0, ms=8, color="#1b9e77", zorder=5,
        label="NoPE-GSSM (bounded, O(1) memory)  —  100% to 128×")
# Transformer where it runs
ax.plot([t for t, _ in tf_acc], [a * 100 for _, a in tf_acc], "-D", lw=2.0, ms=6,
        color="#666666", zorder=4,
        label="Transformer (where it can run)")
# crash region
if tf_crash:
    first_crash = min(tf_crash)
    ax.axvspan(first_crash * 0.85, max(Ts) * 1.1, color="#d62728", alpha=0.06, zorder=0)
    ax.text(first_crash * 1.7, 50,
            "Transformer\nforward pass\nCRASHES here\n(PE buffer, max_len=1024)",
            fontsize=9.5, color="#b22222", ha="center", va="center", fontweight="bold")
    for T in tf_crash:
        ax.plot([T], [2], "X", ms=13, mew=3, color="#d62728", zorder=6)

# chance line
ax.axhline(100 / 8, ls=":", color="#999999", lw=1.0)
ax.text(Ts[0] * 1.1, 100 / 8 + 2, "chance (1/8)", fontsize=8, color="#888888")

# training length
ax.axvline(TRAIN_T, ls=":", color="#aaaaaa", lw=1.2)
ax.text(TRAIN_T * 1.1, 8, "trained here\n(T=64)", fontsize=8.5, color="#666666", va="bottom")

ax.set_xscale("log", base=2)
ax.set_xticks(Ts)
ax.xaxis.set_major_locator(FixedLocator(Ts))
ax.set_xticklabels([f"{t}\n({t//TRAIN_T}×)" for t in Ts], fontsize=9)
ax.set_xlabel("evaluation sequence length  (× training length)", fontsize=11)
ax.set_ylabel("flip-flop recall accuracy  (%)", fontsize=11)
ax.set_title("Capability boundary: long-range state tracking\n"
             "NoPE-GSSM stays at 100%; the Transformer degrades, then cannot run at all",
             fontsize=12.5, pad=12)
ax.set_ylim(-3, 108)
ax.grid(True, which="both", ls="-", alpha=0.12)
ax.legend(loc="center left", fontsize=9.5, framealpha=0.96,
          bbox_to_anchor=(0.012, 0.30))

fig.tight_layout()
out = os.path.join(REPO, "plots", "capability_flipflop.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=160, bbox_inches="tight")
print(f"wrote {out}")
