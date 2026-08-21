#!/usr/bin/env python3 -u
"""
Length-invariance figure — the one plot that tells the whole story.
NoPE-Selective is the only flat line; everything else breaks.
Data: results/length_extrap_v2_extreme.json (+ Transformer from the run log, it crashed at T=4096).
"""
import os, sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# --- data (verified from results/length_extrap_v2_extreme.json + run log) ---
T = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
nope   = [165, 161, 154, 160, 163, 155, 160, 159, 160]
sel_pe = [169, 165, 160, 168, 181, 196, 305, 473, 714]
pure   = [231, 272, 333, 981, 2467, 2855, 2810, 2774, 2603]
# Transformer crashed at T=4096 (fixed PE buffer max_len=2048 → tensor-size error)
tf_T   = [32, 64, 128, 256, 512, 1024, 2048]
tf     = [226, 258, 275, 303, 317, 307, 341]

TRAIN_T = 32

fig, ax = plt.subplots(figsize=(9, 5.6))

# the hero line — NoPE, thick, on top
ax.plot(T, nope, "-o", lw=3.0, ms=7, color="#1b9e77", zorder=5,
        label="Selective-NoPE  (selective gate, no PE)  →  ×0.97 at 256×")
ax.plot(T, sel_pe, "-s", lw=2.0, ms=5, color="#d95f02", zorder=4,
        label="Selective + PE  →  ×4.2 (the PE drifts)")
ax.plot(T, pure, "-^", lw=2.0, ms=5, color="#7570b3", zorder=3,
        label="Pure  (bounded, no selective gate)  →  ×11–12 (gate is necessary)")
ax.plot(tf_T, tf, "-D", lw=2.0, ms=5, color="#666666", zorder=4,
        label="Transformer  →  drifts, then CANNOT RUN past T=2048")

# mark the transformer crash
ax.annotate("Transformer cannot run past T=2048\n(fixed PE buffer, max_len=2048)",
            xy=(2048, 341), xytext=(2150, 205),
            fontsize=8.5, color="#444444", ha="left",
            arrowprops=dict(arrowstyle="->", color="#666666", lw=1.2))
ax.plot([2048], [341], "x", ms=13, mew=3.0, color="#444444", zorder=6)

# training length marker
ax.axvline(TRAIN_T, ls=":", color="#999999", lw=1.2)
ax.text(TRAIN_T * 1.15, 2900, "trained here\n(T=32)", fontsize=8.5, color="#666666",
        va="top")

# shade the extrapolation region
ax.axvspan(TRAIN_T, max(T), color="#000000", alpha=0.025, zorder=0)

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(T)
ax.xaxis.set_major_locator(FixedLocator(T))
ax.set_xticklabels([f"{t}\n({t//TRAIN_T}×)" if t > TRAIN_T else f"{t}\n(1×)" for t in T],
                   fontsize=8.5)
ax.set_xlabel("evaluation sequence length  (× training length)", fontsize=11)
ax.set_ylabel("validation perplexity  (log scale)", fontsize=11)
ax.set_title("Length invariance: train at T=32, evaluate to 256× (T=8192)\n"
             "NoPE-Selective is the only architecture that stays flat",
             fontsize=12.5, pad=12)

ax.grid(True, which="both", ls="-", alpha=0.12)
ax.legend(loc="center left", bbox_to_anchor=(0.012, 0.42), fontsize=9.0,
          framealpha=0.96)
ax.set_ylim(120, 3600)

# annotate the headline gap at T=8192
ax.annotate("", xy=(8192, 160), xytext=(8192, 714),
            arrowprops=dict(arrowstyle="<->", color="#d95f02", lw=1.4, alpha=0.7))
ax.text(8192 * 0.62, 350, "4.4×\ngap", fontsize=9, color="#d95f02", ha="center",
        fontweight="bold")

fig.tight_layout()
out_png = os.path.join(REPO, "plots", "length_invariance.png")
os.makedirs(os.path.dirname(out_png), exist_ok=True)
fig.savefig(out_png, dpi=160, bbox_inches="tight")
print(f"wrote {out_png}")
# also a public-repo copy path hint
print("→ copy to gssm-public/plots/ for the README")
