#!/usr/bin/env python3 -u
"""
Scale-to-a-million figure: flat PPL to 16.7M tokens (524,288×) at CONSTANT memory.
Twin axes: PPL stays flat (even improves) while RSS stays flat as length grows 2000×.
Data: results/scale_to_a_million.json
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
with open(os.path.join(REPO, "results", "scale_to_a_million.json")) as f:
    d = json.load(f)

TRAIN_T = d["train_T"]
Ls = sorted(int(x) for x in d["curve"])
ppl = [d["curve"][str(L)]["ppl"] for L in Ls]
rss = [d["curve"][str(L)]["peak_rss_gb"] for L in Ls]
base = d["base_ppl"]

fig, ax1 = plt.subplots(figsize=(9.6, 5.4))

# PPL (left axis) — the flat (improving) line
c1 = "#1b9e77"
ax1.plot(Ls, ppl, "-o", lw=3.0, ms=9, color=c1, zorder=5, label="perplexity (left)")
ax1.axhspan(base * 0.70, base * 1.05, color=c1, alpha=0.06, zorder=0)
ax1.set_ylabel("validation perplexity", fontsize=11, color=c1)
ax1.tick_params(axis="y", labelcolor=c1)
ax1.set_ylim(base * 0.62, base * 1.12)

# RSS (right axis) — the flat memory line
ax2 = ax1.twinx()
c2 = "#d95f02"
ax2.plot(Ls, rss, "--s", lw=2.0, ms=6, color=c2, zorder=4, label="peak memory (right)")
ax2.set_ylabel("peak process memory (GB)", fontsize=11, color=c2)
ax2.tick_params(axis="y", labelcolor=c2)
ax2.set_ylim(0, 16)
ax2.axhline(16, ls=":", color="#999999", lw=1.0)
ax2.text(Ls[0], 16.3, "machine RAM = 16 GB", fontsize=8, color="#888888")

# annotate the headline point
ax1.annotate(f"16.7M tokens = 524,288× training length\n×{ppl[-1]/base:.2f} PPL — "
             f"flat (better!), {rss[-1]:.1f} GB",
             xy=(Ls[-1], ppl[-1]), xytext=(Ls[0] * 3, base * 0.74),
             fontsize=10, color="#147a5a", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=c1, lw=1.5))

ax1.set_xscale("log", base=2)
ax1.set_xticks(Ls)
ax1.xaxis.set_major_locator(FixedLocator(Ls))
def fmt(L):
    if L >= 1_000_000: return f"{L//1_000_000}M\n({L//TRAIN_T//1000}k×)"
    if L >= 1000:      return f"{L//1000}k\n({L//TRAIN_T}×)"
    return f"{L}\n({L//TRAIN_T}×)"
ax1.set_xticklabels([fmt(L) for L in Ls], fontsize=8.5)
ax1.set_xlabel("effective sequence length  (tokens streamed, log scale)", fontsize=11)
ax1.set_title("No length wall: flat PPL to 16.7M tokens (524,288×) at CONSTANT memory\n"
              "chunked O(1)-state streaming — length is time-limited, never RAM-limited",
              fontsize=12, pad=12)
ax1.grid(True, which="both", ls="-", alpha=0.12)

# combined legend
lines1, lab1 = ax1.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, lab1 + lab2, loc="lower left", fontsize=9.5, framealpha=0.95)

fig.tight_layout()
out = os.path.join(REPO, "plots", "scale_to_a_million.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=160, bbox_inches="tight")
print(f"wrote {out}")
