#!/usr/bin/env python3 -u
"""
Scale-to-a-BILLION figure: flat PPL to 1,000,000,000 tokens of effective sequence length
at CONSTANT memory. The corpus is streamed from C4 (never downloaded) and the eval is
chunked (activations never materialized) — doubly O(1). Length is time-limited, never
RAM-limited.

Twin axes: running PPL stays flat while RSS stays pinned as the streamed length grows past
1B (30,000,000× the T=32 training length).
Data: results/scale_to_a_billion.json  (checkpoints = list of [streamed, ppl, rss])
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
with open(os.path.join(REPO, "results", "scale_to_a_billion.json")) as f:
    d = json.load(f)

TRAIN_T = d["train_T"]
ckpts = d["checkpoints"]                       # [[streamed, ppl, rss], ...]
streamed = [c[0] for c in ckpts]
ppl = [c[1] for c in ckpts]
rss = [c[2] for c in ckpts]
# include the final point if it extends past the last checkpoint
if d["tokens_streamed"] > streamed[-1]:
    streamed.append(d["tokens_streamed"])
    ppl.append(d["final_ppl"])
    rss.append(d["peak_rss_gb"])
base = ppl[0]

fig, ax1 = plt.subplots(figsize=(10.2, 5.6))

# PPL (left axis) — the flat line
c1 = "#1b9e77"
ax1.plot(streamed, ppl, "-o", lw=3.0, ms=7, color=c1, zorder=5, label="running perplexity (left)")
ax1.axhspan(base * 0.85, base * 1.15, color=c1, alpha=0.06, zorder=0)
ax1.set_ylabel("running perplexity (C4-en, streamed)", fontsize=11, color=c1)
ax1.tick_params(axis="y", labelcolor=c1)
lo = min(ppl) * 0.85
hi = max(ppl) * 1.15
ax1.set_ylim(lo, hi)

# RSS (right axis) — the flat memory line
ax2 = ax1.twinx()
c2 = "#d95f02"
ax2.plot(streamed, rss, "--s", lw=2.0, ms=5, color=c2, zorder=4, label="peak memory (right)")
ax2.set_ylabel("peak process memory (GB)", fontsize=11, color=c2)
ax2.tick_params(axis="y", labelcolor=c2)
ax2.set_ylim(0, 18)
ax2.axhline(16, ls=":", color="#999999", lw=1.0)
ax2.text(streamed[0], 16.3, "machine RAM = 16 GB", fontsize=8, color="#888888")

# headline annotation
final = streamed[-1]
ax1.annotate(f"{final/1e9:.2f}B tokens = {final//TRAIN_T:,}× training length\n"
             f"×{ppl[-1]/base:.2f} PPL — FLAT, {rss[-1]:.1f} GB constant",
             xy=(final, ppl[-1]), xytext=(streamed[0] * 6, base * 1.10),
             fontsize=10.5, color="#147a5a", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=c1, lw=1.5))

ax1.set_xscale("log")
def fmt(x, _):
    if x >= 1e9: return f"{x/1e9:.1f}B"
    if x >= 1e6: return f"{x/1e6:.0f}M"
    if x >= 1e3: return f"{x/1e3:.0f}k"
    return f"{x:.0f}"
ax1.xaxis.set_major_formatter(FuncFormatter(fmt))
ax1.set_xlabel("effective sequence length  (tokens streamed through one O(1) state, log scale)",
               fontsize=11)
ax1.set_title("No length wall: flat PPL to 1 BILLION tokens at CONSTANT memory\n"
              "doubly-O(1) — C4 streamed (never downloaded) + chunked eval (never materialized); "
              "length is time-limited, never RAM-limited",
              fontsize=11.5, pad=12)
ax1.grid(True, which="both", ls="-", alpha=0.12)

lines1, lab1 = ax1.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, lab1 + lab2, loc="lower center", fontsize=9.5, framealpha=0.95)

fig.tight_layout()
out = os.path.join(REPO, "plots", "scale_to_a_billion.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=160, bbox_inches="tight")
print(f"wrote {out}  ({len(streamed)} points, {final:,} tokens, "
      f"PPL ×{ppl[-1]/base:.2f}, {rss[-1]:.1f}GB)")
