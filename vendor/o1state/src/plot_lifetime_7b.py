#!/usr/bin/env python3 -u
"""
The 7-BILLION-token life: one online-learning organism, one process stretch of
12 days, constant memory across almost four orders of magnitude of experience.

Twin axes in the house style of plot_billion.py: loss EMA (left) stays level
while RSS (right) stays pinned far below the run's own self-pause threshold
(4 GB) as the life grows from 0.5B to 7B+ streamed tokens. The one stall
(2026-07-24, HF-stream hang) and its checkpoint self-heal are marked — the
organism resumed ~51k tokens behind and lost about three minutes of stream.

Data: results/lifetime_7b_series.json (downsampled from the run's own logs;
the life predates the oldest retained log, so the series starts at 512M).
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
with open(os.path.join(REPO, "results", "lifetime_7b_series.json")) as f:
    d = json.load(f)

s = d["series"]
tok = [p["tok"] for p in s]
loss = [p["loss_ema"] for p in s]
rss = [p["rss_gb"] for p in s]

STALL_TOK = 869_734_400          # last logged token before the 2026-07-24 restart

fig, ax1 = plt.subplots(figsize=(10.2, 5.6))

c1 = "#1b9e77"
ax1.plot(tok, loss, "-", lw=1.6, color=c1, zorder=5, label="loss EMA (left)")
ax1.set_ylabel("online loss EMA (C4-en, streamed)", fontsize=11, color=c1)
ax1.tick_params(axis="y", labelcolor=c1)
ax1.set_ylim(min(loss) - 0.25, max(loss) + 0.35)

ax2 = ax1.twinx()
c2 = "#d95f02"
ax2.plot(tok, rss, "-", lw=1.6, color=c2, zorder=4, label="process memory (right)")
ax2.set_ylabel("process memory RSS (GB)", fontsize=11, color=c2)
ax2.tick_params(axis="y", labelcolor=c2)
ax2.set_ylim(0, 4.6)
ax2.axhline(4.0, ls=":", color="#999999", lw=1.0)
ax2.text(tok[0], 4.12, "run's own self-pause threshold = 4 GB", fontsize=8, color="#888888")

# the one stall + checkpoint self-heal
ax1.axvline(STALL_TOK, ls="--", color="#7570b3", lw=1.0, alpha=0.7)
ax1.annotate("HF-stream stall, 2026-07-24 —\nself-healed from own checkpoint\n(resumed 51k tokens back, ~3 min lost)",
             xy=(STALL_TOK, max(loss) + 0.05), xytext=(1.15e9, max(loss) + 0.02),
             fontsize=8.5, color="#5e5a9e",
             arrowprops=dict(arrowstyle="->", color="#7570b3", lw=1.0))

final = tok[-1]
ax1.annotate(f"{final/1e9:.2f}B tokens in one life — {d['n_params']:,} parameters\n"
             f"12 days in ONE process since the heal, RSS {rss[-1]:.2f} GB",
             xy=(final, loss[-1]), xytext=(3.4e9, min(loss) + 0.02),
             fontsize=10.5, color="#147a5a", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=c1, lw=1.5))

def fmt(x, _):
    return f"{x/1e9:.1f}B" if x >= 1e9 else f"{x/1e6:.0f}M"
ax1.xaxis.set_major_formatter(FuncFormatter(fmt))
ax1.set_xlabel("tokens streamed through one O(1) state (one continuous life)", fontsize=11)
ax1.set_title("The 7-billion-token life: constant memory across the whole of experience\n"
              "1.7M-parameter organism, online learning on a live C4 stream — no dataset, "
              "no epochs, no growing context; one stall, self-healed",
              fontsize=11.5, pad=12)
ax1.grid(True, which="both", ls="-", alpha=0.12)

lines1, lab1 = ax1.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, lab1 + lab2, loc="center right", fontsize=9.5, framealpha=0.95)

fig.tight_layout()
for out in [os.path.join(REPO, "plots", "lifetime_7b.png"),
            os.path.join(REPO, "paper", "figures", "fig_lifetime_7b.pdf")]:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")
print(f"({len(tok)} points, {final:,} tokens, loss {loss[0]} -> {loss[-1]}, "
      f"rss min {min(rss)} max {max(rss)})")
