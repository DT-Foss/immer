#!/usr/bin/env python3 -u
"""
LIVING-STREAM figure — the constant-memory streaming-training flag in four panels.
==================================================================================
  (A) constant-memory TRAINING: held-out loss falls while RSS stays flat
  (D) idle-persistence: a planted bit survives an input gap; zero the state → recall dies
  (E) the mechanism: the model grew a dedicated long-memory channel (γ≈1, input-gate shut
      in the gap, open at the beacon) — a learned bit-vault. mean-γ hid it.
Data: results/streaming_train.json, results/idle_persistence.json, results/carrier_probe.json
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
def load(name):
    return json.load(open(os.path.join(REPO, "results", name)))

A = load("streaming_train.json")
D = load("idle_persistence.json")
E = load("carrier_probe.json")

fig = plt.figure(figsize=(13.5, 9.0))
gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.26)

# ── Panel A: constant-memory training ──────────────────────────────────────
axA = fig.add_subplot(gs[0, 0])
toks = [c[0] for c in A["curve"]]
hl = [c[1] for c in A["curve"]]
rss = [c[2] for c in A["curve"]]
cL, cR = "#1b9e77", "#d95f02"
axA.plot([t / 1e6 for t in toks], hl, "-o", color=cL, lw=2.5, ms=5, label="held-out loss")
axA.set_ylabel("held-out loss (never streamed)", color=cL, fontsize=10)
axA.tick_params(axis="y", labelcolor=cL)
axA.set_xlabel("tokens streamed through training (millions)", fontsize=10)
axA2 = axA.twinx()
axA2.plot([t / 1e6 for t in toks], rss, "--s", color=cR, lw=2.0, ms=4, label="peak RSS")
axA2.set_ylabel("peak RSS (GB)", color=cR, fontsize=10)
axA2.tick_params(axis="y", labelcolor=cR)
axA2.set_ylim(0, max(2.0, max(rss) * 1.5))
axA.set_title(f"(A) Constant-memory streaming TRAINING\n"
              f"held-out {A['base_heldout']:.2f}→{A['final_heldout']:.2f}, "
              f"RSS flat ~{A['peak_rss_gb']:.1f} GB", fontsize=11, fontweight="bold")

# ── Panel D: idle-persistence recall vs gap ─────────────────────────────────
axD = fig.add_subplot(gs[0, 1])
gaps = sorted(int(g) for g in D["gaps"])
carried = [D["gaps"][str(g)]["carried"] for g in gaps]
zeroed = [D["gaps"][str(g)]["zeroed_at_gap"] for g in gaps]
noplant = [D["gaps"][str(g)]["no_plant"] for g in gaps]
axD.plot(gaps, carried, "-o", color="#1b9e77", lw=2.5, ms=6, label="carried state")
axD.plot(gaps, zeroed, "--s", color="#d95f02", lw=2.0, ms=5, label="state zeroed at gap")
axD.plot(gaps, noplant, ":^", color="#999999", lw=1.5, ms=4, label="no beacon (control)")
axD.axhline(0.5, color="#cccccc", lw=1.0, ls=":")
axD.set_ylim(0.4, 1.05)
axD.set_xlabel("input-gap length (filler tokens)", fontsize=10)
axD.set_ylabel("beacon recall accuracy", fontsize=10)
axD.legend(fontsize=9, loc="center left")
sep = D.get("decisive_separation", carried[-1] - zeroed[-1])
axD.set_title(f"(D) Idle-persistence: the state lives through silence\n"
              f"bit survives to G*={D.get('horizon_G_star')}, "
              f"zeroing the state kills it (sep +{sep:.2f})", fontsize=11, fontweight="bold")

# ── Panel E: the carrier channel state trajectory ───────────────────────────
axE = fig.add_subplot(gs[1, 0])
carrier = max(E["layers"], key=lambda L: abs(L["corr"]))
z0 = carrier["traj_z0"]; z1 = carrier["traj_z1"]
xs = list(range(len(z0)))
axE.plot(xs, z1, "-", color="#c0392b", lw=2.6, label="state z (beacon=1)")
axE.plot(xs, z0, "-", color="#3b6fb6", lw=2.6, label="state z (beacon=0)")
axE.fill_between(xs, z0, z1, color="#888888", alpha=0.10)
zlo, zhi = min(min(z0), min(z1)), max(max(z0), max(z1))
pad = 0.15 * (zhi - zlo + 1e-6)
axE.set_ylim(zlo - pad, zhi + pad)
axE.annotate(f"margin {abs(z1[0]-z0[0]):.1f} → {abs(z1[-1]-z0[-1]):.1f}\n"
             f"(×{carrier['margin_ratio_end_start']:.2f} over {len(z0)} tokens)",
             xy=(len(z0) * 0.5, (z0[0] + z1[0]) / 2), fontsize=9.5, ha="center",
             color="#444444", fontweight="bold")
axE.set_xlabel("position into the input gap (tokens)", fontsize=10)
axE.set_ylabel(f"carrier-channel state z  (L{carrier['layer']} H{carrier['carrier_head']}C{carrier['carrier_chan']})",
               fontsize=10)
axE.legend(fontsize=9, loc="center right")
axE.set_title(f"(E) The learned bit-vault: a γ≈{carrier['gamma_carrier_max']:.2f} channel\n"
              f"the two classes stay fully separated across {len(z0)} tokens (margin ×{carrier['margin_ratio_end_start']:.2f})",
              fontsize=11, fontweight="bold")

# ── Panel F: the carrier gates (γ high, α shut in gap, open at beacon) ───────
axF = fig.add_subplot(gs[1, 1])
g_tr = carrier["traj_gamma"]; a_tr = carrier["traj_alpha"]
axF.plot(xs, g_tr, "-", color="#6a3d9a", lw=2.2, label="γ forget-gate (hold)")
axF.plot(xs, a_tr, "-", color="#e08214", lw=2.2, label="α input-gate (write)")
axF.axhline(1.0, color="#cccccc", lw=1.0, ls=":")
axF.scatter([0], [carrier["alpha_carrier_beacon"]], color="#e08214", s=80, zorder=5,
            marker="*", label=f"α at beacon = {carrier['alpha_carrier_beacon']:.2f}")
axF.set_ylim(-0.05, 1.1)
axF.set_xlabel("position into the input gap (tokens)", fontsize=10)
axF.set_ylabel("gate value", fontsize=10)
axF.legend(fontsize=9, loc="center right")
axF.set_title(f"(E) Why it holds: γ≈1 (never forgets), α≈{carrier['alpha_carrier_gap']:.2f} in gap\n"
              f"(input gated OUT — frozen, not decaying), opens at the beacon",
              fontsize=11, fontweight="bold")

fig.suptitle("LIVING-STREAM: a GSSM trains on an unbounded stream at constant memory, "
             "and its persistent state carries a bit through silence",
             fontsize=13, fontweight="bold", y=0.98)
out = os.path.join(REPO, "plots", "living_stream.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
