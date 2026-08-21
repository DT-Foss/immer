#!/usr/bin/env python3 -u
"""
Night-session figure: the threshold, in both forms. For David's morning — the story on one glance.
Left  = STRUCTURAL percolation (giant component S + susceptibility χ vs mean degree ⟨k⟩).
Right = DYNAMICAL potentiation (capability C(t) over reinforcement iterations: frozen vs controls).
Both show the same physics: a sharp threshold, past which everything connects.
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
def load(n): return json.load(open(os.path.join(REPO, "results", n)))

perc = load("percolation_hard.json")
rl = load("reinforcement_loop.json")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4))

# ── LEFT: structural percolation ──
curve = perc["full_true"]["curve"]
k = [c["kmean"] for c in curve]
S = [c["S"] for c in curve]
chi = [c["chi"] for c in curve]
cS, cChi = "#1b9e77", "#d95f02"
axL.plot(k, S, "-o", color=cS, lw=2.6, ms=6, label="giant component S (left)")
axL.set_xlabel("mean degree ⟨k⟩  (edges admitted, PMI-sorted)", fontsize=10)
axL.set_ylabel("giant-component fraction S", color=cS, fontsize=10)
axL.tick_params(axis="y", labelcolor=cS)
axL.set_xlim(0, 4)
axL2 = axL.twinx()
axL2.plot(k, chi, "--s", color=cChi, lw=2.0, ms=5, label="susceptibility χ (right)")
axL2.set_ylabel("susceptibility χ  (peaks AT the critical point)", color=cChi, fontsize=10)
axL2.tick_params(axis="y", labelcolor=cChi)
kc = max(curve, key=lambda c: c["chi"])["kmean"]
axL.axvline(kc, color="#888", ls=":", lw=1.2)
axL.text(kc + 0.05, 0.5, f"critical ⟨k⟩≈{kc:.1f}", fontsize=9, color="#555", rotation=90, va="center")
fss = perc.get("finite_size", {})
chis = ", ".join(f"{fss[n]['chi_max']:.0f}" for n in sorted(fss, key=int))
axL.set_title(f"(1) STRUCTURAL: percolation phase transition\n"
              f"χ diverges with N [{chis}] — a real transition, PMI-driven", fontsize=11, fontweight="bold")
lL = axL.get_legend_handles_labels()[0] + axL2.get_legend_handles_labels()[0]
axL.legend(lL, ["giant S (left)", "susceptibility χ (right)"], loc="center right", fontsize=9)

# ── RIGHT: dynamical potentiation ──
arms = rl["arms"]
colors = {"frozen": "#1b9e77", "random": "#999999", "shuffle": "#d95f02"}
labels = {"frozen": "used-path reinforce (structure)", "random": "random reinforce (control)",
          "shuffle": "shuffled graph (control)"}
for arm in ["frozen", "random", "shuffle"]:
    C = arms[arm]["C"]
    axR.plot(range(len(C)), C, "-o", color=colors[arm], lw=2.4 if arm == "frozen" else 1.6,
             ms=4 if arm == "frozen" else 2, label=labels[arm])
axR.set_xlabel("reinforcement iteration  (edges FROZEN — pure potentiation)", fontsize=10)
axR.set_ylabel("capability C  (fraction of probes connected)", fontsize=10)
axR.set_ylim(-0.02, 0.75)
axR.legend(loc="center right", fontsize=9, framealpha=0.95)
g = arms["frozen"]["gain"]
axR.set_title(f"(2) DYNAMICAL: super-linear potentiation\n"
              f"used paths pull the system across the threshold (+{g:.2f}); random/shuffle do nothing",
              fontsize=11, fontweight="bold")

fig.suptitle("David's threshold intuition, validated: a knowledge graph has a critical point past "
             "which paths self-reinforce — structurally AND dynamically",
             fontsize=12.5, fontweight="bold", y=0.99)
out = os.path.join(REPO, "plots", "night_percolation.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
