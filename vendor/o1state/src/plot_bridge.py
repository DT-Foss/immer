#!/usr/bin/env python3 -u
"""
M5 bridge figure: the threshold in the REAL GSSM readout lives in the NONLINEAR GATE.
Left  = capacity vs load K/D: LINEAR readout rolls off smoothly (no threshold) while the GATED m·tanh
        readout CLIFFS at K/D≈1 — the neural counterpart of the percolation jump, in the state itself.
Right = the dynamical dissociation: potentiation needs LATENT structure; the bounded O(1) state has no
        latent-revivable regime (present⇒already clean, over-capacity⇒erased), which is the mechanistic
        reason the dynamical threshold belongs to the graph/index (cortex), not the state (hippocampus).
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
d = json.load(open(os.path.join(REPO, "results", "gssm_potentiation.json")))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4))

# ── LEFT: linear (smooth) vs gated (cliff) ──
curve = d["structural"]["curve"]
load = [c["load"] for c in curve]
lin = [c["fidelity"] for c in curve]
gat = [c["fidelity_gated"] for c in curve]
axL.plot(load, lin, "-o", color="#7570b3", lw=2.4, ms=6, label="LINEAR readout (rolls off — no threshold)")
axL.plot(load, gat, "-s", color="#d95f02", lw=2.8, ms=6, label="GATED m·tanh readout (cliffs)")
ga = d["structural"]["gated"]; li = d["structural"]["linear"]
kc = ga["cliff_at_load"]
if kc is not None:
    axL.axvline(kc, color="#888", ls=":", lw=1.2)
    axL.text(kc + 0.03, 0.5, f"critical K/D≈{kc}", rotation=90, va="center", fontsize=9, color="#555")
axL.set_xlabel("load  K/D  (facts written into one bounded GSSM state)", fontsize=10)
axL.set_ylabel("readout fidelity (corr with true value)", fontsize=10)
axL.set_ylim(-0.02, 1.04)
axL.legend(loc="upper right", fontsize=9)
axL.set_title(f"(1) the threshold is REAL in the neural state — in the GATE\n"
              f"gated slope {ga['max_slope']:.2f}/load (span {ga['transition_span']}) vs "
              f"linear {li['max_slope']:.2f}/load — sharp={ga['sharp_cliff']}",
              fontsize=11, fontweight="bold")

# ── RIGHT: the dynamical dissociation (latent vs deleted structure) ──
arms = d["dynamical"]["arms"]
def series(name): return arms[name]
g09 = series("gated"); gov = series("gated_over"); gsh = series("gated_shuffle")
x09 = range(len(g09)); xov = range(len(gov))
axR.plot(x09, g09, "-o", color="#1b9e77", lw=2.4, ms=3,
         label="gated @load0.9  (structure PRESENT → already clean)")
axR.plot(xov, gov, "-s", color="#d95f02", lw=2.4, ms=3,
         label="gated @load1.4  (over capacity → structure ERASED)")
axR.plot(range(len(gsh)), gsh, "--", color="#999999", lw=1.6,
         label="gated @0.9 shuffled keys (null)")
axR.set_xlabel("use→reinforce iteration  (gate sharpen + key de-correlate, structure FROZEN)", fontsize=10)
axR.set_ylabel("readout fidelity", fontsize=10)
axR.set_ylim(-0.05, 1.05)
axR.legend(loc="center right", fontsize=8.5)
axR.set_title("(2) why dynamical potentiation belongs to the GRAPH, not the O(1) state\n"
              "no latent-revivable regime: present⇒clean, over-capacity⇒gone — nothing to potentiate",
              fontsize=11, fontweight="bold")

fig.suptitle("M5 BRIDGE — David's threshold in the real GSSM readout: a sharp GATED capacity cliff "
             "(structural), and the honest reason potentiation lives in the index (dynamical)",
             fontsize=12, fontweight="bold", y=0.99)
out = os.path.join(REPO, "plots", "bridge_gssm_threshold.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
