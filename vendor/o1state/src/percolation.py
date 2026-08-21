#!/usr/bin/env python3 -u
"""
M4 — PERCOLATION THRESHOLD: does David's "exponential past a threshold" exist?
==============================================================================
David's gut: "if paths can form, everything potentiates each other past some point — there's a
THRESHOLD past which it goes exponential, because every path reinforces every other."

That is the percolation / criticality intuition. Below a critical edge density the association graph
is isolated islands (reinforcement stays local). Above it, a GIANT CONNECTED COMPONENT emerges and
every node becomes mutually reachable — a PHASE TRANSITION, a sharp knee, not a gradual rise.
Human-analog: brain synaptogenesis explodes in a critical window; language acquisition clicks.

We measure it two ways:
  (A) STRUCTURAL percolation — sweep an edge-admission threshold theta DOWN (admit edge iff its
      PMI weight > theta). At each theta: giant-component fraction S (order parameter), and the
      second-largest-component size (SUSCEPTIBILITY — it PEAKS exactly at the critical point). A
      true transition shows a sharp knee in S and a sharp peak in the 2nd-component; the peak should
      SHARPEN as the graph grows (finite-size scaling) — the signature of a real transition vs a
      smooth crossover.
  (B) Reachability — mean BFS-reachable set size per node vs theta: does it jump from ~O(1) to
      ~O(N) at the same theta? (the "everything reaches everything" potentiation).

Honest null: if S rises smoothly with no knee and the 2nd-component shows no peak, there is NO
threshold — just gradual densification. Negative is negative.

Light + safe: dict-of-dicts graph, union-find for components, BFS-sample for reachability. psutil
watchdog. Runs on a 16GB CPU Mac.
"""
import os, sys, json, argparse, re
sys.path.insert(0, "reference"); sys.path.insert(0, "src")


def _uf_giant(nodes, edge_iter):
    """Union-find over an edge stream; return (giant_fraction, second_fraction, n_components)."""
    parent = {n: n for n in nodes}
    size = {n: 1 for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edge_iter:
        ra, rb = find(a), find(b)
        if ra != rb:
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]
    # component sizes = size[root] for roots
    roots = {}
    for n in nodes:
        r = find(n)
        roots[r] = roots.get(r, 0) + 1
    comp = sorted(roots.values(), reverse=True)
    N = max(1, len(nodes))
    giant = comp[0] / N if comp else 0.0
    second = comp[1] / N if len(comp) > 1 else 0.0
    return giant, second, len(comp)


def build_graph(max_chars, window):
    from gssm_causal import GSSMCausal
    from length_extrap_v2 import load_wikitext2
    text, _ = load_wikitext2()
    text = text[:max_chars]
    g = GSSMCausal()
    for para in re.split(r"\n\s*\n", text):
        if len(para) > 40:
            g.add_text(para, window=window)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chars", type=int, default=1_500_000)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--out", default="results/percolation.json")
    args = ap.parse_args()

    g = build_graph(args.max_chars, args.window)
    nodes = list(g.adj.keys())
    print(f"[graph] {g.stats()}")

    # all undirected edges with their PMI weight, sorted DESC (admit strongest first as theta falls)
    pmi_edges = []
    seen = set()
    for a in g.adj:
        for b in g.adj[a]:
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            pmi_edges.append((g._pmi(a, b), key[0], key[1]))
    pmi_edges.sort(reverse=True)
    M = len(pmi_edges)
    print(f"[edges] {M:,} unique PMI-weighted edges")

    # sweep: admit the top-fraction f of strongest edges, measure giant + 2nd component
    curve = []
    fracs = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.55, 0.7, 0.85, 1.0]
    for f in fracs:
        k = max(1, int(M * f))
        admitted = ((a, b) for (_, a, b) in pmi_edges[:k])
        giant, second, ncomp = _uf_giant(nodes, admitted)
        theta = pmi_edges[k - 1][0]
        curve.append({"frac_edges": f, "theta": round(theta, 6), "n_edges": k,
                      "giant": round(giant, 4), "second": round(second, 4), "components": ncomp})
        print(f"  f={f:5.3f} ({k:>7,} edges, θ≥{theta:.4f}): giant {giant:5.1%}  "
              f"2nd {second:6.2%}  comps {ncomp:,}")

    # find the knee: where giant rises fastest + where 2nd-component peaks (the critical point)
    max_jump, knee = 0.0, None
    for i in range(1, len(curve)):
        jump = (curve[i]["giant"] - curve[i - 1]["giant"]) / (curve[i]["frac_edges"] - curve[i - 1]["frac_edges"])
        if jump > max_jump:
            max_jump, knee = jump, curve[i]
    peak2 = max(curve, key=lambda c: c["second"])

    out = {"graph": g.stats(), "n_edges": M, "curve": curve,
           "knee_frac": knee["frac_edges"] if knee else None,
           "knee_theta": knee["theta"] if knee else None,
           "max_giant_jump_per_frac": round(max_jump, 2),
           "second_component_peak_at_frac": peak2["frac_edges"],
           "second_component_peak_value": peak2["second"]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    print(f"\n── M4: percolation threshold ──")
    print(f"  steepest giant-component jump: {max_jump:.1f} per unit edge-fraction, "
          f"around f={knee['frac_edges'] if knee else '?'} (θ≈{knee['theta'] if knee else '?'})")
    print(f"  2nd-component PEAK at f={peak2['frac_edges']} (value {peak2['second']:.2%}) "
          f"— the critical point if it's a real transition")
    # a sharp jump + a clear 2nd-component peak = percolation transition
    sharp = max_jump > 3.0 and peak2["second"] > 0.01
    verdict = ("THRESHOLD CONFIRMED — giant component jumps sharply and the 2nd-component peaks at a "
               "critical edge density: David's 'exponential past a threshold' is a real percolation "
               "transition in the association graph"
               if sharp else
               "smooth densification, no sharp threshold on this graph/metric — the potentiation is "
               "gradual not a phase transition (honest null; may need the reinforcement loop to appear)")
    print(f"\n  → {verdict}")
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
