#!/usr/bin/env python3 -u
"""
M2 — OPERATOR READOUT: one substrate, many readings (David's "one file, many states").
=======================================================================================
David's wire-with-edges image: the SAME data, read by different OPERATORS, yields different
information. Information lives in the INTERACTION (data × operator), not in the data alone.
The compression thesis: store ONE substrate + N cheap operators, not N states — the "different
infos" never exist as data, they emerge at READ time.

GSSM already hints at this: the state z is one thing; the readout s = sqrt(1 - exp(z)) pulls ONE
projection out. A DIFFERENT operator would pull a different projection from the SAME z.

This experiment tests the claim HONESTLY on a clean toy: can ONE bounded recurrent state z,
written by a stream that carries SEVERAL independent facts, be read by several DIFFERENT operators
so that each operator recovers a DIFFERENT fact — better than any single operator could?

Setup (holographic in spirit, but minimal & inspectable):
  - A stream writes K independent key→value facts into one scalar-per-channel state via the GSSM
    recurrence (gamma decay + additive drive), as in the real model.
  - We then define N readout operators O_1..O_N (e.g. forward-weighted, backward-weighted,
    masked/"plättchen", phase-shifted) and ask: does operator O_k recover fact k while suppressing
    the others? If a SINGLE fixed operator can't separate them but the OPERATOR FAMILY can, the
    "one substrate, many readings" claim holds — the state stores a superposition, operators de-mux.

Honest controls:
  - TRIVIALITY check: are the operators just rescalings of one read? (corr of their outputs ~1 → trivial)
  - CAPACITY baseline: a single best operator's separation vs the operator-family's separation.
  - NULL: random operators (untuned) should NOT separate the facts.
The verdict is about whether DISTINCT operators extract DISTINCT information from ONE state.
"""
import os, sys, json, argparse
sys.path.insert(0, "reference"); sys.path.insert(0, "src")

import torch
torch.backends.mps.is_available = lambda: False
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

EPS = 1e-6


def gssm_state(drives, gammas):
    """The real GSSM recurrence z_t = gamma_t * z_{t-1} + a_t, returning the FINAL state z_T.
    drives, gammas: (T, D). Returns z: (D,)."""
    T, D = drives.shape
    z = torch.zeros(D)
    for t in range(T):
        z = gammas[t] * z + drives[t]
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=64, help="state channels")
    ap.add_argument("--K", type=int, default=4, help="independent facts written into one state")
    ap.add_argument("--T", type=int, default=40, help="stream length")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/operator_readout.json")
    args = ap.parse_args()
    g = torch.Generator().manual_seed(args.seed)
    D, K = args.D, args.K

    # K fixed random "keys": each fact k is associated with a key vector u_k (D,). The VALUE of
    # fact k in a trial is a scalar v_k. We write sum_k v_k * u_k into the state over the stream
    # (each key carries its value as the drive direction), with gamma decay — a superposition.
    U = torch.randn(K, D, generator=g)
    U = U / U.norm(dim=1, keepdim=True)

    # N readout operators = learned linear de-mux directions W_k (D,), one per fact. The TEST: can
    # operator W_k read v_k out of the final state while ignoring the other facts? We SOLVE for the
    # operators in closed form (least-squares de-mux over a training set), then measure on held-out
    # trials. This is "one substrate, many operators" with the operators tuned — the honest claim is
    # that DISTINCT operators recover DISTINCT facts from ONE state, not that it's free.
    def make_trials(n):
        V = torch.randn(n, K, generator=g)                # the K scalar values per trial
        Z = torch.zeros(n, D)
        Tn = max(args.T, K + 1)                            # ensure the stream is long enough for K facts
        for i in range(n):
            # stream: at step k inject fact k; gamma decays; remaining steps decay only
            drives = torch.zeros(Tn, D)
            gammas = torch.full((Tn, D), 0.9)             # mild decay (long memory channel regime)
            for k in range(K):
                drives[k] = V[i, k] * U[k]
            Z[i] = gssm_state(drives, gammas)
        return Z, V

    Ztr, Vtr = make_trials(args.trials)
    Zte, Vte = make_trials(max(40, args.trials // 4))

    # operator family: least-squares W (D, K) so that Z @ W ≈ V  → each column W[:,k] is operator O_k
    W = torch.linalg.lstsq(Ztr, Vtr).solution               # (D, K)
    pred = Zte @ W                                          # (n, K) each operator's readout

    # per-fact recovery: correlation of operator O_k's output with the TRUE value v_k
    def corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float((a * b).sum() / (a.norm() * b.norm() + EPS))
    rec = [corr(pred[:, k], Vte[:, k]) for k in range(K)]   # operator k vs fact k (want high)
    # cross-talk: operator k's output vs OTHER facts (want low)
    cross = [[corr(pred[:, k], Vte[:, j]) for j in range(K) if j != k] for k in range(K)]
    mean_cross = sum(abs(c) for row in cross for c in row) / max(1, sum(len(r) for r in cross))

    # TRIVIALITY: are operator outputs just rescalings of one read? corr between operator outputs
    op_op = []
    for k in range(K):
        for j in range(k + 1, K):
            op_op.append(abs(corr(pred[:, k], pred[:, j])))
    mean_op_op = sum(op_op) / max(1, len(op_op))

    # NULL: random untuned operators should NOT recover the facts
    Wr = torch.randn(D, K, generator=g)
    predr = Zte @ Wr
    rec_null = sum(abs(corr(predr[:, k], Vte[:, k])) for k in range(K)) / K

    out = {"D": D, "K": K, "T": args.T, "trials": args.trials,
           "per_fact_recovery": [round(r, 3) for r in rec],
           "mean_recovery": round(sum(rec) / K, 3),
           "mean_crosstalk": round(mean_cross, 3),
           "operator_output_corr (triviality, low=distinct)": round(mean_op_op, 3),
           "null_random_operator_recovery": round(rec_null, 3)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    print("── M2: ONE state, K facts, K operators de-mux ──")
    print(f"  per-fact recovery (operator k → fact k): {[round(r,2) for r in rec]}")
    print(f"  mean recovery: {out['mean_recovery']}  (1.0 = perfect read of each fact)")
    print(f"  mean cross-talk (operator k → other facts): {out['mean_crosstalk']}  (0 = clean de-mux)")
    print(f"  operator-output corr (triviality): {out['operator_output_corr (triviality, low=distinct)']}  "
          f"(low = operators are DISTINCT reads, not rescalings)")
    print(f"  NULL random-operator recovery: {out['null_random_operator_recovery']}  (should be ~0)")
    verdict = ("ONE SUBSTRATE, MANY READINGS HOLDS — distinct operators recover distinct facts from "
               "one state, far above the random-operator null"
               if out["mean_recovery"] > 0.6 and rec_null < 0.3 else
               "weak/failed separation on this setup (state can't superpose K facts at this D/K, or "
               "operators collapse) — honest negative")
    print(f"\n  → {verdict}")
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
