#!/usr/bin/env python3 -u
"""
HOLO-INDEX-HYBRID — STATE+INDEX breaks the O(1) capacity wall (P16).
======================================================================
The O(1) holographic state (src/holo_stream_recall.py) can only hold a few bindings:
the 1/sqrt(P) wall measured there (P=2 strong, P=4 already weak). The O1 two-system
thesis says capacity does NOT have to live in the state at all — the state only needs
to deliver a SHARP READ once the right key/value pair has been reminded to it. Bulk
capacity belongs to an external index (Contribution 6's closed-loop mechanism,
src/closed_loop.py), consulted at runtime, no gradient, no adaptation.

This file measures that claim directly: MQAR-across-gap at P far above the state's own
capacity (P=8, 16 against P_max=32, V_max=16), comparing three arms —

  BASE    — read the query straight from the carried O(1) state (no index).
  HYBRID  — if the query key was written to a surprise-gated external index during the
            write phase, inject [key, value] as a "reminder span" into a CLONE of the
            carried state immediately before the query is read (paired injection,
            exactly closed_loop.py's fork-inject-read pattern, using pos_index.py's
            vendor-free _advance/_read). Gate misses fall back to BASE — honest, no
            free lunch for keys the index chose not to keep.
  RANDOM  — identical injection mechanics, but the injected value is WRONG (a random
            in-range value != the true one). This is the guard against the injection
            being an artifact: if RANDOM helped as much as HYBRID, the effect would be
            "any span injected before the query helps" rather than "the correct value
            helps" — so RANDOM must not beat BASE by more than noise.

The model is StreamingHolographicLM (use_phase=True), trained ONLY on the ordinary
P=2 gap-recall curriculum (holo_stream_recall.py's own recipe: bar 0.8, 2000 iters,
train-short-eval-long). It never sees the hybrid mechanism, P=8, or P=16 during
training — the index is a pure runtime add-on, not a learned behavior. If the hybrid
wins at P=16 it is because the two-system split (state=sharp local read, index=bulk
capacity) is real, not because the model was fit to the eval task.

CPU-only, single-thread, low process priority (os.nice(19)). Imports ONLY from
holo_stream_recall.py (StreamingHolographicLM, chunked_forward, train_gap_curriculum,
_gap_vocab, make_gap_mqar_batch) and pos_index.py (_advance, for the no-scoring
injection push — the paired reads themselves need an ARGMAX prediction, not
pos_index._read's scalar NLL, so those go through the model directly after
_advance, same underlying forward()) — both vendor-free. closed_loop.py is READ as
a design reference only, never imported (it pulls in the attic/streaming_train
vendor chain, which this file has no business needing).

Outputs -> results/holo_hybrid*.json only.
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    os.nice(19)
except (AttributeError, OSError, PermissionError):
    pass

import torch

torch.backends.mps.is_available = lambda: False   # force CPU (repo convention)
torch.set_num_threads(1)

from holo_stream_recall import (        # noqa: E402
    StreamingHolographicLM, chunked_forward, train_gap_curriculum,
    _gap_vocab, make_gap_mqar_batch, _build_lm,
)
from pos_index import _advance          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(REPO, "results")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Surprise-gated write-phase index. Runtime-only: no_grad, dict per trial.
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def build_trial_index(model, row, P, gate_rate):
    """Walk the write phase [k1,v1,...,kP,vP] of ONE trial's token row (one shared
    state, each (key,value) pair advanced exactly once), measuring the per-token NLL
    of predicting each value token FROM THE STATE CARRIED UP THROUGH ALL PRIOR PAIRS
    (i.e. the state a live stream would actually have at write time) — the write-phase
    analogue of pos_index.py's spike detector / closed_loop.py's surprise signal.

    A (key, value) pair is written to the index iff its value's surprise ranks among
    the top `gate_rate * P` most-surprising pairs of the trial:
      gate_rate == 1.0 -> keep everything (upper reference — "gate 1.0" in the mission)
      gate_rate == 0.5 -> keep the more-surprising half (the median-split gate)
    gate_rate is a literal keep-FRACTION, matching the mission's 0.25/0.5/1.0 sweep.

    Returns (index dict {key_id -> value_id}, states_after_write_phase, per-pair NLLs).
    """
    import torch.nn.functional as F
    states = None
    nlls = []
    pairs = []
    for pi in range(P):
        k_tok, v_tok = row[2 * pi], row[2 * pi + 1]
        x = torch.tensor([k_tok, v_tok], dtype=torch.long).unsqueeze(0)
        logits, states = model(x, states)
        # logits[:,0,:] predicts token at position 1 (the value) from the key
        nll = F.cross_entropy(logits[:, 0, :], torch.tensor([v_tok], dtype=torch.long))
        nlls.append(float(nll))
        pairs.append((k_tok, v_tok))

    order = sorted(range(P), key=lambda i: -nlls[i])   # most surprising first
    n_keep = max(0, min(P, round(gate_rate * P)))
    kept_ranks = set(order[:n_keep])

    index = {}
    for pi in range(P):
        if pi in kept_ranks:
            k_tok, v_tok = pairs[pi]
            index[k_tok] = v_tok

    return index, states, nlls


# ═══════════════════════════════════════════════════════════════════════════
# 2. Paired query-time injection — exact closed_loop.py fork/inject/read logic,
#    reimplemented on pos_index._advance/_read (vendor-free, chunk-boundary exact
#    since P,G here stay within chunk length so a single _advance/_read is exact).
# ═══════════════════════════════════════════════════════════════════════════
def _clone_states(states):
    if states is None:
        return None
    return [{k: v.clone() for k, v in st.items()} for st in states]


@torch.no_grad()
def _query_argmax(model, q_key, states, V_max):
    """Push the query key through `states` (via pos_index._advance, no scoring — we
    need the ARGMAX prediction, not an NLL, so this does not go through pos_index._read,
    whose job is exactly a scalar NLL over a target continuation) and return the argmax
    prediction at that final position, restricted to the FIRST V_max logit columns.

    NOTE (matches holo_stream_recall.eval_gap_recall's `pred = logits[:, -1, :V_max]`
    exactly): make_gap_mqar_batch already returns targets as `q_vals - val_lo`, i.e.
    value CLASS indices in [0, V_max) — not absolute token ids. The model's head was
    trained against those class-index targets, so the value read-out lives in logit
    columns [0, V_max), NOT [val_lo, val_lo+V_max). Slicing at val_lo (an earlier, now
    corrected, bug in this file) reads the KEY-id logit block instead and produces
    near-chance predictions regardless of what was injected — a silent-failure trap,
    not a state/index capacity result.
    """
    x = torch.tensor([q_key], dtype=torch.long).unsqueeze(0)
    logits, _ = model(x, states)
    return int(logits[0, -1, :V_max].argmax())


@torch.no_grad()
def eval_hybrid_batch(model, P, G, gen, P_max, V_max, F_fillers, chunk, gate_rate, seed):
    """One eval batch, all three arms (base, hybrid, random), scored per-trial (the
    index and the injected span differ per row, so this loops over the batch instead
    of vectorizing — eval_batch is small enough, O(few hundred), that this is cheap).
    """
    key_lo, val_lo, fill_lo, _ = _gap_vocab(P_max, V_max, F_fillers)
    x, y = make_gap_mqar_batch(gen.batch, P, G, gen.g, P_max, V_max, F_fillers,
                                key_lo, val_lo, fill_lo)
    B = x.size(0)
    kv_len = 2 * P
    q_pos = x.size(1) - 1
    q_keys = x[:, q_pos]

    correct_base = 0
    correct_hybrid = 0
    correct_random = 0
    gate_hits = 0

    rand_gen = torch.Generator().manual_seed(seed + 777)

    for b in range(B):
        row = x[b]
        target = int(y[b])
        q_key = int(q_keys[b])

        # index for this trial, gated on write-phase surprise
        index, _, _ = build_trial_index(model, row.tolist(), P, gate_rate)

        # carried state through the FULL pre-query sequence (write phase + fillers),
        # via the model's own chunked_forward so this matches training/eval exactly.
        pre_q = row[:q_pos].unsqueeze(0)
        _, states_pre = chunked_forward(model, pre_q, chunk)

        # ── BASE: read the query directly from the carried state ──
        pred_base = _query_argmax(model, q_key, states_pre, V_max)
        correct_base += int(pred_base == target)

        if q_key in index:
            gate_hits += 1
            v_true = index[q_key]

            # ── HYBRID: inject [key, true_value] reminder span into a CLONE, then
            # read the query — exact closed_loop.py fork/inject/read pattern, using
            # pos_index._advance for the (no-scoring) injection push.
            st_hyb = _advance(model, [q_key, v_true], _clone_states(states_pre))
            pred_hyb = _query_argmax(model, q_key, st_hyb, V_max)

            # ── RANDOM control: inject [key, WRONG_value] — must not help ──
            wrong_choices = [v for v in range(val_lo, val_lo + V_max) if v != v_true]
            v_rand = wrong_choices[int(torch.randint(0, len(wrong_choices), (1,),
                                                      generator=rand_gen).item())]
            st_rnd = _advance(model, [q_key, v_rand], _clone_states(states_pre))
            pred_rnd = _query_argmax(model, q_key, st_rnd, V_max)
        else:
            # gate miss: hybrid and random both degrade honestly to BASE (no free
            # lunch for keys the surprise gate chose not to keep).
            pred_hyb = pred_base
            pred_rnd = pred_base

        correct_hybrid += int(pred_hyb == target)
        correct_random += int(pred_rnd == target)

    return {
        "base": correct_base / B, "hybrid": correct_hybrid / B,
        "random": correct_random / B, "gate_hit_rate": gate_hits / B, "n": B,
    }


class _GenBundle:
    """Small holder so eval_hybrid_batch can take one object for (batch_size, rng)."""
    def __init__(self, batch, g):
        self.batch = batch
        self.g = g


# ═══════════════════════════════════════════════════════════════════════════
# 3. Orchestration — train ONCE per seed on the ordinary P=2 curriculum (v3
#    recipe: bar 0.8, 2000 iters, train-short-eval-long), then sweep (P, G, gate).
# ═══════════════════════════════════════════════════════════════════════════
def run(args):
    P_max, V_max, F_fillers = args.p_max, args.v_max, args.f_fillers
    key_lo, val_lo, fill_lo, vocab_size = _gap_vocab(P_max, V_max, F_fillers)
    mask_idx = vocab_size
    chance = 1.0 / V_max

    Ps = [int(p) for p in args.pairs.split(",")]
    Gs = [int(g) for g in args.gaps.split(",")]
    gates = [float(g) for g in args.gates.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 78)
    print("HOLO-INDEX-HYBRID — STATE+INDEX vs STATE-ALONE, P far above state capacity")
    print(f"P_max={P_max} V_max={V_max} F={F_fillers} vocab={vocab_size}  chance=1/{V_max}={chance:.4f}")
    print(f"pairs(P)={Ps}  gaps(G)={Gs}  gate_rates={gates}  seeds={seeds}  chunk={args.chunk}")
    print("=" * 78)

    t0 = time.time()
    results = {"config": vars(args), "chance": chance, "sweep": {}, "curriculum": {}}

    for seed in seeds:
        # ── train ONCE per seed on the ordinary P=2 gap-recall curriculum (v3
        # recipe) — the model NEVER sees the hybrid mechanism or P>2 at train time.
        torch.manual_seed(seed)
        model = _build_lm(vocab_size, mask_idx, use_phase=True,
                          d_model=args.d_model, n_layers=args.n_layers,
                          n_heads=args.n_heads, d_head=args.d_head)
        train_P = 2
        cap = args.train_gap_cap if args.train_gap_cap > 0 else args.chunk - 2 * train_P - 2
        Gmax_train = min(max(Gs) if Gs else 0, max(2, cap))
        curr = train_gap_curriculum(
            model, train_P, Gmax_train, args.iters, args.lr, seed, args.batch, args.chunk,
            P_max, V_max, F_fillers, log_every=args.log_every, g_start=args.g_start,
            patience=args.patience)
        curr["train_gap_cap"] = Gmax_train
        curr["train_P"] = train_P
        results["curriculum"][f"seed{seed}"] = curr
        print(f"\nseed={seed}: curriculum done (train P={train_P}) "
              f"final_train_gap={curr['final_train_gap']} final_train_acc={curr['final_train_acc']:.3f}")
        model.eval()

        for P in Ps:
            for G in Gs:
                for gate_rate in gates:
                    gen = torch.Generator().manual_seed(seed + 5000 + P * 97 + G * 13)
                    bundle = _GenBundle(args.eval_batch, gen)
                    out = eval_hybrid_batch(model, P, G, bundle, P_max, V_max, F_fillers,
                                            args.chunk, gate_rate, seed)
                    sk = f"seed{seed}|P{P}|G{G}|gate{gate_rate}"
                    results["sweep"][sk] = {
                        "seed": seed, "P": P, "G": G, "gate_rate": gate_rate,
                        "base": round(out["base"], 4), "hybrid": round(out["hybrid"], 4),
                        "random": round(out["random"], 4),
                        "gate_hit_rate": round(out["gate_hit_rate"], 4),
                        "chance": round(chance, 4), "n": out["n"],
                    }
                    print(f"  P={P:>3} G={G:>3} gate={gate_rate:.2f}: "
                          f"base={out['base']:.3f} hybrid={out['hybrid']:.3f} "
                          f"random={out['random']:.3f} gate_hit_rate={out['gate_hit_rate']:.3f}")

    # ── verdict ──
    def _mean_at(field, P, G, gate_rate):
        vals = [v[field] for v in results["sweep"].values()
                if v["P"] == P and v["G"] == G and v["gate_rate"] == gate_rate]
        return sum(vals) / len(vals) if vals else None

    lines = []
    for P in Ps:
        for G in Gs:
            for gate_rate in gates:
                b = _mean_at("base", P, G, gate_rate)
                h = _mean_at("hybrid", P, G, gate_rate)
                r = _mean_at("random", P, G, gate_rate)
                if b is None:
                    continue
                lines.append(f"P={P} G={G} gate={gate_rate}: base={b:.3f} hybrid={h:.3f} "
                            f"random={r:.3f} chance={chance:.3f}")

    def _get(P, G, gate_rate, field):
        v = _mean_at(field, P, G, gate_rate)
        return v

    gate_top = max(gates)
    checks = {}
    if Ps and Gs:
        P_big = max(Ps)
        G_ref = Gs[0]
        hybrid_p16 = _get(P_big, G_ref, gate_top, "hybrid")
        base_p16 = _get(P_big, G_ref, gate_top, "base")
        random_p16 = _get(P_big, G_ref, gate_top, "random")
        P_small = min(Ps)
        hybrid_p2 = _get(P_small, G_ref, gate_top, "hybrid")
        base_p2 = _get(P_small, G_ref, gate_top, "base")
        checks = {
            f"hybrid@P{P_big}_gate{gate_top}>=0.9": bool(hybrid_p16 is not None and hybrid_p16 >= 0.9),
            f"base@P{P_big}<=0.15": bool(base_p16 is not None and base_p16 <= 0.15),
            f"random@P{P_big}~=base": bool(random_p16 is not None and base_p16 is not None
                                            and abs(random_p16 - base_p16) < 0.10),
            f"hybrid@P{P_small}>=base@P{P_small}": bool(hybrid_p2 is not None and base_p2 is not None
                                                          and hybrid_p2 >= base_p2),
        }

    all_pass = all(checks.values()) if checks else False
    verdict = ("STATE+INDEX BREAKS THE CAPACITY WALL — hybrid clears the state-alone "
               "baseline by a wide margin at P far above state capacity, random-value "
               "injection does not (guards against an injection artifact), all acceptance "
               "checks pass"
               if all_pass else
               "hybrid does not clear all acceptance checks in this sweep — see 'checks' "
               "and 'verdict_detail' for the per-cell numbers")
    results["verdict"] = verdict
    results["checks"] = checks
    results["verdict_detail"] = lines
    results["elapsed_s"] = round(time.time() - t0, 1)

    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "=" * 78)
    print("VERDICT")
    for ln in lines:
        print("  " + ln)
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f">>> {verdict}")
    print(f"\n-> {args.out}  ({results['elapsed_s']}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="fast sanity sweep: 1 seed, P in {2,16}, G=8, gate=1.0, 800 iters")
    ap.add_argument("--full", action="store_true",
                    help="the full sweep: P in {2,8,16}, G in {8,32}, gate in {0.5,1.0}, 2 seeds")
    ap.add_argument("--p-max", type=int, default=32, help="distinct key ids available")
    ap.add_argument("--v-max", type=int, default=16, help="distinct value ids available")
    ap.add_argument("--f-fillers", type=int, default=16, help="distinct filler ids available")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-head", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=64, help="streaming chunk length (detach-carry boundary)")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-batch", type=int, default=200)
    ap.add_argument("--g-start", type=int, default=2, help="curriculum starting gap")
    ap.add_argument("--patience", type=int, default=25,
                    help="consecutive train-P acc>0.9 iters required before curriculum grows the gap")
    ap.add_argument("--train-gap-cap", type=int, default=0,
                    help="max TRAINING gap for the P=2 curriculum (0 = auto: chunk-2*2-2)")
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--pairs", default="2,16", help="comma list of eval n_pairs P to sweep")
    ap.add_argument("--gaps", default="8,32", help="comma list of eval gap lengths G to sweep")
    ap.add_argument("--gates", default="0.5,1.0", help="comma list of index gate keep-rates to sweep")
    ap.add_argument("--seeds", default="0,1", help="comma list of seeds")
    ap.add_argument("--out", default=os.path.join(RESULTS, "holo_hybrid.json"))
    args = ap.parse_args()

    if args.smoke:
        args.pairs = "2,16"
        args.gaps = "8"
        args.gates = "1.0"
        # seed=1, not 0: holo_stream_recall_v3.json shows the P=2 write-phase
        # curriculum itself is seed-sensitive (seed=0 -> final_train_acc 0.34,
        # never consolidates past-chance recall even at P=2 G=0; seed=1 -> 0.84,
        # matches the v3 bar-0.8 recipe). --full still sweeps seeds 0,1 and will
        # show this instability honestly; --smoke's job is a fast sanity check of
        # the HYBRID MECHANISM, which needs a consolidated base state to probe.
        args.seeds = "1"
        args.iters = min(args.iters, 800) if args.iters == 2000 else args.iters
        if args.out == os.path.join(RESULTS, "holo_hybrid.json"):
            args.out = os.path.join(RESULTS, "holo_hybrid_smoke.json")
    elif args.full:
        args.pairs = "2,8,16"
        args.gaps = "8,32"
        args.gates = "0.5,1.0"
        args.seeds = "0,1"
        args.iters = max(args.iters, 2000)
        if args.out == os.path.join(RESULTS, "holo_hybrid.json"):
            pass  # keep default full-run filename

    # chunk must comfortably contain a write phase for the LARGEST P plus some
    # margin — with P_max eval P up to 16, kv_len=32, so chunk defaults to 64.
    run(args)


if __name__ == "__main__":
    main()
