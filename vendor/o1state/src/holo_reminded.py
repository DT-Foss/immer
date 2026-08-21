#!/usr/bin/env python3 -u
"""
HOLO-REMINDED — LEARNED TO BE REMINDED (P18): breaking M5's 0.84 consultation ceiling.
========================================================================================
M5 (src/holo_index_hybrid.py, results/holo_hybrid.json) showed the state+index split
works — hybrid clears state-alone by a wide margin at P far above state capacity — but
diagnosed a ceiling: a model trained ONLY on the ordinary P=2 gap-recall curriculum
(holo_stream_recall.py's ordinary recipe) never learned to CONSULT a reminder. Injecting
the correct [key,value] span into a clone of its state before the query lifts P=16
recall only 0.15 -> 0.51 (gate 1.0) and P=8 only 0.20 -> 0.66 — nowhere near the ~1.0
a model that could perfectly exploit a correct injection should reach. The mechanism is
real (random-injection guard holds: wrong values hurt, they don't help) but the model is
reading an unfamiliar kind of span through eyes trained for a different distribution.

P18's moonshot: make consultation part of TRAINING itself. Instead of bolting the index
onto a naive model at eval time (M5's design, and a legitimate "zero-shot consultation"
lower bound), the model here is trained on a curriculum where a reminder — right, wrong,
or absent — sometimes appears immediately before the query token, AS PART OF THE INPUT
SEQUENCE. The loss is still the ordinary next-token loss at the last position; nothing
about the objective changes. What changes is the model gets to see, and learn to
ARBITRATE, three regimes:
  p=0.5  a CORRECT reminder [k_q, v_true] sits right before the query
  p=0.1  a WRONG reminder [k_q, v_wrong] (uniform other value) sits right before it
  p=0.4  no reminder — the query is unprepended, exactly the plain M5/curriculum trial
If this works, hybrid@P16 should clear M5's 0.51 by a wide margin (target >=0.85), base
(no-reminder arm) should barely move from M5's own base numbers (the skill being learned
is arbitration on top of existing recall, not a different task), and — the sharpest test
— accuracy under a WRONG reminder at P=2 (where the carried state alone already knows
the answer) should beat M5's random arm (0.30): a model that learned to consult should
also have learned when NOT to.

DESIGN DECISIONS (read before touching anything)
--------------------------------------------------------------------------------------
1. INJECTION FORM. M5 injected [k_q, v] into a CLONE of the state via pos_index._advance
   AFTER the write phase was already read into a carried state (a "state-trick": the
   query is scored from a state that saw the reminder, but the reminder never appeared
   in the token stream the model was trained to read). Here the reminder is literally
   two more tokens in the sequence x itself — [..., write-phase, fillers, k_q, v_?, k_q]
   — i.e. the SAME query key appears twice: once inside the reminder span (if present)
   and once as the actual query token being scored. This is the "deployment-realistic"
   form per the mission: a real closed-loop index (closed_loop.py) writes its answer
   INTO the stream (as retrieved context, a tool-call result, a RAG chunk...), it does
   not reach into the model's hidden state. M5's clone-injection is a clean state-space
   probe; this file's in-sequence injection is what a model actually sees in production,
   and it is trainable, which the state-trick is not (nothing before it in the compute
   graph reaches a gradient). Documented here, not glossed over: these are two different
   experiments answering two different questions, and their eval numbers are not the
   same measurement even at matched (P, G, gate) cells — the comparison table below is
   an M5-vs-P18 STORYLINE ("did learning fix the ceiling"), not an apples-to-apples
   ablation of one changed variable.

2. PADDING FILLER TOKENS. Three regimes (correct / wrong / none) must produce equal-
   length rows in the same training batch (torch.stack needs a fixed shape). The
   "no reminder" case inserts 2 filler tokens (drawn from the ordinary filler id range,
   same distribution as the gap fillers) at the position the reminder would have
   occupied, immediately before the query key. This was chosen over (a) padding with a
   dedicated MASK id, which would let the model learn a trivial "is this MASK -> ignore"
   shortcut that has nothing to do with arbitration, and (b) left-padding the whole
   batch to a global max length, which would shift the query position per-row and break
   chunked_forward's fixed chunk-boundary carry. Same-distribution fillers make the
   "no reminder" trial indistinguishable, AT THE TOKEN-TYPE level, from an ordinary gap
   filler run — the model has to actually attend to whether the two tokens immediately
   before the query look like [k_q, some_v] (a reminder pair) or generic filler noise,
   not just pattern-match on token identity.

3. CURRICULUM / RECIPE. v3 recipe verbatim (this mission's explicit instruction): bar
   0.8, patience 25, 2000 iters, chunked, cap = one chunk (train_gap_cap default, same
   auto rule as M5: chunk - 2*P - 2), lr 3e-3, batch 32. Seed trap: M5's own smoke mode
   documents that seed=0 does not consolidate the P=2 write phase (final_train_acc 0.34,
   stuck at chance) while seed=1 does (0.84) — this file uses seeds {1,2}, never 0, for
   exactly that reason (holo_index_hybrid.py's --smoke seed comment, verbatim rationale
   reused here).

4. EVAL. Same (P, G) grid, same three arms (base / hybrid / random), same eval_batch and
   2-seed protocol as M5, so the headline table lines up cell-by-cell against
   results/holo_hybrid.json. The injection MECHANICS differ (in-sequence tokens here,
   per point 1) but the ARM SEMANTICS are identical: base = no reminder ever shown,
   hybrid = correct [k_q,v_true] reminder shown, random = wrong [k_q,v_wrong] reminder
   shown. Gate: unlike M5 (which gates on write-phase surprise against a per-trial
   index), this file's hybrid/random arms ALWAYS inject (gate_rate is not part of this
   design — training never saw a gate, only a stochastic presence/absence of a
   reminder), so there is no gate_hit_rate here; the M5 comparison column uses M5's
   gate=1.0 cells (its "reminder always available" reference), the fair match for
   "reminder always shown."

5. ARBITRATION TEST. At P=2, the carried state ALONE already predicts the query well
   above chance (M5's own base@P2 ~ 0.55-0.59) — the state has spare capacity for 2
   pairs. Feeding a WRONG reminder at P=2 pits the model's own (correct) internal state
   against an adversarial hint. M5 never trained for this scenario at all, so its
   random@P2 (~0.21-0.30 across gates, results/holo_hybrid.json) is a naive model
   simply obeying the last thing it read. If P18's training regime (which includes
   p=0.1 wrong-reminder trials with the SAME loss target as always: the true value)
   taught arbitration rather than obedience, accuracy_under_wrong_reminder at P=2 here
   should exceed M5's random@P2 by a wide margin — evidence the model learned to weigh
   its own state against a contradicting hint instead of just copying the hint forward.

Imports ONLY from holo_index_hybrid.py (which itself imports only from
holo_stream_recall.py and pos_index.py, both vendor-free) — no new dependencies, no
attic/streaming_train vendor chain. Outputs -> results/holo_reminded*.json only.
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
import torch.nn as nn

torch.backends.mps.is_available = lambda: False   # force CPU (repo convention)
torch.set_num_threads(1)

from holo_index_hybrid import (         # noqa: E402
    _gap_vocab, make_gap_mqar_batch, chunked_forward, _build_lm,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(REPO, "results")
M5_PATH = os.path.join(RESULTS, "holo_hybrid.json")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Reminder-augmented batch builder. Builds the ordinary gap-MQAR batch, then
#    prepends a stochastic 2-token reminder span [k_q, v_?] immediately before
#    the query token — CORRECT (p_correct), WRONG (p_wrong), or PADDING-FILLER
#    (1 - p_correct - p_wrong), so every row in the batch has equal length.
# ═══════════════════════════════════════════════════════════════════════════
def make_reminded_batch(B, P, G, gen, P_max, V_max, F, key_lo, val_lo, fill_lo,
                         p_correct=0.5, p_wrong=0.1, regime=None):
    """Returns (x, y, regimes) where x has 2 extra columns (the reminder/padding
    span) inserted right before the final query-key column, and regimes is a
    LongTensor in {0=correct, 1=wrong, 2=none}, one per row.

    `regime`: if given (a length-B LongTensor or a python int), FORCE that regime
    for every row instead of sampling — used by eval (each arm is a fixed regime,
    not a stochastic mixture) while training always samples (regime=None).
    """
    x, y = make_gap_mqar_batch(B, P, G, gen, P_max, V_max, F, key_lo, val_lo, fill_lo)
    q_pos = x.size(1) - 1
    q_keys = x[:, q_pos]
    v_true = y + val_lo   # y is a value CLASS index in [0, V_max); token id = y + val_lo

    if regime is None:
        u = torch.rand(B, generator=gen)
        regimes = torch.full((B,), 2, dtype=torch.long)     # default: none
        regimes[u < p_correct] = 0                          # correct
        regimes[(u >= p_correct) & (u < p_correct + p_wrong)] = 1   # wrong
    elif isinstance(regime, int):
        regimes = torch.full((B,), regime, dtype=torch.long)
    else:
        regimes = regime

    # column 0 of the reminder span: k_q where a reminder is shown, filler otherwise
    fillers2 = torch.randint(0, F, (B, 2), generator=gen) + fill_lo
    span0 = torch.where(regimes != 2, q_keys, fillers2[:, 0])

    # column 1: v_true where correct, a uniformly-sampled WRONG value where wrong,
    # filler otherwise. Sampled per-row so "wrong" is not a single fixed offset.
    wrong_offset = torch.randint(1, V_max, (B,), generator=gen)   # in [1, V_max-1]
    v_wrong = val_lo + (y + wrong_offset) % V_max                 # != v_true by construction
    span1 = torch.where(regimes == 0, v_true,
             torch.where(regimes == 1, v_wrong, fillers2[:, 1]))

    pre_q = x[:, :q_pos]
    q_col = x[:, q_pos:q_pos + 1]
    x_aug = torch.cat([pre_q, span0.unsqueeze(1), span1.unsqueeze(1), q_col], dim=1)
    return x_aug, y, regimes


# ═══════════════════════════════════════════════════════════════════════════
# 2. Training — v3 recipe (bar 0.8, patience 25, 2000 iters, chunked, cap =
#    one chunk), P=2 curriculum on GAP, stochastic reminder regime per trial.
# ═══════════════════════════════════════════════════════════════════════════
def train_reminded_curriculum(model, P, Gmax, iters, lr, seed, batch, chunk,
                               P_max, V_max, F, p_correct=0.5, p_wrong=0.1,
                               log_every=0, g_start=2, patience=25):
    """Same gap-growth curriculum as holo_stream_recall.train_gap_curriculum
    (grow G only after acc>0.9 sustained `patience` iters — the v3 fix for the
    diagnosed v1 P=1 collapse), plus the stochastic reminder span on every
    trial. Loss is the ordinary next-token CE at the last (query) position —
    unchanged by the reminder mechanism, exactly per the mission spec."""
    key_lo, val_lo, fill_lo, _ = _gap_vocab(P_max, V_max, F)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    model.train()
    Gcur = min(g_start, Gmax) if Gmax > 0 else 0
    acc = 0.0
    good = 0
    regime_counts = {"correct": 0, "wrong": 0, "none": 0}
    for it in range(iters):
        x, y, regimes = make_reminded_batch(batch, P, Gcur, gen, P_max, V_max, F,
                                             key_lo, val_lo, fill_lo, p_correct, p_wrong)
        logits, _ = chunked_forward(model, x, chunk)
        pred = logits[:, -1, :V_max]
        loss = lossf(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        acc = float((pred.argmax(-1) == y).float().mean())
        good = good + 1 if acc > 0.9 else 0
        if good >= patience and Gcur < Gmax:
            Gcur = min(Gmax, int(Gcur * 1.5) + 1)
            good = 0
        regime_counts["correct"] += int((regimes == 0).sum())
        regime_counts["wrong"] += int((regimes == 1).sum())
        regime_counts["none"] += int((regimes == 2).sum())
        if log_every and (it + 1) % log_every == 0:
            print(f"    it {it+1:>4}/{iters}: loss {float(loss):.3f} acc {acc:.3f} (train-gap {Gcur})")
    n_total = sum(regime_counts.values())
    regime_frac = {k: round(v / n_total, 4) for k, v in regime_counts.items()}
    return {"final_train_gap": Gcur, "final_train_acc": round(acc, 4),
            "train_regime_frac": regime_frac}


# ═══════════════════════════════════════════════════════════════════════════
# 3. Eval — same (P, G) grid / arms / eval_batch as M5, using the IN-SEQUENCE
#    injection form (point 1 of the module docstring), forced regime per arm.
# ═══════════════════════════════════════════════════════════════════════════
_ARM_SEED_OFFSET = {"base": 0, "hybrid": 1, "random": 2}   # deterministic per-arm fork


@torch.no_grad()
def eval_reminded_batch(model, P, G, eval_batch, base_seed, P_max, V_max, F, chunk):
    """All three arms, each scored on its OWN eval_batch-size batch with a FORCED
    regime (not sampled): base=none(2), hybrid=correct(0), random=wrong(1). Each
    arm draws from an independently-seeded generator (deterministic fork of
    base_seed) so the three arms see different trial content — same spirit as
    M5's per-(P,G) independent draw, extended one level so a forced-regime arm
    can't accidentally reuse another arm's exact trials.

    Returns per-arm accuracy; 'random' doubles as accuracy_under_wrong_reminder,
    the P18 arbitration metric (named explicitly in the caller at P=2)."""
    key_lo, val_lo, fill_lo, _ = _gap_vocab(P_max, V_max, F)
    regime_of = {"base": 2, "hybrid": 0, "random": 1}
    out = {}
    for arm, regime in regime_of.items():
        arm_gen = torch.Generator().manual_seed(base_seed + _ARM_SEED_OFFSET[arm])
        x, y, _ = make_reminded_batch(eval_batch, P, G, arm_gen, P_max, V_max, F,
                                       key_lo, val_lo, fill_lo, regime=regime)
        logits, _ = chunked_forward(model, x, chunk)
        pred = logits[:, -1, :V_max].argmax(-1)
        out[arm] = float((pred == y).float().mean())
    out["n"] = eval_batch
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 4. M5 comparison — load results/holo_hybrid.json (if present) and pull the
#    matching (P, G, gate=1.0) cells (M5's "reminder always available"
#    reference — the fair match for this file's always-inject hybrid/random).
# ═══════════════════════════════════════════════════════════════════════════
def load_m5_reference():
    if not os.path.exists(M5_PATH):
        return None
    with open(M5_PATH) as f:
        m5 = json.load(f)
    gate_top = max(float(g) for g in m5["config"]["gates"].split(",")) \
        if isinstance(m5["config"].get("gates"), str) else 1.0
    cells = {}
    for v in m5["sweep"].values():
        if v["gate_rate"] != gate_top:
            continue
        key = f"P{v['P']}|G{v['G']}"
        cells.setdefault(key, []).append(v)
    m5_mean = {}
    for key, rows in cells.items():
        m5_mean[key] = {
            "base": round(sum(r["base"] for r in rows) / len(rows), 4),
            "hybrid": round(sum(r["hybrid"] for r in rows) / len(rows), 4),
            "random": round(sum(r["random"] for r in rows) / len(rows), 4),
            "gate_rate": gate_top, "n_seeds": len(rows),
        }
    return {"gate_used": gate_top, "cells": m5_mean, "source": M5_PATH}


# ═══════════════════════════════════════════════════════════════════════════
# 5. Orchestration — train ONCE per seed on the reminder-augmented P=2
#    curriculum, then sweep (P, G) exactly matching M5's grid.
# ═══════════════════════════════════════════════════════════════════════════
def run(args):
    P_max, V_max, F_fillers = args.p_max, args.v_max, args.f_fillers
    key_lo, val_lo, fill_lo, vocab_size = _gap_vocab(P_max, V_max, F_fillers)
    mask_idx = vocab_size
    chance = 1.0 / V_max

    Ps = [int(p) for p in args.pairs.split(",")]
    Gs = [int(g) for g in args.gaps.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 78)
    print("HOLO-REMINDED — P18: training WITH stochastic consultation (learned arbitration)")
    print(f"P_max={P_max} V_max={V_max} F={F_fillers} vocab={vocab_size}  chance=1/{V_max}={chance:.4f}")
    print(f"pairs(P)={Ps}  gaps(G)={Gs}  seeds={seeds}  chunk={args.chunk}  "
          f"p_correct={args.p_correct} p_wrong={args.p_wrong}")
    print("=" * 78)

    t0 = time.time()
    results = {"config": vars(args), "chance": chance, "sweep": {}, "curriculum": {}}

    for seed in seeds:
        torch.manual_seed(seed)
        model = _build_lm(vocab_size, mask_idx, use_phase=True,
                          d_model=args.d_model, n_layers=args.n_layers,
                          n_heads=args.n_heads, d_head=args.d_head)
        train_P = 2
        # cap = one chunk, same auto rule as M5: chunk - 2*P - 2 leaves room for
        # the write phase + the 2-token reminder/padding span + the query token.
        cap = args.train_gap_cap if args.train_gap_cap > 0 else args.chunk - 2 * train_P - 2
        Gmax_train = min(max(Gs) if Gs else 0, max(2, cap))
        curr = train_reminded_curriculum(
            model, train_P, Gmax_train, args.iters, args.lr, seed, args.batch, args.chunk,
            P_max, V_max, F_fillers, p_correct=args.p_correct, p_wrong=args.p_wrong,
            log_every=args.log_every, g_start=args.g_start, patience=args.patience)
        curr["train_gap_cap"] = Gmax_train
        curr["train_P"] = train_P
        results["curriculum"][f"seed{seed}"] = curr
        print(f"\nseed={seed}: curriculum done (train P={train_P}) "
              f"final_train_gap={curr['final_train_gap']} final_train_acc={curr['final_train_acc']:.3f} "
              f"regime_frac={curr['train_regime_frac']}")
        model.eval()

        for P in Ps:
            for G in Gs:
                base_seed = seed + 5000 + P * 97 + G * 13
                out = eval_reminded_batch(model, P, G, args.eval_batch, base_seed,
                                          P_max, V_max, F_fillers, args.chunk)
                sk = f"seed{seed}|P{P}|G{G}"
                results["sweep"][sk] = {
                    "seed": seed, "P": P, "G": G,
                    "base": round(out["base"], 4), "hybrid": round(out["hybrid"], 4),
                    "random": round(out["random"], 4),
                    "chance": round(chance, 4), "n": out["n"],
                }
                print(f"  P={P:>3} G={G:>3}: base={out['base']:.3f} "
                      f"hybrid={out['hybrid']:.3f} random={out['random']:.3f}")

    # ── M5 side-by-side comparison table ──
    m5_ref = load_m5_reference()
    results["m5_reference"] = m5_ref

    def _mean_at(field, P, G):
        vals = [v[field] for v in results["sweep"].values() if v["P"] == P and v["G"] == G]
        return sum(vals) / len(vals) if vals else None

    comparison = {}
    lines = []
    for P in Ps:
        for G in Gs:
            b = _mean_at("base", P, G)
            h = _mean_at("hybrid", P, G)
            r = _mean_at("random", P, G)
            if b is None:
                continue
            key = f"P{P}|G{G}"
            m5c = m5_ref["cells"].get(key) if m5_ref else None
            comparison[key] = {
                "p18_base": round(b, 4), "p18_hybrid": round(h, 4), "p18_random": round(r, 4),
                "m5_base": m5c["base"] if m5c else None,
                "m5_hybrid": m5c["hybrid"] if m5c else None,
                "m5_random": m5c["random"] if m5c else None,
            }
            m5_str = (f" | M5: base={m5c['base']:.3f} hybrid={m5c['hybrid']:.3f} "
                      f"random={m5c['random']:.3f}" if m5c else " | M5: n/a")
            lines.append(f"P={P} G={G}: base={b:.3f} hybrid={h:.3f} random={r:.3f} "
                        f"chance={chance:.3f}{m5_str}")
    results["comparison_vs_m5"] = comparison

    # ── P18 scoring (mission's three acceptance checks) ──
    P_big = max(Ps) if Ps else None
    G_ref = Gs[0] if Gs else None
    P_small = min(Ps) if Ps else None

    hybrid_p16 = _mean_at("hybrid", P_big, G_ref) if P_big else None
    base_p16 = _mean_at("base", P_big, G_ref) if P_big else None
    random_p2 = _mean_at("random", P_small, G_ref) if P_small else None   # accuracy_under_wrong_reminder @ P2

    m5_base_p16 = None
    m5_random_p2 = None
    if m5_ref:
        c16 = m5_ref["cells"].get(f"P{P_big}|G{G_ref}") if P_big else None
        c2 = m5_ref["cells"].get(f"P{P_small}|G{G_ref}") if P_small else None
        m5_base_p16 = c16["base"] if c16 else None
        m5_random_p2 = c2["random"] if c2 else None

    checks = {}
    if hybrid_p16 is not None:
        checks[f"hybrid@P{P_big}>=0.85"] = bool(hybrid_p16 >= 0.85)
    if base_p16 is not None and m5_base_p16 is not None:
        checks[f"base@P{P_big}_within_5pp_of_m5_base"] = bool(abs(base_p16 - m5_base_p16) <= 0.05)
    if random_p2 is not None and m5_random_p2 is not None:
        checks[f"wrong_reminder@P{P_small}>m5_random@P{P_small}({m5_random_p2:.2f})"] = \
            bool(random_p2 > m5_random_p2)
    elif random_p2 is not None:
        # M5 reference unavailable — fall back to the mission's literal number (0.30)
        checks[f"wrong_reminder@P{P_small}>0.30(m5_ref_missing)"] = bool(random_p2 > 0.30)

    all_pass = all(checks.values()) if checks else False
    verdict = ("LEARNED TO BE REMINDED — training with stochastic consultation breaks "
               "M5's untrained-consultation ceiling: hybrid clears 0.85 at the largest P, "
               "base stays within 5pp of M5 (arbitration learned, not a different task), "
               "and wrong-reminder accuracy at P=2 beats M5's naive-obedience random arm "
               "(the model weighs its own state against a contradicting hint) — all "
               "acceptance checks pass"
               if all_pass else
               "training with stochastic consultation does not clear all P18 acceptance "
               "checks in this sweep — see 'checks' and 'comparison_vs_m5' for the "
               "per-cell numbers against M5's own reference")
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
                    help="fast sanity sweep: 1 seed, P in {2,16}, G=8, 800 iters")
    ap.add_argument("--full", action="store_true",
                    help="the full sweep: P in {2,8,16}, G in {8,32}, 2 seeds, 2000 iters")
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
    ap.add_argument("--p-correct", type=float, default=0.5, help="train-time P(correct reminder)")
    ap.add_argument("--p-wrong", type=float, default=0.1, help="train-time P(wrong reminder)")
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--pairs", default="2,16", help="comma list of eval n_pairs P to sweep")
    ap.add_argument("--gaps", default="8,32", help="comma list of eval gap lengths G to sweep")
    ap.add_argument("--seeds", default="1,2", help="comma list of seeds (never 0: seed trap, see docstring)")
    ap.add_argument("--out", default=os.path.join(RESULTS, "holo_reminded.json"))
    args = ap.parse_args()

    if args.smoke:
        args.pairs = "2,16"
        args.gaps = "8"
        args.seeds = "1"
        args.iters = min(args.iters, 800) if args.iters == 2000 else args.iters
        if args.out == os.path.join(RESULTS, "holo_reminded.json"):
            args.out = os.path.join(RESULTS, "holo_reminded_smoke.json")
    elif args.full:
        args.pairs = "2,8,16"
        args.gaps = "8,32"
        args.seeds = "1,2"
        args.iters = max(args.iters, 2000)

    run(args)


if __name__ == "__main__":
    main()
