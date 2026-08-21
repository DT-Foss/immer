#!/usr/bin/env python3 -u
"""
HOLO-HELDOUT-KEYS — compositional binding: does the phase write generalize to
NEVER-TRAINED-AS-KEY ids?
======================================================================================
v3 of src/holo_stream_recall.py (see analysis/HOLO_STREAM_VERDICT.md) found that at
P=2 with a 16-key vocabulary the phase-carried arm (holo_carried, use_phase=True)
beats the magnitude-only baseline (holo_off, use_phase=False) by +25-30pp. The open
question left at the end of that verdict: is this REAL content-addressable binding,
or can the baseline "cheat" via channel allocation? With only 16 keys and 512 state
channels (n_heads*d_head*n_layers), input-conditioned gates (gamma(x), alpha(x)) can
in principle learn a per-key REGISTER FILE — reserve a fixed set of channels for each
of the 16 key identities and route writes there. That is a lookup table over TRAINED
keys, and a lookup table cannot answer a query for a key it never saw at write time.
The phase mechanism is different in kind: the read is De-rotation, Re(S . e^{-i phi_q}),
a FUNCTION of the query embedding — it does not need to have seen that specific key
id occupy the "key" role before, only to have learned the embedding->angle map. This
is the defining property of content-based (attention-style) binding, and if it shows
up in an O(1) recurrent state, that is a qualitatively different claim than "wins at
P=2, 16 keys."

THE TEST: split the P_max key ids into K_train (trained AS KEYS) and K_test (NEVER
used in key position, at training time). At eval, present queries drawn from EACH
set separately (never mixed within a trial) and compare accuracy. A channel-allocation
lookup table has no channel reserved for a K_test id -> must fail (near chance). The
phase de-rotation reads out whatever the embedding maps to -> CAN generalize.

*** THE CENTRAL DESIGN ELEMENT (read this before touching the vocab code) ***
It is not enough to simply withhold K_test ids from key position during training.
If a K_test id NEVER appears anywhere in training, held-out failure is trivially
explained by an untrained embedding row (pure noise-in, noise-out) — that would
confound "the key ROLE is new" with "the token ID is new," and the experiment would
prove nothing about binding at all. To isolate the ROLE as the only untrained
variable, K_test ids are mixed into the GAP FILLER stream during training at a fixed
rate (HELDOUT_FILLER_RATE, default 0.20): with probability 0.20 each filler slot in
the gap is drawn from K_test instead of the ordinary filler range F. This trains the
K_test embeddings (they are seen, gradients flow into their rows, they participate in
gamma(x)/alpha(x)/phi(x) many times) while their KEY ROLE — appearing at an odd
write-block position, being a possible query target — is strictly never exercised
during training. Only the ROLE is held out, not the token. This is the whole point
of the experiment and must not be diluted (e.g. never let a K_test id appear as a
value or as the query key during training).

Design (fixed exactly to v3 / holo_stream_recall.py, only the key split and the eval
protocol are new):
  - Vocab: P_max=64 key ids, V_max=16 value ids, F=16 filler ids, disjoint ranges
    (the _gap_vocab idiom, imported unchanged).
  - K_train = key ids [0, 40), K_test = key ids [40, 64) (held out from key position).
    NOTE the smoke already fired the pre-written honest-negative branch (both arms
    generalize at 600 iters); --full decides whether that survives full consolidation.
  - Task/recipe: identical to v3 — P=2 pairs, gap-MQAR, train-short-eval-long (train
    gap capped at the single-chunk max, chunk-2P-2, chunk=16), patience=25 gap
    curriculum, 2000 iters, lr 3e-3, batch 32, d_model 64. The query key (and both
    written keys) are drawn ONLY from K_train during training.
  - Arms, same init per seed: holo_carried (use_phase=True, state carried across the
    gap), holo_off (use_phase=False, the Selective/channel-allocation baseline),
    holo_zeroed_at_gap (use_phase=True, decisive null — state zeroed at gap onset).
  - Eval (chunked, carried forward, matches training's chunking): accuracy on (a)
    K_train-key trials, (b) K_test-key trials. EVERY key used within one trial
    (both written keys and the query key) comes from exactly ONE of the two sets —
    trials are never mixed (see make_heldout_gap_batch: key_pool selects the whole
    trial's key pool up front, sampled without replacement, so a K_test trial's
    fillers can still legitimately include OTHER K_test ids consistent with the
    training-time mixing distribution, but a given trial's key ROLE positions are
    homogeneous by construction; eval fillers are ordinary F-range ids only —
    no held-out mixing at eval). G in {0, 8, 32}, eval_batch 200, 2 seeds.

Output: results/holo_heldout_keys.json (or *_smoke.json under --smoke): config, per
(arm, key_set, G, seed) accuracy, mean over seeds, chance=1/V_max=1/16, and a
"generalization_gap" = acc(K_train) - acc(K_test) per arm per G, plus a verdict
string.

PREDICTION (registered before --full is launched):
  holo_carried: K_test accuracy >= 3x chance AND (K_test - K_test_off) >= +10pp at
  G in {0, 8} (compositional binding: phase generalizes where a lookup table cannot).
  holo_off: K_test accuracy ~= chance (a channel-allocation register file has no slot
  reserved for an id that never occupied key position during training).
  Both arms' K_train accuracy should look like v3's ignited P=2 numbers (0.7-0.9 at
  G<=32). If holo_off ALSO generalizes to K_test, the channel-allocation story for
  the v3 P11 tie-break is wrong and the true explanation is something else (embedding
  geometry alone might carry key identity without gating specialization) — that
  would be a genuinely interesting negative, reported as such, not hedged away.

CPU-only, single-thread (torch.set_num_threads(1)), os.nice(19) set at process start.
Repo idiom, code/comments in English. Results -> results/holo_heldout_keys*.json only.
"""
import os
os.nice(19)

import sys
import json
import math
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference"))

import torch
import torch.nn as nn

torch.backends.mps.is_available = lambda: False   # force CPU (repo convention)
torch.set_num_threads(1)

from holo_stream_recall import (   # noqa: E402
    StreamingHolographicLM, chunked_forward, _gap_vocab, _build_lm,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(REPO, "results")

HELDOUT_FILLER_RATE = 0.20   # rate at which K_test ids replace ordinary fillers in training gaps


# ═══════════════════════════════════════════════════════════════════════════
# 1. Held-out-key batch generator. Same [k1,v1,...,kP,vP][G fillers][query_key]
#    layout as make_gap_mqar_batch (holo_stream_recall.py), but:
#      - key ROLE positions (write keys + query key) are drawn from a single
#        caller-specified key_pool (K_train during training; K_train OR K_test,
#        never mixed within a trial, at eval).
#      - during TRAINING ONLY (heldout_ids is not None), each filler slot is,
#        independently with probability HELDOUT_FILLER_RATE, replaced by a
#        uniformly-drawn K_test id instead of an ordinary filler id. This is
#        the embedding-confound control described in the module docstring.
# ═══════════════════════════════════════════════════════════════════════════
def make_heldout_gap_batch(B, P, G, gen, key_pool, V_max, F, val_lo, fill_lo,
                            heldout_ids=None, heldout_rate=HELDOUT_FILLER_RATE):
    """key_pool: 1-D LongTensor of key ids this trial's keys/query are drawn from
    (K_train at train time; K_train xor K_test at eval time — callers pass exactly
    one set, never a union). heldout_ids: 1-D LongTensor of K_test ids to mix into
    the filler stream (only passed during training; None at eval, where fillers are
    ordinary F-range ids only, matching the eval distribution used everywhere else
    in this repo's gap-MQAR tasks).
    """
    n_keys = key_pool.numel()
    keys = torch.stack(
        [key_pool[torch.randperm(n_keys, generator=gen)[:P]] for _ in range(B)])
    vals = torch.randint(0, V_max, (B, P), generator=gen) + val_lo
    fillers = torch.randint(0, F, (B, max(G, 0)), generator=gen) + fill_lo
    if heldout_ids is not None and G > 0 and heldout_rate > 0:
        n_ho = heldout_ids.numel()
        swap_mask = torch.rand(B, G, generator=gen) < heldout_rate
        ho_draw = heldout_ids[torch.randint(0, n_ho, (B, G), generator=gen)]
        fillers = torch.where(swap_mask, ho_draw, fillers)
    q_idx = torch.randint(0, P, (B,), generator=gen)
    q_keys = keys.gather(1, q_idx.unsqueeze(1)).squeeze(1)
    q_vals = vals.gather(1, q_idx.unsqueeze(1)).squeeze(1)

    kv = torch.stack([keys, vals], dim=2).reshape(B, 2 * P)
    if G > 0:
        x = torch.cat([kv, fillers, q_keys.unsqueeze(1)], dim=1)
    else:
        x = torch.cat([kv, q_keys.unsqueeze(1)], dim=1)
    return x, q_vals - val_lo   # target = value INDEX in [0, V_max)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Training: gap-curriculum (patience-gated), exactly v3's train_gap_curriculum,
#    but drawing keys only from K_train and mixing K_test into fillers.
# ═══════════════════════════════════════════════════════════════════════════
def train_gap_curriculum_heldout(model, P, Gmax, iters, lr, seed, batch, chunk,
                                  key_train, key_test, V_max, F, val_lo, fill_lo,
                                  log_every=0, g_start=2, patience=25,
                                  heldout_rate=HELDOUT_FILLER_RATE):
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    model.train()
    Gcur = min(g_start, Gmax) if Gmax > 0 else 0
    acc = 0.0
    good = 0
    for it in range(iters):
        x, y = make_heldout_gap_batch(
            batch, P, Gcur, gen, key_train, V_max, F, val_lo, fill_lo,
            heldout_ids=key_test, heldout_rate=heldout_rate)
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
        if log_every and (it + 1) % log_every == 0:
            print(f"    it {it+1:>4}/{iters}: loss {float(loss):.3f} acc {acc:.3f} (train-gap {Gcur})")
    return {"final_train_gap": Gcur, "final_train_acc": round(acc, 4)}


# ═══════════════════════════════════════════════════════════════════════════
# 3. Eval: accuracy on K_train-key trials vs K_test-key trials, per (arm, G, seed).
#    Fillers at eval time are ordinary F-range only (no heldout mixing) — matches
#    the eval distribution used everywhere else in the repo's gap tasks and avoids
#    conflating "held-out key in filler position" with "held-out key in key position."
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def eval_heldout(model, P, G, NB, seed, key_pool, V_max, F, val_lo, fill_lo, chunk,
                  zero_at_gap=False):
    gen = torch.Generator().manual_seed(seed)
    x, y = make_heldout_gap_batch(NB, P, G, gen, key_pool, V_max, F, val_lo, fill_lo,
                                   heldout_ids=None)
    model.eval()
    kv_len = 2 * P
    if not zero_at_gap:
        logits, _ = chunked_forward(model, x, chunk)
    else:
        logits_kv, states = chunked_forward(model, x[:, :kv_len], chunk)
        states = model.zero_states(states)
        logits_rest, _ = chunked_forward(model, x[:, kv_len:], chunk)
        logits = torch.cat([logits_kv, logits_rest], dim=1)
    pred = logits[:, -1, :V_max]
    acc = float((pred.argmax(-1) == y).float().mean())
    return acc


# ═══════════════════════════════════════════════════════════════════════════
# 4. Orchestration.
# ═══════════════════════════════════════════════════════════════════════════
def run(args):
    P_max, V_max, F = args.p_max, args.v_max, args.f_fillers
    key_lo, val_lo, fill_lo, vocab_size = _gap_vocab(P_max, V_max, F)
    mask_idx = vocab_size
    chance = 1.0 / V_max

    n_train_keys = args.train_keys
    key_train = torch.arange(key_lo, key_lo + n_train_keys)
    key_test = torch.arange(key_lo + n_train_keys, key_lo + P_max)
    assert key_test.numel() > 0, "no held-out keys left — check --p-max/--train-keys"

    Gs = [int(g) for g in args.gaps.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    P = args.pairs

    print("=" * 78)
    print("HOLO-HELDOUT-KEYS — compositional binding: phase write on unseen key ids")
    print(f"P_max={P_max} V_max={V_max} F={F} vocab={vocab_size}  chance=1/{V_max}={chance:.4f}")
    print(f"K_train=[{key_lo},{key_lo+n_train_keys}) ({n_train_keys} ids)  "
          f"K_test=[{key_lo+n_train_keys},{key_lo+P_max}) ({key_test.numel()} ids)")
    print(f"P={P}  gaps(G)={Gs}  seeds={seeds}  chunk={args.chunk}  "
          f"heldout_filler_rate={args.heldout_rate}")
    print("=" * 78)

    arms = {
        "holo_carried": dict(use_phase=True, zero_at_gap=False),
        "holo_off": dict(use_phase=False, zero_at_gap=False),
        "holo_zeroed_at_gap": dict(use_phase=True, zero_at_gap=True),
    }

    t0 = time.time()
    results = {"config": vars(args),
               "key_split": {"key_lo": key_lo, "n_train_keys": n_train_keys,
                             "n_test_keys": int(key_test.numel())},
               "chance": chance, "sweep": {}, "curriculum": {}}

    for seed in seeds:
        print(f"\n{'='*78}\nseed={seed}  P={P}\n{'='*78}")
        for arm_name, arm_cfg in arms.items():
            torch.manual_seed(seed)
            model = _build_lm(vocab_size, mask_idx, use_phase=arm_cfg["use_phase"],
                              d_model=args.d_model, n_layers=args.n_layers,
                              n_heads=args.n_heads, d_head=args.d_head)
            Gmax_eval = max(Gs) if Gs else 0
            cap = args.train_gap_cap if args.train_gap_cap > 0 else args.chunk - 2 * P - 2
            Gmax_train = min(Gmax_eval, max(2, cap))
            curr = train_gap_curriculum_heldout(
                model, P, Gmax_train, args.iters, args.lr, seed, args.batch, args.chunk,
                key_train, key_test, V_max, F, val_lo, fill_lo,
                log_every=args.log_every, g_start=args.g_start, patience=args.patience,
                heldout_rate=args.heldout_rate)
            curr["train_gap_cap"] = Gmax_train
            key = f"seed{seed}_{arm_name}"
            results["curriculum"][key] = curr
            print(f"  [{arm_name:20s}] curriculum done: final_train_gap={curr['final_train_gap']} "
                  f"final_train_acc={curr['final_train_acc']:.3f}")

            for G in Gs:
                for set_name, pool in (("train_keys", key_train), ("test_keys", key_test)):
                    acc = eval_heldout(
                        model, P, G, args.eval_batch, seed + 1000 + G, pool, V_max, F,
                        val_lo, fill_lo, args.chunk, zero_at_gap=arm_cfg["zero_at_gap"])
                    sk = f"{arm_name}|{set_name}|G{G}|seed{seed}"
                    results["sweep"][sk] = {
                        "seed": seed, "arm": arm_name, "key_set": set_name, "P": P, "G": G,
                        "accuracy": round(acc, 4), "chance": round(chance, 4),
                        "beats_chance_3x": bool(acc > 3 * chance)}
                    print(f"    [{arm_name:20s}] {set_name:10s} G={G:>4}: acc={acc:.4f} "
                          f"(chance {chance:.4f}, 3x {'YES' if acc > 3*chance else 'no'})")

    # ── verdict ──
    def _mean_at(arm, set_name, G):
        vals = [v["accuracy"] for k, v in results["sweep"].items()
                if v["arm"] == arm and v["key_set"] == set_name and v["G"] == G]
        return sum(vals) / len(vals) if vals else None

    detail = []
    gen_gaps = {}
    verdict_hits = []
    for arm_name in arms:
        gen_gaps[arm_name] = {}
        for G in Gs:
            tr = _mean_at(arm_name, "train_keys", G)
            te = _mean_at(arm_name, "test_keys", G)
            if tr is None or te is None:
                continue
            gap = round(tr - te, 4)
            gen_gaps[arm_name][f"G{G}"] = gap
            detail.append(f"{arm_name:20s} G={G:>4}: train_keys={tr:.3f} test_keys={te:.3f} "
                          f"gen_gap={gap:+.3f}  chance={chance:.3f}")
        results["generalization_gap"] = gen_gaps

    for G in Gs:
        holo_te = _mean_at("holo_carried", "test_keys", G)
        off_te = _mean_at("holo_off", "test_keys", G)
        if holo_te is None or off_te is None:
            continue
        holo_generalizes = holo_te > 3 * chance
        off_generalizes = off_te > 3 * chance
        holo_beats_off = (holo_te - off_te) >= 0.10
        verdict_hits.append((G, holo_generalizes, off_generalizes, holo_beats_off, holo_te, off_te))

    any_compositional = any(hg and hbo for _, hg, _, hbo, _, _ in verdict_hits)
    both_generalize = any(hg and og for _, hg, og, _, _, _ in verdict_hits)
    neither_generalize = len(verdict_hits) > 0 and all(
        not hg and not og for _, hg, og, _, _, _ in verdict_hits)
    off_only = any((not hg) and og for _, hg, og, _, _, _ in verdict_hits)

    if any_compositional:
        verdict = ("COMPOSITIONAL BINDING: the phase generalizes to unseen keys where "
                   "channel allocation cannot — holo_carried clears 3x chance AND beats "
                   "holo_off by >=10pp on held-out-key queries in at least one G cell")
    elif both_generalize:
        verdict = ("BOTH ARMS GENERALIZE to held-out keys — the channel-allocation "
                   "hypothesis for the v3 P11 tie-break is wrong (or incomplete); "
                   "embedding geometry alone appears to carry enough key identity for "
                   "the magnitude-only baseline to bind unseen keys too")
    elif neither_generalize:
        verdict = ("NEITHER ARM GENERALIZES to held-out keys — compositional binding did "
                   "not emerge at this P_max/train-key-count/training budget; both arms "
                   "behave as lookup tables over trained key identities")
    elif off_only:
        verdict = ("UNEXPECTED: holo_off generalizes to held-out keys but holo_carried does "
                   "not — the phase write is NOT the generalizing mechanism here; report as-is")
    else:
        verdict = ("MIXED: no cell clears the compositional-binding bar cleanly; see "
                   "'generalization_gap' and per-cell numbers for the actual pattern")

    results["verdict"] = verdict
    results["verdict_detail"] = detail
    results["elapsed_s"] = round(time.time() - t0, 1)

    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "=" * 78)
    print("VERDICT")
    for ln in detail:
        print("  " + ln)
    print(f">>> {verdict}")
    print(f"\n→ {args.out}  ({results['elapsed_s']}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="fast sanity run (1 seed, 600 iters, G in {0,8})")
    ap.add_argument("--full", action="store_true", help="the full sweep (2 seeds, 2000 iters, G in {0,8,32})")
    ap.add_argument("--p-max", type=int, default=64, help="total distinct key ids")
    ap.add_argument("--train-keys", type=int, default=40, help="size of K_train (rest = K_test, held out)")
    ap.add_argument("--v-max", type=int, default=16, help="distinct value ids")
    ap.add_argument("--f-fillers", type=int, default=16, help="distinct ordinary filler ids")
    ap.add_argument("--heldout-rate", type=float, default=HELDOUT_FILLER_RATE,
                    help="rate at which K_test ids replace ordinary fillers during TRAINING")
    ap.add_argument("--pairs", type=int, default=2, help="n_pairs P (fixed per v3's ignited regime)")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-head", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=16, help="streaming chunk length (detach-carry boundary)")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-batch", type=int, default=200)
    ap.add_argument("--g-start", type=int, default=2, help="curriculum starting gap")
    ap.add_argument("--patience", type=int, default=25,
                    help="consecutive acc>0.9 iters required before the curriculum grows the gap")
    ap.add_argument("--train-gap-cap", type=int, default=0,
                    help="max TRAINING gap (0 = auto: single-chunk, chunk-2P-2)")
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--gaps", default="0,8,32", help="comma list of eval gap lengths G")
    ap.add_argument("--seeds", default="0,1", help="comma list of seeds")
    ap.add_argument("--out", default=os.path.join(RESULTS, "holo_heldout_keys.json"))
    args = ap.parse_args()

    if args.smoke:
        args.gaps = "0,8"
        args.seeds = "0"
        args.iters = min(args.iters, 600)
        if args.out == os.path.join(RESULTS, "holo_heldout_keys.json"):
            args.out = os.path.join(RESULTS, "holo_heldout_keys_smoke.json")
    elif args.full:
        args.gaps = "0,8,32"
        args.seeds = "0,1"
        args.iters = max(args.iters, 2000)

    run(args)


if __name__ == "__main__":
    main()
