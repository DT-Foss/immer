#!/usr/bin/env python3 -u
"""
HOLO-CAPACITY-RETURN — does the phase advantage return with capacity?
======================================================================================
M1 (results/holo_heldout_keys.json / _8k.json) ran the compositional-binding test at
P_max=64 keys, d_model=64 and found the v3 phase advantage (+27pp over holo_off at
P_max=16, d_model=64) had VANISHED: both holo_carried and holo_off plateau at ~0.55,
even at 8k iterations (4x budget), with the curriculum never leaving G=2
(final_train_acc 0.38-0.63). Two readings were left open in
analysis/HOLO_STREAM_VERDICT.md ("Phase B+ / open question"):
  (a) phase binding is a SMALL-KEY-SPACE phenomenon — it vanishes for good once the
      key space outgrows some fixed channel budget, regardless of how much capacity
      you add elsewhere.
  (b) it is a CAPACITY LAW — d_model=64 (256 total state channels: n_heads*d_head*
      n_layers = 4*16*2) is simply too small to represent P_max=64 keys distinctly,
      and scaling d_model back up to the key count restores the advantage.

THIS RUN (P17, registered): fix everything else at M1's setup and sweep d_model in
{64, 128, 256} (n_heads=4 fixed, d_head=d_model/4, so total state channels scale as
n_heads*d_head*n_layers = d_model/4 * 4 * 2 = 2*d_model -> 128/256/512 channels for
P_max=64 keys). If (b) is right, holo_carried should pull back away from holo_off as
d_model grows, and the curriculum should finally consolidate (final_train_acc climbing
toward v3's 0.8-0.9 and the gap curriculum growing past G=2) at d_model=256. If (a) is
right, both arms stay pinned near chance/~0.55 regardless of d_model.

PREDICTION (P17, registered before this file ran): holo_carried - holo_off >= +15pp
at d_model=256 (at least one of G in {0,8,32}, on at least one of train_keys/test_keys)
= the capacity story. Anything short of that (advantage stays absent, or only appears
on train_keys and not test_keys) is reported verbatim, not hedged.

Design — held fixed to M1 exactly except where noted:
  - P=2 pairs, P_max=64 keys, 40 train-key ids / 24 held-out (test) key ids,
    V_max=16 values, F=16 ordinary fillers (the _gap_vocab idiom, same ranges as M1).
  - heldout-filler-rate 0.2 (K_test ids mixed into training fillers at this rate,
    embedding-confound control — see holo_heldout_keys.py docstring, unchanged here).
  - v3 recipe: lr 3e-3, batch 32, chunk 16, g_start 2, patience 25.
  - NEW vs M1: curriculum bar 0.8 instead of the hardcoded 0.9. M1's bar was fixed at
    0.9 (hardcoded `acc > 0.9` in train_gap_curriculum_heldout); holo_gap_knee.py's
    M2 run found bar=0.9 NEVER FIRES for P=2 (P=2 consolidates around ~0.85, so the
    curriculum silently never grows past g_start=2 -- exactly the "curriculum never
    left G=2" symptom M1 also reports). We reimplement the same bar mechanism locally
    (train_gap_curriculum_heldout_bar, a bar-parameterized copy of the imported
    train_gap_curriculum_heldout loop) so the d=256 cell gets a genuine chance to
    consolidate past G=2 instead of silently reproducing M1's non-firing curriculum.
  - d_model sweep: d in {64, 128, 256}, n_heads=4 fixed, d_head=d_model//4 (16/32/64).
  - iters 3000 (up from M1's 2000, still well short of M1's 8k consolidation control --
    this is a capacity sweep, not a second consolidation study).
  - 2 seeds (0, 1).
  - Arms: holo_carried (use_phase=True), holo_off (use_phase=False) at every d; plus
    holo_zeroed_at_gap (the decisive null, use_phase=True + state zeroed at gap onset)
    as a spot-check ONLY at d=256 (cheapest place to confirm the carried-state result
    still holds at the top of the capacity sweep, without tripling the cost of every
    cell).
  - Eval: accuracy on train_keys AND test_keys, G in {0, 8, 32}, eval_batch 200.

Output: results/holo_capreturn.json (or *_smoke.json under --smoke). Per (d_model, arm,
key_set, G): accuracy (per-seed + seed-mean). Per (d_model, arm, seed): curriculum
final state (final_train_acc, final_train_gap -- the M1 consolidation check, does
d=256 finally leave G=2?). phase_advantage = holo_carried_acc - holo_off_acc per
(d_model, key_set, G) (seed-meaned). Verdict against P17 stated for all three possible
outcomes up front (returns / stays absent / train-keys-only), decided mechanically from
the advantage trend across d, not narrated after the fact.

CLI: --smoke (1 seed, d in {64,256}, 1000 iters, G in {0,8}, ~cheapest cross-section
of the sweep) / --full (2 seeds, d in {64,128,256}, 3000 iters, G in {0,8,32}).

CPU-only, single-thread (torch.set_num_threads(1)), os.nice(19) set at process start.
Repo idiom, code/comments in English. Results -> results/holo_capreturn*.json only.
"""
import os
os.nice(19)

import sys
import json
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference"))

import torch
import torch.nn as nn

torch.backends.mps.is_available = lambda: False   # force CPU (repo convention)
torch.set_num_threads(1)

from holo_stream_recall import _gap_vocab, _build_lm, chunked_forward   # noqa: E402
from holo_heldout_keys import make_heldout_gap_batch, eval_heldout      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(REPO, "results")

HELDOUT_FILLER_RATE = 0.20


# ═══════════════════════════════════════════════════════════════════════════
# 1. Curriculum with a caller-set bar (M2's holo_gap_knee.py fix: `acc > 0.9`
#    hardcoded never fires for P=2, which consolidates ~0.85 -- the curriculum
#    silently never grows past g_start. Same loop as holo_heldout_keys.py's
#    train_gap_curriculum_heldout, parameterized on `bar` instead of the literal.)
# ═══════════════════════════════════════════════════════════════════════════
def train_gap_curriculum_heldout_bar(model, P, Gmax, iters, lr, seed, batch, chunk,
                                      key_train, key_test, V_max, F, val_lo, fill_lo,
                                      bar=0.8, log_every=0, g_start=2, patience=25,
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
        good = good + 1 if acc > bar else 0
        if good >= patience and Gcur < Gmax:
            Gcur = min(Gmax, int(Gcur * 1.5) + 1)
            good = 0
        if log_every and (it + 1) % log_every == 0:
            print(f"    it {it+1:>4}/{iters}: loss {float(loss):.3f} acc {acc:.3f} (train-gap {Gcur})")
    return {"final_train_gap": Gcur, "final_train_acc": round(acc, 4)}


# ═══════════════════════════════════════════════════════════════════════════
# 2. Orchestration: d_model sweep x arm x key_set x G, 2 seeds.
# ═══════════════════════════════════════════════════════════════════════════
def run(args):
    P_max, V_max, F = args.p_max, args.v_max, args.f_fillers
    key_lo, val_lo, fill_lo, vocab_size = _gap_vocab(P_max, V_max, F)
    mask_idx = vocab_size
    chance = 1.0 / V_max

    n_train_keys = args.train_keys
    key_train = torch.arange(key_lo, key_lo + n_train_keys)
    key_test = torch.arange(key_lo + n_train_keys, key_lo + P_max)
    assert key_test.numel() > 0, "no held-out keys left -- check --p-max/--train-keys"

    Gs = [int(g) for g in args.gaps.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    d_models = [int(d) for d in args.d_models.split(",")]
    P = args.pairs

    print("=" * 78)
    print("HOLO-CAPACITY-RETURN — does the phase advantage return with capacity? (P17)")
    print(f"P_max={P_max} V_max={V_max} F={F} vocab={vocab_size}  chance=1/{V_max}={chance:.4f}")
    print(f"K_train=[{key_lo},{key_lo+n_train_keys}) ({n_train_keys} ids)  "
          f"K_test=[{key_lo+n_train_keys},{key_lo+P_max}) ({key_test.numel()} ids)")
    print(f"P={P}  d_models={d_models}  gaps(G)={Gs}  seeds={seeds}  chunk={args.chunk}  "
          f"bar={args.curriculum_bar}  heldout_filler_rate={args.heldout_rate}  iters={args.iters}")
    print("=" * 78)

    t0 = time.time()
    results = {"config": vars(args),
               "key_split": {"key_lo": key_lo, "n_train_keys": n_train_keys,
                             "n_test_keys": int(key_test.numel())},
               "chance": chance, "sweep": {}, "curriculum": {}, "timing": {}}

    for d_model in d_models:
        n_heads = args.n_heads
        d_head = d_model // n_heads
        assert d_head * n_heads == d_model, \
            f"d_model={d_model} not divisible by n_heads={n_heads}"

        arms = {
            "holo_carried": dict(use_phase=True, zero_at_gap=False),
            "holo_off": dict(use_phase=False, zero_at_gap=False),
        }
        if d_model == args.null_check_d_model:
            arms["holo_zeroed_at_gap"] = dict(use_phase=True, zero_at_gap=True)

        for seed in seeds:
            print(f"\n{'='*78}\nd_model={d_model} (d_head={d_head})  seed={seed}  P={P}\n{'='*78}")
            for arm_name, arm_cfg in arms.items():
                t_arm = time.time()
                torch.manual_seed(seed)
                model = _build_lm(vocab_size, mask_idx, use_phase=arm_cfg["use_phase"],
                                  d_model=d_model, n_layers=args.n_layers,
                                  n_heads=n_heads, d_head=d_head)
                Gmax_eval = max(Gs) if Gs else 0
                cap = args.train_gap_cap if args.train_gap_cap > 0 else args.chunk - 2 * P - 2
                Gmax_train = min(Gmax_eval, max(2, cap))
                curr = train_gap_curriculum_heldout_bar(
                    model, P, Gmax_train, args.iters, args.lr, seed, args.batch, args.chunk,
                    key_train, key_test, V_max, F, val_lo, fill_lo,
                    bar=args.curriculum_bar, log_every=args.log_every,
                    g_start=args.g_start, patience=args.patience,
                    heldout_rate=args.heldout_rate)
                curr["train_gap_cap"] = Gmax_train
                curr["d_model"] = d_model
                curr["d_head"] = d_head
                train_s = time.time() - t_arm
                key = f"d{d_model}_seed{seed}_{arm_name}"
                results["curriculum"][key] = curr
                print(f"  [d={d_model:>3} {arm_name:20s}] curriculum done: "
                      f"final_train_gap={curr['final_train_gap']} "
                      f"final_train_acc={curr['final_train_acc']:.3f}  "
                      f"({train_s:.1f}s, {train_s/args.iters*1000:.1f}ms/iter)")

                for G in Gs:
                    for set_name, pool in (("train_keys", key_train), ("test_keys", key_test)):
                        acc = eval_heldout(
                            model, P, G, args.eval_batch, seed + 1000 + G, pool, V_max, F,
                            val_lo, fill_lo, args.chunk, zero_at_gap=arm_cfg["zero_at_gap"])
                        sk = f"d{d_model}|{arm_name}|{set_name}|G{G}|seed{seed}"
                        results["sweep"][sk] = {
                            "d_model": d_model, "seed": seed, "arm": arm_name,
                            "key_set": set_name, "P": P, "G": G,
                            "accuracy": round(acc, 4), "chance": round(chance, 4),
                            "beats_chance_3x": bool(acc > 3 * chance)}
                        print(f"    [d={d_model:>3} {arm_name:20s}] {set_name:10s} G={G:>4}: "
                              f"acc={acc:.4f} (chance {chance:.4f}, "
                              f"3x {'YES' if acc > 3*chance else 'no'})")
                results["timing"][key] = round(train_s, 2)

    # ── phase_advantage = holo_carried - holo_off, per (d_model, key_set, G) ──
    def _mean_at(d_model, arm, set_name, G):
        vals = [v["accuracy"] for k, v in results["sweep"].items()
                if v["d_model"] == d_model and v["arm"] == arm
                and v["key_set"] == set_name and v["G"] == G]
        return sum(vals) / len(vals) if vals else None

    def _seed_vals_at(d_model, arm, set_name, G):
        return {v["seed"]: v["accuracy"] for k, v in results["sweep"].items()
                if v["d_model"] == d_model and v["arm"] == arm
                and v["key_set"] == set_name and v["G"] == G}

    phase_advantage = {}
    detail = []
    for d_model in d_models:
        phase_advantage[str(d_model)] = {}
        for set_name in ("train_keys", "test_keys"):
            for G in Gs:
                holo = _mean_at(d_model, "holo_carried", set_name, G)
                off = _mean_at(d_model, "holo_off", set_name, G)
                if holo is None or off is None:
                    continue
                adv = round(holo - off, 4)
                phase_advantage[str(d_model)][f"{set_name}_G{G}"] = adv
                detail.append(
                    f"d={d_model:>3} {set_name:10s} G={G:>4}: "
                    f"holo_carried={holo:.3f} holo_off={off:.3f} "
                    f"phase_advantage={adv:+.3f}  chance={chance:.3f}")
    results["phase_advantage"] = phase_advantage
    results["verdict_detail"] = detail

    # ── consolidation check: does d=256 leave G=2 where M1 (d=64) never did? ──
    consolidation = {}
    for d_model in d_models:
        accs = [v["final_train_acc"] for k, v in results["curriculum"].items()
                if v.get("d_model") == d_model]
        gaps = [v["final_train_gap"] for k, v in results["curriculum"].items()
                if v.get("d_model") == d_model]
        if accs:
            consolidation[str(d_model)] = {
                "final_train_acc_range": [round(min(accs), 4), round(max(accs), 4)],
                "final_train_gap_range": [min(gaps), max(gaps)],
                "left_g_start": any(g > args.g_start for g in gaps),
            }
    results["consolidation_check"] = consolidation

    # ── verdict against P17: holo_carried - holo_off >= +15pp at d_model=256,
    #    at some G, on some key_set ──
    top_d = max(d_models)
    hits_train = [phase_advantage.get(str(top_d), {}).get(f"train_keys_G{G}")
                  for G in Gs]
    hits_test = [phase_advantage.get(str(top_d), {}).get(f"test_keys_G{G}")
                 for G in Gs]
    hits_train = [h for h in hits_train if h is not None]
    hits_test = [h for h in hits_test if h is not None]

    returns_train = any(h >= 0.15 for h in hits_train)
    returns_test = any(h >= 0.15 for h in hits_test)

    # trend across d: does the best-G advantage at each d move monotonically up?
    trend = []
    for d_model in d_models:
        best = None
        for set_name in ("train_keys", "test_keys"):
            for G in Gs:
                v = phase_advantage.get(str(d_model), {}).get(f"{set_name}_G{G}")
                if v is not None and (best is None or v > best):
                    best = v
        trend.append((d_model, best))
    results["advantage_trend_best_per_d"] = trend

    if returns_train and returns_test:
        verdict = (f"P17 CONFIRMED: the phase advantage RETURNS with capacity -- at "
                   f"d_model={top_d}, holo_carried beats holo_off by >=15pp on BOTH "
                   f"train_keys and test_keys in at least one G cell. Capacity law, "
                   f"not a small-key-space-only phenomenon.")
    elif returns_train and not returns_test:
        verdict = (f"P17 PARTIAL: the phase advantage returns on train_keys at "
                   f"d_model={top_d} (>=15pp) but NOT on test_keys -- capacity restores "
                   f"raw binding fidelity but not (yet, at this budget) compositional "
                   f"generalization to held-out key ids. Reported as-is, not hedged.")
    elif returns_test and not returns_train:
        verdict = (f"P17 UNEXPECTED: advantage clears +15pp on test_keys at "
                   f"d_model={top_d} but not train_keys -- inspect for a bug or a "
                   f"genuinely surprising asymmetry before trusting this cell.")
    else:
        verdict = (f"P17 NOT CONFIRMED at d_model up to {top_d}: the phase advantage did "
                   f"NOT return to >=15pp on any key_set/G cell -- either the effect is a "
                   f"small-key-space phenomenon after all (reading (a)), or the capacity "
                   f"ceiling needed is beyond {top_d} (still open, needs a further d sweep "
                   f"point, e.g. d=512, before reading (a) is safe to conclude).")

    results["verdict"] = verdict
    results["elapsed_s"] = round(time.time() - t0, 1)

    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "=" * 78)
    print("PHASE ADVANTAGE (holo_carried - holo_off) BY d_model")
    for ln in detail:
        print("  " + ln)
    print("\nCONSOLIDATION CHECK (final_train_acc / final_train_gap range per d_model)")
    for d_model in d_models:
        c = consolidation.get(str(d_model))
        if c:
            print(f"  d={d_model:>3}: acc_range={c['final_train_acc_range']} "
                  f"gap_range={c['final_train_gap_range']} "
                  f"left_g_start={c['left_g_start']}")
    print(f"\n>>> {verdict}")
    print(f"\n-> {args.out}  ({results['elapsed_s']}s)")


# ═══════════════════════════════════════════════════════════════════════════
# 3. lr-control: is holo_off's d=256 collapse (0.219 final_train_acc in the smoke,
#    well below holo_carried's 0.625 AND below holo_off's OWN d=64 number of 0.469)
#    an optimization artifact of the larger model at the same lr=3e-3, or does it
#    genuinely fail to bind regardless of lr? Trains ONLY holo_off, at ONE d_model,
#    across a list of lr candidates, same eval protocol as the main sweep. Written
#    as an independent CLI path (does not touch run()/the main sweep) so the
#    reported phase_advantage in the main sweep is never silently redefined --
#    the honest comparison this produces is holo_carried (main sweep, lr=3e-3)
#    minus best-of-off (max accuracy across lr candidates here), computed and
#    reported separately, never conflated with the single-lr phase_advantage field.
# ═══════════════════════════════════════════════════════════════════════════
def run_offlr_control(args):
    P_max, V_max, F = args.p_max, args.v_max, args.f_fillers
    key_lo, val_lo, fill_lo, vocab_size = _gap_vocab(P_max, V_max, F)
    mask_idx = vocab_size
    chance = 1.0 / V_max

    n_train_keys = args.train_keys
    key_train = torch.arange(key_lo, key_lo + n_train_keys)
    key_test = torch.arange(key_lo + n_train_keys, key_lo + P_max)

    Gs = [int(g) for g in args.gaps.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    lrs = [float(lr) for lr in args.offlr_candidates.split(",")]
    d_model = args.offlr_d_model
    n_heads = args.n_heads
    d_head = d_model // n_heads
    assert d_head * n_heads == d_model
    P = args.pairs

    print("=" * 78)
    print("HOLO-CAPACITY-RETURN — holo_off lr-control at d_model="
          f"{d_model} (optimization-artifact check)")
    print(f"lr candidates={lrs}  seeds={seeds}  gaps(G)={Gs}  iters={args.iters}")
    print("=" * 78)

    t0 = time.time()
    results = {"config": vars(args), "d_model": d_model, "d_head": d_head,
               "chance": chance, "sweep": {}, "curriculum": {}}

    for lr in lrs:
        for seed in seeds:
            t_arm = time.time()
            torch.manual_seed(seed)
            model = _build_lm(vocab_size, mask_idx, use_phase=False,
                              d_model=d_model, n_layers=args.n_layers,
                              n_heads=n_heads, d_head=d_head)
            Gmax_eval = max(Gs) if Gs else 0
            cap = args.train_gap_cap if args.train_gap_cap > 0 else args.chunk - 2 * P - 2
            Gmax_train = min(Gmax_eval, max(2, cap))
            curr = train_gap_curriculum_heldout_bar(
                model, P, Gmax_train, args.iters, lr, seed, args.batch, args.chunk,
                key_train, key_test, V_max, F, val_lo, fill_lo,
                bar=args.curriculum_bar, log_every=args.log_every,
                g_start=args.g_start, patience=args.patience,
                heldout_rate=args.heldout_rate)
            curr["train_gap_cap"] = Gmax_train
            curr["lr"] = lr
            train_s = time.time() - t_arm
            key = f"lr{lr}_seed{seed}"
            results["curriculum"][key] = curr
            print(f"  [lr={lr:<8} seed={seed}] curriculum done: "
                  f"final_train_gap={curr['final_train_gap']} "
                  f"final_train_acc={curr['final_train_acc']:.3f}  "
                  f"({train_s:.1f}s, {train_s/args.iters*1000:.1f}ms/iter)")

            for G in Gs:
                for set_name, pool in (("train_keys", key_train), ("test_keys", key_test)):
                    acc = eval_heldout(
                        model, P, G, args.eval_batch, seed + 1000 + G, pool, V_max, F,
                        val_lo, fill_lo, args.chunk, zero_at_gap=False)
                    sk = f"lr{lr}|{set_name}|G{G}|seed{seed}"
                    results["sweep"][sk] = {
                        "lr": lr, "seed": seed, "arm": "holo_off", "key_set": set_name,
                        "P": P, "G": G, "accuracy": round(acc, 4),
                        "chance": round(chance, 4), "beats_chance_3x": bool(acc > 3 * chance)}
                    print(f"    [lr={lr:<8}] {set_name:10s} G={G:>4}: acc={acc:.4f}")

    # best-of-off per (key_set, G): max accuracy across the lr candidates (seed-meaned)
    def _mean_at(lr, set_name, G):
        vals = [v["accuracy"] for v in results["sweep"].values()
                if v["lr"] == lr and v["key_set"] == set_name and v["G"] == G]
        return sum(vals) / len(vals) if vals else None

    best_of_off = {}
    detail = []
    for set_name in ("train_keys", "test_keys"):
        for G in Gs:
            per_lr = {lr: _mean_at(lr, set_name, G) for lr in lrs}
            per_lr = {lr: v for lr, v in per_lr.items() if v is not None}
            if not per_lr:
                continue
            best_lr = max(per_lr, key=per_lr.get)
            best_of_off[f"{set_name}_G{G}"] = {"best_lr": best_lr,
                                                "best_acc": round(per_lr[best_lr], 4),
                                                "per_lr": {str(k): round(v, 4)
                                                           for k, v in per_lr.items()}}
            detail.append(f"{set_name:10s} G={G:>4}: best_of_off={per_lr[best_lr]:.3f} "
                          f"(at lr={best_lr})  per_lr={per_lr}")
    results["best_of_off"] = best_of_off
    results["verdict_detail"] = detail
    results["elapsed_s"] = round(time.time() - t0, 1)

    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "=" * 78)
    print("BEST-OF-OFF (max accuracy across lr candidates) BY key_set/G")
    for ln in detail:
        print("  " + ln)
    print(f"\n-> {args.out}  ({results['elapsed_s']}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="fast sanity run (1 seed, d in {64,256}, 1000 iters, G in {0,8})")
    ap.add_argument("--full", action="store_true",
                    help="the full sweep (2 seeds, d in {64,128,256}, 3000 iters, G in {0,8,32})")
    ap.add_argument("--offlr-control", action="store_true",
                    help="run ONLY the holo_off lr-control at --offlr-d-model "
                         "(optimization-artifact check; writes a separate output file)")
    ap.add_argument("--offlr-d-model", type=int, default=256,
                    help="d_model at which the lr-control sweep runs (offlr-control mode)")
    ap.add_argument("--offlr-candidates", default="1e-3,3e-3",
                    help="comma list of lr candidates for holo_off (offlr-control mode)")
    ap.add_argument("--p-max", type=int, default=64, help="total distinct key ids")
    ap.add_argument("--train-keys", type=int, default=40, help="size of K_train (rest = K_test, held out)")
    ap.add_argument("--v-max", type=int, default=16, help="distinct value ids")
    ap.add_argument("--f-fillers", type=int, default=16, help="distinct ordinary filler ids")
    ap.add_argument("--heldout-rate", type=float, default=HELDOUT_FILLER_RATE,
                    help="rate at which K_test ids replace ordinary fillers during TRAINING")
    ap.add_argument("--pairs", type=int, default=2, help="n_pairs P (fixed per v3/M1's ignited regime)")
    ap.add_argument("--d-models", default="64,128,256", help="comma list of d_model sweep points")
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--null-check-d-model", type=int, default=256,
                    help="d_model at which the holo_zeroed_at_gap decisive-null spot-check also runs")
    ap.add_argument("--chunk", type=int, default=16, help="streaming chunk length (detach-carry boundary)")
    ap.add_argument("--curriculum-bar", type=float, default=0.8,
                    help="acc bar the curriculum must sustain before growing the gap "
                         "(0.9 never fires for P=2, which consolidates ~0.85 -- the M1/M2 diagnosis)")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-batch", type=int, default=200)
    ap.add_argument("--g-start", type=int, default=2, help="curriculum starting gap")
    ap.add_argument("--patience", type=int, default=25,
                    help="consecutive acc>bar iters required before the curriculum grows the gap")
    ap.add_argument("--train-gap-cap", type=int, default=0,
                    help="max TRAINING gap (0 = auto: single-chunk, chunk-2P-2)")
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--gaps", default="0,8,32", help="comma list of eval gap lengths G")
    ap.add_argument("--seeds", default="0,1", help="comma list of seeds")
    ap.add_argument("--out", default=None,
                    help="output path (default: holo_capreturn.json, or "
                         "holo_capreturn_offlr_control.json in --offlr-control mode)")
    args = ap.parse_args()

    if args.out is None:
        args.out = os.path.join(
            RESULTS, "holo_capreturn_offlr_control.json" if args.offlr_control
            else "holo_capreturn.json")

    if args.smoke:
        args.d_models = "64,256"
        args.gaps = "0,8"
        args.seeds = "0"
        args.iters = min(args.iters, 1000)
        args.null_check_d_model = 256
        if args.out == os.path.join(RESULTS, "holo_capreturn.json"):
            args.out = os.path.join(RESULTS, "holo_capreturn_smoke.json")
    elif args.full:
        args.d_models = "64,128,256"
        args.gaps = "0,8,32"
        args.seeds = "0,1"
        args.iters = max(args.iters, 3000)

    if args.offlr_control:
        run_offlr_control(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
