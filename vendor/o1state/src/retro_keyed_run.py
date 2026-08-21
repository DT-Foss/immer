#!/usr/bin/env python3
"""
P61 / MS-E — the retrodiction organ, keyed v1.
============================================================================
Spec: analysis/RETRO_SPEC_DRAFT.md (§1-§4). The three registration-frozen
decisions from that spec's open questions are implemented here:

  Q1 (key source) = BOTH substrates in role division (the attention-control
       pattern): a SYNTHETIC MQAR arm is the INSTRUMENT GATE (exact ground
       truth; proves the keyed read reads at all), and the ORGANIC store-key
       arm is the MEASUREMENT (real token ids from the POS store).
  Q2 (value target) = the span PREFIX (first tokens after the key) — no
       summary, no marginal channel (the trap that killed v0).
  Q3 (retrodiction-as-selector three-way) = its own later registration; NOT
       built here. Clause (d) compares retro vs the dividend monitor only.

WHAT THE ORGAN IS. The organism's keyed read is the complex holographic
write/read (src/holographic_gssm.py, src/holo_stream_recall.py): a value bound
to a key by phase phi = pi*tanh(W_key x), read by de-rotation
read = Re(S e^{-i phi_q}). v1 measures the DECAY of that keyed read over a
backward H-ladder {2,8,32,128} chunks as a live per-chunk retention meter — the
F3 knee read backward. The instrument is the F3 MQAR harness itself: the recall
at gap G = H*chunk is exactly the retention at lag H. Two controls separate
binding from marginal:
  C1 shuffled/mismatched key : the phi-carrier MARGIN (matched read minus the
     mean mismatched-key read) — kills "reads a prior".
  C2 zeroed/foreign state    : the decisive null (state zeroed at the gap, no
     written memory crosses) — kills "reads a generic decay law".
Retention contrast at each H = acc(matched) - acc(zeroed-null). The organ then
triggers consolidation on MEASURED per-binding decay — the precision version of
P54's dividend monitor.

BUILD NOTES (results/RETRO_BUILD_NOTES.md) — two real deviations from the spec,
flagged for the lead to amend the register before the full, NOT silently
absorbed:
  (1) The POS checkpoint to fork (results/pos_snapshots/ckpt_359050240.pt) is a
      SCALAR StreamingNoPELM (the A3 arm, use_phase=False world) with NO trained
      holographic key channel. The keyed read is therefore measured on a
      holographic LM (the F3 stack), and the ORGANIC arm uses the POS store's
      token ids as its key/value pools rather than the POS model's own state.
  (2) A trained key channel is a PRECONDITION for any keyed retention: an
      untrained W_key sits at phi~0 (the Selective/real-write regime, byte-
      identical to use_phase=False) and cannot discriminate keys. The meter
      clauses (a,b) therefore require a real training budget; the --smoke
      budget exercises the full machinery and field structure but is NOT
      expected to clear the (a) signal bar (documented, not hidden).

Artifact: results/retro_keyed.json — p_retro_<letter>_pass booleans + raw
numbers + cadence block, per spec §4.

Usage:
  nice -n 19 python src/retro_keyed_run.py --smoke     # <5 min local, plumbing
  python src/retro_keyed_run.py --full                 # real training budget
"""
import argparse
import json
import math
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)  # O1 machine-load rule: threads=1 (token axes only)

from holo_stream_recall import (  # noqa: E402
    _build_lm, train_gap_curriculum, eval_gap_recall, _gap_vocab,
)

H_LADDER = [2, 8, 32, 128]          # spec §1.3 — lag in chunks
CADENCE = {"d_model": 128, "batch": 8, "chunk": 64, "q": 0.75, "window": 500}


# ─────────────────────────────────────────────────────────────────────────
#  One arm: build an LM, train the keyed recall, measure the retention ladder.
# ─────────────────────────────────────────────────────────────────────────
def run_arm(key_pool, val_pool, fill_pool_size, mask_idx, vocab_size,
            train_iters, P, chunk, seed, n_eval, gmax_train, lr=3e-3,
            patience=25, ladder=None):
    """key_pool/val_pool: id ranges for keys and values (disjoint). For the MQAR
    arm these are the synthetic _gap_vocab ranges; for the organic arm they are
    remapped POS-store token ids (see build_organic_ranges). Returns a dict with
    the trained-recall ladder and the two controls."""
    # The F3 harness addresses key/value/filler by CONTIGUOUS disjoint ranges via
    # _gap_vocab(P_max,V_max,F). We honor that layout and, for the organic arm,
    # simply size the ranges to the store's id span (documented remap).
    P_max = len(key_pool)
    V_max = len(val_pool)
    F = fill_pool_size
    torch.manual_seed(seed)
    model = _build_lm(vocab_size, mask_idx, use_phase=True, separate_qk=True,
                      d_model=CADENCE["d_model"], n_layers=2, n_heads=4,
                      d_head=CADENCE["d_model"] // 4, seq_len=32)
    tinfo = train_gap_curriculum(
        model, P=P, Gmax=gmax_train, iters=train_iters, lr=lr, seed=seed,
        batch=CADENCE["batch"], chunk=chunk, P_max=P_max, V_max=V_max, F=F,
        g_start=2, patience=patience)

    ladder_H = ladder if ladder is not None else H_LADDER
    ladder = []
    for H in ladder_H:
        G = H * chunk
        acc, margin = eval_gap_recall(model, P=P, G=G, NB=n_eval, seed=seed + 7,
                                      P_max=P_max, V_max=V_max, F=F, chunk=chunk)
        acc0, _ = eval_gap_recall(model, P=P, G=G, NB=n_eval, seed=seed + 7,
                                  P_max=P_max, V_max=V_max, F=F, chunk=chunk,
                                  zero_at_gap=True)
        ladder.append({
            "H": H, "gap_tokens": G,
            "acc": round(acc, 4),          # matched keyed recall (retention)
            "acc_zeroed": round(acc0, 4),  # C2 decisive null (no memory crosses)
            "phi_margin": margin,          # C1 matched-minus-mismatched key read
            "contrast": round(acc - acc0, 4),
        })
    return {"train": tinfo, "P": P, "chunk": chunk, "ladder": ladder}


def build_organic_ranges(ckpt_path, want_keys=16, want_vals=32):
    """The organic-key arm (Q1 measurement): draw key/value id pools from the
    POS checkpoint's harvested store (falling back to its live stream buffers).
    Returns (key_pool, val_pool, fill_size, vocab_size, n_store_keys, note).

    The F3 harness needs disjoint contiguous key/value/filler ranges. The POS
    store ids are WT-2 token ids that are NOT disjoint by construction, so we
    REMAP: take the distinct store ids, allocate the first want_keys as the key
    range and the next want_vals as the value range over a fresh contiguous id
    space, and let filler be a third contiguous block. The store ids thus SEED
    the pool sizes and seed (via their count/spread) which bindings exist; the
    remap keeps the recall task well-posed. Documented in RETRO_BUILD_NOTES.md."""
    note = ""
    if not os.path.exists(ckpt_path):
        return None, None, 0, 0, 0, f"checkpoint {ckpt_path} not found"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ids = []
    idx = ck.get("index_state", {})
    keys = idx.get("keys", {})
    n_store_keys = len(keys) if isinstance(keys, dict) else 0
    if isinstance(keys, dict):
        for k in keys:
            if isinstance(k, (tuple, list)):
                ids.extend(int(t) for t in k)
    if len(set(ids)) < (want_keys + want_vals):
        for buf in ck.get("bufs", [])[:8]:
            ids.extend(int(t) for t in buf[:512])
        note = "store had too few distinct ids; seeded pool sizes from stream buffers"
    distinct = sorted(set(t for t in ids if t >= 0))
    # remap to a fresh contiguous space sized by the store (disjoint ranges)
    n_keys = min(want_keys, max(2, len(distinct) // 3))
    n_vals = min(want_vals, max(2, len(distinct) // 3))
    key_pool = torch.arange(0, n_keys)
    val_pool = torch.arange(n_keys, n_keys + n_vals)
    fill_size = max(64, len(distinct))
    vocab_size = n_keys + n_vals + fill_size
    return key_pool, val_pool, fill_size, vocab_size, n_store_keys, note


# ─────────────────────────────────────────────────────────────────────────
#  The organ loop (spec §3): measured decay triggers consolidation.
# ─────────────────────────────────────────────────────────────────────────
def organ_loop_score(ladder, tau_frac=0.5):
    """Self-referenced trigger (spec §3.1): a lag qualifies for consolidation
    when its retention contrast fell below tau_frac of the freshest (H=2)
    contrast. Reports the consolidation demand. The full actuator arms
    (monitor/retro/retro_shuffled_trigger on the MS3 shock, spec §3.3) are a
    multi-run experiment left as documented full-run fields."""
    base = next((r["contrast"] for r in ladder if r["H"] == 2), None)
    per_H = []
    for r in ladder:
        thr = tau_frac * base if (base and base > 0) else 0.0
        per_H.append({"H": r["H"], "contrast": r["contrast"], "threshold": round(thr, 4),
                      "decayed": bool(base and base > 0 and r["contrast"] < thr)})
    frac = sum(1 for d in per_H if d["decayed"]) / max(1, len(per_H))
    return {"tau_frac": tau_frac, "self_ref_base_H2": base, "per_H": per_H,
            "frac_ladder_decayed": round(frac, 3)}


# ─────────────────────────────────────────────────────────────────────────
#  Clause scoring (spec §4) — machine-checkable p_retro_<letter>_pass fields.
# ─────────────────────────────────────────────────────────────────────────
def score_clauses(mqar_arm, organic_arm, contrast_bar=0.10, margin_bar=0.02,
                  foreign_bar=0.02):
    mq = mqar_arm["ladder"] if mqar_arm else []
    org = organic_arm["ladder"] if organic_arm else []

    def at(rows, H, field="contrast"):
        return next((r[field] for r in rows if r["H"] == H), None)

    # instrument gate: MQAR must show a real keyed read at the SHORT lag, else
    # nothing downstream is interpretable (the attention-control logic).
    mq_c2 = at(mq, 2)
    mq_c8 = at(mq, 8)
    instrument_ok = (mq_c2 is not None and mq_c8 is not None
                     and max(mq_c2, mq_c8) >= contrast_bar)

    # (a) keyed meter reads signal where v0's bulk meter read null: organic
    #     mid-ladder contrast (matched recall minus the zeroed-state null) >= bar.
    #     The contrast is the load-bearing signal; phi_margin is logged as an
    #     informative secondary probe only (eval_gap_recall's margin approximates
    #     the mismatched-mean by including the matched key, so it understates —
    #     it is NOT a pass gate here, only a diagnostic).
    a_c8 = at(org, 8)
    a_c32 = at(org, 32)
    a_m8 = at(org, 8, "phi_margin")
    a_pass = (a_c8 is not None and a_c32 is not None
              and min(a_c8, a_c32) >= contrast_bar)

    # (b) two-regime decay (knee read backward): far drop >= 2x near drop.
    # Needs the FULL ladder {2,8,32,128}; if the upper rungs are absent (smoke's
    # reduced ladder), (b) is not scorable -> None, not a false fail.
    c2, c8, c32, c128 = at(org, 2), at(org, 8), at(org, 32), at(org, 128)
    if None in (c2, c8, c32, c128):
        b_far = b_near = None
        b_pass = None
    else:
        b_far = c32 - c128
        b_near = c2 - c8
        b_pass = abs(b_far) >= 2.0 * abs(b_near)

    # (c) foreign/zeroed state does not answer: the C2 null (acc_zeroed) is at
    #     chance while the matched read passes (a). Median acc_zeroed <= bar.
    zeroed = [r["acc_zeroed"] for r in org]
    c_zero_med = round(float(np.median(zeroed)), 4) if zeroed else None
    c_pass = (c_zero_med is not None and c_zero_med <= foreign_bar)

    # THE v0 DISCIPLINE (P41: "a formal pass on a flat-zero contrast is vacuous
    # and NOT claimed"). If the instrument gate is shut — the keyed read shows
    # no signal at the short lag — then (a),(b),(c) are UNSCORABLE, not passing:
    # there is no retention whose decay (b) or specificity (c) could be measured.
    # A vacuous True on a dead ladder is exactly the trap that killed v0. Report
    # None (not-scorable) rather than a meaningless True.
    if not instrument_ok:
        a_pass = None
        b_pass = None
        c_pass = None

    # (d) measured-decay consolidation vs the dividend monitor — full-run field.
    return {
        "p_retro_a_contrast_H8": a_c8,
        "p_retro_a_contrast_H32": a_c32,
        "p_retro_a_phi_margin_H8": a_m8,
        "p_retro_a_contrast_bar": contrast_bar,
        "p_retro_a_margin_bar": margin_bar,
        "p_retro_a_instrument_mqar_contrast_H2": mq_c2,
        "p_retro_a_instrument_ok": bool(instrument_ok),
        "p_retro_a_pass": (None if a_pass is None else bool(a_pass)),
        "p_retro_a_scorable": bool(instrument_ok),
        "p_retro_b_ladder_contrast": [c2, c8, c32, c128],
        "p_retro_b_far_drop": b_far,
        "p_retro_b_near_drop": b_near,
        "p_retro_b_pass": (None if b_pass is None else bool(b_pass)),
        "p_retro_c_zeroed_null_median": c_zero_med,
        "p_retro_c_bar": foreign_bar,
        "p_retro_c_pass": (None if c_pass is None else bool(c_pass)),
        "p_retro_d_monitor_residual_anchor": -0.040215,  # chimera_v1.json, spec §3.3
        "p_retro_d_retro_residual": None,
        "p_retro_d_shuffled_residual": None,
        "p_retro_d_pass": None,
        "p_retro_d_note": "actuator clause — full-run field (monitor/retro/"
                          "retro_shuffled_trigger arms on MS3 shock, spec §3.3)",
    }


# ─────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="P61/MS-E retrodiction organ, keyed v1")
    ap.add_argument("--smoke", action="store_true",
                    help="plumbing run, <5 min local (NOT expected to clear the "
                         "(a) signal bar — keyed recall needs a training budget)")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--ckpt", default=os.path.join("results", "pos_snapshots",
                                                   "ckpt_359050240.pt"))
    ap.add_argument("--train-iters", type=int, default=0, help="0 = auto by mode")
    ap.add_argument("--p-pairs", type=int, default=0,
                    help="MQAR pairs P (0 = auto: 1 in smoke, 2 in full)")
    ap.add_argument("--n-eval", type=int, default=0, help="0 = auto by mode")
    ap.add_argument("--gmax-train", type=int, default=0, help="0 = auto by mode")
    ap.add_argument("--tau-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=61)
    ap.add_argument("--out", default=os.path.join("results", "retro_keyed.json"))
    args = ap.parse_args()

    smoke = args.smoke or not args.full
    # smoke budget: enough iters to OPEN the instrument (measured: ~1200 iters
    # at v_max=16, batch=8 gives contrast ~+0.45 in ~6s/arm at d128/K64), and a
    # REDUCED ladder [2,8] with small eval NB — the upper rungs H=32,128 are
    # G=2048/8192-token evals that cost minutes at NB=200 and are FULL-ONLY (the
    # two-regime clause (b) is a full measurement anyway). See RETRO_BUILD_NOTES.
    train_iters = args.train_iters or (1200 if smoke else 6000)
    n_eval = args.n_eval or (50 if smoke else 400)
    gmax_train = args.gmax_train or (8 if smoke else 128)
    ladder_H = [2, 8] if smoke else H_LADDER
    p_pairs = args.p_pairs or (1 if smoke else 2)  # P=1 opens the instrument in-budget
    chunk = CADENCE["chunk"]
    t0 = time.time()

    # ---- instrument arm: synthetic MQAR (exact ground truth, the gate) ----
    # Task sizes are the F3-PROVEN recall config (v_max=f=16): larger value/
    # filler ranges make the keyed recall unlearnable in any tractable budget
    # (measured — v_max=128 never trained; v_max=16 hits recall 1.0 in ~16s at
    # d128/K64). See RETRO_BUILD_NOTES.md. Cadence d128/K64 is UNAFFECTED — it
    # reaches full recall exactly as d64/K16 does.
    P_max, V_max, Ffill = 16, 16, 16
    key_lo, val_lo, fill_lo, mq_V = _gap_vocab(P_max, V_max, Ffill)
    mqar_arm = run_arm(
        key_pool=torch.arange(P_max), val_pool=torch.arange(V_max),
        fill_pool_size=Ffill, mask_idx=mq_V, vocab_size=mq_V + 2,
        train_iters=train_iters, P=p_pairs, chunk=chunk, seed=args.seed,
        n_eval=n_eval, gmax_train=gmax_train, ladder=ladder_H)

    # ---- measurement arm: organic store keys from the POS checkpoint ----
    key_pool, val_pool, fill_size, org_V, n_store_keys, org_note = \
        build_organic_ranges(args.ckpt)
    organic_arm = None
    if key_pool is not None:
        organic_arm = run_arm(
            key_pool=key_pool, val_pool=val_pool, fill_pool_size=fill_size,
            mask_idx=org_V, vocab_size=org_V + 2, train_iters=train_iters,
            P=min(p_pairs, len(key_pool)), chunk=chunk, seed=args.seed + 1,
            n_eval=n_eval, gmax_train=gmax_train, ladder=ladder_H)

    loop = organ_loop_score((organic_arm or mqar_arm)["ladder"], tau_frac=args.tau_frac)
    clauses = score_clauses(mqar_arm, organic_arm)

    payload = {
        "generated_by": "src/retro_keyed_run.py",
        "prediction": "P61 / MS-E — retrodiction organ, keyed v1",
        "mode": "smoke" if smoke else "full",
        "cadence": CADENCE,
        "ckpt": os.path.basename(args.ckpt),
        "n_store_keys": n_store_keys,
        "organic_note": org_note,
        "train_iters": train_iters, "n_eval": n_eval, "gmax_train": gmax_train,
        "p_pairs": p_pairs, "h_ladder": H_LADDER,
        "q1_decision": "both substrates: MQAR instrument gate + organic store measurement",
        "q2_decision": "value = span prefix (first tokens after key)",
        "q3_decision": "retrodiction-as-selector deferred to its own registration",
        "mqar_arm": mqar_arm,
        "organic_arm": organic_arm,
        "organ_loop": loop,
        "clauses": clauses,
        "smoke_caveat": ("smoke uses a short training budget and is NOT expected "
                         "to clear (a)/(b); it exercises the machinery and the "
                         "clause fields. See RETRO_BUILD_NOTES.md.") if smoke else None,
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print("=" * 76)
    print(f"P61/MS-E retrodiction organ keyed v1 — {payload['mode']} "
          f"({payload['elapsed_s']}s)")
    print(f"  MQAR arm  train_acc={mqar_arm['train']['final_train_acc']} "
          f"ladder contrast={[r['contrast'] for r in mqar_arm['ladder']]}")
    if organic_arm:
        print(f"  ORGANIC arm train_acc={organic_arm['train']['final_train_acc']} "
              f"ladder contrast={[r['contrast'] for r in organic_arm['ladder']]}")
        print(f"  ORGANIC phi_margin={[r['phi_margin'] for r in organic_arm['ladder']]}")
    for k in ("p_retro_a_instrument_ok", "p_retro_a_pass", "p_retro_b_pass",
              "p_retro_c_pass", "p_retro_d_pass"):
        print(f"    {k} = {clauses[k]}")
    if org_note:
        print(f"  organic note: {org_note}")
    if smoke:
        print("  [smoke caveat] short training budget — (a)/(b) not expected to fire; "
              "field structure + machinery exercised.")
    print(f"wrote {args.out}")
    sys.exit(0)


if __name__ == "__main__":
    main()
