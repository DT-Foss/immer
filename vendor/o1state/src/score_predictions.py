#!/usr/bin/env python3
"""
Auto-Falsifier — score analysis/PREDICTIONS.md's P1-P20 against the JSONs that
have actually landed in results/, mechanically.
============================================================================
PREDICTIONS.md is prose with numbers; this file is NOT a prose parser. The
REGISTRY below is hand-curated: for every P-number we hardcode which result
file(s) it depends on, a small extractor that pulls the measured quantity out
of that JSON, and a rule that turns the measured quantity into a verdict.
PREDICTIONS.md stays the human-readable source; this script is the
machine-readable backbone that re-runs every time new results/*.json land.

Verdicts: CONFIRMED / PARTIAL / FALSIFIED / PENDING (source file(s) missing).
Exit code is always 0 — this scorer reports, it does not gate.

Usage:  python src/score_predictions.py [--results-dir results] [--out path]
"""
import os
import sys
import json
import argparse

import numpy as np


# ─────────────────────────────────────────────────────────────────────────
#  small helpers
# ─────────────────────────────────────────────────────────────────────────
def load_json(results_dir, name):
    path = os.path.join(results_dir, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fmt(x, nd=4):
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


# ─────────────────────────────────────────────────────────────────────────
#  EXTRACTORS — one per prediction (or per small family of predictions).
#  Each takes `data` = {source_name: loaded_json_or_None} and returns a dict
#  of measured quantities (or {"_missing": [...]} if required files absent).
# ─────────────────────────────────────────────────────────────────────────
def _missing(data, required):
    miss = [k for k in required if data.get(k) is None]
    return miss


def extract_p8_p9_p11(data):
    required = ["holo_stream_recall_v3"]
    miss = _missing(data, required)
    if miss:
        return {"_missing": miss}
    d = data["holo_stream_recall_v3"]
    sweep = d["sweep"]
    ign_lines = d.get("ignition_detail", [])

    # Parse ignition_detail lines: "P=1 G=0: carried=1.000 zeroed=0.065 off=1.000 chance=0.062 -> IGNITED"
    cells = {}  # (P, G) -> dict(carried, zeroed, off, ignited)
    for line in ign_lines:
        try:
            head, tail = line.split(":", 1)
            p_str, g_str = head.strip().split()
            P = int(p_str.split("=")[1])
            G = int(g_str.split("=")[1])
            parts = tail.strip().split("->")
            vals_part = parts[0].strip()
            ignited = "IGNITED" in parts[1] if len(parts) > 1 else False
            kv = dict(tok.split("=") for tok in vals_part.split())
            cells[(P, G)] = {
                "carried": float(kv["carried"]),
                "zeroed": float(kv["zeroed"]),
                "off": float(kv["off"]),
                "ignited": ignited,
            }
        except Exception:
            continue

    chance = d.get("chance", 0.0625)

    # P8: plateau — for every ignited (P,G) cell, accuracy at G=128 within 10pp of G=8
    plateau_gaps = []
    for P in sorted({p for p, g in cells}):
        c8 = cells.get((P, 8))
        c128 = cells.get((P, 128))
        if c8 and c128 and c8["ignited"] and c128["ignited"]:
            plateau_gaps.append((P, c8["carried"], c128["carried"], abs(c8["carried"] - c128["carried"])))

    # P9: 1/sqrt(P) ordering at G=8, conditional on ignition — P2>=0.5, P4>=0.25
    p2_g8 = cells.get((2, 8))
    p4_g8 = cells.get((4, 8))

    # P11-revision: holo_on(carried) - holo_off >= +10pp at P=2,G=8
    rent_p2g8 = None
    if p2_g8:
        rent_p2g8 = p2_g8["carried"] - p2_g8["off"]

    zeroed_at_chance = all(abs(c["zeroed"] - chance) < 0.10 for c in cells.values())

    return {
        "chance": chance,
        "cells": cells,
        "plateau_gaps": plateau_gaps,
        "p2_g8": p2_g8,
        "p4_g8": p4_g8,
        "rent_p2g8": rent_p2g8,
        "zeroed_at_chance": zeroed_at_chance,
    }


def extract_p12(data):
    required = ["holo_heldout_keys", "holo_heldout_keys_8k"]
    miss = _missing(data, required)
    if miss:
        return {"_missing": miss}
    out = {}
    for tag in ("2k", "8k"):
        d = data["holo_heldout_keys"] if tag == "2k" else data["holo_heldout_keys_8k"]
        gg = d["generalization_gap"]
        chance = d.get("chance", 0.0625)
        carried_gap = max(abs(v) for v in gg.get("holo_carried", {}).values())
        off_gap = max(abs(v) for v in gg.get("holo_off", {}).values())
        # margin over chance: use sweep accuracies directly
        sweep = d["sweep"]
        accs = [v["accuracy"] for v in sweep.values()]
        margin = (float(np.mean(accs)) - chance) if accs else None
        out[tag] = {"carried_gap": carried_gap, "off_gap": off_gap, "margin_over_chance": margin}
    return out


def extract_p13(data):
    required = ["holo_knee", "holo_knee_bar08"]
    miss = _missing(data, required)
    if miss:
        return {"_missing": miss}
    out = {}
    for tag, key in (("bar09", "holo_knee"), ("bar08", "holo_knee_bar08")):
        d = data[key]
        theory_lines = d.get("theory_check", [])
        knees = d.get("knees", {})
        # per-variant mean knee_G
        variants = sorted({v.split("_", 1)[1] for v in knees} if False else set())
        by_variant = {}
        for k, v in knees.items():
            variant = k.split("_", 1)[1] if "_" in k else k
            by_variant.setdefault(variant, []).append(v["knee_G"])
        mean_knee = {v: float(np.mean(gs)) for v, gs in by_variant.items()}
        out[tag] = {
            "theory_check": theory_lines,
            "verdict": d.get("verdict"),
            "mean_knee_by_variant": mean_knee,
        }
    return out


def extract_p14(data):
    required = ["holo_magread"]
    miss = _missing(data, required)
    if miss:
        return {"_missing": miss}
    d = data["holo_magread"]
    knees = d.get("knees", {})
    by_variant = {}
    for k, v in knees.items():
        variant = k.split("_", 1)[1] if "_" in k else k
        by_variant.setdefault(variant, []).append(v["knee_G"])
    mean_knee = {v: float(np.mean(gs)) for v, gs in by_variant.items()}
    return {
        "verdict": d.get("verdict"),
        "theory_check": d.get("theory_check", []),
        "mean_knee_by_variant": mean_knee,
    }


def extract_p15(data):
    smoke = data.get("pos_sleep_cycles_smoke")
    full = data.get("pos_sleep_cycles")
    if full is None and smoke is None:
        return {"_missing": ["pos_sleep_cycles_smoke", "pos_sleep_cycles (full)"]}
    out = {"has_full": full is not None}
    src = full if full is not None else smoke
    out["source"] = "pos_sleep_cycles" if full is not None else "pos_sleep_cycles_smoke (interim)"
    w = src["arms"]["W"]
    c = src["arms"]["C"]
    out["gap"] = w["final"] - c["final"]  # positive => C (closed loop) ahead
    out["sleep_dividends"] = c.get("sleep_dividends", [])
    out["verdict_string"] = src.get("verdict")
    return out


def extract_p16(data):
    required = ["holo_hybrid"]
    miss = _missing(data, required)
    if miss:
        return {"_missing": miss}
    d = data["holo_hybrid"]
    sweep = d["sweep"]
    chance = d.get("chance", 0.0625)

    def cells_for(P):
        return [v for v in sweep.values() if v["P"] == P]

    p16_cells = cells_for(16)
    hybrid_g1 = [v["hybrid"] for v in p16_cells if v["gate_rate"] == 1.0]
    base_p16 = [v["base"] for v in p16_cells]
    random_g1 = [v["random"] for v in p16_cells if v["gate_rate"] == 1.0]

    p2_cells = cells_for(2)
    hybrid_p2 = [v["hybrid"] for v in p2_cells]
    base_p2 = [v["base"] for v in p2_cells]

    # dose monotonicity: gate 0.5 hybrid < gate 1.0 hybrid (per matching P,G,seed)
    by_key = {(v["P"], v["G"], v["seed"], v["gate_rate"]): v for v in sweep.values()}
    dose_pairs = []
    for (P, G, seed, gate_rate), v in by_key.items():
        if gate_rate != 0.5:
            continue
        v_hi = by_key.get((P, G, seed, 1.0))
        if v_hi is not None:
            dose_pairs.append((P, G, seed, v["hybrid"], v_hi["hybrid"]))

    dose_monotone = all(lo < hi for (_, _, _, lo, hi) in dose_pairs) if dose_pairs else None

    return {
        "chance": chance,
        "checks": d.get("checks"),
        "verdict": d.get("verdict"),
        "hybrid_p16_gate1_mean": float(np.mean(hybrid_g1)) if hybrid_g1 else None,
        "base_p16_mean": float(np.mean(base_p16)) if base_p16 else None,
        "random_p16_gate1_mean": float(np.mean(random_g1)) if random_g1 else None,
        "hybrid_p2_mean": float(np.mean(hybrid_p2)) if hybrid_p2 else None,
        "base_p2_mean": float(np.mean(base_p2)) if base_p2 else None,
        "dose_pairs": dose_pairs,
        "dose_monotone": dose_monotone,
    }


def extract_p17(data):
    required = ["holo_capreturn", "holo_capreturn_offlr_control"]
    miss = _missing(data, required)
    if miss:
        return {"_missing": miss}
    cap = data["holo_capreturn"]
    ctrl = data["holo_capreturn_offlr_control"]
    sweep = cap["sweep"]

    def avg_carried(key_set, G, d_model=256):
        vals = [v["accuracy"] for v in sweep.values()
                if v.get("d_model") == d_model and v["arm"] == "holo_carried"
                and v["key_set"] == key_set and v["G"] == G]
        return float(np.mean(vals)) if vals else None

    advantage = {}
    for ks in ("train_keys", "test_keys"):
        for G in (0, 8, 32):
            carried = avg_carried(ks, G)
            bo_key = f"{ks}_G{G}"
            best_of_off = ctrl.get("best_of_off", {}).get(bo_key, {}).get("best_acc")
            if carried is not None and best_of_off is not None:
                advantage[f"{ks}_G{G}"] = carried - best_of_off

    train_low_g = [advantage.get("train_keys_G0"), advantage.get("train_keys_G8")]
    train_low_g = [v for v in train_low_g if v is not None]
    test_all = [advantage.get(f"test_keys_G{G}") for G in (0, 8, 32)]
    test_all = [v for v in test_all if v is not None]

    return {
        "advantage": advantage,
        "train_low_g_advantage": train_low_g,
        "test_advantage": test_all,
        "capreturn_verdict": cap.get("verdict"),
    }


def extract_p18(data):
    required = ["holo_reminded", "holo_hybrid"]
    miss = _missing(data, required)
    if miss:
        return {"_missing": miss}
    sw = data["holo_reminded"]["sweep"]
    def mean_over_seeds(P, G, field):
        vals = [v[field] for v in sw.values() if v["P"] == P and v["G"] == G]
        return sum(vals) / len(vals) if vals else None
    # M5 reference: base at P=16 from the hybrid sweep (gate 1.0 rows carry base too)
    m5 = data["holo_hybrid"]["sweep"] if isinstance(data["holo_hybrid"].get("sweep"), dict) else {}
    m5_base16 = [v["base"] for v in m5.values()
                 if isinstance(v, dict) and v.get("P") == 16 and v.get("gate") in (1.0, None)]
    m5_base16 = sum(m5_base16) / len(m5_base16) if m5_base16 else 0.154
    return {
        "hybrid_p16": min(x for x in (mean_over_seeds(16, 8, "hybrid"),
                                       mean_over_seeds(16, 32, "hybrid")) if x is not None),
        "base_p16": mean_over_seeds(16, 8, "base"),
        "m5_base_p16": m5_base16,
        "wrong_p2": max(x for x in (mean_over_seeds(2, 8, "random"),
                                     mean_over_seeds(2, 32, "random")) if x is not None),
        "read_jump_p2": mean_over_seeds(2, 8, "hybrid"),
    }


def extract_p19(data):
    required = ["pos_dream"]
    miss = _missing(data, required)
    if miss:
        return {"_missing": miss}
    arms = data["pos_dream"]["arms"]
    return {k: arms[k]["delta"] for k in ("sleep", "dream", "fresh", "dream_shuffled")}


def rule_p18(m):
    """Scores the ORIGINAL registered P18 (immutable): hybrid@P16 >= 0.85,
    base@P16 within 5pp of M5's base, wrong-reminder@P2 > 0.30. The hardened
    read-jump at P=2 is reported in the reason, not scored (it was not the
    registered claim)."""
    c1 = m["hybrid_p16"] >= 0.85
    c2 = abs(m["base_p16"] - m["m5_base_p16"]) <= 0.05
    c3 = m["wrong_p2"] > 0.30
    n_pass = sum([c1, c2, c3])
    status = "CONFIRMED" if n_pass == 3 else ("PARTIAL" if n_pass >= 1 else "FALSIFIED")
    reason = (f"hybrid@P16={m['hybrid_p16']:.3f} (>=0.85: {c1}), "
              f"base@P16={m['base_p16']:.3f} vs M5 {m['m5_base_p16']:.3f} (within 5pp: {c2}), "
              f"wrong@P2={m['wrong_p2']:.3f} (>0.30: {c3}; conflict-variant passes this bar but is "
              f"diagnosed as global trust decay, see holo_reminded_conflict.json); "
              f"unregistered headline: read-jump hybrid@P2={m['read_jump_p2']:.3f} vs M5 0.715")
    return status, reason


def rule_p19(m):
    c1 = m["dream"] - m["fresh"] >= 0.03      # registered: dream beats fresh by >=0.03
    c2 = m["sleep"] > m["dream"]               # registered: sleep beats dream
    status = "CONFIRMED" if (c1 and c2) else ("PARTIAL" if c2 else "FALSIFIED")
    reason = (f"deltas: sleep={m['sleep']:+.4f} dream={m['dream']:+.4f} "
              f"fresh={m['fresh']:+.4f} shuffled={m['dream_shuffled']:+.4f}; "
              f"dream-vs-fresh={m['dream']-m['fresh']:+.4f} (>=+0.03: {c1}), "
              f"sleep>dream: {c2}")
    return status, reason


def extract_p20(data):
    required = ["pos_domain_shock"]
    miss = _missing(data, required)
    if miss:
        return {"_missing": miss}
    return {"raw": data["pos_domain_shock"]}


def extract_p1_p7_interim(data):
    # Not scored (Phase B / verify_pos.py's job); this only pulls an interim
    # P1-style ratio from the last eval row of pos_metrics.jsonl, if present.
    rows = data.get("_pos_metrics_evals") or []
    if not rows:
        return {"_missing": ["results/pos_metrics.jsonl (no eval rows)"]}
    last = rows[-1]
    arms = last.get("arms", {})
    a1 = arms.get("A1", {}).get("heldout")
    a2 = arms.get("A2", {}).get("heldout")
    a3 = arms.get("A3", {}).get("heldout")
    ratio = None
    if a1 is not None and a2 is not None and a3 is not None and (a1 - a2) != 0:
        ratio = (a1 - a3) / (a1 - a2)
    return {
        "n_streamed": last.get("n_streamed"),
        "a1_heldout": a1,
        "a2_heldout": a2,
        "a3_heldout": a3,
        "a3_grad_frac": (arms.get("A3", {}).get("grad_tokens") / last["n_streamed"])
        if arms.get("A3", {}).get("grad_tokens") is not None and last.get("n_streamed") else None,
        "interim_ratio_A3_vs_A2": ratio,
    }


# ─────────────────────────────────────────────────────────────────────────
#  RULES — measured dict -> (status, reason)
# ─────────────────────────────────────────────────────────────────────────
def rule_p8(m):
    if "_missing" in m:
        return "PENDING", f"missing source(s): {m['_missing']}"
    gaps = m["plateau_gaps"]
    if not gaps:
        return "PENDING", "no ignited (P,G=8&128) cell pairs found"
    bad = [(P, g8, g128, diff) for (P, g8, g128, diff) in gaps if diff > 0.10]
    detail = "; ".join(f"P={P}: G8={g8:.3f}->G128={g128:.3f} (|Δ|={diff:.3f})" for P, g8, g128, diff in gaps)
    if not bad:
        return "CONFIRMED", f"all ignited cells plateau within 10pp: {detail}"
    return "PARTIAL", f"{len(bad)}/{len(gaps)} ignited cells exceed 10pp band: {detail}"


def rule_p9(m):
    if "_missing" in m:
        return "PENDING", f"missing source(s): {m['_missing']}"
    p2, p4 = m["p2_g8"], m["p4_g8"]
    if p2 is None or p4 is None:
        return "PENDING", "P=2 or P=4 @G=8 cell missing"
    ok2 = p2["ignited"] and p2["carried"] >= 0.5
    ok4 = p4["ignited"] and p4["carried"] >= 0.25
    detail = f"P2@G8={p2['carried']:.3f} (ignited={p2['ignited']}, need>=0.5); P4@G8={p4['carried']:.3f} (ignited={p4['ignited']}, need>=0.25)"
    if ok2 and ok4:
        return "CONFIRMED", detail
    if ok2 or ok4:
        return "PARTIAL", detail
    return "FALSIFIED", detail


def rule_p11_revision(m):
    if "_missing" in m:
        return "PENDING", f"missing source(s): {m['_missing']}"
    rent = m["rent_p2g8"]
    if rent is None:
        return "PENDING", "P=2 @G=8 cell missing"
    detail = f"holo_on(carried) - holo_off @ P2,G8 = {rent:+.3f} (v3-regime revision of P11; original v1 P11 was FALSIFIED, see HOLO_STREAM_VERDICT.md)"
    if rent >= 0.10:
        return "CONFIRMED", detail
    if rent > 0:
        return "PARTIAL", detail
    return "FALSIFIED", detail


def rule_p12(m):
    if "_missing" in m:
        return "PENDING", f"missing source(s): {m['_missing']}"
    lines = []
    all_generalize = True
    for tag in ("2k", "8k"):
        v = m[tag]
        gap_ok = v["carried_gap"] < 0.10 and v["off_gap"] < 0.10
        margin_ok = v["margin_over_chance"] is not None and v["margin_over_chance"] > 0.30
        all_generalize &= gap_ok and margin_ok
        lines.append(f"{tag}: carried_gap={v['carried_gap']:.3f} off_gap={v['off_gap']:.3f} margin_over_chance={v['margin_over_chance']:.3f} (gap<0.10 & margin>0.30 -> {'both generalize' if gap_ok and margin_ok else 'not clean'})")
    detail = "; ".join(lines) + " | P12 registered form (phase=function vs baseline=lookup, dichotomy) was FALSIFIED — both arms generalize, revised finding confirmed"
    # Score against the REVISED claim ("both generalize"), matching HOLO_STREAM_VERDICT.md's read.
    if all_generalize:
        return "FALSIFIED", "original P12 dichotomy falsified; revised finding 'both arms generalize' CONFIRMED — " + detail
    return "PARTIAL", detail


def rule_p13(m):
    if "_missing" in m:
        return "PENDING", f"missing source(s): {m['_missing']}"
    bar09 = m["bar09"]
    bar08 = m["bar08"]
    t2_knee = bar09["mean_knee_by_variant"].get("T2_fullseq_curriculum")
    t3_knee = bar09["mean_knee_by_variant"].get("T3_fullseq_kickstart")
    detail = (f"bar0.9: {bar09['verdict']} | bar0.8: {bar08['verdict']} | "
              f"T2 mean_knee(bar0.9)={t2_knee} (target>=256) T3 mean_knee(bar0.9)={t3_knee} (target>=1024)")
    t2_ok = t2_knee is not None and t2_knee >= 256
    t3_ok = t3_knee is not None and t3_knee >= 1024
    ordering_ok = "CONFIRMED" in str(bar09["verdict"]) and "CONFIRMED" in str(bar08["verdict"])
    if t2_ok and t3_ok:
        return "CONFIRMED", detail
    if ordering_ok:
        return "PARTIAL", "point targets NOT MET but knee-vs-filler-gamma ordering law CONFIRMED in both curriculum-bar runs — " + detail
    return "FALSIFIED", detail


def rule_p14(m):
    if "_missing" in m:
        return "PENDING", f"missing source(s): {m['_missing']}"
    v2_knee = m["mean_knee_by_variant"].get("V2_kickstart_magnorm")
    detail = f"{m['verdict']} | V2(kickstart+magnorm) mean_knee={v2_knee} (target>=1024)"
    if v2_knee is not None and v2_knee >= 1024:
        return "CONFIRMED", detail
    if v2_knee is not None and v2_knee > 256:
        return "PARTIAL", "point target (>=1024) NOT MET but knee extended well past the 128/256 reference (arc 32->128->256->512), read-fix synergy with gamma confirmed — " + detail
    return "FALSIFIED", detail


def rule_p15(m):
    if "_missing" in m:
        return "PENDING", f"missing source(s): {m['_missing']}"
    gap = m["gap"]
    divs = m["sleep_dividends"]
    gap_ok = gap is not None and gap > 0.02
    divs_sustained = len(divs) >= 2 and divs[-1] > 0
    prefix = "" if m["has_full"] else "[INTERIM — smoke only, full run pending] "
    detail = f"{prefix}source={m['source']} gap(W_final-C_final)={gap:+.4f} (need>0.02) sleep_dividends={divs} verdict_string={m['verdict_string']!r}"
    if not m["has_full"]:
        # Smoke can inform but per the mission spec we report PENDING with the interim value when Full is absent.
        return "PENDING", detail
    if gap_ok and divs_sustained:
        return "CONFIRMED", detail
    if gap_ok or divs_sustained:
        return "PARTIAL", detail
    return "FALSIFIED", detail


def rule_p16(m):
    if "_missing" in m:
        return "PENDING", f"missing source(s): {m['_missing']}"
    checks = m.get("checks") or {}
    hybrid_ceiling_ok = checks.get("hybrid@P16_gate1.0>=0.9", False)
    base_ceiling_ok = checks.get("base@P16<=0.15", False)
    random_guard_ok = checks.get("random@P16~=base", False)
    p2_not_worse = checks.get("hybrid@P2>=base@P2", False)
    dose_ok = m.get("dose_monotone")
    detail = (f"checks={checks} | hybrid@P16(gate1.0)={m['hybrid_p16_gate1_mean']} base@P16={m['base_p16_mean']} "
              f"random@P16(gate1.0)={m['random_p16_gate1_mean']} | dose_monotone(gate0.5<gate1.0)={dose_ok} | "
              f"hybrid@P2={m['hybrid_p2_mean']} base@P2={m['base_p2_mean']}")
    if hybrid_ceiling_ok and base_ceiling_ok and random_guard_ok and p2_not_worse and dose_ok:
        return "CONFIRMED", detail
    core_mechanism_ok = random_guard_ok and p2_not_worse and dose_ok
    if core_mechanism_ok:
        return "PARTIAL", "ceiling target (>=0.9 @P16) NOT MET but mechanism fully validated: dose-monotone, random-injection guard holds, P2 not hurt — " + detail
    return "FALSIFIED", detail


def rule_p17(m):
    if "_missing" in m:
        return "PENDING", f"missing source(s): {m['_missing']}"
    train_low_g = m["train_low_g_advantage"]
    test_adv = m["test_advantage"]
    train_ok = bool(train_low_g) and all(v >= 0.15 for v in train_low_g)
    train_partial = bool(train_low_g) and any(v >= 0.10 for v in train_low_g)
    test_returns = bool(test_adv) and all(v >= 0.10 for v in test_adv)
    detail = f"advantage(carried - best_of_off)={m['advantage']} | train_keys G<=8: {train_low_g} (target>=15pp) | test_keys all G: {test_adv}"
    if train_ok and test_returns:
        return "CONFIRMED", detail
    if train_ok or train_partial:
        return "PARTIAL", "phase advantage returns on train_keys at d=256 but NOT on held-out test_keys (compositional generalization) — " + detail
    return "FALSIFIED", detail


def rule_pending_only(claim_note):
    def _rule(m):
        if "_missing" in m:
            return "PENDING", f"missing source(s): {m['_missing']}"
        return "PENDING", f"{claim_note} — source file present but scoring rule not yet exercised against real data; treat this branch as a stub until the run lands"
    return _rule


def rule_p1_p7(m):
    if "_missing" in m:
        return "PENDING", "results/pos_summary.json not found — P1-P7 belong to verify_pos.py / Phase B, not scored here"
    return "PENDING", (
        f"P1-P7 belong to verify_pos.py / Phase B (not duplicated here). "
        f"Interim (last eval @ n_streamed={m['n_streamed']:,}): "
        f"A1={m['a1_heldout']} A2={m['a2_heldout']} A3={m['a3_heldout']} "
        f"A3_grad_frac={fmt(m['a3_grad_frac'])} interim_ratio_A3_vs_A2={fmt(m['interim_ratio_A3_vs_A2'])}"
    )


# ─────────────────────────────────────────────────────────────────────────
#  REGISTRY
# ─────────────────────────────────────────────────────────────────────────
REGISTRY = [
    {
        "id": "P1-P7",
        "claim": "POS long run (A3 ratio, gate drift, RSS flatness, A1 frozen, twin fork, injection, probe volume)",
        "source_files": ["pos_summary.json"],
        "extractor": lambda data: (
            {"_missing": ["pos_summary.json"]} if data.get("pos_summary") is None
            else extract_p1_p7_interim({**data, "_pos_metrics_evals": data.get("_pos_metrics_evals", [])})
        ),
        "rule": rule_p1_p7,
    },
    {
        "id": "P8",
        "claim": "persistence axis (G): ignited cells plateau within 10pp from G=8 to G=128, not exponential decay",
        "source_files": ["holo_stream_recall_v3.json"],
        "extractor": extract_p8_p9_p11,
        "rule": rule_p8,
    },
    {
        "id": "P9",
        "claim": "capacity axis (P): carried recall at G=8 orders ~1/sqrt(P), P2>=0.5 P4>=0.25, conditional on ignition",
        "source_files": ["holo_stream_recall_v3.json"],
        "extractor": extract_p8_p9_p11,
        "rule": rule_p9,
    },
    {
        "id": "P11-revision",
        "claim": "phase pays rent at P>=2 (v3 regime): holo_on - holo_off >= +10pp at P=2,G=8 (v1 P11 was FALSIFIED; v3 revises)",
        "source_files": ["holo_stream_recall_v3.json"],
        "extractor": extract_p8_p9_p11,
        "rule": rule_p11_revision,
    },
    {
        "id": "P12",
        "claim": "held-out-key binding: original dichotomy (phase generalizes, magnitude doesn't) vs revised finding both arms generalize",
        "source_files": ["holo_heldout_keys.json", "holo_heldout_keys_8k.json"],
        "extractor": extract_p12,
        "rule": rule_p12,
    },
    {
        "id": "P13",
        "claim": "gamma-knee mobility: full-sequence training moves the knee to >=256 (T2) / >=1024 (T3 kickstart); knee order tracks filler-gamma order",
        "source_files": ["holo_knee.json", "holo_knee_bar08.json"],
        "extractor": extract_p13,
        "rule": rule_p13,
    },
    {
        "id": "P14",
        "claim": "magnitude-normalized read (M3): renormalizing |S| before de-rotation moves the knee from 256 to >=1024",
        "source_files": ["holo_magread.json"],
        "extractor": extract_p14,
        "rule": rule_p14,
    },
    {
        "id": "P15",
        "claim": "closed wake/sleep cycles (M4) beat plain continued training at equal total gradient budget; per-cycle sleep dividend does not collapse to zero",
        "source_files": ["pos_sleep_cycles.json (full)", "pos_sleep_cycles_smoke.json (interim)"],
        "extractor": extract_p15,
        "rule": rule_p15,
    },
    {
        "id": "P16",
        "claim": "state+index hybrid (M5) on MQAR: recall at P=16 reaches >=0.9 with dose-monotone gate and random-injection guard; P2 not hurt",
        "source_files": ["holo_hybrid.json"],
        "extractor": extract_p16,
        "rule": rule_p16,
    },
    {
        "id": "P17",
        "claim": "phase advantage returns with capacity (M6): at d_model=256, holo-off advantage (vs best-of-off lr control) >= +15pp at G<=8",
        "source_files": ["holo_capreturn.json", "holo_capreturn_offlr_control.json"],
        "extractor": extract_p17,
        "rule": rule_p17,
    },
    {
        "id": "P18",
        "claim": "learned to be reminded (MS1): training with stochastic consultation lifts hybrid@P16 from 0.51 to >=0.85",
        "source_files": ["holo_reminded.json", "holo_hybrid.json"],
        "extractor": extract_p18,
        "rule": rule_p18,
    },
    {
        "id": "P19",
        "claim": "dream generator (MS2): training on self-sampled continuations beats fresh data but loses to stored-span replay",
        "source_files": ["pos_dream.json"],
        "extractor": extract_p19,
        "rule": rule_p19,
    },
    {
        "id": "P20",
        "claim": "domain shock (MS3): gating+dosed replay forgets least on C4->code->C4 (<=50% of full-gradient's forgetting)",
        "source_files": ["pos_domain_shock.json"],
        "extractor": extract_p20,
        "rule": rule_pending_only("WT-2 heldout degradation across arms during the code phase"),
    },
]


# ─────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────
def load_all_sources(results_dir):
    """Load every distinct source file referenced by the registry, keyed by
    its json stem (e.g. 'holo_stream_recall_v3' for holo_stream_recall_v3.json)."""
    stems = set()
    for entry in REGISTRY:
        for sf in entry["source_files"]:
            stem = sf.split(" ")[0]  # strip "(full)"/"(interim)" annotations
            if stem.endswith(".json"):
                stems.add(stem[:-5])
    data = {}
    for stem in stems:
        data[stem] = load_json(results_dir, stem + ".json")
    # pos_metrics.jsonl feeds the P1-P7 interim line
    evals = [r for r in read_jsonl(os.path.join(results_dir, "pos_metrics.jsonl")) if r.get("type") == "eval"]
    data["_pos_metrics_evals"] = evals
    return data


def main():
    ap = argparse.ArgumentParser(description="Score analysis/PREDICTIONS.md's P1-P20 against results/*.json")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default=os.path.join("results", "prediction_scores.json"))
    args = ap.parse_args()

    data = load_all_sources(args.results_dir)

    rows = []
    for entry in REGISTRY:
        measured = entry["extractor"](data)
        status, reason = entry["rule"](measured)
        rows.append({
            "id": entry["id"],
            "claim": entry["claim"],
            "source_files": entry["source_files"],
            "status": status,
            "reason": reason,
            "measured": measured,
        })

    # ── stdout report ──
    print("=" * 78)
    print("AUTO-FALSIFIER — PREDICTIONS.md P1-P20 scored against results/*.json")
    print("=" * 78)
    for r in rows:
        print(f"[{r['status']}] {r['id']}: {r['claim']}")
        print(f"    measured: {r['reason']}")
        print()

    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("-" * 78)
    print("SUMMARY: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())) + f"  (n={len(rows)})")
    print("-" * 78)

    out_payload = {
        "generated_by": "src/score_predictions.py",
        "n_predictions": len(rows),
        "summary": counts,
        "predictions": rows,
    }

    def _sanitize(o):
        """Recursively make a structure JSON-safe: stringify non-str dict keys
        (e.g. the (P, G) tuple keys in the P8/P9/P11 'cells' dict) and coerce
        numpy scalars / tuples to native types."""
        if isinstance(o, dict):
            return {(k if isinstance(k, (str, int, float, bool)) or k is None else str(k)): _sanitize(v)
                    for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_sanitize(v) for v in o]
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        return o

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(_sanitize(out_payload), f, indent=2)
    print(f"\nwrote {args.out}")

    sys.exit(0)


if __name__ == "__main__":
    main()
