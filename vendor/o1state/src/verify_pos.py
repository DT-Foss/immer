#!/usr/bin/env python3
"""
Self-verify a POS run from its raw logs + pos_analyze.py's summary JSON.
==========================================================================
Two sections, same split as verify_billion.py:
  INTEGRITY   hard PASS/FAIL — re-derives every summary number from the raw
              metrics/chunks/probes files and checks the run's own invariants
              (A1 never learns, A2 is full-gradient, A3's grad_tokens is
              exactly reconstructible from its gate trace). Any FAIL -> exit 1.
  HEADLINES   measured numbers printed for context (INFO only, never FAIL) —
              these are what the run is FOR, not what makes it valid.

Usage:  python src/verify_pos.py [--tag T] [--results-dir results] [--summary path]
"""
import os, sys, json, argparse

import numpy as np

# reference bands for the headline section only — informational, not gates
REF_RATIO_A3_A2   = 0.75    # target: A3 captures >= 75% of A2's improvement
REF_GRAD_FRAC     = (0.15, 0.30)   # target band for A3 grad-token fraction
REF_RSS_BAND_GB   = 0.10


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


def read_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def rolling_mean(xs, window):
    out = np.empty(len(xs))
    c = np.cumsum(np.insert(np.asarray(xs, dtype=np.float64), 0, 0.0))
    for i in range(len(xs)):
        lo = max(0, i + 1 - window)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def evals_of(metrics):
    return [m for m in metrics if m.get("type") == "eval"]


def fork_event(metrics):
    forks = [m for m in metrics if m.get("type") == "event" and m.get("event") == "twin_fork"]
    return forks[0] if forks else None, forks


# ───────────────────────────────────────────────────────────────────────────
#  INTEGRITY checks — each returns (name, ok, detail)
# ───────────────────────────────────────────────────────────────────────────
def check_metrics_monotonic(metrics, chunks, config):
    checks = []
    evals = evals_of(metrics)
    ok = len(evals) >= 2
    checks.append(("at least 2 eval records", ok, f"{len(evals)} evals"))

    ns = [e["n_streamed"] for e in evals]
    ok = all(ns[i] <= ns[i + 1] for i in range(len(ns) - 1))
    checks.append(("eval n_streamed monotone non-decreasing", ok, f"{ns[:3]}...{ns[-3:] if len(ns)>3 else ''}"))

    step = config.get("batch", 8) * config.get("chunk", 64)
    cn = [c["n"] for c in chunks]
    ok = all(cn[i] < cn[i + 1] for i in range(len(cn) - 1))
    checks.append(("chunk n strictly increasing", ok, f"{len(cn)} chunks"))
    ok2 = all(cn[i + 1] - cn[i] == step for i in range(len(cn) - 1))
    checks.append((f"chunk step size == batch*chunk ({step})", ok2,
                   f"observed steps {'constant' if ok2 else 'VARY'}"))
    return checks


def check_a1_a2_invariants(metrics, config):
    checks = []
    evals = evals_of(metrics)
    step = config.get("batch", 8) * config.get("chunk", 64)

    bad_a1 = [e["n_streamed"] for e in evals if "A1" in e.get("arms", {}) and e["arms"]["A1"]["grad_tokens"] != 0]
    checks.append(("A1.grad_tokens == 0 in every eval", len(bad_a1) == 0,
                   "OK" if not bad_a1 else f"violations at n={bad_a1}"))

    bad_a2 = []
    for e in evals:
        if "A2" not in e.get("arms", {}):
            continue
        gt, n = e["arms"]["A2"]["grad_tokens"], e["n_streamed"]
        if abs(gt - n) > step:
            bad_a2.append((n, gt))
    checks.append(("A2.grad_tokens == n_streamed (full-gradient, ±batch*chunk)", len(bad_a2) == 0,
                   "OK" if not bad_a2 else f"violations {bad_a2}"))
    return checks


def reconstruct_grad_tokens(chunks, config, gate_key, n_key, upto_n):
    """Sum of batch*chunk over rows with gate_key==1 and n_key <= upto_n."""
    step = config.get("batch", 8) * config.get("chunk", 64)
    return step * sum(1 for c in chunks if gate_key in c and c[gate_key] == 1 and c[n_key] <= upto_n)


def check_a3_consistency(metrics, chunks, config):
    checks = []
    evals = evals_of(metrics)
    mismatches = []
    for e in evals:
        if "A3" not in e.get("arms", {}):
            continue
        expect = reconstruct_grad_tokens(chunks, config, "g", "n", e["n_streamed"])
        got = e["arms"]["A3"]["grad_tokens"]
        if expect != got:
            mismatches.append((e["n_streamed"], got, expect))
    checks.append(("A3.grad_tokens reconstructs exactly from chunk gate trace", len(mismatches) == 0,
                   "OK" if not mismatches else f"mismatches (n, got, expect)={mismatches}"))

    mismatches_r = []
    for e in evals:
        if "A3R" not in e.get("arms", {}):
            continue
        expect = reconstruct_grad_tokens(chunks, config, "gr", "n", e["n_streamed"])
        got = e["arms"]["A3R"]["grad_tokens"]
        if expect != got:
            mismatches_r.append((e["n_streamed"], got, expect))
    if any("A3R" in e.get("arms", {}) for e in evals):
        checks.append(("A3R.grad_tokens reconstructs exactly from chunk gate trace (post-fork)",
                       len(mismatches_r) == 0,
                       "OK" if not mismatches_r else f"mismatches={mismatches_r}"))
    return checks


def rebuild_summary(metrics, chunks, probes, status):
    """Same derivation pos_analyze.py uses — kept in lockstep so check 4 is a real
    independent re-derivation from raw logs, not a copy of the summary file."""
    evals = evals_of(metrics)
    config = status.get("config", {})

    def arm_summary(name):
        rows = [e for e in evals if name in e.get("arms", {})]
        if not rows:
            return None
        first = next((e for e in rows if e["n_streamed"] == 0), rows[0])
        last = rows[-1]
        hf, hl = first["arms"][name]["heldout"], last["arms"][name]["heldout"]
        return {"heldout_first": hf, "heldout_final": hl, "improvement": hf - hl,
                "grad_tokens_final": last["arms"][name]["grad_tokens"]}

    arms = {name: arm_summary(name) for name in ("A1", "A2", "A3", "A3R")}
    last_eval = evals[-1] if evals else None
    n_streamed = last_eval["n_streamed"] if last_eval else status.get("n_streamed", 0)
    wall_s = last_eval["wall_s"] if last_eval else status.get("wall_s", 0.0)

    ratio = None
    if arms["A2"] and arms["A3"] and arms["A2"]["improvement"]:
        ratio = arms["A3"]["improvement"] / arms["A2"]["improvement"]
    a3_frac = None
    if arms["A3"] and n_streamed:
        a3_frac = arms["A3"]["grad_tokens_final"] / n_streamed

    ignition = config.get("ignition_chunks", 100)
    gate_frac = None
    if len(chunks) > ignition:
        gate_frac = float(np.mean([c.get("g", 0) for c in chunks[ignition:]]))

    rss_vals = [e["rss_gb"] for e in evals]
    rss = ({"min": min(rss_vals), "max": max(rss_vals), "span": max(rss_vals) - min(rss_vals)}
          if rss_vals else {"min": None, "max": None, "span": None})

    injection = None
    if probes:
        d_inj = np.array([p["d_inj"] for p in probes])
        d_rand = np.array([p["d_rand"] for p in probes])
        injection = {"n_probes": len(probes), "mean_d_inj": float(d_inj.mean()),
                     "mean_d_rand": float(d_rand.mean()),
                     "frac_inj_helped": float((d_inj > 0).mean()),
                     "frac_rand_helped": float((d_rand > 0).mean()),
                     "mean_paired_diff": float((d_inj - d_rand).mean())}

    fork, _ = fork_event(metrics)
    twin = None
    if fork is not None:
        fork_n, fork_w = fork["n_streamed"], fork["wall_s"]
        post = [c for c in chunks if c.get("n", 0) >= fork_n and "s3r" in c]
        excess_rows = [c for c in post if c["w"] <= fork_w + 7200]
        surprise_excess = (float(np.mean([c["s3r"] - c["s3"] for c in excess_rows]))
                           if excess_rows else None)
        post_gate_a3 = float(np.mean([c.get("g", 0) for c in post])) if post else None
        post_gate_a3r = float(np.mean([c.get("gr", 0) for c in post])) if post else None
        converged_at = None
        if post:
            n = [c["n"] for c in post]
            s3 = [c["s3"] for c in post]
            s3r = [c["s3r"] for c in post]
            diff = np.abs(rolling_mean(s3r, 50) - rolling_mean(s3, 50))
            for i in range(len(diff)):
                if diff[i] < 0.05 and np.all(diff[i:i + 50] < 0.05):
                    converged_at = n[i]
                    break
        twin = {"forked_at_n": fork_n, "forked_at_wall_s": fork_w,
               "surprise_excess_first_2h": surprise_excess,
               "post_fork_gate_frac_A3": post_gate_a3, "post_fork_gate_frac_A3R": post_gate_a3r,
               "converged_at_n": converged_at}

    return {"tag": status.get("config", {}).get("tag", ""), "n_streamed": n_streamed,
            "wall_hours": wall_s / 3600.0, "n_evals": len(evals), "arms": arms,
            "ratio_A3_vs_A2_improvement": ratio, "A3_grad_token_frac": a3_frac,
            "gate_frac_post_ignition": gate_frac, "rss": rss,
            "injection": injection, "twin": twin}


def _num_close(a, b, rel=1e-6, abs_tol=1e-9):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b)))


def _deep_compare(expect, got, path=""):
    """Returns list of mismatch descriptions (empty if equal within tolerance)."""
    diffs = []
    if isinstance(expect, dict) and isinstance(got, dict):
        keys = set(expect) | set(got)
        for k in keys:
            diffs += _deep_compare(expect.get(k), got.get(k), f"{path}.{k}")
    elif isinstance(expect, (int, float)) and isinstance(got, (int, float)) and not isinstance(expect, bool):
        if not _num_close(expect, got):
            diffs.append(f"{path}: expect={expect} got={got}")
    else:
        if expect != got:
            diffs.append(f"{path}: expect={expect!r} got={got!r}")
    return diffs


def check_summary_matches(summary_path, metrics, chunks, probes, status, tag):
    checks = []
    if not os.path.exists(summary_path):
        checks.append(("summary JSON exists", False, f"not found: {summary_path} — run pos_analyze.py first"))
        return checks, None
    got = read_json(summary_path)
    expect = rebuild_summary(metrics, chunks, probes, status)
    expect["tag"] = tag         # pos_analyze.py stamps its own --tag verbatim into the summary
    diffs = _deep_compare(expect, got)
    ok = len(diffs) == 0
    checks.append(("every summary field matches raw-log re-derivation", ok,
                   "OK" if ok else f"{len(diffs)} mismatch(es): {diffs[:5]}"))
    return checks, got


def check_probes(probes, config):
    checks = []
    if not probes:
        return checks
    req = {"key", "n", "w", "dt_h", "L", "s_base", "s_inj", "s_rand", "d_inj", "d_rand"}
    missing = [i for i, p in enumerate(probes) if not req.issubset(p.keys())]
    checks.append(("every probe row has all required fields", len(missing) == 0,
                   "OK" if not missing else f"rows missing fields: {missing[:5]}"))

    bad_inj = [i for i, p in enumerate(probes)
              if abs((p["s_base"] - p["s_inj"]) - p["d_inj"]) > 2e-4]
    checks.append(("d_inj == s_base - s_inj (±2e-4)", len(bad_inj) == 0,
                   "OK" if not bad_inj else f"rows {bad_inj[:5]}"))

    bad_rand = [i for i, p in enumerate(probes)
               if abs((p["s_base"] - p["s_rand"]) - p["d_rand"]) > 2e-4]
    checks.append(("d_rand == s_base - s_rand (±2e-4)", len(bad_rand) == 0,
                   "OK" if not bad_rand else f"rows {bad_rand[:5]}"))

    min_dt = config.get("recur_min_hours", 2.0)
    bad_dt = [i for i, p in enumerate(probes) if p["dt_h"] < min_dt - 1e-9]
    checks.append((f"dt_h >= recur_min_hours ({min_dt})", len(bad_dt) == 0,
                   "OK" if not bad_dt else f"rows {bad_dt[:5]}"))
    return checks


def check_twin(metrics):
    checks = []
    evals = evals_of(metrics)
    has_a3r_eval = any("A3R" in e.get("arms", {}) for e in evals)
    if not has_a3r_eval:
        return checks
    fork, forks = fork_event(metrics)
    checks.append(("exactly one twin_fork event", len(forks) == 1, f"{len(forks)} found"))
    if fork is None:
        return checks
    fork_n = fork["n_streamed"]
    before = [e for e in evals if e["n_streamed"] < fork_n]
    after = [e for e in evals if e["n_streamed"] >= fork_n]
    bad_before = [e["n_streamed"] for e in before if "A3R" in e.get("arms", {})]
    checks.append(("A3R absent from every eval before the fork", len(bad_before) == 0,
                   "OK" if not bad_before else f"present at n={bad_before}"))
    bad_after = [e["n_streamed"] for e in after if "A3R" not in e.get("arms", {})]
    checks.append(("A3R present in every eval at/after the fork", len(bad_after) == 0,
                   "OK" if not bad_after else f"missing at n={bad_after}"))
    return checks


# ───────────────────────────────────────────────────────────────────────────
#  Headlines (INFO only)
# ───────────────────────────────────────────────────────────────────────────
def print_headlines(summary):
    print("\n  MEASURED HEADLINES (informational — reference bands are targets, not gates)")
    r = summary.get("ratio_A3_vs_A2_improvement")
    if r is not None:
        print(f"  [INFO] ratio_A3_vs_A2_improvement = {r:.4f}  (reference target >= {REF_RATIO_A3_A2})")
    f = summary.get("A3_grad_token_frac")
    if f is not None:
        lo, hi = REF_GRAD_FRAC
        print(f"  [INFO] A3_grad_token_frac = {f:.4f}  (reference target <= 0.25, band {lo}-{hi})")
    rss = summary.get("rss") or {}
    if rss.get("span") is not None:
        print(f"  [INFO] RSS span = {rss['span']:.3f} GB  (reference band ±{REF_RSS_BAND_GB} GB, "
              f"min={rss['min']:.3f} max={rss['max']:.3f})")
    inj = summary.get("injection")
    if inj:
        print(f"  [INFO] injection: n_probes={inj['n_probes']}  mean_d_inj={inj['mean_d_inj']:.4f} "
              f"vs mean_d_rand={inj['mean_d_rand']:.4f}  "
              f"(helped: inj={inj['frac_inj_helped']:.2f} rand={inj['frac_rand_helped']:.2f})")
    else:
        print("  [INFO] injection: no probes recorded")
    twin = summary.get("twin")
    if twin:
        print(f"  [INFO] twin: forked at n={twin['forked_at_n']:,} "
              f"surprise_excess_first_2h={twin['surprise_excess_first_2h']}  "
              f"post_fork_gate_frac A3={twin['post_fork_gate_frac_A3']} A3R={twin['post_fork_gate_frac_A3R']}  "
              f"converged_at_n={twin['converged_at_n']}")
    else:
        print("  [INFO] twin: no fork in this run")


# ───────────────────────────────────────────────────────────────────────────
#  Main
# ───────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Verify POS run integrity + print measured headlines")
    ap.add_argument("--tag", default="")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--summary", default="")
    args = ap.parse_args()

    tag = f"{args.tag}_" if args.tag else ""
    R = lambda stem: os.path.join(args.results_dir, f"pos_{tag}{stem}")
    summary_path = args.summary or os.path.join(args.results_dir, f"pos_{tag}summary.json")

    metrics = read_jsonl(R("metrics.jsonl"))
    chunks = read_jsonl(R("chunks.jsonl"))
    probes = read_jsonl(R("probes.jsonl"))
    status = read_json(R("status.json"))
    config = status.get("config", {})

    print(f"verifying POS run: results-dir={args.results_dir} tag={args.tag or '(none)'}")
    print(f"  metrics={len(metrics)} rows  chunks={len(chunks)} rows  probes={len(probes)} rows\n")

    all_checks = []
    all_checks += check_metrics_monotonic(metrics, chunks, config)
    all_checks += check_a1_a2_invariants(metrics, config)
    all_checks += check_a3_consistency(metrics, chunks, config)
    summary_checks, summary = check_summary_matches(summary_path, metrics, chunks, probes, status, args.tag)
    all_checks += summary_checks
    all_checks += check_probes(probes, config)
    all_checks += check_twin(metrics)

    print("  INTEGRITY")
    all_ok = True
    for name, ok, detail in all_checks:
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")

    if summary is not None:
        print_headlines(summary)

    print(f"\n{'ALL INTEGRITY CHECKS PASS' if all_ok else 'SOME CHECKS FAILED'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
