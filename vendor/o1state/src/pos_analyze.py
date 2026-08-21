#!/usr/bin/env python3
"""
POS analysis — Phase B, runs AFTER the long stream against results/pos_<tag_>*.
=================================================================================
Turns the raw event log (metrics/chunks/index/probes/status) into the six plots
and the summary JSON that verify_pos.py then re-derives and checks bit-for-bit.
Tolerant by design: this reads a run that may still be going, may have crashed
early, or may never have had index/probes/twin enabled — every optional piece
degrades to null/placeholder rather than raising.

Usage:  python src/pos_analyze.py [--tag T] [--results-dir results] [--plots-dir plots] [--out path]
"""
import os, sys, json, argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np


# ───────────────────────────────────────────────────────────────────────────
#  Loading — every reader tolerates a missing/empty file (index & probes are
#  optional by spec; metrics/chunks/status are expected but we still degrade).
# ───────────────────────────────────────────────────────────────────────────
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


def fmt_millions(x, _pos=None):
    return f"{x/1e6:.1f}M"


MTICK = matplotlib.ticker.FuncFormatter(fmt_millions)


# ───────────────────────────────────────────────────────────────────────────
#  Derived series
# ───────────────────────────────────────────────────────────────────────────
def evals_of(metrics):
    return [m for m in metrics if m.get("type") == "eval"]


def fork_event(metrics):
    for m in metrics:
        if m.get("type") == "event" and m.get("event") == "twin_fork":
            return m
    return None


def rolling_mean(xs, window):
    """Trailing rolling mean, same length as xs (short prefix uses what's available)."""
    out = np.empty(len(xs))
    c = np.cumsum(np.insert(np.asarray(xs, dtype=np.float64), 0, 0.0))
    for i in range(len(xs)):
        lo = max(0, i + 1 - window)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


# ───────────────────────────────────────────────────────────────────────────
#  Plot 1 — heldout loss vs n_streamed, all arms
# ───────────────────────────────────────────────────────────────────────────
def plot_loss_vs_streamed(evals, fork, path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    names = ["A1", "A2", "A3", "A3R"]
    colors = {"A1": "#888888", "A2": "#1f77b4", "A3": "#d62728", "A3R": "#9467bd"}
    for name in names:
        xs = [e["n_streamed"] for e in evals if name in e.get("arms", {})]
        ys = [e["arms"][name]["heldout"] for e in evals if name in e.get("arms", {})]
        if xs:
            ax.plot(xs, ys, marker="o", ms=3, lw=1.5, label=name, color=colors[name])
    if fork:
        ax.axvline(fork["n_streamed"], color="black", ls="--", lw=1, alpha=0.6, label="twin fork")
    ax.set_xlabel("tokens streamed")
    ax.set_ylabel("heldout loss (WT-2 val, nats/token)")
    ax.set_title("POS — heldout loss vs. tokens streamed")
    ax.xaxis.set_major_formatter(MTICK)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────────
#  Plot 2 — THE core plot: heldout loss vs cumulative grad_tokens per arm
# ───────────────────────────────────────────────────────────────────────────
def plot_loss_vs_grad_tokens(evals, path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, color in (("A2", "#1f77b4"), ("A3", "#d62728"), ("A3R", "#9467bd")):
        xs = [e["arms"][name]["grad_tokens"] for e in evals if name in e.get("arms", {})]
        ys = [e["arms"][name]["heldout"] for e in evals if name in e.get("arms", {})]
        if xs:
            ax.plot(xs, ys, marker="o", ms=3, lw=1.5, label=name, color=color)
    a1_vals = [e["arms"]["A1"]["heldout"] for e in evals if "A1" in e.get("arms", {})]
    if a1_vals:
        ax.axhline(a1_vals[-1], color="#888888", ls="--", lw=1.5,
                   label="A1 (forward-only floor, grad_tokens=0)")
    ax.set_xlabel("cumulative gradient tokens (arm-local)")
    ax.set_ylabel("heldout loss (WT-2 val, nats/token)")
    ax.set_title("POS — heldout loss vs. gradient tokens spent (the core comparison)")
    ax.xaxis.set_major_formatter(MTICK)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────────
#  Plot 3 — RSS timeline
# ───────────────────────────────────────────────────────────────────────────
def plot_rss_timeline(evals, path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = [e["wall_s"] / 3600.0 for e in evals]
    ys = [e["rss_gb"] for e in evals]
    if xs:
        ax.plot(xs, ys, marker="o", ms=3, lw=1.5, color="#2ca02c")
    else:
        ax.text(0.5, 0.5, "no eval records", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("wall-clock (hours)")
    ax.set_ylabel("RSS (GB)")
    ax.set_title("POS — memory timeline (flat RSS is the claim)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────────
#  Plot 4 — surprise timeline (rolling s3, threshold overlay, gated scatter)
# ───────────────────────────────────────────────────────────────────────────
def plot_surprise_timeline(chunks, path, window=50):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if chunks:
        n = np.array([c["n"] for c in chunks])
        s3 = np.array([c["s3"] for c in chunks])
        g = np.array([c.get("g", 0) for c in chunks])
        th = [c.get("th") for c in chunks]
        ax.plot(n, rolling_mean(s3, window), lw=1.5, color="#d62728",
               label=f"A3 s3 (rolling mean, w={window})")
        th_n = [ni for ni, t in zip(n, th) if t is not None]
        th_v = [t for t in th if t is not None]
        if th_v:
            ax.plot(th_n, th_v, lw=1, color="#ff7f0e", alpha=0.7, label="gate threshold")
        gated_n = n[g == 1]
        gated_s3 = s3[g == 1]
        if len(gated_n):
            ax.scatter(gated_n, gated_s3, s=4, alpha=0.15, color="#d62728", label="gated chunks")
        if "s3r" in chunks[-1] or any("s3r" in c for c in chunks):
            fork_chunks = [c for c in chunks if "s3r" in c]
            if fork_chunks:
                fn = np.array([c["n"] for c in fork_chunks])
                fs3r = np.array([c["s3r"] for c in fork_chunks])
                ax.plot(fn, rolling_mean(fs3r, window), lw=1.5, color="#9467bd",
                       label=f"A3R s3r (rolling mean, w={window})")
    else:
        ax.text(0.5, 0.5, "no chunk records", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("tokens streamed")
    ax.set_ylabel("chunk-mean NLL (nats/token)")
    ax.set_title("POS — surprise timeline (A3 gate)")
    ax.xaxis.set_major_formatter(MTICK)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────────
#  Plot 5 — injection probe bars
# ───────────────────────────────────────────────────────────────────────────
def plot_injection_bars(probes, path):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    if not probes:
        ax.text(0.5, 0.5, "no injection probes recorded\n(index warmup not reached, or index off)",
               ha="center", va="center", transform=ax.transAxes)
        ax.set_title("POS — injection probes (d_inj vs d_rand)")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return
    d_inj = np.array([p["d_inj"] for p in probes])
    d_rand = np.array([p["d_rand"] for p in probes])
    means = [d_inj.mean(), d_rand.mean()]
    ax.bar(["d_inj", "d_rand"], means, color=["#d62728", "#888888"], alpha=0.75, zorder=1)
    jx = np.random.RandomState(0).uniform(-0.08, 0.08, size=len(d_inj))
    ax.scatter(0 + jx, d_inj, s=14, color="black", alpha=0.5, zorder=2)
    ax.scatter(1 + jx, d_rand, s=14, color="black", alpha=0.5, zorder=2)
    ax.axhline(0.0, color="black", lw=0.8)
    n_probes = len(probes)
    frac_inj = float((d_inj > 0).mean())
    frac_rand = float((d_rand > 0).mean())
    ax.text(0.02, 0.97, f"n_probes={n_probes}\nfrac(d_inj>0)={frac_inj:.2f}\nfrac(d_rand>0)={frac_rand:.2f}",
           transform=ax.transAxes, va="top", ha="left",
           bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax.set_ylabel("s_base − s_inj|rand (nats/token; >0 = probe helped)")
    ax.set_title("POS — paired injection probes (index recurrence)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────────
#  Plot 6 — twin recovery (post-fork s3 vs s3r, cumulative grad_tokens)
# ───────────────────────────────────────────────────────────────────────────
def plot_twin_recovery(chunks, fork, path, window=50, bk=512):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if not fork:
        for ax in axes:
            ax.axis("off")
        fig.text(0.5, 0.5, "no twin fork in this run", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return
    post = [c for c in chunks if c.get("n", 0) >= fork["n_streamed"] and "s3r" in c]
    ax0, ax1 = axes
    if post:
        n = np.array([c["n"] for c in post])
        s3 = np.array([c["s3"] for c in post])
        s3r = np.array([c["s3r"] for c in post])
        ax0.plot(n, rolling_mean(s3, window), lw=1.5, color="#d62728", label="A3 (living)")
        ax0.plot(n, rolling_mean(s3r, window), lw=1.5, color="#9467bd", label="A3R (reset twin)")
        ax0.xaxis.set_major_formatter(MTICK)
        ax0.set_xlabel("tokens streamed"); ax0.set_ylabel("rolling chunk-mean NLL")
        ax0.set_title(f"post-fork surprise (rolling mean, w={window})")
        ax0.legend(); ax0.grid(alpha=0.3)

        g = np.array([c.get("g", 0) for c in post], dtype=np.float64)
        gr = np.array([c.get("gr", 0) for c in post], dtype=np.float64)
        cum_g = np.cumsum(g) * bk       # B*K stream tokens per chunk == grad tokens when gated
        cum_gr = np.cumsum(gr) * bk
        ax1.plot(n, cum_g, lw=1.5, color="#d62728", label="A3 cum. grad tokens (post-fork)")
        ax1.plot(n, cum_gr, lw=1.5, color="#9467bd", label="A3R cum. grad tokens (post-fork)")
        ax1.xaxis.set_major_formatter(MTICK)
        ax1.yaxis.set_major_formatter(MTICK)
        ax1.set_xlabel("tokens streamed"); ax1.set_ylabel("cumulative gradient tokens")
        ax1.set_title("restart warmup cost (A3R over-gates while re-learning)")
        ax1.legend(); ax1.grid(alpha=0.3)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "no post-fork chunks with s3r yet", ha="center", va="center",
                   transform=ax.transAxes)
    fig.suptitle("POS — twin recovery (reset-twin vs. living state)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────────
#  Summary JSON
# ───────────────────────────────────────────────────────────────────────────
def arm_summary(evals, name):
    rows = [e for e in evals if name in e.get("arms", {})]
    if not rows:
        return None
    first = next((e for e in rows if e["n_streamed"] == 0), rows[0])
    last = rows[-1]
    hf, hl = first["arms"][name]["heldout"], last["arms"][name]["heldout"]
    return {"heldout_first": hf, "heldout_final": hl, "improvement": hf - hl,
            "grad_tokens_final": last["arms"][name]["grad_tokens"]}


def gate_frac_post_ignition(chunks, ignition_chunks):
    if len(chunks) <= ignition_chunks:
        return None
    post = chunks[ignition_chunks:]
    return float(np.mean([c.get("g", 0) for c in post]))


def injection_summary(probes):
    if not probes:
        return None
    d_inj = np.array([p["d_inj"] for p in probes])
    d_rand = np.array([p["d_rand"] for p in probes])
    return {"n_probes": len(probes), "mean_d_inj": float(d_inj.mean()),
            "mean_d_rand": float(d_rand.mean()),
            "frac_inj_helped": float((d_inj > 0).mean()),
            "frac_rand_helped": float((d_rand > 0).mean()),
            "mean_paired_diff": float((d_inj - d_rand).mean())}


def twin_summary(metrics, chunks, evals):
    fork = fork_event(metrics)
    if fork is None:
        return None
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

    return {"forked_at_n": fork_n, "forked_at_wall_s": fork_w,
            "surprise_excess_first_2h": surprise_excess,
            "post_fork_gate_frac_A3": post_gate_a3, "post_fork_gate_frac_A3R": post_gate_a3r,
            "converged_at_n": converged_at}


def build_summary(tag, metrics, chunks, probes, status):
    evals = evals_of(metrics)
    config = status.get("config", {})
    last_eval = evals[-1] if evals else None
    n_streamed = last_eval["n_streamed"] if last_eval else status.get("n_streamed", 0)
    wall_s = last_eval["wall_s"] if last_eval else status.get("wall_s", 0.0)

    arms = {name: arm_summary(evals, name) for name in ("A1", "A2", "A3", "A3R")}

    ratio = None
    if arms["A2"] and arms["A3"] and arms["A2"]["improvement"]:
        ratio = arms["A3"]["improvement"] / arms["A2"]["improvement"]

    a3_frac = None
    if arms["A3"] and n_streamed:
        a3_frac = arms["A3"]["grad_tokens_final"] / n_streamed

    ignition = config.get("ignition_chunks", 100)
    gate_frac = gate_frac_post_ignition(chunks, ignition)

    rss_vals = [e["rss_gb"] for e in evals]
    rss = ({"min": min(rss_vals), "max": max(rss_vals), "span": max(rss_vals) - min(rss_vals)}
          if rss_vals else {"min": None, "max": None, "span": None})

    return {"tag": tag, "n_streamed": n_streamed, "wall_hours": wall_s / 3600.0,
            "n_evals": len(evals), "arms": arms,
            "ratio_A3_vs_A2_improvement": ratio, "A3_grad_token_frac": a3_frac,
            "gate_frac_post_ignition": gate_frac, "rss": rss,
            "injection": injection_summary(probes), "twin": twin_summary(metrics, chunks, evals)}


# ───────────────────────────────────────────────────────────────────────────
#  Console summary
# ───────────────────────────────────────────────────────────────────────────
def print_summary(s):
    print(f"[pos_analyze] tag={s['tag'] or '(none)'}")
    print(f"  n_streamed={s['n_streamed']:,}  wall={s['wall_hours']:.2f}h  n_evals={s['n_evals']}")
    for name in ("A1", "A2", "A3", "A3R"):
        a = s["arms"].get(name)
        if a:
            print(f"  {name}: heldout {a['heldout_first']:.4f} -> {a['heldout_final']:.4f} "
                  f"(Δ{a['improvement']:.4f})  grad_tokens={a['grad_tokens_final']:,}")
    if s["ratio_A3_vs_A2_improvement"] is not None:
        print(f"  ratio_A3_vs_A2_improvement = {s['ratio_A3_vs_A2_improvement']:.4f}")
    if s["A3_grad_token_frac"] is not None:
        print(f"  A3_grad_token_frac = {s['A3_grad_token_frac']:.4f}")
    if s["gate_frac_post_ignition"] is not None:
        print(f"  gate_frac_post_ignition = {s['gate_frac_post_ignition']:.4f}")
    if s["rss"]["span"] is not None:
        print(f"  RSS span = {s['rss']['span']:.3f} GB ({s['rss']['min']:.3f}-{s['rss']['max']:.3f})")
    if s["injection"]:
        i = s["injection"]
        print(f"  injection: n={i['n_probes']} mean_d_inj={i['mean_d_inj']:.4f} "
              f"mean_d_rand={i['mean_d_rand']:.4f} frac_helped={i['frac_inj_helped']:.2f}")
    if s["twin"]:
        t = s["twin"]
        print(f"  twin: forked at n={t['forked_at_n']:,} "
              f"surprise_excess_2h={t['surprise_excess_first_2h']}  converged_at={t['converged_at_n']}")


# ───────────────────────────────────────────────────────────────────────────
#  Main
# ───────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="POS Phase-B analysis: plots + summary JSON")
    ap.add_argument("--tag", default="")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--plots-dir", default="plots")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    tag = f"{args.tag}_" if args.tag else ""
    R = lambda stem: os.path.join(args.results_dir, f"pos_{tag}{stem}")
    out_path = args.out or os.path.join(args.results_dir, f"pos_{tag}summary.json")

    metrics = read_jsonl(R("metrics.jsonl"))
    chunks = read_jsonl(R("chunks.jsonl"))
    probes = read_jsonl(R("probes.jsonl"))
    status = read_json(R("status.json"))

    os.makedirs(args.plots_dir, exist_ok=True)
    PL = lambda stem: os.path.join(args.plots_dir, f"pos_{tag}{stem}")

    evals = evals_of(metrics)
    fork = fork_event(metrics)

    plot_loss_vs_streamed(evals, fork, PL("loss_vs_streamed.png"))
    plot_loss_vs_grad_tokens(evals, PL("loss_vs_grad_tokens.png"))
    plot_rss_timeline(evals, PL("rss_timeline.png"))
    plot_surprise_timeline(chunks, PL("surprise_timeline.png"))
    plot_injection_bars(probes, PL("injection_bars.png"))
    cfg = status.get("config", {})
    plot_twin_recovery(chunks, fork, PL("twin_recovery.png"),
                       bk=cfg.get("batch", 8) * cfg.get("chunk", 64))

    summary = build_summary(args.tag, metrics, chunks, probes, status)
    d = os.path.dirname(out_path) or "."
    os.makedirs(d, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print_summary(summary)
    print(f"[pos_analyze] wrote {out_path}")
    print(f"[pos_analyze] wrote 6 plots to {args.plots_dir}/")


if __name__ == "__main__":
    main()
