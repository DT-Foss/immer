#!/usr/bin/env python3 -u
"""
THE REPLAY LAW DECOMPOSED (MS-C+I / P50) — is catch-up really 2x live?

P39 filed "catch-up replay ~2x live CPU" as a candidate law (2.01/2.04/
2.22 across machines). The registered decomposition bet: no law —
cpu_ratio = wall_ratio x thread_inflation, where the wall overhang is
ds.skip() re-downloading the organism's prior life (ratio = f(skip/T),
constant only because every measurement shared skip/T), and the cpu
overhang on top is pool threads burning clock without doing work.

One harness, five cells, every cell a FRESH subprocess (honest coldstart,
env pinned before import):

  A_ref    live source, no env pins        -> historical live anchor
  A_pin    live source, OMP/MKL=1 pinned   -> pinned live anchor +
           snapshots at 300/1500/3000 + exact token cache of each
           T-window (for B_cache and for parity checks)
  Z1  B_replay snap300  env none  -> must reproduce cpu_ratio ~2.0
  Z2  B_replay snap300  env pin   -> inflation term isolated
  Z3a B_replay snap1500 env pin   -> skip linearity, point 2
  Z3b B_replay snap3000 env pin   -> skip linearity, point 3
  Z4  B_cache  snap300  env pin   -> compute parity (clause c):
      same chunks from local cache, no HF; heldout must EQUAL Z2's
      (same tokens, same updates) — built-in determinism check.

Phase timers in every B cell: subprocess_total (parent wall), in-main
wall/cpu, load_snapshot, first_doc (the ds.skip fast-forward), stream
(next_xy), model (step_gated), harvest. Cadence recorded per the
2026-08-05 audit rule.
"""
import argparse
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

PIN_ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
           "NUMEXPR_NUM_THREADS": "1"}


# ─── subprocess entries ─────────────────────────────────────────────────────
def _timers():
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF)
    return time.time(), r.ru_utime + r.ru_stime


def cell_a(args):
    """Live source: N_pre + T chunks. Snapshots at each --snap-at (before the
    window that follows it), token cache of the T window after the LAST snap
    point requested with --cache-window. Per-chunk wall+cpu series after
    warmup -> live rate."""
    t_main0, c_main0 = _timers()
    import torch
    import portable_organism as po
    _apply_cadence(po, args)
    vocab, stoi, unk, mask, val_ids = po.get_vocab()
    V = len(vocab)
    t_setup, c_setup = _timers()

    torch.manual_seed(args.seed)
    org = po.Organism("A", V, mask, seed=args.seed)
    stream = po.make_stream(stoi, unk, skip_docs=0)
    feeder = po.ChunkFeeder(stream, po.BATCH, po.CHUNK)
    snap_at = sorted(int(s) for s in args.snap_at.split(",")) if args.snap_at else []
    total = (snap_at[-1] if snap_at else 0) + args.T

    per_chunk = []          # (wall_s, cpu_s, stream_wall, model_wall) after warmup
    caches = {}             # snap_n -> list of (x,y) int lists for T chunks
    active_cache = None
    os.makedirs(args.out_dir, exist_ok=True)

    for ci in range(1, total + 1):
        tw0, tc0 = _timers()
        x, y = feeder.next_xy()
        tw1, tc1 = _timers()
        s, gated, nll = org.step_gated(x, y)
        if gated:
            po.harvest_spans(x, nll)
        tw2, tc2 = _timers()
        if ci > args.warmup:
            per_chunk.append((tw2 - tw0, tc2 - tc0, tw1 - tw0, tw2 - tw1))
        if active_cache is not None and len(active_cache) < args.T:
            active_cache.append((x.tolist(), y.tolist()))
        if ci in snap_at:
            p = os.path.join(args.out_dir, f"snap_{ci}.pt")
            po.save_snapshot(p, org, stream, feeder.bufs, 0.0)
            if args.cache_windows:
                caches[ci] = []
                active_cache = caches[ci]

    import torch as _t
    for n, cache in caches.items():
        _t.save(cache, os.path.join(args.out_dir, f"cache_{n}.pt"))

    t_end, c_end = _timers()
    ws = sorted(r[0] for r in per_chunk)
    cs = sorted(r[1] for r in per_chunk)
    sw = sorted(r[2] for r in per_chunk)
    mw = sorted(r[3] for r in per_chunk)
    med = lambda a: a[len(a) // 2] if a else None
    out = {"cell": args.cell, "env_pinned": bool(os.environ.get("OMP_NUM_THREADS")),
           "n_chunks": total, "T": args.T, "snap_at": snap_at,
           "stream_docs_at_snaps": {str(n): None for n in snap_at},
           "setup_wall_s": round(t_setup - t_main0, 3),
           "main_wall_s": round(t_end - t_main0, 3),
           "main_cpu_s": round(c_end - c_main0, 3),
           "live_rate": {"n_sampled": len(per_chunk),
                         "median_wall_per_chunk": round(med(ws), 6) if ws else None,
                         "median_cpu_per_chunk": round(med(cs), 6) if cs else None,
                         "median_stream_wall": round(med(sw), 6) if sw else None,
                         "median_model_wall": round(med(mw), 6) if mw else None},
           "final_stream_docs": stream.docs, "reconnects": stream.reconnects}
    with open(args.result_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{args.cell}] {total} chunks | med wall/chunk "
          f"{out['live_rate']['median_wall_per_chunk']}s | docs {stream.docs}", flush=True)


def _apply_cadence(po, args):
    if args.d_model:
        po.D_MODEL = args.d_model
    if args.batch:
        po.BATCH = args.batch
    if args.chunk_size:
        po.CHUNK = args.chunk_size


def cell_b(args):
    """Catch-up: load snapshot, replay exactly T chunks — either through the
    HF stream (mode=replay: pays ds.skip over the prior life) or from the
    exact token cache (mode=cache: pays nothing but compute). v2: keeps a
    per-chunk wall series so B gets a warmup-robust MEDIAN rate — v1 only
    had phase sums, which fold process warmup into the ratio."""
    t_main0, c_main0 = _timers()
    import torch
    import portable_organism as po
    _apply_cadence(po, args)
    vocab, stoi, unk, mask, val_ids = po.get_vocab()
    V = len(vocab)
    t_setup, c_setup = _timers()

    org, ck = po.load_snapshot(args.resume, V, mask, name="B")
    t_load, c_load = _timers()

    phases = {"stream_wall": 0.0, "stream_cpu": 0.0,
              "model_wall": 0.0, "model_cpu": 0.0, "harvest_wall": 0.0}
    first_doc_wall = first_doc_cpu = None

    if args.mode == "replay":
        stream = po.make_stream(stoi, unk, skip_docs=ck["stream_docs"])
        stream.pending = list(ck["stream_pending"])
        feeder = po.ChunkFeeder(stream, po.BATCH, po.CHUNK)
        feeder.bufs = [list(b) for b in ck["bufs"]]
        # the first next_xy pays the entire ds.skip fast-forward; time it apart
        tw0, tc0 = _timers()
        x, y = feeder.next_xy()
        tw1, tc1 = _timers()
        first_doc_wall, first_doc_cpu = tw1 - tw0, tc1 - tc0
        batches = None
    else:
        batches = torch.load(args.cache, weights_only=False)
        assert len(batches) >= args.T, f"cache has {len(batches)} < T={args.T}"
        x = torch.tensor(batches[0][0], dtype=torch.long)
        y = torch.tensor(batches[0][1], dtype=torch.long)
        stream = None

    done = 0
    per_chunk_model = []
    while done < args.T:
        if done > 0:
            if args.mode == "replay":
                tw0, tc0 = _timers()
                x, y = feeder.next_xy()
                tw1, tc1 = _timers()
                phases["stream_wall"] += tw1 - tw0
                phases["stream_cpu"] += tc1 - tc0
            else:
                x = torch.tensor(batches[done][0], dtype=torch.long)
                y = torch.tensor(batches[done][1], dtype=torch.long)
        tw1, tc1 = _timers()
        s, gated, nll = org.step_gated(x, y)
        tw2, tc2 = _timers()
        phases["model_wall"] += tw2 - tw1
        phases["model_cpu"] += tc2 - tc1
        if done >= args.warmup:
            per_chunk_model.append(tw2 - tw1)
        if gated:
            po.harvest_spans(x, nll)
            phases["harvest_wall"] += _timers()[0] - tw2
        done += 1
    pcm = sorted(per_chunk_model)
    b_median_model_wall = pcm[len(pcm) // 2] if pcm else None

    t_end, c_end = _timers()
    evX, evY = po.build_eval_set(val_ids, po.EVAL_TOKENS, po.CHUNK)
    h = po.heldout(org.model, evX, evY)
    out = {"cell": args.cell, "mode": args.mode,
           "env_pinned": bool(os.environ.get("OMP_NUM_THREADS")),
           "resume": os.path.basename(args.resume), "T": args.T,
           "snapshot_stream_docs": ck["stream_docs"],
           "setup_wall_s": round(t_setup - t_main0, 3),
           "load_wall_s": round(t_load - t_setup, 3),
           "first_doc_wall_s": round(first_doc_wall, 3) if first_doc_wall is not None else None,
           "first_doc_cpu_s": round(first_doc_cpu, 3) if first_doc_cpu is not None else None,
           "phases": {k: round(v, 3) for k, v in phases.items()},
           "b_median_model_wall_per_chunk": round(b_median_model_wall, 6) if b_median_model_wall else None,
           "b_warmup_excluded": args.warmup,
           "main_wall_s": round(t_end - t_main0, 3),
           "main_cpu_s": round(c_end - c_main0, 3),
           "final_heldout": round(h, 6),
           "reconnects": stream.reconnects if stream else 0}
    with open(args.result_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{args.cell}] T={args.T} {args.mode} | main {out['main_wall_s']}s wall "
          f"{out['main_cpu_s']}s cpu | first_doc {out['first_doc_wall_s']}s | "
          f"heldout {h:.4f}", flush=True)


# ─── orchestrator ───────────────────────────────────────────────────────────
def production(args):
    """Clause-(c) at production cadence (default 8/64/128, ~81ms/chunk on the
    reference cadence probe): the model term dominates the environment, so
    'is replay COMPUTE more expensive than live' becomes measurable. Three
    cells, all env-pinned, warmup-robust medians on BOTH sides."""
    B, K, D = args.batch or 8, args.chunk_size or 64, args.d_model or 128
    cad = ["--batch", str(B), "--chunk-size", str(K), "--d-model", str(D)]
    T = args.T
    out_dir = args.out_dir + "_prod"
    os.makedirs(out_dir, exist_ok=True)
    tmo = 7200

    print(f"[replay_law:prod] A_pin cadence {B}/{K}/{D}, snap 300, T={T} ...", flush=True)
    a_pin = spawn("A_pin", ["--T", str(T), "--snap-at", "300", "--cache-windows",
                            "--warmup", "50", "--out-dir", os.path.join(out_dir, "a_pin")] + cad,
                  pin=True, out_dir=out_dir, timeout=tmo)
    snap = os.path.join(out_dir, "a_pin", "snap_300.pt")
    cache = os.path.join(out_dir, "a_pin", "cache_300.pt")

    print("[replay_law:prod] Z2p (replay, pin) ...", flush=True)
    z2 = spawn("Z2p", ["--mode", "replay", "--resume", snap, "--T", str(T),
                       "--warmup", "50"] + cad, pin=True, out_dir=out_dir, timeout=tmo)
    print("[replay_law:prod] Z4p (cache, pin) ...", flush=True)
    z4 = spawn("Z4p", ["--mode", "cache", "--resume", snap, "--cache", cache,
                       "--T", str(T), "--warmup", "50"] + cad,
               pin=True, out_dir=out_dir, timeout=tmo)

    live = a_pin["live_rate"]["median_model_wall"]
    out = {"p50_production": True,
           "cadence": {"d_model": D, "batch": B, "chunk": K}, "T": T,
           "a_pin_live_rate": a_pin["live_rate"],
           "cells": {"Z2p": z2, "Z4p": z4},
           "clause_c": {
               "live_median_model_wall": live,
               "replay_median_model_wall": z2["b_median_model_wall_per_chunk"],
               "cache_median_model_wall": z4["b_median_model_wall_per_chunk"],
               "replay_over_live": round(z2["b_median_model_wall_per_chunk"] / live, 4) if live else None,
               "cache_over_live": round(z4["b_median_model_wall_per_chunk"] / live, 4) if live else None,
               "bar": "cache_over_live <= 1.1"},
           "parity_heldout": {"Z2p": z2["final_heldout"], "Z4p": z4["final_heldout"],
                              "equal": z2["final_heldout"] == z4["final_heldout"]}}
    path = os.path.join(REPO_ROOT, "results", "replay_law_prod.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    c = out["clause_c"]
    print(f"[replay_law:prod] live {c['live_median_model_wall']} | replay/live "
          f"{c['replay_over_live']} | cache/live {c['cache_over_live']} | "
          f"parity {out['parity_heldout']['equal']} -> {path}", flush=True)


def spawn(cell, mode_args, pin, out_dir, timeout=3600):
    rj = os.path.join(out_dir, f"{cell}.json")
    cmd = [sys.executable, "-u", os.path.abspath(__file__), "--cell", cell,
           "--result-json", rj] + mode_args
    env = dict(os.environ)
    for k in PIN_ENV:
        env.pop(k, None)
    if pin:
        env.update(PIN_ENV)
    t0 = time.time()
    r = subprocess.run(cmd, env=env, timeout=timeout)
    wall = time.time() - t0
    if r.returncode != 0:
        raise RuntimeError(f"cell {cell} failed rc={r.returncode}")
    with open(rj) as f:
        res = json.load(f)
    res["subprocess_total_wall_s"] = round(wall, 3)
    return res


def main():
    ap = argparse.ArgumentParser(description="MS-C+I / P50: the replay law decomposed")
    ap.add_argument("--cell", default=None, help="internal: run as this cell")
    ap.add_argument("--result-json", default=None)
    ap.add_argument("--mode", choices=["replay", "cache"], default="replay")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--snap-at", default="")
    ap.add_argument("--cache-windows", action="store_true")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=0, help="0 = portable_organism default")
    ap.add_argument("--chunk-size", type=int, default=0)
    ap.add_argument("--d-model", type=int, default=0)
    ap.add_argument("--production", action="store_true",
                    help="clause-(c) cell at production cadence (8/64/128): "
                         "A_pin + Z2p replay + Z4p cache, per-chunk B medians")
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "results", "replay_law"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "replay_law.json"))
    args = ap.parse_args()

    if args.cell and args.cell.startswith("A"):
        return cell_a(args)
    if args.cell:
        return cell_b(args)

    if args.production:
        return production(args)

    snaps = [60] if args.smoke else [300, 1500, 3000]
    T = 40 if args.smoke else 200
    out_dir = args.out_dir + ("_smoke" if args.smoke else "")
    os.makedirs(out_dir, exist_ok=True)
    snap_str = ",".join(map(str, snaps))
    tmo = 1800 if args.smoke else 7200

    print(f"[replay_law] A_ref (env none) ...", flush=True)
    a_ref = spawn("A_ref", ["--T", str(T), "--snap-at", str(snaps[0]),
                            "--warmup", "20", "--out-dir", os.path.join(out_dir, "a_ref")],
                  pin=False, out_dir=out_dir, timeout=tmo)
    print(f"[replay_law] A_pin (env pinned, snaps {snap_str}, caches) ...", flush=True)
    a_pin = spawn("A_pin", ["--T", str(T), "--snap-at", snap_str, "--cache-windows",
                            "--warmup", "20", "--out-dir", os.path.join(out_dir, "a_pin")],
                  pin=True, out_dir=out_dir, timeout=tmo)

    ref_snap = os.path.join(out_dir, "a_ref", f"snap_{snaps[0]}.pt")
    pin_snap0 = os.path.join(out_dir, "a_pin", f"snap_{snaps[0]}.pt")
    pin_cache0 = os.path.join(out_dir, "a_pin", f"cache_{snaps[0]}.pt")

    cells = {}
    print("[replay_law] Z1 repro (replay, env none) ...", flush=True)
    cells["Z1"] = spawn("Z1", ["--mode", "replay", "--resume", ref_snap, "--T", str(T)],
                        pin=False, out_dir=out_dir, timeout=tmo)
    print("[replay_law] Z2 (replay, env pin) ...", flush=True)
    cells["Z2"] = spawn("Z2", ["--mode", "replay", "--resume", pin_snap0, "--T", str(T)],
                        pin=True, out_dir=out_dir, timeout=tmo)
    if not args.smoke:
        for label, sn in (("Z3a", snaps[1]), ("Z3b", snaps[2])):
            print(f"[replay_law] {label} (replay, env pin, snap {sn}) ...", flush=True)
            cells[label] = spawn(label, ["--mode", "replay",
                                         "--resume", os.path.join(out_dir, "a_pin", f"snap_{sn}.pt"),
                                         "--T", str(T)],
                                 pin=True, out_dir=out_dir, timeout=tmo)
    print("[replay_law] Z4 (cache, env pin) ...", flush=True)
    cells["Z4"] = spawn("Z4", ["--mode", "cache", "--resume", pin_snap0,
                               "--cache", pin_cache0, "--T", str(T)],
                        pin=True, out_dir=out_dir, timeout=tmo)

    import portable_organism as po
    ref_rate = a_ref["live_rate"]
    pin_rate = a_pin["live_rate"]

    def ratios(cell, rate):
        lw, lc = rate["median_wall_per_chunk"], rate["median_cpu_per_chunk"]
        return {"wall_ratio": round(cell["main_wall_s"] / (cell["T"] * lw), 4) if lw else None,
                "cpu_ratio": round(cell["main_cpu_s"] / (cell["T"] * lc), 4) if lc else None}

    summary = {
        "p50": True, "smoke": args.smoke,
        "cadence": {"d_model": po.D_MODEL, "batch": po.BATCH, "chunk": po.CHUNK},
        "T": T, "snaps": snaps,
        "a_ref_live_rate": ref_rate, "a_pin_live_rate": pin_rate,
        "cells": cells,
        "ratios": {"Z1_vs_ref": ratios(cells["Z1"], ref_rate),
                   "Z2_vs_pin": ratios(cells["Z2"], pin_rate),
                   **({f"{z}_vs_pin": ratios(cells[z], pin_rate)
                       for z in ("Z3a", "Z3b") if z in cells}),
                   "Z4_vs_pin": ratios(cells["Z4"], pin_rate)},
        "parity_Z2_Z4_heldout": {"Z2": cells["Z2"]["final_heldout"],
                                 "Z4": cells["Z4"]["final_heldout"],
                                 "equal": cells["Z2"]["final_heldout"] == cells["Z4"]["final_heldout"]},
        "skip_points": [
            {"snap": s,
             "snapshot_stream_docs": cells[z]["snapshot_stream_docs"],
             "first_doc_wall_s": cells[z]["first_doc_wall_s"]}
            for z, s in (("Z2", snaps[0]),) + ((("Z3a", snaps[1]), ("Z3b", snaps[2])) if not args.smoke else ())
            if z in cells],
    }
    path = args.out if not args.smoke else args.out.replace(".json", "_smoke.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    z1r = summary["ratios"]["Z1_vs_ref"]
    z4r = summary["ratios"]["Z4_vs_pin"]
    print(f"[replay_law] Z1 repro cpu_ratio {z1r['cpu_ratio']} wall {z1r['wall_ratio']} | "
          f"Z4 cache cpu_ratio {z4r['cpu_ratio']} | parity {summary['parity_Z2_Z4_heldout']['equal']} "
          f"-> {path}", flush=True)


if __name__ == "__main__":
    main()
