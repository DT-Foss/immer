#!/usr/bin/env python3
"""Per-chunk cost measured IN-PROCESS, after warmup.

The first probe measured subprocess wall-clock over 60 chunks, which is
dominated by fixed startup (vocab build, WikiText load, torch import) and
produced nonsense: 237 ms/chunk at toy scale, and d=128 apparently FASTER
than d=64. This version times the step loop itself, after a warmup, so the
number is the per-chunk cost P39 (a)/(c) are ratios against.
"""
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP = os.path.join(REPO_ROOT, "results")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch
torch.set_num_threads(1)          # house rule: threads=1, comparable numbers

import portable_organism as po

WARMUP = 10
MEASURE = 40
SNAP_SAMPLES = 7      # median over repeats; a single snapshot timing is noise-dominated

GRID = [
    (64, 4, 32, "toy (every P38/P39 result so far)"),
    (128, 4, 32, "d128 only (what P39 FINAL measured)"),
    (64, 8, 64, "chunk weight only"),
    (128, 8, 64, "production (the 40h POS organism)"),
]

vocab, stoi, unk, mask, val_ids = po.get_vocab()
V = len(vocab)

results = []
for d, b, k, label in GRID:
    po.D_MODEL, po.BATCH, po.CHUNK = d, b, k

    organism = po.Organism(f"probe_d{d}_b{b}_k{k}", V, mask)
    stream = po.make_stream(stoi, unk)
    feeder = po.ChunkFeeder(stream, b, k)

    for _ in range(WARMUP):
        x, y = feeder.next_xy()
        organism.step_gated(x, y)

    t0 = time.perf_counter()
    c0 = time.process_time()
    for _ in range(MEASURE):
        x, y = feeder.next_xy()
        organism.step_gated(x, y)
    wall_ms = (time.perf_counter() - t0) / MEASURE * 1000.0
    cpu_ms = (time.process_time() - c0) / MEASURE * 1000.0

    # snapshot cost: the fixed constant P39 (a) is a ratio against. A single
    # timing is far too noisy to compare across configs — repeated writes of
    # the SAME checkpoint varied 17.9-26.7 ms (filesystem cache, allocator
    # state), which is the size of the effect being measured. Median of N.
    snap_path = os.path.join(SP, f"probe_snap_d{d}_b{b}_k{k}.pt")
    snap_samples = []
    for _ in range(SNAP_SAMPLES):
        ts = time.perf_counter()
        po.save_snapshot(snap_path, organism, stream, feeder.bufs, 0.0)
        snap_samples.append((time.perf_counter() - ts) * 1000.0)
    snap_ms = float(sorted(snap_samples)[len(snap_samples) // 2])
    snap_mb = os.path.getsize(snap_path) / 1e6
    os.remove(snap_path)

    rec = {"d_model": d, "batch": b, "chunk": k, "label": label,
           "tokens_per_chunk": b * k,
           "wall_ms_per_chunk": round(wall_ms, 3),
           "cpu_ms_per_chunk": round(cpu_ms, 3),
           "snapshot_ms": round(snap_ms, 2),
           "snapshot_mb": round(snap_mb, 2),
           "snapshot_in_chunk_slots": round(snap_ms / wall_ms, 3)}
    results.append(rec)
    print(f"d={d:>4} B={b} K={k:>3}  {b*k:>4} tok/chunk | "
          f"chunk {wall_ms:>7.2f} ms wall / {cpu_ms:>7.2f} ms cpu | "
          f"snapshot {snap_ms:>7.1f} ms ({snap_mb:.1f} MB) "
          f"= {snap_ms/wall_ms:>6.2f} chunk slots  | {label}", flush=True)

with open(os.path.join(SP, "cadence_probe.json"), "w") as f:
    json.dump({"warmup_chunks": WARMUP, "measured_chunks": MEASURE,
               "torch_threads": 1, "results": results}, f, indent=2)
print(f"\n-> {SP}/cadence_probe.json")
