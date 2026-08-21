#!/usr/bin/env python3 -u
"""
SOURCE HOT-SWAP (MS-L / P43) — can a living organism change its food source
mid-life, with short idle pauses, and come back unharmed?

One gated organism streams C4 -> idle pause -> WT-103 -> idle pause -> C4.
A control arm (same seed, same cadence, same idle pattern) stays on pure C4
at equal total streamed chunks. Registered expectations live in
analysis/PREDICTIONS.md P43 — including the SIGNED transient: WT-103 is the
vocab/eval domain sibling, so swap-in should QUIET the gate (rate <= 0.6x
baseline in the first 100 chunks) and the return to C4 should SPIKE it
(>= 1.5x). Full per-chunk trace (source, s, gate, threshold) — this run is
its own instrument.

Reuses portable_organism's machinery (vocab, Organism, eval, feeder); the
gate globals are set to the POS values (q=0.75, window 500, min 100,
ignition 100) as registered, and the cadence is explicit and RECORDED in
the artifact per the 2026-08-05 audit rule.

Stream continuity: each source keeps ONE stream object for the whole run,
so returning to C4 resumes the exact stream position. Feeder buffers are
flushed at every boundary (counted in the artifact) so no tokens of the old
source bleed into the new segment.
"""
import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch
torch.set_num_threads(1)

import portable_organism as po


class HFStream(po.C4Stream):
    """C4Stream with the dataset made explicit. Everything else — pending
    buffer, reconnect-with-backoff, doc skipping, tokenization — is
    inherited unchanged."""

    SOURCES = {
        "c4": ("allenai/c4", "en"),
        "wt103": ("Salesforce/wikitext", "wikitext-103-raw-v1"),
    }

    def __init__(self, source, stoi, unk, block=8192, skip_docs=0):
        super().__init__(stoi, unk, block=block, skip_docs=skip_docs)
        self.source = source

    def _connect(self):
        from datasets import load_dataset
        path, name = self.SOURCES[self.source]
        ds = load_dataset(path, name, split="train", streaming=True)
        if self.docs:
            ds = ds.skip(self.docs)
        self._it = iter(ds)


def run_arm(name, schedule, seed, args, trace_path):
    """schedule: list of (kind, n_chunks) with kind in {'c4','wt103','idle'}.
    Returns the summary dict. One stream object per source, kept alive
    across segments."""
    po.D_MODEL, po.BATCH, po.CHUNK = args.d_model, args.batch, args.chunk_size
    po.GATE_Q, po.GATE_WINDOW, po.MIN_WINDOW, po.IGNITION_CHUNKS = \
        args.q, args.window, args.min_window, args.ignition_chunks

    vocab, stoi, unk, mask, val_ids = po.get_vocab()
    evX, evY = po.build_eval_set(val_ids, po.EVAL_TOKENS, po.CHUNK)

    torch.manual_seed(seed)
    organism = po.Organism(name, len(vocab), mask, seed=seed)
    streams, feeders = {}, {}
    flushed_tokens = 0
    boundary_evals = []
    t0 = time.time()

    tf = open(trace_path, "w")
    ci = 0                                  # global chunk index across segments
    for seg_idx, (kind, n) in enumerate(schedule):
        if kind == "idle":
            for _ in range(n):
                ci += 1
                organism.n_chunks += 1      # an idle chunk is a chunk that passed
            continue
        if kind not in streams:
            streams[kind] = HFStream(kind, stoi, unk)
            feeders[kind] = po.ChunkFeeder(streams[kind], po.BATCH, po.CHUNK)
        else:
            # flush stale buffered tokens so nothing of the paused segment's
            # tail bleeds across the boundary; the stream position advances
            flushed_tokens += sum(len(b) for b in feeders[kind].bufs)
            feeders[kind] = po.ChunkFeeder(streams[kind], po.BATCH, po.CHUNK)
        feeder = feeders[kind]
        for _ in range(n):
            ci += 1
            x, y = feeder.next_xy()
            s, gated, nll = organism.step_gated(x, y)
            thresh = None
            if len(organism.window) >= po.MIN_WINDOW:
                import numpy as np
                thresh = float(np.quantile(np.fromiter(organism.window, dtype=np.float64), po.GATE_Q))
            tf.write(json.dumps({"i": ci, "seg": seg_idx, "src": kind,
                                 "s": round(float(s), 6), "g": int(gated),
                                 "th": round(thresh, 6) if thresh is not None else None}) + "\n")
        tf.flush()
        h = po.heldout(organism.model, evX, evY)
        boundary_evals.append({"after_seg": seg_idx, "kind": kind, "chunk": ci,
                               "heldout": round(float(h), 6)})
        print(f"[{name}] seg{seg_idx} ({kind}×{n}) done @chunk {ci} | heldout {h:.4f}", flush=True)
    tf.close()

    return {"name": name, "schedule": schedule, "seed": seed,
            "cadence": {"d_model": po.D_MODEL, "batch": po.BATCH, "chunk": po.CHUNK,
                        "q": po.GATE_Q, "window": po.GATE_WINDOW,
                        "min_window": po.MIN_WINDOW, "ignition_chunks": po.IGNITION_CHUNKS},
            "final_heldout": boundary_evals[-1]["heldout"],
            "boundary_evals": boundary_evals,
            "flushed_tokens_at_boundaries": flushed_tokens,
            "stream_docs": {k: streams[k].docs for k in streams},
            "stream_reconnects": {k: streams[k].reconnects for k in streams},
            "grad_tokens": organism.grad_tokens, "n_chunks": organism.n_chunks,
            "wall_s": round(time.time() - t0, 1), "trace": trace_path}


def main():
    ap = argparse.ArgumentParser(description="MS-L / P43: live source hot-swap")
    ap.add_argument("--segment-chunks", type=int, default=1500, help="S: chunks per source segment")
    ap.add_argument("--pause-chunks", type=int, default=200, help="idle chunks at each swap")
    ap.add_argument("--smoke", action="store_true", help="S=60, pause=20")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--q", type=float, default=0.75)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--min-window", type=int, default=100)
    ap.add_argument("--ignition-chunks", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "source_swap.json"))
    args = ap.parse_args()

    S = 60 if args.smoke else args.segment_chunks
    P = 20 if args.smoke else args.pause_chunks
    tag = "smoke" if args.smoke else "full"
    tdir = os.path.join(REPO_ROOT, "results", f"source_swap_{tag}_traces")
    os.makedirs(tdir, exist_ok=True)

    swap_sched = [("c4", S), ("idle", P), ("wt103", S), ("idle", P), ("c4", S)]
    ctrl_sched = [("c4", S), ("idle", P), ("c4", S), ("idle", P), ("c4", S)]

    swap = run_arm("swap", swap_sched, args.seed, args, os.path.join(tdir, "swap.jsonl"))
    ctrl = run_arm("ctrl", ctrl_sched, args.seed, args, os.path.join(tdir, "ctrl.jsonl"))

    out = {"p43": True, "smoke": args.smoke, "S": S, "pause": P,
           "swap": swap, "control": ctrl,
           "heldout_delta_swap_minus_ctrl": round(swap["final_heldout"] - ctrl["final_heldout"], 6)}
    path = args.out if not args.smoke else args.out.replace(".json", "_smoke.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[source_swap] swap {swap['final_heldout']} vs ctrl {ctrl['final_heldout']} "
          f"(delta {out['heldout_delta_swap_minus_ctrl']:+.4f}) -> {path}", flush=True)


if __name__ == "__main__":
    main()
