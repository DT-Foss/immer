#!/usr/bin/env python3 -u
"""
THE ARBITRATION ORGAN v0 (P51) — selective ingestion by measured benefit.

The hub's named hard problem: the reader does not defend itself (MS1),
while the file interface CAN separate poison post-hoc (P47c, shuffle
0.494x). P51 closes the loop in-run: four frozen producer files, one
poisoned by within-span token shuffling; a consumer with a probe budget
measures each file's benefit on the shared C4-slice instrument and doses
only what measurably helps.

Arms (one deepcopied init, both seeds):
  ctrl     own stream only
  naive    replay events round-robin over ALL four files
  arbiter  probe phase (2 events/file = 10% of budget, heldout delta
           per probe), then round-robin over files with benefit > 0
  oracle   round-robin over the three intact files

Cadence d128/B8/K64 (the P46/P47 cadence), recorded in the artifact.
"""
import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch
torch.set_num_threads(1)

import portable_organism as po
from knowledge_file_run import build_c4_eval, heldout_c4


def make_organism(seed, V, mask, name):
    torch.manual_seed(seed)
    return po.Organism(name, V, mask, seed=seed)


def produce_file(name, seed, skip_docs, n_chunks, stoi, unk, mask, V):
    org = make_organism(seed, V, mask, name)
    stream = po.C4Stream(stoi, unk, skip_docs=skip_docs)
    feeder = po.ChunkFeeder(stream, po.BATCH, po.CHUNK)
    spans = []
    t0 = time.time()
    for _ in range(n_chunks):
        x, y = feeder.next_xy()
        s, gated, nll = org.step_gated(x, y)
        if gated:
            spans.extend([list(map(int, sp)) for sp in po.harvest_spans(x, nll)])
    sha = hashlib.sha256(json.dumps(spans).encode()).hexdigest()
    print(f"[{name}] {n_chunks} chunks @doc {skip_docs:,} | {len(spans)} spans | "
          f"sha {sha[:12]} | {time.time()-t0:.0f}s", flush=True)
    return spans, sha


def poison_shuffle(spans, seed):
    """P47's measured poison: shuffle tokens WITHIN each span — length,
    unigram stats, and coordinates survive; order (the content) dies."""
    rng = random.Random(seed)
    out = []
    for sp in spans:
        q = list(sp)
        rng.shuffle(q)
        out.append(q)
    return out


def replay_event(org, spans, seed):
    sp = po.SpanFeeder(po.SpanStream(spans, seed=seed), po.BATCH, po.CHUNK)
    sx, sy = sp.next_xy()
    org.sleep_step(sx, sy)


def consumer_arm(base, name, files, schedule, T, reader_offset, stoi, unk,
                 replay_every, seed, evX, evY, probe_files=None, probe_per_file=0):
    """schedule: 'none' | 'roundrobin' | 'arbiter'. files: list of span-lists.
    arbiter: first len(probe_files)*probe_per_file replay events probe one
    file each (heldout before/after), then round-robin over files whose mean
    probe benefit > 0 (fallback: the single best)."""
    org = copy.deepcopy(base)
    stream = po.C4Stream(stoi, unk, skip_docs=reader_offset)
    feeder = po.ChunkFeeder(stream, po.BATCH, po.CHUNK)
    ev_i = 0
    probes = []                       # (file_idx, delta)
    chosen = None
    for ci in range(1, T + 1):
        x, y = feeder.next_xy()
        org.step_gated(x, y)
        if schedule == "none" or ci % replay_every:
            continue
        if schedule == "roundrobin":
            replay_event(org, files[ev_i % len(files)], seed + ci)
        else:                          # arbiter
            n_probe_events = len(probe_files) * probe_per_file
            if ev_i < n_probe_events:
                fi = probe_files[ev_i % len(probe_files)]
                h0 = heldout_c4(org.model, evX, evY)
                replay_event(org, files[fi], seed + ci)
                h1 = heldout_c4(org.model, evX, evY)
                probes.append((fi, round(h0 - h1, 6)))
            else:
                if chosen is None:
                    agg = {}
                    for fi, d in probes:
                        agg.setdefault(fi, []).append(d)
                    benefit = {fi: sum(v) / len(v) for fi, v in agg.items()}
                    chosen = [fi for fi in probe_files if benefit[fi] > 0] \
                        or [max(benefit, key=lambda k: benefit[k])]
                replay_event(org, files[chosen[ev_i % len(chosen)]], seed + ci)
        ev_i += 1
    h = heldout_c4(org.model, evX, evY)
    out = {"heldout": round(h, 6), "replay_events": ev_i}
    if schedule == "arbiter":
        agg = {}
        for fi, d in probes:
            agg.setdefault(fi, []).append(d)
        out["probe_benefit"] = {str(fi): round(sum(v) / len(v), 6) for fi, v in agg.items()}
        out["probes_raw"] = probes
        out["chosen_files"] = chosen
        out["probe_share"] = round(len(probes) / max(1, ev_i), 4)
    print(f"[{name}] heldout {h:.4f} | events {ev_i}"
          + (f" | chosen {out.get('chosen_files')}" if schedule == "arbiter" else ""),
          flush=True)
    return out


def run_seed(seed, args, stoi, unk, mask, V, evX, evY):
    S, T, N = args.segment_chunks, args.consumer_chunks, 4
    files, shas = [], []
    for k in range(N):
        spans, sha = produce_file(f"s{seed}_prod{k}", seed, k * args.offset_docs,
                                  S, stoi, unk, mask, V)
        files.append(spans)
        shas.append(sha)
    poison_idx = N - 1
    files[poison_idx] = poison_shuffle(files[poison_idx], seed)
    sha_p = hashlib.sha256(json.dumps(files[poison_idx]).encode()).hexdigest()
    print(f"[seed {seed}] file {poison_idx} POISONED (shuffle) sha {sha_p[:12]}", flush=True)

    base = make_organism(seed, V, mask, f"s{seed}_consumer")
    common = dict(T=T, reader_offset=args.reader_offset_docs, stoi=stoi, unk=unk,
                  replay_every=args.replay_every, seed=seed, evX=evX, evY=evY)
    arms = {}
    arms["ctrl"] = consumer_arm(base, f"s{seed}_ctrl", files, "none", **common)
    arms["naive"] = consumer_arm(base, f"s{seed}_naive", files, "roundrobin", **common)
    arms["arbiter"] = consumer_arm(base, f"s{seed}_arbiter", files, "arbiter",
                                   probe_files=list(range(N)),
                                   probe_per_file=args.probe_per_file, **common)
    clean = [files[i] for i in range(N) if i != poison_idx]
    arms["oracle"] = consumer_arm(base, f"s{seed}_oracle", clean, "roundrobin", **common)

    b = arms["arbiter"].get("probe_benefit", {})
    ranked = sorted(b, key=lambda k: b[k]) if b else []
    poisoned_last = bool(ranked) and int(ranked[0]) == poison_idx
    return {"seed": seed, "file_shas": shas, "poison_idx": poison_idx,
            "poisoned_sha": sha_p, "file_sizes": [len(f) for f in files],
            "arms": arms,
            "p51a_poisoned_ranked_last": poisoned_last,
            "p51b_arbiter_minus_naive": round(arms["arbiter"]["heldout"] - arms["naive"]["heldout"], 6),
            "p51c_arbiter_minus_oracle": round(arms["arbiter"]["heldout"] - arms["oracle"]["heldout"], 6),
            "naive_minus_oracle": round(arms["naive"]["heldout"] - arms["oracle"]["heldout"], 6),
            "p51d_probe_share": arms["arbiter"].get("probe_share")}


def main():
    ap = argparse.ArgumentParser(description="P51: the arbitration organ v0")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--segment-chunks", type=int, default=1500)
    ap.add_argument("--consumer-chunks", type=int, default=1600)
    ap.add_argument("--replay-every", type=int, default=20)
    ap.add_argument("--probe-per-file", type=int, default=2)
    ap.add_argument("--offset-docs", type=int, default=200_000)
    ap.add_argument("--reader-offset-docs", type=int, default=1_000_000)
    ap.add_argument("--eval-offset-docs", type=int, default=1_200_000)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--seeds", default="42,43")
    ap.add_argument("--q", type=float, default=0.75)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--min-window", type=int, default=100)
    ap.add_argument("--ignition-chunks", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "arbiter.json"))
    args = ap.parse_args()

    if args.smoke:
        args.segment_chunks, args.consumer_chunks = 60, 60
        args.replay_every, args.probe_per_file = 10, 1
        args.offset_docs = 20_000
        args.seeds = "42"

    po.D_MODEL, po.BATCH, po.CHUNK = args.d_model, args.batch, args.chunk_size
    po.GATE_Q, po.GATE_WINDOW, po.MIN_WINDOW, po.IGNITION_CHUNKS = \
        args.q, args.window, args.min_window, args.ignition_chunks

    vocab, stoi, unk, mask, val_ids = po.get_vocab()
    V = len(vocab)
    evX, evY = build_c4_eval(stoi, unk, args.eval_offset_docs, po.EVAL_TOKENS, po.CHUNK)

    per_seed = [run_seed(int(s), args, stoi, unk, mask, V, evX, evY)
                for s in args.seeds.split(",")]

    both_a = all(r["p51a_poisoned_ranked_last"] for r in per_seed)
    out = {"p51": True, "smoke": args.smoke,
           "cadence": {"d_model": po.D_MODEL, "batch": po.BATCH, "chunk": po.CHUNK,
                       "q": po.GATE_Q, "window": po.GATE_WINDOW,
                       "min_window": po.MIN_WINDOW, "ignition_chunks": po.IGNITION_CHUNKS},
           "config": {"S": args.segment_chunks, "T": args.consumer_chunks,
                      "replay_every": args.replay_every,
                      "probe_per_file": args.probe_per_file},
           "per_seed": per_seed,
           "p51a_pass": both_a,
           "p51b_pass": all(r["p51b_arbiter_minus_naive"] <= -0.005 for r in per_seed),
           "p51c_pass": all(abs(r["p51c_arbiter_minus_oracle"]) <= 0.01
                            or r["p51c_arbiter_minus_oracle"] < 0 for r in per_seed),
           "p51d_pass": all((r["p51d_probe_share"] or 1) <= 0.10 + 1e-9 for r in per_seed),
           "poison_inert_flag": all(abs(r["naive_minus_oracle"]) < 0.005 for r in per_seed)}
    path = args.out if not args.smoke else args.out.replace(".json", "_smoke.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[arbiter] a:{out['p51a_pass']} b:{out['p51b_pass']} c:{out['p51c_pass']} "
          f"d:{out['p51d_pass']} inert:{out['poison_inert_flag']} -> {path}", flush=True)


if __name__ == "__main__":
    main()
