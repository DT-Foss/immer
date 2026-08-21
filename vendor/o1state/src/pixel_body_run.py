#!/usr/bin/env python3 -u
"""
THE PIXEL BODY (P48) — the surprise calculus asked in a second modality.

A deterministic procedural maze (rooms with distinct floor colors, joined by
corridors), 8 deterministic wall-following walkers, egocentric 15×15 render
rotated heading-up → 225 color tokens + separator, padded to 256 = exactly
4 chunks of K=64. Fed through the UNCHANGED organism machinery: same
Organism, same gate, same harvest — only the world is new. No RL, no
reward; this measures perception learning and the calculus, nothing else.

Registered expectations: analysis/PREDICTIONS.md P48 —
  (a) the stack runs unchanged (loss falls ≥30% below ignition plateau,
      RSS flat),
  (b) the transition detector transfers (room-entry chunks gate ≥2× the
      corridor-steady rate post-ignition),
  (c) provenance is BIT-EXACT (5/5 sampled entries reproduce from
      (maze_seed, walker, step) by re-simulating the world),
  (d) habituation (a room's second visit spikes ≥20% less than its first).

Runs anywhere torch runs — the world is synthetic, no HF, no corpus.
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch
torch.set_num_threads(1)

import portable_organism as po

# world tokens: 0 corridor-floor, 1 wall, 2..9 room floors, 10 pad, 11 sep
PAD, SEP = 10, 11
VOCAB = 16
VIEW = 15                      # egocentric window (odd)
FRAME_TOKENS = VIEW * VIEW + 1 # 225 + separator
FRAME_PADDED = 256             # 4 chunks of 64


class Maze:
    """Deterministic: rooms on a coarse grid, L-corridors between successive
    rooms. Every cell is wall (1), corridor floor (0), or room floor (2+i)."""

    def __init__(self, seed, size=41, n_rooms=6, hidden=False):
        rng = np.random.default_rng(seed)
        g = np.ones((size, size), dtype=np.int8)
        self.rooms = []
        for i in range(n_rooms):
            w, h = int(rng.integers(5, 9)), int(rng.integers(5, 9))
            x = int(rng.integers(1, size - w - 1))
            y = int(rng.integers(1, size - h - 1))
            g[y:y + h, x:x + w] = 2 + i
            self.rooms.append((x, y, w, h))
        for i in range(1, n_rooms):
            x0, y0, w0, h0 = self.rooms[i - 1]
            x1, y1, w1, h1 = self.rooms[i]
            cx0, cy0 = x0 + w0 // 2, y0 + h0 // 2
            cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
            for x in range(min(cx0, cx1), max(cx0, cx1) + 1):
                if g[cy0, x] == 1:
                    g[cy0, x] = 0
            for y in range(min(cy0, cy1), max(cy0, cy1) + 1):
                if g[y, cx1] == 1:
                    g[y, cx1] = 0
        self.g = g
        self.size = size
        # ── the hidden room (P49): carved from the start, SEALED — masked as
        #    wall in every render until unseal(), so its color has never
        #    appeared in any stream. One corridor connects it to room 0, with
        #    a single seal cell; unseal() opens that cell and lifts the mask.
        self.sealed = False
        self.hidden_mask = np.zeros_like(g, dtype=bool)
        self.seal_cell = None
        if hidden:
            hw, hh = 6, 6
            hx = size - hw - 2
            hy = size - hh - 2
            g[hy:hy + hh, hx:hx + hw] = 2 + n_rooms          # color 8 at n_rooms=6
            self.rooms.append((hx, hy, hw, hh))
            x0, y0, w0, h0 = self.rooms[0]
            cx0, cy0 = x0 + w0 // 2, y0 + h0 // 2
            cxh, cyh = hx + hw // 2, hy + hh // 2
            corridor = []
            for x in range(min(cx0, cxh), max(cx0, cxh) + 1):
                if g[cyh, x] == 1:
                    g[cyh, x] = 0
                    corridor.append((x, cyh))
            for y in range(min(cy0, cyh), max(cy0, cyh) + 1):
                if g[y, cx0] == 1:
                    g[y, cx0] = 0
                    corridor.append((cx0, y))
            self.hidden_mask[hy:hy + hh, hx:hx + hw] = True
            for (x, y) in corridor:
                self.hidden_mask[y, x] = True
            sx, sy = corridor[0] if corridor else (cxh, cyh)
            self.seal_cell = (sx, sy)
            g[sy, sx] = 1                                     # physically sealed
            self.sealed = True

    def unseal(self):
        if self.seal_cell is not None:
            sx, sy = self.seal_cell
            self.g[sy, sx] = 0
        self.sealed = False

    def room_id(self, x, y):
        v = int(self.g[y, x])
        return v - 2 if v >= 2 else None


HEADINGS = [(0, -1), (1, 0), (0, 1), (-1, 0)]      # N E S W


class Walker:
    """Right-hand wall follower. Fully deterministic from (maze, start, h0)."""

    def __init__(self, maze, wid):
        self.m = maze
        rng = np.random.default_rng(1000 + wid)
        x, y, w, h = maze.rooms[wid % len(maze.rooms)]
        self.x = x + int(rng.integers(0, w))
        self.y = y + int(rng.integers(0, h))
        self.h = int(rng.integers(0, 4))
        self.step_i = 0

    def _free(self, hdg):
        dx, dy = HEADINGS[hdg]
        nx, ny = self.x + dx, self.y + dy
        return 0 <= nx < self.m.size and 0 <= ny < self.m.size and self.m.g[ny, nx] != 1

    def step(self):
        for turn in (1, 0, 3, 2):                  # right, straight, left, back
            hdg = (self.h + turn) % 4
            if self._free(hdg):
                self.h = hdg
                dx, dy = HEADINGS[hdg]
                self.x += dx
                self.y += dy
                break
        self.step_i += 1

    def render(self):
        """Egocentric VIEW×VIEW window, rotated heading-up, walls beyond
        the border. Returns FRAME_PADDED tokens."""
        r = VIEW // 2
        patch = np.ones((VIEW, VIEW), dtype=np.int8)
        for vy in range(VIEW):
            for vx in range(VIEW):
                wx, wy = self.x + vx - r, self.y + vy - r
                if 0 <= wx < self.m.size and 0 <= wy < self.m.size:
                    if self.m.sealed and self.m.hidden_mask[wy, wx]:
                        patch[vy, vx] = 1                     # sealed = you see wall
                    else:
                        patch[vy, vx] = self.m.g[wy, wx]
        patch = np.rot90(patch, k=self.h)          # heading-up
        toks = patch.flatten().tolist() + [SEP]
        toks += [PAD] * (FRAME_PADDED - len(toks))
        return toks


def harvest_with_rows(x, nll):
    """po.harvest_spans' exact selection logic, returning (row, span) so the
    span can be attributed to ITS walker — the bare version discards the row,
    which would make provenance attribution wrong."""
    out = []
    B, K = x.shape
    flat = nll.reshape(-1)
    order = torch.argsort(flat, descending=True)
    taken = set()
    for idx in order.tolist():
        if len(out) >= 2:
            break
        if float(flat[idx]) < po.SPIKE_MIN_NLL:
            break
        b, k = idx // K, idx % K
        if b in taken:
            continue
        lo, hi = max(0, k - po.SPAN_HALF), min(K, k + po.SPAN_HALF + 1)
        span = x[b, lo:hi].tolist()
        if len(span) > 1:
            out.append((b, span))
            taken.add(b)
    return out


def frame_stream(maze_seed, wid, from_step, n_tokens, unseal_at=None):
    """Deterministic re-generation of walker `wid`'s token stream starting at
    from_step — the provenance replay primitive AND the eval generator.
    unseal_at replays the WORLD MUTATION at the exact recorded step, so
    provenance holds THROUGH the change (the P49(c) claim)."""
    m = Maze(maze_seed, hidden=unseal_at is not None)
    w = Walker(m, wid)
    def maybe_unseal():
        if unseal_at is not None and m.sealed and w.step_i >= unseal_at:
            m.unseal()
    for _ in range(from_step):
        maybe_unseal()
        w.step()
    toks = []
    while len(toks) < n_tokens:
        maybe_unseal()
        toks.extend(w.render())
        w.step()
    return toks[:n_tokens]


def main():
    ap = argparse.ArgumentParser(description="P48: the pixel body")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--frames-per-walker", type=int, default=3000)
    ap.add_argument("--maze-seed", type=int, default=7)
    ap.add_argument("--unseal-at", type=int, default=0,
                    help="P49: frame at which the sealed seventh room opens (0 = off, pure P48 world)")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--q", type=float, default=0.75)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--min-window", type=int, default=100)
    ap.add_argument("--ignition-chunks", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "pixel_body.json"))
    args = ap.parse_args()
    if args.smoke:
        args.frames_per_walker = 400

    po.D_MODEL, po.BATCH, po.CHUNK = args.d_model, 8, 64
    po.GATE_Q, po.GATE_WINDOW, po.MIN_WINDOW, po.IGNITION_CHUNKS = \
        args.q, args.window, args.min_window, args.ignition_chunks

    mask = VOCAB - 1
    torch.manual_seed(args.seed)
    org = po.Organism("pixel", VOCAB, mask, seed=args.seed)

    maze = Maze(args.maze_seed, hidden=args.unseal_at > 0)
    walkers = [Walker(maze, k) for k in range(po.BATCH)]
    from collections import deque as _dq
    lane_hist = [_dq(maxlen=200) for _ in range(po.BATCH)]
    first_sight = [None] * po.BATCH                # (step, chunk, steady_mean_before)
    unseal_info = None
    lane_buf = [[] for _ in range(po.BATCH)]
    lane_meta = [[] for _ in range(po.BATCH)]      # (step, room_id) per frame
    room_state = [{"cur": None, "out": 0, "visits": {}} for _ in range(po.BATCH)]

    trace = []
    entries_log = []
    steady_lane = []
    knowledge = []
    rss0 = None
    K = po.CHUNK
    n_chunks_total = args.frames_per_walker * (FRAME_PADDED // K)
    t0 = time.time()

    for ci in range(1, n_chunks_total + 1):
        xs, ys = [], []
        chunk_rooms, chunk_entry = [], []
        for b in range(po.BATCH):
            while len(lane_buf[b]) < K + 1:
                w = walkers[b]
                rid = maze.room_id(w.x, w.y)
                st = room_state[b]
                is_entry, visit_no = False, None
                if rid is not None:
                    if st["cur"] != rid and st["out"] >= 3:
                        st["visits"][rid] = st["visits"].get(rid, 0) + 1
                        is_entry, visit_no = True, st["visits"][rid]
                    st["cur"], st["out"] = rid, 0
                else:
                    st["cur"] = None
                    st["out"] += 1
                lane_meta[b].append((w.step_i, rid, is_entry, visit_no))
                if args.unseal_at and maze.sealed and walkers[0].step_i >= args.unseal_at:
                    maze.unseal()
                    unseal_info = {"at_walker0_step": walkers[0].step_i, "chunk": ci}
                    print(f"[world] UNSEAL at walker0 step {walkers[0].step_i} (chunk {ci})", flush=True)
                toks = w.render()
                if args.unseal_at and first_sight[b] is None and (2 + 6) in toks:
                    import numpy as _np
                    steady = float(_np.mean(lane_hist[b])) if lane_hist[b] else None
                    first_sight[b] = {"step": w.step_i, "chunk": ci, "steady_before": steady}
                lane_buf[b].extend(toks)
                w.step()
            xs.append(lane_buf[b][:K])
            ys.append(lane_buf[b][1:K + 1])
            del lane_buf[b][:K]
            m = lane_meta[b][-1]
            chunk_rooms.append(m[1])
            chunk_entry.append((m[2], m[3], m[0]))
        x = torch.tensor(xs, dtype=torch.long)
        y = torch.tensor(ys, dtype=torch.long)
        s, gated, nll = org.step_gated(x, y)
        lane_s = nll.mean(dim=1)                   # per-walker surprise — the
                                                   # batch-mean gate is 8x
                                                   # diluted against single-
                                                   # lane events; this is not
        if gated:
            for b, sp in harvest_with_rows(x, nll):
                knowledge.append({"tokens": [int(t) for t in sp],
                                  "maze_seed": args.maze_seed, "walker": b,
                                  "step": lane_meta[b][-1][0], "chunk_i": ci})
        for b in range(po.BATCH):
            ent, vn, st = chunk_entry[b]
            if ent:
                entries_log.append({"chunk": ci, "walker": b,
                                    "surprise": float(s),
                                    "surprise_lane": round(float(lane_s[b]), 5),
                                    "gated": int(gated), "visit_no": vn})
        if ci % 7 == 0:                            # steady per-lane baseline sample
            for b in range(po.BATCH):
                if not chunk_entry[b][0]:
                    steady_lane.append(float(lane_s[b]))
        for b in range(po.BATCH):
            fs = first_sight[b]
            if fs is not None and "surprise" not in fs and fs["chunk"] == ci:
                fs["surprise"] = round(float(lane_s[b]), 5)
                fs["gated"] = int(gated)
                fs["ratio"] = (round(float(lane_s[b]) / fs["steady_before"], 3)
                               if fs.get("steady_before") else None)
            lane_hist[b].append(float(lane_s[b]))
        trace.append({"i": ci, "s": round(float(s), 5), "g": int(gated),
                      "in_room": sum(1 for r in chunk_rooms if r is not None)})
        if ci == 200:
            rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1e9 if sys.platform == "darwin" else 1e6
    print(f"[pixel] {n_chunks_total} chunks in {time.time()-t0:.0f}s | "
          f"{len(knowledge)} knowledge entries | {len(entries_log)} room entries", flush=True)

    # ── scoring ────────────────────────────────────────────────────────────
    S = np.array([t["s"] for t in trace])
    ign_end = po.IGNITION_CHUNKS
    plateau = float(S[:ign_end].mean())
    late = float(S[-max(200, n_chunks_total // 10):].mean())
    p48a_drop = (plateau - late) / plateau
    rss_span_gb = abs(rss1 - rss0) / scale if rss0 else None

    post = [t for t in trace if t["i"] > ign_end + po.MIN_WINDOW]
    entry_chunks = {e["chunk"] for e in entries_log}
    g_entry = [t["g"] for t in post if t["i"] in entry_chunks]
    g_steady = [t["g"] for t in post if t["i"] not in entry_chunks]
    rate_entry = float(np.mean(g_entry)) if g_entry else None
    rate_steady = float(np.mean(g_steady)) if g_steady else None

    first = [e["surprise"] for e in entries_log if e["visit_no"] == 1 and e["chunk"] > ign_end]
    second = [e["surprise"] for e in entries_log if e["visit_no"] == 2 and e["chunk"] > ign_end]
    hab = (1 - np.mean(second) / np.mean(first)) if first and second else None
    # per-lane (undiluted) instruments — logging only, the stack is untouched
    fl = [e["surprise_lane"] for e in entries_log if e["visit_no"] == 1 and e["chunk"] > ign_end]
    sl = [e["surprise_lane"] for e in entries_log if e["visit_no"] == 2 and e["chunk"] > ign_end]
    hab_lane = (1 - np.mean(sl) / np.mean(fl)) if fl and sl else None
    entry_lane_mean = float(np.mean([e["surprise_lane"] for e in entries_log
                                     if e["chunk"] > ign_end])) if entries_log else None
    steady_lane_mean = float(np.mean(steady_lane)) if steady_lane else None

    # (c) provenance: re-simulate and compare bit-exact
    prov = []
    rng = np.random.default_rng(7)
    for e in (rng.choice(knowledge, size=min(5, len(knowledge)), replace=False)
              if knowledge else []):
        regen = frame_stream(e["maze_seed"], e["walker"], max(0, e["step"] - 2),
                             FRAME_PADDED * 5,
                             unseal_at=(args.unseal_at or None))
        span = e["tokens"]
        found = any(regen[i:i + len(span)] == span
                    for i in range(len(regen) - len(span) + 1))
        prov.append({"walker": e["walker"], "step": e["step"], "found": bool(found)})
        print(f"[provenance] walker {e['walker']} step {e['step']} -> "
              f"{'BIT-EXACT' if found else 'NOT FOUND'}", flush=True)
    n_found = sum(1 for p in prov if p["found"])

    out = {"p48": True, "smoke": args.smoke, "maze_seed": args.maze_seed,
           "cadence": {"d_model": po.D_MODEL, "batch": po.BATCH, "chunk": po.CHUNK,
                       "q": po.GATE_Q, "window": po.GATE_WINDOW,
                       "min_window": po.MIN_WINDOW,
                       "ignition_chunks": po.IGNITION_CHUNKS},
           "n_chunks": n_chunks_total, "n_knowledge": len(knowledge),
           "n_room_entries": len(entries_log),
           "loss_plateau_ignition": round(plateau, 4), "loss_late": round(late, 4),
           "p48a_drop": round(float(p48a_drop), 4),
           "p48a_pass": bool(p48a_drop >= 0.30),
           "rss_span_gb": round(rss_span_gb, 4) if rss_span_gb is not None else None,
           "gate_rate_entry": round(rate_entry, 4) if rate_entry is not None else None,
           "gate_rate_steady": round(rate_steady, 4) if rate_steady is not None else None,
           "p48b_ratio": (round(rate_entry / rate_steady, 3)
                          if rate_entry and rate_steady else None),
           "p48b_pass": bool(rate_entry and rate_steady and
                             rate_entry >= 2 * rate_steady and len(g_entry) >= 20),
           "n_entry_events_post": len(g_entry),
           "habituation_first_n": len(first), "habituation_second_n": len(second),
           "p48d_drop": round(float(hab), 4) if hab is not None else None,
           "lane_instrument": {
               "entry_surprise_lane_mean": round(entry_lane_mean, 5) if entry_lane_mean else None,
               "steady_surprise_lane_mean": round(steady_lane_mean, 5) if steady_lane_mean else None,
               "entry_over_steady": (round(entry_lane_mean / steady_lane_mean, 3)
                                     if entry_lane_mean and steady_lane_mean else None),
               "habituation_lane": round(float(hab_lane), 4) if hab_lane is not None else None,
               "note": "per-walker surprise, logging only — the registered (b)/(d) used the batch-mean gate, which is 8x diluted against asynchronous single-walker events"},
           "p48d_pass": bool(hab is not None and hab >= 0.20),
           "p48c_provenance": prov, "p48c_found": f"{n_found}/{len(prov)}",
           "p48c_pass": bool(prov and n_found == len(prov))}
    if args.unseal_at:
        contacts = [f for f in first_sight if f is not None and "surprise" in f]
        ratios = [f["ratio"] for f in contacts if f.get("ratio")]
        gates = [f["gated"] for f in contacts]
        r6 = [e for e in entries_log if e.get("visit_no") and e["chunk"] > ign_end]
        # room-6 habituation via the lane instrument, novel room only:
        # entries into the novel room = entries recorded AFTER the unseal chunk
        uc = unseal_info["chunk"] if unseal_info else None
        nov1 = [e["surprise_lane"] for e in entries_log if uc and e["chunk"] > uc and e["visit_no"] == 1]
        nov2 = [e["surprise_lane"] for e in entries_log if uc and e["chunk"] > uc and e["visit_no"] >= 2]
        hab6 = (1 - np.mean(nov2) / np.mean(nov1)) if nov1 and nov2 else None
        out["p49"] = {
            "unseal": unseal_info,
            "n_walkers_contacted": len(contacts),
            "first_contacts": [f for f in first_sight if f is not None],
            "median_contact_ratio": (round(float(np.median(ratios)), 3) if ratios else None),
            "contact_gate_rate": (round(float(np.mean(gates)), 3) if gates else None),
            "p49a_pass": bool(ratios and np.median(ratios) >= 3.0 and
                              gates and np.mean(gates) >= 0.5),
            "novel_entries_visit1_n": len(nov1), "novel_entries_visit2plus_n": len(nov2),
            "p49b_habituation": (round(float(hab6), 4) if hab6 is not None else None),
            "p49b_pass": bool(hab6 is not None and hab6 >= 0.20),
            "p49c_note": "provenance sampling above runs WITH the mutation replayed (unseal_at forwarded)"}
    path = args.out if not args.smoke else args.out.replace(".json", "_smoke.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    tdir = path.replace(".json", "_trace.jsonl")
    with open(tdir, "w") as f:
        for t in trace:
            f.write(json.dumps(t) + "\n")
    print(f"[pixel_body] drop {p48a_drop:.1%} | entry {rate_entry} vs steady "
          f"{rate_steady} | habituation {hab} | provenance {n_found}/{len(prov)} "
          f"-> {path}", flush=True)


if __name__ == "__main__":
    main()
