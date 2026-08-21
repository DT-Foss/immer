#!/usr/bin/env python3 -u
"""
P52 — THE ORGANISM ENTERS A REAL GAME ENGINE (VizDoom, my_way_home).

Eight bodies = eight independent headless games. Per-episode re-seeding
(set_seed(base*10000+episode) before each new_episode) makes every episode
independently addressable: the provenance coordinate is (lane, episode,
frame). Engine determinism verified before registration (bit-identical
frames, same seed + same actions).

Frames: GRAY8 120x160 -> 8x10 block means -> 15x16 grid at 12 gray levels
-> 240 tokens + 16 SEP pad = 256 tokens = 4 x K64 chunks through the
UNCHANGED stack at d128/B8/K64. Actions: seeded scripted walker
(0.6 forward / 0.2 left / 0.2 right), Random(lane_seed*100000+episode),
one draw per frame — replayable by construction.

Clauses (registered 3da8d0e): (a) learn-over-life >=30% below plateau at
flat RSS; (b) own routes < never-seen route; (c) provenance 5/5 bit-exact
through the engine; (d) frozen file carries >=0.02 on the fresh-route
instrument. Gate novelty-transfer NOT registered (P49 promise); lane
surprise logs only.
"""
import argparse
import hashlib
import json
import os
import random
import resource
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import numpy as np
import torch
torch.set_num_threads(1)

import portable_organism as po

SEP = 12          # pad token closing each 240-token frame to 256
LEVELS = 12       # gray buckets 0..11
FRAME_TOKENS = 256
ACTIONS = {"L": [1, 0, 0], "R": [0, 1, 0], "F": [0, 0, 1]}


def rss_gb():
    denom = 1024 ** 2 if sys.platform.startswith("linux") else 1024 ** 3
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / denom


def make_game(seed):
    """Fresh engine at a seed — the ONLY way an episode is addressable:
    measured 2026-08-05, set_seed+new_episode on a RUNNING engine yields a
    DIFFERENT episode than on a fresh one (engine state survives
    new_episode). Life and replay both use this factory, so they are
    identical by construction."""
    import vizdoom as vzd
    scen = os.path.join(os.path.dirname(vzd.__file__), "scenarios")
    g = vzd.DoomGame()
    g.load_config(os.path.join(scen, "my_way_home.cfg"))
    g.set_window_visible(False)
    g.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    g.set_screen_format(vzd.ScreenFormat.GRAY8)
    g.set_seed(seed)
    g.init()
    g.new_episode()
    return g


def tokenize_frame(buf):
    """(120,160) uint8 -> 15x16 block means -> 12 levels -> 240 + 16 SEP."""
    g = buf.reshape(15, 8, 16, 10).mean(axis=(1, 3))
    toks = np.minimum((g // (256.0 / LEVELS)).astype(np.int64), LEVELS - 1)
    return list(toks.flatten()) + [SEP] * (FRAME_TOKENS - toks.size)


def action_for(lane_seed, episode, frame_idx, rng_cache):
    """One RNG per (lane, episode), one draw per frame — replayable."""
    key = (lane_seed, episode)
    if key not in rng_cache:
        rng_cache[key] = random.Random(lane_seed * 100000 + episode)
    p = rng_cache[key].random()
    return ACTIONS["F"] if p < 0.6 else (ACTIONS["L"] if p < 0.8 else ACTIONS["R"])


class DoomBody:
    def __init__(self, lane_seed, frame_skip=4):
        self.lane_seed, self.skip = lane_seed, frame_skip
        self.episode, self.frame_idx = 0, 0
        self.rng_cache = {}
        self.g = make_game(lane_seed * 10000 + self.episode)

    def _advance_episode(self):
        self.g.close()
        self.episode += 1
        self.frame_idx = 0
        self.g = make_game(self.lane_seed * 10000 + self.episode)

    def next_frame(self):
        """Returns (tokens, episode, frame_idx) and steps the body once."""
        if self.g.is_episode_finished():
            self._advance_episode()
        st = self.g.get_state()
        if st is None:
            self._advance_episode()
            st = self.g.get_state()
        toks = tokenize_frame(st.screen_buffer)
        meta = (self.episode, self.frame_idx)
        a = action_for(self.lane_seed, self.episode, self.frame_idx, self.rng_cache)
        self.g.make_action(a, self.skip)
        self.frame_idx += 1
        return toks, meta

    def close(self):
        self.g.close()


class DoomFeeder:
    """(B, K) chunks from B bodies; tracks per-lane frame provenance."""
    def __init__(self, bodies, K):
        self.bodies, self.K = bodies, K
        self.bufs = [[] for _ in bodies]
        self.tok_count = [0] * len(bodies)          # tokens consumed per lane
        self.frame_meta = [[] for _ in bodies]      # (start_tok, episode, frame)

    def next_xy(self):
        B, K = len(self.bodies), self.K
        for b in range(B):
            while len(self.bufs[b]) < K + 1:
                start = self.tok_count[b] + len(self.bufs[b])
                toks, (ep, fi) = self.bodies[b].next_frame()
                self.frame_meta[b].append((start, ep, fi))
                self.bufs[b].extend(toks)
        x = torch.tensor([self.bufs[b][:K] for b in range(B)], dtype=torch.long)
        y = torch.tensor([self.bufs[b][1:K + 1] for b in range(B)], dtype=torch.long)
        for b in range(B):
            del self.bufs[b][:K]
            self.tok_count[b] += K
        return x, y

    def locate(self, lane, tok_pos):
        """Global lane token position -> (episode, frame, offset_in_frame)."""
        metas = self.frame_meta[lane]
        lo, hi = 0, len(metas) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if metas[mid][0] <= tok_pos:
                lo = mid
            else:
                hi = mid - 1
        start, ep, fi = metas[lo]
        return ep, fi, tok_pos - start


def harvest_rows(x, nll, min_nll=1.0, half=8):
    """Row-aware span harvest: per-token NLL peaks, keeps the batch row."""
    out = []
    flat = nll.reshape(-1)
    K = x.shape[1]
    top = torch.topk(flat, k=min(4, flat.numel())).indices.tolist()
    for idx in top:
        if flat[idx].item() < min_nll:
            continue
        b, t = idx // K, idx % K
        lo, hi = max(0, t - half), min(K, t + half)
        out.append((b, lo, [int(v) for v in x[b, lo:hi]]))
    return out


def eval_frames(model, frame_sets, K):
    """Mean NLL over a list of 256-token frames, chunked at K."""
    xs, ys = [], []
    for toks in frame_sets:
        t = torch.tensor(toks, dtype=torch.long)
        for c in range(0, FRAME_TOKENS - K, K):
            xs.append(t[c:c + K])
            ys.append(t[c + 1:c + K + 1])
    X, Y = torch.stack(xs), torch.stack(ys)
    model.eval()
    with torch.no_grad():
        tot, n = 0.0, 0
        for i in range(0, X.shape[0], 64):
            logits, _ = model(X[i:i + 64], None)
            l = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), Y[i:i + 64].reshape(-1))
            tot += l.item() * Y[i:i + 64].numel()
            n += Y[i:i + 64].numel()
    model.train()
    return tot / n


def collect_route_frames(seed, n_frames, skip=4):
    body = DoomBody(seed, frame_skip=skip)
    frames = [body.next_frame()[0] for _ in range(n_frames)]
    body.close()
    return frames


def replay_entry(entry, skip=4):
    """Fresh engine, reseed the episode, replay actions to the frame —
    the same factory the life path uses, identical by construction."""
    g = make_game(entry["lane_seed"] * 10000 + entry["episode"])
    rng_cache = {}
    toks = None
    for fi in range(entry["frame"] + 1):
        if g.is_episode_finished():
            break
        st = g.get_state()
        if st is None:
            break
        if fi == entry["frame"]:
            toks = tokenize_frame(st.screen_buffer)
            break
        a = action_for(entry["lane_seed"], entry["episode"], fi, rng_cache)
        g.make_action(a, skip)
    g.close()
    if toks is None:
        return False
    off, span = entry["offset"], entry["tokens"]
    frag = (toks + toks)[off:off + len(span)] if off + len(span) > len(toks) else toks[off:off + len(span)]
    return frag[:len(span)] == span


def main():
    ap = argparse.ArgumentParser(description="P52: the organism in VizDoom")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--chunks", type=int, default=24000)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--fresh-seed", type=int, default=777)
    ap.add_argument("--transfer-chunks", type=int, default=3000)
    ap.add_argument("--replay-every", type=int, default=25)
    ap.add_argument("--q", type=float, default=0.75)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--min-window", type=int, default=100)
    ap.add_argument("--ignition-chunks", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "vizdoom_life.json"))
    ap.add_argument("--file-out", default=os.path.join(REPO_ROOT, "results", "vizdoom_knowledge.jsonl"))
    args = ap.parse_args()
    if args.smoke:
        args.chunks, args.transfer_chunks = 400, 200

    po.D_MODEL, po.BATCH, po.CHUNK = args.d_model, args.batch, args.chunk_size
    po.GATE_Q, po.GATE_WINDOW, po.MIN_WINDOW, po.IGNITION_CHUNKS = \
        args.q, args.window, args.min_window, args.ignition_chunks
    vocab, stoi, unk, mask, val_ids = po.get_vocab()
    V = len(vocab)
    B, K = args.batch, args.chunk_size

    torch.manual_seed(args.base_seed)
    org = po.Organism("doom", V, mask, seed=args.base_seed)
    bodies = [DoomBody(args.base_seed * 100 + b) for b in range(B)]
    feeder = DoomFeeder(bodies, K)

    knowledge, own_pool, series = [], [], []
    t0 = time.time()
    for ci in range(1, args.chunks + 1):
        x, y = feeder.next_xy()
        s, gated, nll = org.step_gated(x, y)
        if gated:
            for b, lo, toks in harvest_rows(x, nll):
                tok_pos = feeder.tok_count[b] - K + lo
                ep, fi, off = feeder.locate(b, tok_pos)
                knowledge.append({"lane": b, "lane_seed": args.base_seed * 100 + b,
                                  "episode": ep, "frame": fi, "offset": off,
                                  "tokens": toks, "surprise": round(s, 4),
                                  "n_chunk": ci})
        if ci % 200 == 0:
            series.append({"chunk": ci, "s": round(s, 4), "rss": round(rss_gb(), 3)})
        if ci % 30 == 0 and len(own_pool) < 400:
            own_pool.append(list(feeder.bufs[ci % B]) if len(feeder.bufs[ci % B]) >= FRAME_TOKENS
                            else None)
            own_pool = [f for f in own_pool if f]
        if ci % 2000 == 0:
            print(f"[life] {ci}/{args.chunks} | s {s:.3f} | entries {len(knowledge)} | "
                  f"rss {rss_gb():.2f} | {time.time()-t0:.0f}s", flush=True)
    for b in bodies:
        b.close()

    # own_pool via buffer snapshots is unreliable below FRAME_TOKENS —
    # rebuild it deterministically instead: frames from mid-life episodes
    own_frames = []
    for b in range(B):
        eps = sorted({m[1] for m in feeder.frame_meta[b]})
        mid = eps[len(eps) // 2]
        rng_cache = {}
        body = DoomBody(args.base_seed * 100 + b)
        body.episode = 0
        while body.episode < mid:
            body._advance_episode()
        for _ in range(50 // B + 7):
            own_frames.append(body.next_frame()[0])
        body.close()
    fresh_frames = collect_route_frames(args.fresh_seed, len(own_frames))

    nll_own = eval_frames(org.model, own_frames, K)
    nll_fresh = eval_frames(org.model, fresh_frames, K)

    plateau = [r["s"] for r in series if 100 <= r["chunk"] <= 600]
    tail = [r["s"] for r in series if r["chunk"] > args.chunks * 0.9]
    p52a_drop = 1 - (sum(tail) / len(tail)) / (sum(plateau) / len(plateau)) if plateau and tail else None
    rss_span = max(r["rss"] for r in series) - min(r["rss"] for r in series[1:]) if len(series) > 2 else None

    # freeze the file
    with open(args.file_out, "w") as f:
        for e in knowledge:
            f.write(json.dumps(e) + "\n")
    sha = hashlib.sha256(open(args.file_out, "rb").read()).hexdigest()

    # (c) provenance
    rng = random.Random(args.base_seed)
    picks = rng.sample(knowledge, min(5, len(knowledge))) if knowledge else []
    prov = [{"lane": e["lane"], "episode": e["episode"], "frame": e["frame"],
             "exact": bool(replay_entry(e))} for e in picks]
    n_exact = sum(p["exact"] for p in prov)

    # (d) file transfer: fresh reader, new routes, with vs without the file
    spans = [e["tokens"] for e in knowledge if len(e["tokens"]) >= 8]
    import copy
    torch.manual_seed(999)
    fresh_org = po.Organism("fresh", V, mask, seed=999)
    twin = copy.deepcopy(fresh_org)
    for arm, use_file in ((fresh_org, True), (twin, False)):
        arm_bodies = [DoomBody(500 + b) for b in range(B)]
        arm_feeder = DoomFeeder(arm_bodies, K)
        for ci in range(1, args.transfer_chunks + 1):
            x, y = arm_feeder.next_xy()
            arm.step_gated(x, y)
            if use_file and spans and ci % args.replay_every == 0:
                sp = po.SpanFeeder(po.SpanStream(spans, seed=999 + ci), B, K)
                sx, sy = sp.next_xy()
                arm.sleep_step(sx, sy)
        for bd in arm_bodies:
            bd.close()
    nll_with = eval_frames(fresh_org.model, fresh_frames, K)
    nll_without = eval_frames(twin.model, fresh_frames, K)

    out = {"p52": True, "smoke": args.smoke,
           "cadence": {"d_model": po.D_MODEL, "batch": B, "chunk": K,
                       "q": po.GATE_Q, "window": po.GATE_WINDOW,
                       "min_window": po.MIN_WINDOW, "ignition_chunks": po.IGNITION_CHUNKS},
           "engine": "vizdoom my_way_home GRAY8 160x120 skip4",
           "token_map": f"15x16 @ {LEVELS} levels + SEP pad = {FRAME_TOKENS}",
           "n_chunks": args.chunks, "n_entries": len(knowledge),
           "file_sha256": sha, "series_tail": series[-5:],
           "p52a": {"plateau_s": round(sum(plateau) / max(1, len(plateau)), 4) if plateau else None,
                    "tail_s": round(sum(tail) / max(1, len(tail)), 4) if tail else None,
                    "drop": round(p52a_drop, 4) if p52a_drop is not None else None,
                    "rss_span_gb": round(rss_span, 4) if rss_span is not None else None,
                    "pass": bool(p52a_drop is not None and p52a_drop >= 0.30
                                 and rss_span is not None and rss_span <= 0.1)},
           "p52b": {"nll_own_routes": round(nll_own, 4), "nll_fresh_route": round(nll_fresh, 4),
                    "delta": round(nll_fresh - nll_own, 4),
                    "pass": bool(nll_fresh - nll_own > 0)},
           "p52c": {"picks": prov, "exact": f"{n_exact}/{len(prov)}",
                    "pass": bool(prov) and n_exact == len(prov)},
           "p52d": {"nll_with_file": round(nll_with, 4), "nll_without": round(nll_without, 4),
                    "delta": round(nll_without - nll_with, 4),
                    "pass": bool(nll_without - nll_with >= 0.02)}}
    path = args.out if not args.smoke else args.out.replace(".json", "_smoke.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[p52] a:{out['p52a']['pass']} (drop {out['p52a']['drop']}) | "
          f"b:{out['p52b']['pass']} (Δ{out['p52b']['delta']}) | "
          f"c:{out['p52c']['exact']} | d:{out['p52d']['pass']} (Δ{out['p52d']['delta']}) "
          f"-> {path}", flush=True)


if __name__ == "__main__":
    main()
