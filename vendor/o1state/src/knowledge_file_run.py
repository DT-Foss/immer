#!/usr/bin/env python3 -u
"""
THE FROZEN KNOWLEDGE FILE (MS-N / P47) — the refinery loop, closed.

Organism A streams C4 and harvests surprise-selected spans (F1 decides what
is worth keeping). The harvest is DISTILLED into a frozen, sha256-hashed,
provenance-carrying knowledge file: one entry per span — tokens, the chunk
index, the stream doc-coordinate at harvest, the surprise that selected it.
Built once; never mutated after (the hash is the artifact's identity).

Organism B — fresh, never saw A's stream — then lives a WT-103 domain shock
with dosed sleep-replay from that file. Arms, all forked from ONE shared
pre-phase state (deepcopy, so the P45c trajectory lottery cannot differ the
arms before the intervention):
    intact     replay from A's frozen file
    shuffled   same file, tokens shuffled WITHIN each span (keys+coords
               intact) — the poisoning control
    none       no file
Measured: C4-competence forgetting on a fixed far-offset C4 eval slice
(h_post − h_pre per arm), and the provenance replay — five sampled entries
re-found in the deterministic stream near their recorded coordinate.

Registered expectations: analysis/PREDICTIONS.md P47. Cadence explicit and
recorded per the 2026-08-05 audit rule.
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
from source_swap_run import HFStream


def build_c4_eval(stoi, unk, skip_docs, n_tokens, chunk):
    """A fixed C4 eval slice from a far stream offset — the C4-competence
    instrument. Deterministic: same offset -> same tokens, forever."""
    stream = po.C4Stream(stoi, unk, skip_docs=skip_docs)
    toks = []
    while len(toks) < n_tokens + 1:
        toks.extend(stream.next_block())
    toks = toks[:n_tokens + 1]
    n = (len(toks) - 1) // chunk
    X = torch.tensor([toks[i * chunk:(i + 1) * chunk] for i in range(n)], dtype=torch.long)
    Y = torch.tensor([toks[i * chunk + 1:(i + 1) * chunk + 1] for i in range(n)], dtype=torch.long)
    return X, Y


def heldout_c4(model, X, Y, bs=64):
    import torch.nn.functional as F
    model.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, X.size(0), bs):
            x, y = X[i:i + bs], Y[i:i + bs]
            logits, _ = model(x, None)
            tot += float(F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                         y.reshape(-1), reduction="sum"))
            cnt += y.numel()
    model.train()
    return tot / cnt


def freeze_file(entries, path):
    body = "\n".join(json.dumps(e) for e in entries) + "\n"
    with open(path, "w") as f:
        f.write(body)
    sha = hashlib.sha256(body.encode()).hexdigest()
    with open(path + ".sha256", "w") as f:
        f.write(sha + "\n")
    return sha


def provenance_replay(entries, stoi, unk, n_samples, back_docs=1500, fwd_docs=800):
    """P47(b): re-instantiate the deterministic stream near each sampled
    entry's recorded doc-coordinate and search for the exact span token
    subsequence. Found == the source is replayable from the artifact alone."""
    rng = random.Random(7)
    picks = rng.sample(entries, min(n_samples, len(entries)))
    results = []
    for e in picks:
        lo = max(0, e["doc_coord"] - back_docs)
        stream = po.C4Stream(stoi, unk, skip_docs=lo)
        toks = []
        found = False
        span = e["tokens"]
        while stream.docs < e["doc_coord"] + fwd_docs:
            toks.extend(stream.next_block())
            if len(toks) > 4_000_000:
                break
            # subsequence search over the accumulated window
            s = span
            for i in range(max(0, len(toks) - po.CHUNK * po.BATCH * 40), len(toks) - len(s) + 1):
                if toks[i:i + len(s)] == s:
                    found = True
                    break
            if found:
                break
        results.append({"doc_coord": e["doc_coord"], "n_chunk": e["n_chunk"],
                        "span_len": len(span), "found": found})
        print(f"[provenance] coord {e['doc_coord']:,} span_len {len(span)} -> "
              f"{'FOUND' if found else 'not found'}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser(description="MS-N / P47: the frozen knowledge file")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--producer-chunks", type=int, default=1500)
    ap.add_argument("--pre-chunks", type=int, default=600)
    ap.add_argument("--shock-chunks", type=int, default=800)
    ap.add_argument("--replay-every", type=int, default=25)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--q", type=float, default=0.75)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--min-window", type=int, default=100)
    ap.add_argument("--ignition-chunks", type=int, default=100)
    ap.add_argument("--b-offset-docs", type=int, default=300_000)
    ap.add_argument("--eval-offset-docs", type=int, default=1_200_000)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "knowledge_file.json"))
    args = ap.parse_args()

    if args.smoke:
        args.producer_chunks, args.pre_chunks, args.shock_chunks = 150, 80, 120
        args.replay_every = 20

    po.D_MODEL, po.BATCH, po.CHUNK = args.d_model, args.batch, args.chunk_size
    po.GATE_Q, po.GATE_WINDOW, po.MIN_WINDOW, po.IGNITION_CHUNKS = \
        args.q, args.window, args.min_window, args.ignition_chunks

    vocab, stoi, unk, mask, val_ids = po.get_vocab()
    V = len(vocab)
    evX, evY = build_c4_eval(stoi, unk, args.eval_offset_docs,
                             po.EVAL_TOKENS, po.CHUNK)
    print(f"[eval] C4 slice from doc {args.eval_offset_docs:,}: "
          f"{evX.numel():,} tokens", flush=True)

    tag = "smoke" if args.smoke else "full"
    kdir = os.path.join(REPO_ROOT, "results", f"knowledge_file_{tag}")
    os.makedirs(kdir, exist_ok=True)

    # ── A: the producer ────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    A = po.Organism("producer", V, mask, seed=args.seed)
    a_stream = po.C4Stream(stoi, unk, skip_docs=0)
    a_feeder = po.ChunkFeeder(a_stream, po.BATCH, po.CHUNK)
    entries = []
    t0 = time.time()
    for ci in range(1, args.producer_chunks + 1):
        x, y = a_feeder.next_xy()
        s, gated, nll = A.step_gated(x, y)
        if gated:
            for sp in po.harvest_spans(x, nll):
                entries.append({"tokens": [int(t) for t in sp],
                                "n_chunk": ci,
                                "doc_coord": a_stream.docs,
                                "surprise": round(float(s), 4)})
    if not entries:
        print("[producer] ZERO spans harvested — the spike threshold never fired "
              "at this budget; aborting loudly instead of crashing mid-arm", flush=True)
        sys.exit(2)
    kpath = os.path.join(kdir, "producer.knowledge.jsonl")
    sha = freeze_file(entries, kpath)
    print(f"[producer] {args.producer_chunks} chunks | {len(entries)} spans "
          f"| frozen {kpath} sha256 {sha[:16]}… | {time.time()-t0:.0f}s", flush=True)

    # ── the poisoning control: same file, tokens shuffled within each span ─
    rng = random.Random(1234)
    corrupted = []
    for e in entries:
        t = list(e["tokens"])
        rng.shuffle(t)
        corrupted.append({**e, "tokens": t})

    # ── B: shared pre-phase, then three forked arms ────────────────────────
    torch.manual_seed(args.seed)
    B0 = po.Organism("consumer", V, mask, seed=args.seed)
    b_stream = po.C4Stream(stoi, unk, skip_docs=args.b_offset_docs)
    b_feeder = po.ChunkFeeder(b_stream, po.BATCH, po.CHUNK)
    for _ in range(args.pre_chunks):
        x, y = b_feeder.next_xy()
        B0.step_gated(x, y)
    h_pre = heldout_c4(B0.model, evX, evY)
    print(f"[consumer] pre-phase done ({args.pre_chunks} chunks) | "
          f"C4 heldout {h_pre:.4f}", flush=True)

    def run_arm(name, pool):
        org = copy.deepcopy(B0)                    # identical starting state
        shock = HFStream("wt103", stoi, unk)
        feeder = po.ChunkFeeder(shock, po.BATCH, po.CHUNK)
        replays = 0
        for ci in range(1, args.shock_chunks + 1):
            x, y = feeder.next_xy()
            org.step_gated(x, y)
            if pool and ci % args.replay_every == 0:
                sp_stream = po.SpanStream([e["tokens"] for e in pool],
                                          seed=args.seed + ci)
                sp_feeder = po.SpanFeeder(sp_stream, po.BATCH, po.CHUNK)
                sx, sy = sp_feeder.next_xy()
                org.sleep_step(sx, sy)
                replays += 1
        h_post = heldout_c4(org.model, evX, evY)
        print(f"[{name}] shock done | C4 heldout {h_pre:.4f} -> {h_post:.4f} "
              f"(forgetting {h_post-h_pre:+.4f}) | {replays} replays", flush=True)
        return {"name": name, "h_post": round(h_post, 6),
                "forgetting": round(h_post - h_pre, 6), "replays": replays}

    arms = {a["name"]: a for a in [run_arm("intact", entries),
                                   run_arm("shuffled", corrupted),
                                   run_arm("none", None)]}

    # ── P47(b): provenance replay on the frozen file ───────────────────────
    prov = provenance_replay(entries, stoi, unk, n_samples=5)
    n_found = sum(1 for r in prov if r["found"])

    fn = arms["none"]["forgetting"]
    benefit_intact = fn - arms["intact"]["forgetting"]
    benefit_shuffled = fn - arms["shuffled"]["forgetting"]
    out = {"p47": True, "smoke": args.smoke, "seed": args.seed,
           "cadence": {"d_model": po.D_MODEL, "batch": po.BATCH, "chunk": po.CHUNK,
                       "q": po.GATE_Q, "window": po.GATE_WINDOW,
                       "min_window": po.MIN_WINDOW,
                       "ignition_chunks": po.IGNITION_CHUNKS},
           "producer_chunks": args.producer_chunks, "n_entries": len(entries),
           "file_sha256": sha, "file_path": kpath,
           "h_pre": round(h_pre, 6), "arms": arms,
           "benefit_intact": round(benefit_intact, 6),
           "benefit_shuffled": round(benefit_shuffled, 6),
           "p47a_ratio_intact_vs_none": (round(arms["intact"]["forgetting"] /
                                               fn, 4) if fn > 0 else None),
           "p47a_pass": bool(fn > 0 and arms["intact"]["forgetting"] <= 0.8 * fn),
           "p47b_provenance": prov, "p47b_found": f"{n_found}/5",
           "p47b_pass": bool(n_found >= 4),
           "p47c_pass": bool(benefit_intact > 0 and
                             benefit_shuffled < 0.5 * benefit_intact)}
    path = args.out if not args.smoke else args.out.replace(".json", "_smoke.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[knowledge_file] intact {arms['intact']['forgetting']:+.4f} | "
          f"shuffled {arms['shuffled']['forgetting']:+.4f} | "
          f"none {fn:+.4f} | provenance {n_found}/5 -> {path}", flush=True)


if __name__ == "__main__":
    main()
