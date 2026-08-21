#!/usr/bin/env python3 -u
"""
THE FILE ANSWERS BY KEY (MS-Q / P55) — tappable, not just fertile.

P47/P52 proved diffuse transfer (global NLL). P55 asks the targeted
question: after dosed replay of a frozen file, does a reader complete
THE FILE'S OWN ENTRIES from their first half (the key) measurably better
than its no-file twin — and better than it completes matched
never-harvested spans (the specificity control)?

Producer streams C4 and harvests (the P47 recipe), file frozen with
sha256. Consumer twins fork from one init, stream their own far-offset
budget; one gets dosed file replay. Probe: N harvested entries split
key|value at the midpoint, completion NLL measured on the value half
conditioned on the key half; control: N never-harvested spans of matched
length from the producer's stream region. Cadence recorded.
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


def completion_nll(model, spans, K):
    """Mean NLL on the SECOND half of each span, conditioned on the first
    half (the key). The split sits at each span's own midpoint: harvest
    geometry caps spans at one chunk (33..K tokens for interior spikes),
    so a fixed K-token key would leave no value half. No padding — pad
    tokens would sit inside the measured half."""
    import torch.nn.functional as F
    model.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for sp in spans:
            t = sp[:K + 1]
            if len(t) < 9:
                continue
            x = torch.tensor([t[:-1]], dtype=torch.long)
            y = torch.tensor([t[1:]], dtype=torch.long)
            logits, _ = model(x, None)
            half = (len(t) - 1) // 2
            l = F.cross_entropy(logits[0, half:], y[0, half:])
            tot += float(l)
            cnt += 1
    model.train()
    return tot / max(1, cnt)


def collect_control_spans(stoi, unk, skip_docs, span_lens, seed):
    """Never-harvested spans, length-matched per probe entry, from the
    producer's stream region: seeded random offsets into a fresh read."""
    stream = po.C4Stream(stoi, unk, skip_docs=skip_docs)
    need = max(span_lens, default=1)
    toks = []
    while len(toks) < max(1, len(span_lens)) * need * 20:
        toks.extend(stream.next_block())
    rng = random.Random(seed)
    out = []
    for L in span_lens:
        i = rng.randrange(0, len(toks) - L)
        out.append([int(v) for v in toks[i:i + L]])
    return out


def main():
    ap = argparse.ArgumentParser(description="P55: the file answers by key")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--producer-chunks", type=int, default=1500)
    ap.add_argument("--consumer-chunks", type=int, default=1500)
    ap.add_argument("--replay-every", type=int, default=25)
    ap.add_argument("--n-probe", type=int, default=100)
    ap.add_argument("--producer-offset", type=int, default=0)
    ap.add_argument("--reader-offset", type=int, default=1_000_000)
    ap.add_argument("--eval-offset", type=int, default=1_200_000)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--q", type=float, default=0.75)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--min-window", type=int, default=100)
    ap.add_argument("--ignition-chunks", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "keyed_file.json"))
    args = ap.parse_args()
    if args.smoke:
        args.producer_chunks, args.consumer_chunks = 120, 120
        args.replay_every, args.n_probe = 10, 20

    po.D_MODEL, po.BATCH, po.CHUNK = args.d_model, args.batch, args.chunk_size
    po.GATE_Q, po.GATE_WINDOW, po.MIN_WINDOW, po.IGNITION_CHUNKS = \
        args.q, args.window, args.min_window, args.ignition_chunks
    vocab, stoi, unk, mask, val_ids = po.get_vocab()
    V = len(vocab)
    K = po.CHUNK
    evX, evY = build_c4_eval(stoi, unk, args.eval_offset, po.EVAL_TOKENS, po.CHUNK)

    # ── producer: stream, harvest, freeze ──────────────────────────────────
    prod = make_organism(args.seed, V, mask, "producer")
    stream = po.C4Stream(stoi, unk, skip_docs=args.producer_offset)
    feeder = po.ChunkFeeder(stream, po.BATCH, po.CHUNK)
    spans = []
    t0 = time.time()
    for _ in range(args.producer_chunks):
        x, y = feeder.next_xy()
        s, gated, nll = prod.step_gated(x, y)
        if gated:
            spans.extend([list(map(int, sp)) for sp in po.harvest_spans(x, nll)])
    sha = hashlib.sha256(json.dumps(spans).encode()).hexdigest()
    print(f"[producer] {args.producer_chunks} chunks | {len(spans)} spans | "
          f"sha {sha[:12]} | {time.time()-t0:.0f}s", flush=True)

    # probe material: harvest geometry caps spans at one chunk, interior
    # spikes give 33..K tokens — split key|value at each span's midpoint
    min_probe_len = 33
    long_spans = [sp for sp in spans if len(sp) >= min_probe_len]
    rng = random.Random(args.seed)
    probe = rng.sample(long_spans, min(args.n_probe, len(long_spans)))
    control = collect_control_spans(stoi, unk, args.producer_offset,
                                    [len(sp) for sp in probe], args.seed + 7)
    print(f"[probe] {len(probe)} harvested entries (>= {min_probe_len} tok) | "
          f"{len(control)} matched controls", flush=True)

    # ── consumer twins ─────────────────────────────────────────────────────
    base = make_organism(args.seed + 999, V, mask, "consumer")
    twins = {}
    for name, use_file in (("with_file", True), ("without", False)):
        org = copy.deepcopy(base)
        cstream = po.C4Stream(stoi, unk, skip_docs=args.reader_offset)
        cfeeder = po.ChunkFeeder(cstream, po.BATCH, po.CHUNK)
        for ci in range(1, args.consumer_chunks + 1):
            x, y = cfeeder.next_xy()
            org.step_gated(x, y)
            if use_file and spans and ci % args.replay_every == 0:
                sp = po.SpanFeeder(po.SpanStream(spans, seed=args.seed + ci),
                                   po.BATCH, po.CHUNK)
                sx, sy = sp.next_xy()
                org.sleep_step(sx, sy)
        twins[name] = org
        print(f"[{name}] {args.consumer_chunks} chunks done", flush=True)

    # ── the keyed probe ────────────────────────────────────────────────────
    res = {}
    for name, org in twins.items():
        res[name] = {
            "probe_completion": round(completion_nll(org.model, probe, K), 6),
            "control_completion": round(completion_nll(org.model, control, K), 6),
            "heldout": round(heldout_c4(org.model, evX, evY), 6)}
        print(f"[{name}] probe {res[name]['probe_completion']} | "
              f"control {res[name]['control_completion']} | "
              f"heldout {res[name]['heldout']}", flush=True)

    keyed_gain = res["without"]["probe_completion"] - res["with_file"]["probe_completion"]
    control_gain = res["without"]["control_completion"] - res["with_file"]["control_completion"]
    specificity = keyed_gain - control_gain
    out = {"p55": True, "smoke": args.smoke,
           "cadence": {"d_model": po.D_MODEL, "batch": po.BATCH, "chunk": po.CHUNK,
                       "q": po.GATE_Q, "window": po.GATE_WINDOW,
                       "min_window": po.MIN_WINDOW, "ignition_chunks": po.IGNITION_CHUNKS},
           "config": {"producer_chunks": args.producer_chunks,
                      "consumer_chunks": args.consumer_chunks,
                      "replay_every": args.replay_every, "n_probe": len(probe),
                      "min_probe_len": min_probe_len},
           "file_sha256": sha, "n_spans": len(spans),
           "arms": res,
           "p55a_keyed_gain": round(keyed_gain, 6),
           "p55a_pass": bool(keyed_gain >= 0.05),
           "p55b_specificity": round(specificity, 6),
           "p55b_pass": bool(specificity >= 0.02),
           "diffuse_heldout_gain": round(res["without"]["heldout"] - res["with_file"]["heldout"], 6)}
    path = args.out if not args.smoke else args.out.replace(".json", "_smoke.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[p55] keyed_gain {keyed_gain:+.4f} (bar 0.05) | specificity "
          f"{specificity:+.4f} (bar 0.02) | a:{out['p55a_pass']} b:{out['p55b_pass']} "
          f"-> {path}", flush=True)


if __name__ == "__main__":
    main()
