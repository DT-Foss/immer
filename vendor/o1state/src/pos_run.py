#!/usr/bin/env python3 -u
"""
POS — Plasticity-on-Surprise as an operating mode, measured over wall-clock days.
=================================================================================
Gradient-on-surprise: the forward pass always runs (no_grad, Z-carry), the backward
pass runs ONLY on chunks whose mean surprise clears a rolling quantile threshold.
Expectation: the gated arm captures most of the full-gradient arm's learning at a
fraction of the gradient tokens — at flat RSS, in one CPU process that never restarts.

ONE process, THREE arms, IDENTICAL data stream (every chunk cloned to every arm):
  A1 forward-only    control: no learning (the floor)
  A2 full-gradient   control: standard recipe, backward on every chunk
  A3 surprise-gated  backward only when chunk-mean surprise > rolling q-quantile
plus, forked at --twin-fork-hours from A3:
  A3R reset-twin     same weights, Z and optimizer state zeroed — measures what a
                     restart pays in warmup that the living state never pays.

The model/data path is EXACTLY src/streaming_train.py (StreamingNoPELM, detach-carry
at every chunk boundary — the grad-cos 1.0 path), C4 via HF streaming, CPU, capped
threads, psutil RSS. The index loop (surprise spike -> span store -> paired injection
probe on key recurrence) lives in src/pos_index.py and observes A3 only.

Outputs (results/pos_<tag_>*):
  pos_status.json    heartbeat (tmpfile + os.replace)
  pos_metrics.jsonl  eval + event records (the analysis substrate)
  pos_chunks.jsonl   per-chunk surprise / gate trace
  pos_index.jsonl    stored surprise spans          (written by pos_index.py)
  pos_probes.jsonl   paired injection probes        (written by pos_index.py)
  pos_ckpt.pt        resume checkpoint (tmpfile + os.replace, every --ckpt-every-s)

Determinism: seed 42, no dropout, no RNG in the loop, deterministic C4 order ->
two runs with the same token target produce identical metrics. The run prints a
sha256 digest over all metric fields (wall-clock/RSS excluded); build-gate G2
compares digests of two smoke runs.
"""
import os, sys, json, time, argparse, shutil, hashlib, tempfile
from collections import deque

sys.path.insert(0, "reference"); sys.path.insert(0, "src")

import resource
try:
    import psutil
    _PROC = psutil.Process(os.getpid())
    def _rss_gb(): return _PROC.memory_info().rss / 1e9
except ImportError:
    def _rss_gb():
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1e9 if r > 1e7 else r / 1e6

import numpy as np
import torch
import torch.nn.functional as F
torch.backends.mps.is_available = lambda: False          # force CPU (same as the 1B run)
# ONE thread, not ncpu-2: the ops here are tiny (B=8, d=128) and the machine runs a
# saturating SAT-solver fleet — measured under that load, 8 threads is ~30x SLOWER than 1
# (sync contention), and 1 thread is bit-deterministic. See analysis/DECISIONS.md.
torch.set_num_threads(1)

from streaming_train import StreamingNoPELM
from length_extrap_v2 import load_wikitext2, build_vocab, tokenize

try:
    from pos_index import IndexLoop, fork_twin
except ImportError:
    IndexLoop, fork_twin = None, None


# ───────────────────────────────────────────────────────────────────────────
#  C4 stream with doc-counting (exact resume) and reconnect-on-error (40h runs
#  survive network hiccups; on reconnect we skip the already-consumed docs).
# ───────────────────────────────────────────────────────────────────────────
class C4Stream:
    def __init__(self, stoi, unk, block=65536, skip_docs=0):
        self.stoi, self.unk, self.block = stoi, unk, block
        self.docs = skip_docs
        self.reconnects = 0
        self.pending = []
        self._it = None

    def _connect(self):
        from datasets import load_dataset
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        if self.docs:
            ds = ds.skip(self.docs)
        self._it = iter(ds)

    def next_block(self):
        while len(self.pending) < self.block:
            try:
                if self._it is None:
                    self._connect()
                row = next(self._it)
            except StopIteration:
                self.docs = 0; self._it = None
                continue
            except Exception as e:
                self.reconnects += 1
                print(f"[stream] {type(e).__name__}: {e} — reconnect #{self.reconnects} "
                      f"at doc {self.docs:,}", flush=True)
                self._it = None
                time.sleep(min(300, 5 * self.reconnects))
                continue
            self.docs += 1
            t = row.get("text", "") if isinstance(row, dict) else ""
            if t.strip():
                self.pending.extend(tokenize(t, self.stoi, self.unk))
        out, self.pending = self.pending[:self.block], self.pending[self.block:]
        return out


# ───────────────────────────────────────────────────────────────────────────
#  Arms
# ───────────────────────────────────────────────────────────────────────────
def build_arm(name, V, mask, args):
    torch.manual_seed(args.seed)                       # identical init for every arm
    m = StreamingNoPELM(V, mask, d_model=args.d_model, n_layers=2, n_heads=4,
                        d_head=args.d_model // 4, seq_len=32, dropout=0.0, causal=True)
    m.train()
    return {"name": name, "model": m,
            "opt": None if name == "A1" else torch.optim.Adam(m.parameters(), lr=args.lr),
            "states": None, "grad_tokens": 0, "n_chunks": 0, "n_bwd": 0,
            "window": deque(maxlen=args.window), "recent_gates": deque(maxlen=200)}


def _detach(states):
    return [s.detach() for s in states]


def _fwd_nograd(arm, x, y):
    """Forward under no_grad from the arm's carried state. Returns (nll (B,K), states)."""
    with torch.no_grad():
        logits, st = arm["model"](x, arm["states"])
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                              reduction="none").view(x.shape)
    return nll, st


def step_a1(arm, x, y):
    nll, st = _fwd_nograd(arm, x, y)
    arm["states"] = st
    arm["n_chunks"] += 1
    return float(nll.mean())


def step_a2(arm, x, y, clip=5.0):
    logits, st = arm["model"](x, arm["states"])
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    arm["opt"].zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(arm["model"].parameters(), clip)
    arm["opt"].step()
    arm["states"] = _detach(st)
    arm["grad_tokens"] += x.numel()
    arm["n_chunks"] += 1; arm["n_bwd"] += 1
    return float(loss)


def step_gated(arm, x, y, args, clip=5.0):
    """A3 / A3R: forward always (no_grad); backward only above the rolling quantile.
    The gated backward RECOMPUTES the chunk forward from the same pre-chunk state
    (weights unchanged between the two passes -> identical values), then steps.
    Returns (chunk_mean_surprise, gated, threshold, per_token_nll)."""
    nll, st_ng = _fwd_nograd(arm, x, y)
    s = float(nll.mean())
    thresh = None
    if arm["n_chunks"] < args.ignition_chunks and not arm.get("no_ignition"):
        gated = True                                    # ignition: always learn at first
        # (the reset-twin sets no_ignition — it inherits A3's window and gates from
        #  chunk one; its elevated early gating IS the measured warmup cost)
    elif len(arm["window"]) >= args.min_window:
        thresh = float(np.quantile(np.fromiter(arm["window"], dtype=np.float64), args.q))
        gated = s > thresh
    else:
        gated = True
    if gated:
        logits, st = arm["model"](x, arm["states"])     # recompute WITH graph, same values
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        arm["opt"].zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(arm["model"].parameters(), clip)
        arm["opt"].step()
        arm["states"] = _detach(st)
        arm["grad_tokens"] += x.numel()
        arm["n_bwd"] += 1
    else:
        arm["states"] = st_ng
    arm["window"].append(s)                             # threshold uses PREVIOUS chunks only
    arm["recent_gates"].append(1 if gated else 0)
    arm["n_chunks"] += 1
    return s, gated, thresh, nll


# ───────────────────────────────────────────────────────────────────────────
#  Frozen eval — fixed WT-2 val slice, batched stateless chunks (no_grad)
# ───────────────────────────────────────────────────────────────────────────
def build_eval_set(val_ids, n_tokens, chunk):
    ids = val_ids[:n_tokens + 1]
    rows = (len(ids) - 1) // chunk
    X = torch.tensor([ids[r * chunk:(r + 1) * chunk] for r in range(rows)], dtype=torch.long)
    Y = torch.tensor([ids[r * chunk + 1:(r + 1) * chunk + 1] for r in range(rows)], dtype=torch.long)
    return X, Y


@torch.no_grad()
def heldout(model, evX, evY, bs=64):
    tot, n = 0.0, 0
    for i in range(0, len(evX), bs):
        lg, _ = model(evX[i:i + bs], None)
        tot += float(F.cross_entropy(lg.reshape(-1, lg.size(-1)),
                                     evY[i:i + bs].reshape(-1), reduction="sum"))
        n += evY[i:i + bs].numel()
    return tot / max(1, n)


# ───────────────────────────────────────────────────────────────────────────
#  Atomic writers
# ───────────────────────────────────────────────────────────────────────────
def write_atomic(path, obj):
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def append_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def trim_jsonl(path, key, n_limit):
    """Drop rows past the checkpoint's token count. The jsonl files are appended
    continuously but the checkpoint is only written every --ckpt-every-s, so a
    crash-resume would otherwise leave a duplicated tail (n jumps backwards) —
    which verify_pos.py rightly rejects as an integrity failure."""
    if not os.path.exists(path):
        return 0
    kept, dropped = [], 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if json.loads(line).get(key, 0) <= n_limit:
                kept.append(line)
            else:
                dropped += 1
    if dropped:
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=".tmp_trim_")
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(kept) + ("\n" if kept else ""))
        os.replace(tmp, path)
    return dropped


def save_ckpt(path, arms, stream, n_tok, wall, index, torch_rng, args):
    ck = {"n_streamed": n_tok, "wall_s": wall, "docs": stream.docs,
          "bufs": save_bufs_ref[0], "pending": stream.pending,
          "torch_rng": torch_rng, "config": vars(args),
          "arms": {}, "index_state": index.state_dict() if index else None}
    for a in arms:
        ck["arms"][a["name"]] = {
            "model": a["model"].state_dict(),
            "opt": a["opt"].state_dict() if a["opt"] else None,
            "states": None if a["states"] is None else [s.clone() for s in a["states"]],
            "grad_tokens": a["grad_tokens"], "n_chunks": a["n_chunks"], "n_bwd": a["n_bwd"],
            "window": list(a["window"]), "recent_gates": list(a["recent_gates"])}
    # NOTE: torch.save's PyTorchFileWriter rejects a pre-created extensionless tmpfile,
    # so we use a unique non-pre-created .pt sibling and os.replace it into place.
    tmp = f"{path}.tmp{os.getpid()}.pt"
    torch.save(ck, tmp)
    os.replace(tmp, path)


save_bufs_ref = [None]      # set to the live buffer list inside main (kept out of signatures)


# ───────────────────────────────────────────────────────────────────────────
#  Machine guard: pause on RSS/disk pressure instead of dying
# ───────────────────────────────────────────────────────────────────────────
def guard(args, status_path, status):
    rss = _rss_gb()
    free = shutil.disk_usage(".").free / 1e9
    if rss <= args.pause_rss_gb and free >= args.pause_disk_gb:
        return rss, free
    reason = f"rss={rss:.2f}GB" if rss > args.pause_rss_gb else f"disk_free={free:.1f}GB"
    print(f"[guard] PAUSING ({reason})", flush=True)
    t_pause = time.time()
    while True:
        status["phase"] = "paused"; status["pause_reason"] = reason
        write_atomic(status_path, status)
        time.sleep(60)
        rss = _rss_gb(); free = shutil.disk_usage(".").free / 1e9
        if rss <= args.pause_rss_gb and free >= args.pause_disk_gb + 0.5:
            print(f"[guard] resuming after {time.time()-t_pause:.0f}s pause", flush=True)
            status["phase"] = "running"; status.pop("pause_reason", None)
            return rss, free
        if rss > args.pause_rss_gb and time.time() - t_pause > 1800:
            # our own RSS will not shrink by waiting — save state and exit cleanly
            print(f"[guard] RSS still {rss:.2f}GB after 30min — clean exit", flush=True)
            status["phase"] = "stopped_rss"
            write_atomic(status_path, status)
            sys.exit(2)


# ───────────────────────────────────────────────────────────────────────────
#  Main
# ───────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="POS: plasticity-on-surprise, 3 arms, 1 stream")
    ap.add_argument("--hours", type=float, default=40.0)
    ap.add_argument("--target-tokens", type=int, default=0, help="stop after N streamed tokens (0=off; smoke/G2)")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--q", type=float, default=0.80, help="gating quantile on the rolling window")
    ap.add_argument("--window", type=int, default=500, help="rolling window length (chunks)")
    ap.add_argument("--min-window", type=int, default=100)
    ap.add_argument("--ignition-chunks", type=int, default=100, help="always-backward warmup chunks")
    ap.add_argument("--eval-tokens", type=int, default=200_000)
    ap.add_argument("--eval-every-s", type=float, default=900.0,
                    help="wall-clock eval cadence (long run; throughput varies with machine load)")
    ap.add_argument("--eval-every-tokens", type=int, default=0,
                    help=">0 switches to token-scheduled evals (deterministic; smoke/G2)")
    ap.add_argument("--ckpt-every-s", type=float, default=600.0)
    ap.add_argument("--status-every-s", type=float, default=60.0)
    ap.add_argument("--twin-fork-hours", type=float, default=24.0, help="0 = twin off")
    ap.add_argument("--twin-fork-tokens", type=int, default=0, help="token-scheduled fork (smoke)")
    ap.add_argument("--index-warmup-tokens", type=int, default=5_000_000)
    ap.add_argument("--recur-min-hours", type=float, default=2.0)
    ap.add_argument("--spike-min-nll", type=float, default=7.0)
    ap.add_argument("--span-half", type=int, default=32)
    ap.add_argument("--lookahead", type=int, default=16)
    ap.add_argument("--max-index-entries", type=int, default=20000)
    ap.add_argument("--max-probes-per-hour", type=int, default=30)
    ap.add_argument("--pause-rss-gb", type=float, default=12.0)
    ap.add_argument("--pause-disk-gb", type=float, default=5.0)
    ap.add_argument("--no-index", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tag", default="", help="output-file tag (smoke runs must not clobber the real run)")
    args = ap.parse_args()

    tag = f"{args.tag}_" if args.tag else ""
    P = lambda stem: os.path.join("results", f"pos_{tag}{stem}")
    status_path, metrics_path = P("status.json"), P("metrics.jsonl")
    chunks_path, ckpt_path = P("chunks.jsonl"), P("ckpt.pt")
    os.makedirs("results", exist_ok=True)

    print(f"[pos] 3-arm plasticity-on-surprise | seed {args.seed} q={args.q} "
          f"window={args.window} B={args.batch} K={args.chunk} d={args.d_model} "
          f"| hours={args.hours} target={args.target_tokens or '∞'}", flush=True)

    # data: vocab from WT-2 train; frozen eval = fixed WT-2 val slice (never streamed)
    train_text, val_text = load_wikitext2()
    vocab, stoi, unk, mask = build_vocab(train_text)
    V = len(vocab)
    val_ids = tokenize(val_text, stoi, unk)
    evX, evY = build_eval_set(val_ids, args.eval_tokens, args.chunk)
    print(f"[data] vocab {V} | frozen eval {evY.numel():,} WT-2 val tokens "
          f"({len(evX)} stateless rows)", flush=True)

    arms = [build_arm(n, V, mask, args) for n in ("A1", "A2", "A3")]
    with torch.no_grad():
        p0 = [sum(float(p.abs().sum()) for p in a["model"].parameters()) for a in arms]
    assert max(p0) - min(p0) == 0.0, f"arm inits differ: {p0}"
    print(f"[arms] A1/A2/A3 identical init (param-abs-sum {p0[0]:.4f})", flush=True)

    index = None
    if IndexLoop is not None and not args.no_index:
        index = IndexLoop(P("index.jsonl"), P("probes.jsonl"),
                          warmup_tokens=args.index_warmup_tokens,
                          recur_min_hours=args.recur_min_hours,
                          spike_min_nll=args.spike_min_nll, span_half=args.span_half,
                          lookahead=args.lookahead, max_entries=args.max_index_entries,
                          max_probes_per_hour=args.max_probes_per_hour, seed=args.seed)
        print(f"[index] live (warmup {args.index_warmup_tokens:,} tok, "
              f"recur ≥{args.recur_min_hours}h)", flush=True)
    else:
        print("[index] OFF" + ("" if args.no_index else " (pos_index.py not found)"), flush=True)

    B, K = args.batch, args.chunk
    stream = C4Stream(stoi, unk)
    bufs = [[] for _ in range(B)]
    save_bufs_ref[0] = bufs
    n_tok, wall_off, twin = 0, 0.0, None

    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, weights_only=False)
        n_tok, wall_off = ck["n_streamed"], ck["wall_s"]
        stream.docs, stream.pending = ck["docs"], ck["pending"]
        del bufs[:]; bufs.extend(ck["bufs"])
        torch.set_rng_state(ck["torch_rng"])
        names = list(ck["arms"].keys())
        if "A3R" in names and fork_twin is not None:
            twin = fork_twin([a for a in arms if a["name"] == "A3"][0], args)
            arms.append(twin)
        for a in arms:
            s = ck["arms"][a["name"]]
            a["model"].load_state_dict(s["model"])
            if a["opt"] and s["opt"]: a["opt"].load_state_dict(s["opt"])
            a["states"] = s["states"]
            a["grad_tokens"], a["n_chunks"], a["n_bwd"] = s["grad_tokens"], s["n_chunks"], s["n_bwd"]
            a["window"] = deque(s["window"], maxlen=args.window)
            a["recent_gates"] = deque(s["recent_gates"], maxlen=200)
        if index and ck.get("index_state"):
            index.load_state_dict(ck["index_state"])
        dropped = (trim_jsonl(chunks_path, "n", n_tok)
                   + trim_jsonl(metrics_path, "n_streamed", n_tok)
                   + trim_jsonl(P("index.jsonl"), "n", n_tok)
                   + trim_jsonl(P("probes.jsonl"), "n", n_tok))
        print(f"[resume] n={n_tok:,} wall={wall_off:.0f}s docs={stream.docs:,} "
              f"arms={[a['name'] for a in arms]}"
              + (f" | trimmed {dropped} post-ckpt log rows" if dropped else ""), flush=True)
    else:
        for b in range(B):
            bufs[b].extend(stream.next_block())

    digest = hashlib.sha256()
    t0 = time.time()
    wall = lambda: time.time() - t0 + wall_off
    tok_mode = args.eval_every_tokens > 0
    next_eval = (((n_tok // args.eval_every_tokens) + 1) * args.eval_every_tokens
                 if tok_mode and n_tok else args.eval_every_tokens)
    last_ckpt = last_status = last_eval = time.time()
    status = {"pid": os.getpid(), "phase": "running", "config": vars(args),
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    last_eval_n = [-1]

    def run_eval():
        if last_eval_n[0] == n_tok:
            return None                                 # already evaluated at this point
        last_eval_n[0] = n_tok
        rec = {"type": "eval", "wall_s": round(wall(), 1), "n_streamed": n_tok,
               "rss_gb": round(_rss_gb(), 3), "arms": {}}
        for a in arms:
            hl = heldout(a["model"], evX, evY)
            e = {"heldout": round(hl, 6), "grad_tokens": a["grad_tokens"]}
            if a["name"] in ("A3", "A3R"):
                e["gate_frac_recent"] = round(sum(a["recent_gates"]) / max(1, len(a["recent_gates"])), 4)
                e["gate_frac_cum"] = round(a["n_bwd"] / max(1, a["n_chunks"]), 4)
            rec["arms"][a["name"]] = e
            digest.update(f"E:{n_tok}:{a['name']}:{hl:.6f}:{a['grad_tokens']}".encode())
        append_jsonl(metrics_path, rec)
        line = " | ".join(f"{a['name']} {rec['arms'][a['name']]['heldout']:.4f}" for a in arms)
        print(f"[eval] n={n_tok:>12,} wall={wall()/3600:6.2f}h rss={rec['rss_gb']:.2f}GB | {line} "
              f"| A3 grad {arms[2]['grad_tokens']:,} "
              f"({100*arms[2]['grad_tokens']/max(1,n_tok):.1f}%)", flush=True)
        return rec

    def write_status(extra=None):
        st = dict(status)
        st.update({"wall_s": round(wall(), 1), "n_streamed": n_tok,
                   "tok_per_s": round(n_tok / max(1e-9, wall()), 1),
                   "rss_gb": round(_rss_gb(), 3),
                   "disk_free_gb": round(shutil.disk_usage(".").free / 1e9, 1),
                   "stream_docs": stream.docs, "stream_reconnects": stream.reconnects,
                   "arms": {a["name"]: {"grad_tokens": a["grad_tokens"], "n_chunks": a["n_chunks"],
                                        "n_bwd": a["n_bwd"]} for a in arms},
                   "index": index.stats() if index else None,
                   "twin_forked": twin is not None})
        if extra:
            st.update(extra)
        write_atomic(status_path, st)

    if n_tok == 0:
        run_eval()                                      # baseline at n=0 (all arms identical)
    write_status()

    a1, a2, a3 = arms[0], arms[1], arms[2]
    step = 0
    print("[pos] streaming...", flush=True)
    while True:
        if args.target_tokens and n_tok >= args.target_tokens:
            break
        if not args.target_tokens and wall() >= args.hours * 3600:
            break
        if step % 50 == 0:
            guard(args, status_path, status)

        for b in range(B):
            while len(bufs[b]) < K + 1:
                bufs[b].extend(stream.next_block())
        x = torch.tensor([bufs[b][:K] for b in range(B)], dtype=torch.long)
        y = torch.tensor([bufs[b][1:K + 1] for b in range(B)], dtype=torch.long)
        for b in range(B):
            del bufs[b][:K]
        n_tok += B * K

        sp3 = None if a3["states"] is None else [s.clone() for s in a3["states"]]
        s1 = step_a1(a1, x.clone(), y.clone())
        s2 = step_a2(a2, x.clone(), y.clone())
        s3, g3, th3, nll3 = step_gated(a3, x.clone(), y.clone(), args)
        rec = {"n": n_tok, "w": round(wall(), 1), "s1": round(s1, 6), "s2": round(s2, 6),
               "s3": round(s3, 6), "g": int(g3), "th": None if th3 is None else round(th3, 6)}
        digest.update(f"C:{n_tok}:{s1:.6f}:{s2:.6f}:{s3:.6f}:{int(g3)}".encode())
        if twin is not None:
            s3r, g3r, th3r, _ = step_gated(twin, x.clone(), y.clone(), args)
            rec["s3r"], rec["gr"] = round(s3r, 6), int(g3r)
            digest.update(f"T:{s3r:.6f}:{int(g3r)}".encode())
        append_jsonl(chunks_path, rec)

        if index is not None:
            index.observe_chunk(a3["model"], sp3, x, y, nll3, n_tok, wall(), unk)

        if twin is None and fork_twin is not None and (
                (args.twin_fork_tokens and n_tok >= args.twin_fork_tokens) or
                (not args.twin_fork_tokens and args.twin_fork_hours > 0
                 and wall() >= args.twin_fork_hours * 3600)):
            twin = fork_twin(a3, args)
            arms.append(twin)
            append_jsonl(metrics_path, {"type": "event", "event": "twin_fork",
                                        "wall_s": round(wall(), 1), "n_streamed": n_tok})
            print(f"[twin] forked A3 -> A3R at n={n_tok:,} wall={wall()/3600:.2f}h "
                  f"(weights copied, Z + optimizer zeroed)", flush=True)

        now = time.time()
        if tok_mode:
            if n_tok >= next_eval:
                run_eval()
                next_eval += args.eval_every_tokens
        elif now - last_eval >= args.eval_every_s:
            run_eval()
            last_eval = time.time()
        if now - last_ckpt >= args.ckpt_every_s:
            save_ckpt(ckpt_path, arms, stream, n_tok, wall(), index, torch.get_rng_state(), args)
            last_ckpt = now
        if now - last_status >= args.status_every_s:
            write_status()
            last_status = now
        step += 1

    final = run_eval()
    save_ckpt(ckpt_path, arms, stream, n_tok, wall(), index, torch.get_rng_state(), args)
    dig = digest.hexdigest()
    status["phase"] = "done"
    write_status({"digest": dig, "final": final})
    print(f"\n[pos] DONE n={n_tok:,} wall={wall()/3600:.2f}h  digest={dig}", flush=True)
    print(f"[pos] A3 gradient tokens {a3['grad_tokens']:,} "
          f"({100*a3['grad_tokens']/max(1,n_tok):.1f}% of stream) vs A2 {a2['grad_tokens']:,}", flush=True)


if __name__ == "__main__":
    main()
