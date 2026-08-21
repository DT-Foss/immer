#!/usr/bin/env python3 -u
"""
LIVING-STREAM — constant-memory streaming TRAINING on the O(1) GSSM.
====================================================================
The training-side dual of the proven O(1) EVAL flag (1B tokens at 4.36 GB constant).
The eval harness substitutes a left-context `overlap` for carried state; TRAINING instead
carries the persistent per-layer state `Z` across chunks and cuts the graph with `.detach()`
(truncated BPTT). Because the receptive field is ~5-8 tokens, that truncation is near-exact —
which we MEASURE, not assume (grad-cosine vs full-window BPTT), before any long run.

ONE load-bearing edit, on the SAME NoPE model that streamed a billion tokens:
  - a stateful scan that accepts an incoming Z and returns the final Z (exact reference recurrence)
  - an LM wrapper that threads a per-layer list[Z] in and out
That single persistent object — Z ∈ (B, n_heads, d_head), ~16 KB — is the substrate for:
  (A) constant-memory training      — carry Z.detach() across chunks; loss falls, RSS flat
  (B) source hot-swap mid-stream    — frozen-state twin: carried vs reset recovery
  (C) γ-gate short/long memory       — per-head spectrum, init vs trained
  (D) idle-persistence               — plant a bit, gap the input, read it back; zero Z at the gap = null

The reference module is FROZEN (chmod 444); we subclass, never edit it.

This file builds the CORE first (the §0 edit + grad-exactness check + the A smoke loop).
B/C/D probes are added on top of the same trained model.
"""
import os, sys, json, argparse, time, threading, signal  # noqa: F401 (json/time/threading/signal used in train+watchdog)
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

import torch
import torch.nn as nn
torch.backends.mps.is_available = lambda: False          # force CPU (same as the 1B run)
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

from moebius_scan_transformer_selective import (SelectiveRapiditySqrtScanLayer,
                                                EPS, LOG_COMPLEMENT_CLAMP)
from length_extrap_v2 import SelectiveNoPETransformerLM


# ───────────────────────────────────────────────────────────────────────────
# §0  The one load-bearing edit: a STATEFUL scan (carry Z in, return Z out).
#     Bit-for-bit the reference recurrence z_t = γ_t·z_{t-1} + a_t, only with a
#     non-zero initial Z0 and the final Z returned for the next chunk.
# ───────────────────────────────────────────────────────────────────────────
def stateful_scan(a, gamma, Z0):
    """z_t = γ_t·z_{t-1} + a_t with carried state. a,gamma:(B,T,H,D); Z0:(B,H,D)|None.
    Returns (Z_seq:(B,T,H,D), Z_final:(B,H,D)). Identical to reference
    sequential_linear_scan when Z0 is None (zeros)."""
    B, T, H, D = a.shape
    Z = torch.zeros(B, H, D, dtype=a.dtype) if Z0 is None else Z0
    out = []
    for t in range(T):
        Z = gamma[:, t] * Z + a[:, t]
        out.append(Z)
    return torch.stack(out, dim=1), Z


class StreamingScanLayer(SelectiveRapiditySqrtScanLayer):
    """Causal-only streaming forward: identical v/gate/γ/α/a math as the reference,
    but threads Z in and out. Reuses the parent's weights (we rebuild from a trained
    parent by copying its state_dict), so the math is the SAME operator."""
    def forward(self, x, Z_in=None, return_internals=False):
        B, T, D = x.shape
        v = torch.tanh(self.W_v(x))
        gate = torch.sigmoid(self.W_gate(x))
        gamma = torch.sigmoid(self.W_gamma(x))
        alpha = torch.sigmoid(self.W_alpha(x))
        v_gated = v * gate
        if self.dropout is not None:
            v_gated = self.dropout(v_gated)
        v_gated = v_gated.view(B, T, self.n_heads, self.d_head)
        gamma = gamma.view(B, T, self.n_heads, self.d_head)
        alpha = alpha.view(B, T, self.n_heads, self.d_head)
        w = torch.clamp(v_gated * v_gated, max=LOG_COMPLEMENT_CLAMP)
        a = alpha * torch.log(1.0 - w + EPS)
        Z_seq, Z_fin = stateful_scan(a, gamma, Z_in)        # <<< carried state
        s_sq = torch.clamp(1.0 - torch.exp(Z_seq), min=0.0)
        state = torch.sqrt(s_sq + EPS).view(B, T, self.n_heads * self.d_head)
        out = self.W_out(state)
        if return_internals:
            return out, Z_fin, {"gamma": gamma, "alpha": alpha, "a": a, "Z_seq": Z_seq}
        return out, Z_fin


class StreamingNoPELM(SelectiveNoPETransformerLM):
    """The billion-token NoPE model, made stateful. Swaps each layer's `.scan` for a
    StreamingScanLayer (copying weights), and threads a per-layer list[Z] through forward.
    NoPE guard: self.pos must be Identity (no absolute-position confound)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.pos, nn.Identity), \
            f"NoPE required (pos must be Identity), got {type(self.pos)}"
        # replace each scan with a streaming twin carrying the SAME weights
        for layer in self.layers:
            old = layer.scan
            new = StreamingScanLayer(old.d_model, d_head=old.d_head, n_heads=old.n_heads,
                                     causal=old.causal,
                                     dropout=(old.dropout.p if old.dropout is not None else 0.0))
            new.load_state_dict(old.state_dict())
            layer.scan = new

    def forward(self, x, states=None):
        assert isinstance(self.pos, nn.Identity)
        h = self.embed(x)                                    # pos = Identity → plain embedding
        si = states if states is not None else [None] * len(self.layers)
        new_states = []
        for layer, z in zip(self.layers, si):
            y, z_out = layer.scan(h, z)                      # stateful scan
            h = layer.ln1(h + y)
            h = layer.ln2(h + layer.ffn(h))
            new_states.append(z_out)
        return self.head(h), new_states


# ───────────────────────────────────────────────────────────────────────────
#  Equivalence + grad-exactness checks (HIGHEST blast radius — run before any long loop).
# ───────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def check_equivalence(vocab_size, mask_idx, seed=0):
    """A single full-window forward through the streaming model with states=None must
    equal the original NoPE model's forward (same weights) — proves the stateful scan is
    the SAME operator when started from zero state."""
    torch.manual_seed(seed)
    base = SelectiveNoPETransformerLM(vocab_size, mask_idx, d_model=128, n_layers=2,
                                      n_heads=4, d_head=32, seq_len=32, dropout=0.0, causal=True).eval()
    stream = StreamingNoPELM(vocab_size, mask_idx, d_model=128, n_layers=2,
                             n_heads=4, d_head=32, seq_len=32, dropout=0.0, causal=True).eval()
    stream.load_state_dict(_remap_to_stream(base.state_dict(), stream), strict=False)
    x = torch.randint(0, vocab_size, (2, 48))
    out_base = base(x)
    out_stream, _ = stream(x, None)
    return float((out_base - out_stream).abs().max())


def _remap_to_stream(base_sd, stream_model):
    """Copy matching params from a base NoPE model into the streaming model (the scan
    weights live under layer.scan.* in both; the streaming scan is a subclass so names match)."""
    tgt = stream_model.state_dict()
    out = {}
    for k, v in base_sd.items():
        if k in tgt and tgt[k].shape == v.shape:
            out[k] = v
    return out


def check_grad_exactness(vocab_size, mask_idx, chunk=64, overlap=16, full_T=512, seed=0):
    """Measure how exact truncated-BPTT-with-carried-state is vs full-window BPTT.
    Full: backprop through one full_T-window. Truncated: split into `chunk` pieces,
    carry Z.detach() across them, but recompute `overlap` left-context tokens per chunk
    (warmup, not carried) to absorb the chunk-boundary discontinuity. Report cosine of
    the two gradient vectors over all params. Target > 0.95."""
    torch.manual_seed(seed)
    model = StreamingNoPELM(vocab_size, mask_idx, d_model=128, n_layers=2, n_heads=4,
                            d_head=32, seq_len=32, dropout=0.0, causal=True)
    x = torch.randint(0, vocab_size, (1, full_T))
    y = torch.randint(0, vocab_size, (1, full_T))
    lossf = nn.CrossEntropyLoss()

    # full-window grad
    model.zero_grad()
    logits, _ = model(x, None)
    lossf(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).backward()
    g_full = torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])

    # truncated grad with carried state + overlap warmup
    model.zero_grad()
    states = None
    pos = 0
    while pos < full_T:
        lo = max(0, pos - overlap)
        hi = min(full_T, pos + chunk)
        xc = x[:, lo:hi]
        yc = y[:, lo:hi]
        logits, states = model(xc, states)
        # score only the NEW region [pos, hi); the [lo,pos) overlap is warmup context
        score_from = pos - lo
        lg = logits[:, score_from:, :]
        tg = yc[:, score_from:]
        (lossf(lg.reshape(-1, lg.size(-1)), tg.reshape(-1)) * (hi - pos) / full_T).backward(retain_graph=False)
        states = [s.detach() for s in states]
        pos = hi
    g_trunc = torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])

    cos = float(torch.nn.functional.cosine_similarity(g_full, g_trunc, dim=0))
    rel = float((g_full - g_trunc).norm() / (g_full.norm() + 1e-12))
    return cos, rel


# ───────────────────────────────────────────────────────────────────────────
#  (A) Constant-memory streaming TRAINING loop.
#      Train from scratch on a streamed corpus; carry Z.detach() across chunks.
#      Held-out loss (never in the stream) must FALL; RSS must stay FLAT.
# ───────────────────────────────────────────────────────────────────────────
def _graph_depth(t, cap=100000):
    """Length of the autograd grad-fn chain hanging off tensor `t` (how many ops deep the
    retained graph is). With detach this is ~constant per step; without detach it grows every
    step — the direct fingerprint of an unbounded graph."""
    seen, depth, frontier = set(), 0, []
    fn = getattr(t, "grad_fn", None)
    if fn is not None:
        frontier.append((fn, 0))
    while frontier:
        node, d = frontier.pop()
        if id(node) in seen or d > cap:
            continue
        seen.add(id(node)); depth = max(depth, d)
        for nxt, _ in getattr(node, "next_functions", ()):
            if nxt is not None:
                frontier.append((nxt, d + 1))
    return depth


def _start_watchdog(out, hard_gb):
    def _watch():
        while True:
            if _rss_gb() > hard_gb:
                open(out + ".WATCHDOG_KILL", "w").write(f"rss={_rss_gb():.2f}GB\n")
                os.kill(os.getpid(), signal.SIGKILL)
            time.sleep(0.5)
    threading.Thread(target=_watch, daemon=True).start()


def c4_block_stream(stoi, unk, block=65536):
    """Lazily yield token blocks from C4 streaming (same machinery as the 1B run)."""
    from datasets import load_dataset
    from length_extrap_v2 import tokenize
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    pending = []
    for row in ds:
        t = row["text"] if isinstance(row, dict) else ""
        if not t.strip():
            continue
        pending.extend(tokenize(t, stoi, unk))
        while len(pending) >= block:
            out, pending = pending[:block], pending[block:]
            yield out
    if pending:
        yield pending


@torch.no_grad()
def heldout_loss(model, ids, chunk, mask_idx, device, max_tokens=20000):
    """Causal next-token loss on a fixed held-out id list (eval mode, stateless per chunk)."""
    model.eval()
    lossf = nn.CrossEntropyLoss()
    tot, n = 0.0, 0
    pos = 0
    ids = ids[:max_tokens]
    while pos + 1 < len(ids):
        seg = ids[pos:pos + chunk + 1]
        if len(seg) < 2:
            break
        x = torch.tensor(seg[:-1], dtype=torch.long).unsqueeze(0).to(device)
        y = torch.tensor(seg[1:], dtype=torch.long).unsqueeze(0).to(device)
        logits, _ = model(x, None)
        tot += float(lossf(logits.reshape(-1, logits.size(-1)), y.reshape(-1))) * (len(seg) - 1)
        n += len(seg) - 1
        pos += chunk
    model.train()
    return tot / max(1, n)


def streaming_train(args):
    from length_extrap_v2 import (load_wikitext2, build_vocab, tokenize)
    dev = torch.device("cpu")
    _start_watchdog(args.out, args.mem_hard_gb)
    print(f"[safety] watchdog {args.mem_hard_gb}GB; CONSTANT-MEMORY streaming TRAIN "
          f"(chunk={args.chunk}, B={args.batch}, detach={not args.no_detach})")

    # vocab from WT-2 (cheap, deterministic); held-out = WT-2 val (NEVER streamed)
    train_text, val_text = load_wikitext2()
    vocab, stoi, unk, mask = build_vocab(train_text)
    V = len(vocab)
    val_ids = tokenize(val_text, stoi, unk)
    print(f"  vocab={V}, held-out (WT-2 val) = {len(val_ids):,} tokens (never in the train stream)")

    torch.manual_seed(args.seed)
    model = StreamingNoPELM(V, mask, d_model=args.d_model, n_layers=2, n_heads=4,
                            d_head=args.d_model // 4, seq_len=32, dropout=0.0, causal=True).to(dev)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    lossf = nn.CrossEntropyLoss()

    B, K, OV = args.batch, args.chunk, args.overlap
    bufs = [[] for _ in range(B)]
    block_iter = c4_block_stream(stoi, unk)
    # prime: give each buffer a private block
    for b in range(B):
        bufs[b].extend(next(block_iter))

    states = None
    n_tok = 0
    peak = _rss_gb()
    curve = []
    base_hl = heldout_loss(model, val_ids, K, mask, dev)
    print(f"  step 0: held-out loss {base_hl:.3f}  rss {peak:.1f}GB")
    t0 = time.time()
    step = 0
    while n_tok < args.target_tokens:
        # refill any short buffer (bounded — never grows past one block + K)
        for b in range(B):
            while len(bufs[b]) < K + 1:
                try:
                    bufs[b].extend(next(block_iter))
                except StopIteration:
                    block_iter = c4_block_stream(stoi, unk)  # loop corpus if exhausted
                    bufs[b].extend(next(block_iter))
        x = torch.tensor([bufs[b][:K] for b in range(B)], dtype=torch.long, device=dev)
        y = torch.tensor([bufs[b][1:K + 1] for b in range(B)], dtype=torch.long, device=dev)
        for b in range(B):
            del bufs[b][:K]                                   # advance → CONSTANT buffer size
        n_tok += B * K

        logits, states = model(x, states)
        loss = lossf(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        if args.no_detach:
            # CONTROL: keep the state ATTACHED across steps. The autograd graph then chains every
            # chunk to all previous chunks and can never be freed — its memory grows without bound.
            # We measure that growth directly (graph depth via the live state's grad-fn chain, and
            # RSS) rather than optimizing, because attaching the state across an opt.step() is itself
            # impossible (in-place weight update vs retained graph). Either way the lesson is the
            # same: WITHOUT detach there is no constant-memory streaming — the graph is unbounded.
            depth = _graph_depth(states[0])
            peak = max(peak, _rss_gb())
            del logits, loss
            step += 1
            if step % args.eval_every == 0:
                curve.append((n_tok, depth, round(peak, 3)))
                print(f"    tok {n_tok:>10,} | graph-depth {depth:>5} | rss {peak:5.2f}GB "
                      f"(NO-DETACH control: graph unbounded)", flush=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        states = [s.detach() for s in states]                # THE CARRY: cut graph → constant memory
        peak = max(peak, _rss_gb())
        del logits, loss
        step += 1

        if step % args.eval_every == 0:
            hl = heldout_loss(model, val_ids, K, mask, dev)
            curve.append((n_tok, round(hl, 4), round(peak, 3)))
            print(f"    tok {n_tok:>10,} | held-out {hl:6.3f} | rss {peak:5.2f}GB "
                  f"| {n_tok/(time.time()-t0):,.0f} tok/s", flush=True)

    dt = time.time() - t0
    if args.no_detach:
        # control summary: graph depth + RSS grew (the point — no constant memory without detach)
        d0, dN = curve[0][1], curve[-1][1]
        r0, rN = curve[0][2], curve[-1][2]
        results = {"control": "no_detach", "tokens": n_tok, "elapsed_s": round(dt, 1),
                   "graph_depth_start": d0, "graph_depth_end": dN,
                   "rss_start_gb": r0, "rss_end_gb": rN, "curve": curve}
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"\n→ {args.out}")
        print(f"NO-DETACH CONTROL: graph depth {d0}→{dN} (grows every step), "
              f"RSS {r0:.2f}→{rN:.2f}GB. Without detach the graph is UNBOUNDED — "
              f"this is exactly what the .detach() carry removes.")
        return
    # honest slopes: held-out must fall, RSS must be flat
    import numpy as np
    toks = np.array([c[0] for c in curve], dtype=float)
    hls = np.array([c[1] for c in curve], dtype=float)
    rsss = np.array([c[2] for c in curve], dtype=float)
    half = len(curve) // 2
    hl_slope = float(np.polyfit(toks[half:], hls[half:], 1)[0]) if len(curve) > 3 else 0.0
    rss_slope = float(np.polyfit(toks, rsss, 1)[0]) * 1e7 if len(curve) > 1 else 0.0  # GB / 10M tok
    results = {"d_model": args.d_model, "batch": B, "chunk": K, "overlap": OV,
               "detach": not args.no_detach, "tokens": n_tok, "elapsed_s": round(dt, 1),
               "base_heldout": round(base_hl, 4), "final_heldout": round(curve[-1][1], 4),
               "heldout_slope_back_half": hl_slope, "rss_slope_gb_per_10M": round(rss_slope, 6),
               "peak_rss_gb": round(peak, 3), "curve": curve}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    # save the trained model + vocab so C/D probes run on the SAME trained weights
    ckpt = args.out.replace(".json", ".pt")
    torch.save({"state_dict": model.state_dict(), "vocab_size": V, "mask_idx": mask,
                "d_model": args.d_model, "stoi": stoi, "unk": unk}, ckpt)
    print(f"\n→ {args.out}\n→ {ckpt} (trained model for C/D probes)")
    print(f"HEADLINE: trained on {n_tok:,} streamed tokens — held-out loss "
          f"{base_hl:.3f}→{curve[-1][1]:.3f} (slope {hl_slope:+.2e}/tok), "
          f"RSS {peak:.2f}GB (slope {rss_slope:+.2e} GB/10M tok). "
          f"{'FLAT MEMORY + LEARNING' if hl_slope < 0 and abs(rss_slope) < 0.05 else 'check slopes'}")


# ───────────────────────────────────────────────────────────────────────────
#  (C) γ-gate head SPECTRUM — short vs long memory, per (layer, head).
#      Read-only: reconstruct γ_t = sigmoid(W_gamma x) per head, init vs trained.
#      A long-memory head has γ≈1 (τ=1/(1−γ) large); a short-memory head γ≈0.2 (τ≈1).
# ───────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def gamma_spectrum(model, ids, device, n_tok=4000):
    """Per-(layer,head) mean γ over a sample of real tokens. Returns list of
    arrays [layer][head] = mean γ, plus the implied timescale τ=1/(1−γ)."""
    model.eval()
    x = torch.tensor(ids[:n_tok], dtype=torch.long).unsqueeze(0).to(device)
    h = model.embed(x)
    spec = []
    for layer in model.layers:
        sc = layer.scan
        g = torch.sigmoid(sc.W_gamma(h))                      # (1,T,H*D)
        g = g.view(1, -1, sc.n_heads, sc.d_head).mean(dim=(0, 1, 3))  # per-head mean γ
        spec.append(g.cpu().numpy())
        # advance h through this layer so the next layer sees real activations
        y, _ = sc(h, None)
        h = layer.ln1(h + y); h = layer.ln2(h + layer.ffn(h))
    return spec


def run_gamma_spectrum(args):
    import numpy as np
    from length_extrap_v2 import load_wikitext2, build_vocab, tokenize
    dev = torch.device("cpu")
    ck = torch.load(args.ckpt, weights_only=False)
    train_text, val_text = load_wikitext2()
    vocab, stoi, unk, mask = build_vocab(train_text)
    val_ids = tokenize(val_text, stoi, unk)
    V = ck["vocab_size"]

    torch.manual_seed(0)
    trained = StreamingNoPELM(V, ck["mask_idx"], d_model=ck["d_model"], n_layers=2,
                              n_heads=4, d_head=ck["d_model"] // 4, seq_len=32,
                              dropout=0.0, causal=True).to(dev)
    trained.load_state_dict(ck["state_dict"])
    torch.manual_seed(0)
    untrained = StreamingNoPELM(V, ck["mask_idx"], d_model=ck["d_model"], n_layers=2,
                                n_heads=4, d_head=ck["d_model"] // 4, seq_len=32,
                                dropout=0.0, causal=True).to(dev)

    sp_tr = gamma_spectrum(trained, val_ids, dev)
    sp_un = gamma_spectrum(untrained, val_ids, dev)
    out = {"trained": [], "untrained": []}
    print("── γ-spectrum: per-(layer,head) mean forget-gate, init vs trained ──")
    for li in range(len(sp_tr)):
        for hi in range(len(sp_tr[li])):
            gt, gu = float(sp_tr[li][hi]), float(sp_un[li][hi])
            tau_t = 1.0 / max(1e-3, 1 - gt)
            out["trained"].append({"layer": li, "head": hi, "gamma": round(gt, 4), "tau": round(tau_t, 1)})
            out["untrained"].append({"layer": li, "head": hi, "gamma": round(gu, 4)})
            print(f"   L{li}H{hi}: trained γ={gt:.3f} (τ≈{tau_t:5.1f} tok)  | init γ={gu:.3f}")
    tr_g = np.array([d["gamma"] for d in out["trained"]])
    un_g = np.array([d["gamma"] for d in out["untrained"]])
    out["trained_spread"] = round(float(tr_g.max() - tr_g.min()), 4)
    out["untrained_spread"] = round(float(un_g.max() - un_g.min()), 4)
    out["trained_tau_max"] = round(float(1.0 / max(1e-3, 1 - tr_g.max())), 1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n→ {args.out}")
    print(f"SPECTRUM: trained γ-spread {out['trained_spread']:.3f} (init {out['untrained_spread']:.3f}); "
          f"longest-memory head τ≈{out['trained_tau_max']:.0f} tokens. "
          f"{'LEARNED a short/long-memory spectrum' if out['trained_spread'] > out['untrained_spread'] + 0.05 else 'spectrum ~ init (check)'}")


# ───────────────────────────────────────────────────────────────────────────
#  (D) IDLE-PERSISTENCE — "the state lives through the silence".
#      Train a tiny model on: [beacon β_k][G filler tokens, no beacon][probe ?] → predict k.
#      One bit, 2-way (inside the proven bounded-scalar regime — NOT multi-key recall).
#      Then test recall vs gap length G, with the DECISIVE null: zero the carried state at the
#      gap start. PASS requires carried-state recall to hold while zeroed-state collapses to chance
#      — proving the answer rode the persistent state across the input gap, not local context.
# ───────────────────────────────────────────────────────────────────────────
def _beacon_vocab():
    # minimal synthetic vocab: 0=filler-base ... we use F filler ids + 2 beacons + 1 probe
    F = 16
    fillers = list(range(F))
    beta0, beta1, probe = F, F + 1, F + 2
    V = F + 3
    return F, fillers, beta0, beta1, probe, V


def _make_beacon_batch(B, G, gen, F, beta0, beta1, probe):
    """B trials of [β_k][G random fillers][probe]. Returns x (B, G+2), labels k (B,)."""
    k = torch.randint(0, 2, (B,), generator=gen)
    beac = torch.where(k == 0, torch.tensor(beta0), torch.tensor(beta1))
    fill = torch.randint(0, F, (B, G), generator=gen)
    pr = torch.full((B, 1), probe)
    x = torch.cat([beac.unsqueeze(1), fill, pr], dim=1)
    return x, k


def run_idle_persistence(args):
    import numpy as np
    dev = torch.device("cpu")
    _start_watchdog(args.out, args.mem_hard_gb)
    F, fillers, beta0, beta1, probe, V = _beacon_vocab()
    gen = torch.Generator().manual_seed(args.seed)

    # small model trained on the beacon task at a TRAINING gap, then tested at longer gaps
    torch.manual_seed(args.seed)
    model = StreamingNoPELM(V, V - 1, d_model=64, n_layers=2, n_heads=4, d_head=16,
                            seq_len=32, dropout=0.0, causal=True).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    lossf = nn.CrossEntropyLoss()
    Gmax = args.train_gap
    # CURRICULUM: start at a tiny gap (easy to ignite — the onset lesson from DeltaNet) and grow
    # the training gap toward Gmax as accuracy clears a bar. Learning to hold the bit at G=4 first
    # makes holding it at G=128 reachable; from-scratch at a large gap never ignites.
    print(f"── (D) training beacon-recall with curriculum gap → {Gmax} (1 bit, 2-way) ──")
    model.train()
    Gcur = min(4, Gmax)
    for it in range(args.iters):
        x, k = _make_beacon_batch(args.batch, Gcur, gen, F, beta0, beta1, probe)
        logits, _ = model(x.to(dev), None)               # stateless within a trial (full seq)
        pred = logits[:, -1, :2]                          # readout at the probe: 2-way
        loss = lossf(pred, k.to(dev))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        acc = float((pred.argmax(-1).cpu() == k).float().mean())
        if acc > 0.95 and Gcur < Gmax:                   # mastered current gap → grow it
            Gcur = min(Gmax, int(Gcur * 1.5) + 1)
        if (it + 1) % max(1, args.iters // 8) == 0:
            print(f"   it {it+1:>4}: loss {float(loss):.3f} acc {acc:.3f}  (train-gap {Gcur})")

    # ── test recall vs gap length, with controls ──
    model.eval()
    gaps = [int(g) for g in args.gaps.split(",")]
    NB = 400
    out = {"train_gap": Gmax, "iters": args.iters, "gaps": {}}
    print(f"\n── recall vs gap length (carried vs zeroed-at-gap vs no-plant) ──")
    with torch.no_grad():
        for G in gaps:
            x, k = _make_beacon_batch(NB, G, gen, F, beta0, beta1, probe)
            x = x.to(dev)
            # (1) CARRIED: run beacon token → carry state → fillers → probe, full sequence
            logits, _ = model(x, None)
            acc_carried = float((logits[:, -1, :2].argmax(-1).cpu() == k).float().mean())
            # (2) ZEROED-AT-GAP (decisive null): run [beacon], take state, ZERO it, then [fillers+probe]
            lo, st = model(x[:, :1], None)                # beacon only
            st_zero = [torch.zeros_like(s) for s in st]   # wipe the carried state at the gap
            lo2, _ = model(x[:, 1:], st_zero)             # rest with NO carried info
            acc_zeroed = float((lo2[:, -1, :2].argmax(-1).cpu() == k).float().mean())
            # (3) NO-PLANT: replace beacon with a random filler → must be chance (0.5)
            xnp = x.clone(); xnp[:, 0] = torch.randint(0, F, (NB,), generator=gen)
            lonp, _ = model(xnp, None)
            acc_noplant = float((lonp[:, -1, :2].argmax(-1).cpu() == k).float().mean())
            out["gaps"][G] = {"carried": round(acc_carried, 3), "zeroed_at_gap": round(acc_zeroed, 3),
                              "no_plant": round(acc_noplant, 3)}
            print(f"   G={G:>4}: carried {acc_carried:.3f} | zeroed-at-gap {acc_zeroed:.3f} "
                  f"| no-plant {acc_noplant:.3f}")

    # horizon G* = largest gap where carried recall still ≥ 0.75
    gstar = None
    for G in gaps:
        if out["gaps"][G]["carried"] >= 0.75:
            gstar = G
    out["horizon_G_star"] = gstar

    # (C-on-D) the γ-spectrum the BEACON task grows: holding a bit for G* tokens REQUIRES a
    # long-memory head (γ→1). Contrast with the language model's spectrum (all γ≈0.4, τ≈2): the
    # spectrum is TASK-DEPENDENT — a long-range task grows long-memory heads, a local task doesn't.
    with torch.no_grad():
        xb, _ = _make_beacon_batch(64, max(gaps), gen, F, beta0, beta1, probe)
        h = model.embed(xb.to(dev))
        gammas = []
        for layer in model.layers:
            sc = layer.scan
            g = torch.sigmoid(sc.W_gamma(h)).view(64, -1, sc.n_heads, sc.d_head).mean(dim=(0, 1, 3))
            gammas.extend(float(v) for v in g)
            y, _ = sc(h, None); h = layer.ln1(h + y); h = layer.ln2(h + layer.ffn(h))
    out["beacon_gamma_heads"] = [round(g, 4) for g in gammas]
    out["beacon_tau_max"] = round(1.0 / max(1e-3, 1 - max(gammas)), 1)
    print(f"\n   γ-spectrum (beacon task): heads {[round(g,2) for g in gammas]}  "
          f"→ longest-memory head τ≈{out['beacon_tau_max']:.0f} tokens "
          f"(vs language-task τ≈2 — the spectrum is TASK-DEPENDENT)")
    # decisive separation = the BEST gap at which carried beats zeroed-at-gap (the proof the
    # state — not local context — carries the bit). Measured where carried is still alive, not
    # at the largest gap (where both have already decayed to chance).
    seps = [(G, out["gaps"][G]["carried"] - out["gaps"][G]["zeroed_at_gap"]) for G in gaps]
    G_sep, decisive = max(seps, key=lambda gd: gd[1])
    out["decisive_gap"] = G_sep
    out["decisive_separation"] = round(decisive, 3)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n→ {args.out}")
    print(f"IDLE-PERSISTENCE: at G={G_sep}, carried {out['gaps'][G_sep]['carried']:.2f} vs "
          f"zeroed-at-gap {out['gaps'][G_sep]['zeroed_at_gap']:.2f} (separation {decisive:+.2f}); "
          f"recall holds to horizon G*≈{gstar}. "
          f"{'STATE CARRIES THE BIT THROUGH THE SILENCE' if decisive > 0.30 else 'state not decisive (check)'}")


# ───────────────────────────────────────────────────────────────────────────
#  (E) CARRIER PROBE — HOW does the bit survive the gap when mean-γ says τ≈3?
#      Find the carrier channel (its state correlates with the label), then read ITS
#      γ/α/state trajectory across the gap + two surgical ablations. Discriminates:
#        hidden-long-head (carrier γ≈1) · frozen-by-α (carrier α≈0 in gap) · decay-but-robust.
# ───────────────────────────────────────────────────────────────────────────
def run_carrier_probe(args):
    import numpy as np
    dev = torch.device("cpu")
    F, fillers, beta0, beta1, probe, V = _beacon_vocab()
    gen = torch.Generator().manual_seed(args.seed)

    # retrain the beacon model (same recipe as D) so the probe runs on a known-good model
    torch.manual_seed(args.seed)
    model = StreamingNoPELM(V, V - 1, d_model=64, n_layers=2, n_heads=4, d_head=16,
                            seq_len=32, dropout=0.0, causal=True).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    lossf = nn.CrossEntropyLoss()
    Gmax = args.train_gap
    Gcur = min(4, Gmax)
    print(f"── (E) training beacon model (curriculum → {Gmax}) for the carrier probe ──")
    model.train()
    for it in range(args.iters):
        x, k = _make_beacon_batch(args.batch, Gcur, gen, F, beta0, beta1, probe)
        logits, _ = model(x.to(dev), None)
        loss = lossf(logits[:, -1, :2], k.to(dev))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        acc = float((logits[:, -1, :2].argmax(-1).cpu() == k).float().mean())
        if acc > 0.95 and Gcur < Gmax:
            Gcur = min(Gmax, int(Gcur * 1.5) + 1)

    model.eval()
    Gp = max(int(g) for g in args.gaps.split(","))
    NB = 256

    def fwd_capture(xb):
        h = model.embed(xb.to(dev)); inter = []
        for layer in model.layers:
            y, _, ig = layer.scan(h, None, return_internals=True)
            inter.append(ig); h = layer.ln1(h + y); h = layer.ln2(h + layer.ffn(h))
        return inter

    with torch.no_grad():
        x_mix, k_mix = _make_beacon_batch(NB, Gp, gen, F, beta0, beta1, probe)
        x0 = x_mix.clone(); x0[:, 0] = beta0
        x1 = x_mix.clone(); x1[:, 0] = beta1
        ig_mix, ig0, ig1 = fwd_capture(x_mix), fwd_capture(x0), fwd_capture(x1)
        kf = (k_mix.float() - 0.5)

        out = {"train_gap": Gmax, "gap": Gp, "layers": []}
        print(f"\n── carrier analysis at gap G={Gp} ──")
        for li in range(len(model.layers)):
            Zg = ig_mix[li]["Z_seq"][:, Gp]                       # (B,H,D) state at end of gap
            Zc = Zg - Zg.mean(0, keepdim=True)
            corr = (Zc * kf[:, None, None]).mean(0) / (Zc.std(0) + 1e-6) / (kf.std() + 1e-6)
            flat = corr.abs().reshape(-1)
            idx = int(flat.argmax())
            h_star, c_star = idx // corr.shape[1], idx % corr.shape[1]
            corr_carrier = float(corr[h_star, c_star])
            gap = slice(1, Gp + 1)
            g_c = float(ig_mix[li]["gamma"][:, gap, h_star, c_star].mean())     # mean γ of carrier in gap
            g_c_max = float(ig_mix[li]["gamma"][:, gap, h_star, c_star].mean(0).max())
            al_c = float(ig_mix[li]["alpha"][:, gap, h_star, c_star].mean())    # mean α of carrier in gap
            al_beacon = float(ig_mix[li]["alpha"][:, 0, h_star, c_star].mean())
            a_rms = float(ig_mix[li]["a"][:, gap, h_star, c_star].pow(2).mean().sqrt())
            z0 = ig0[li]["Z_seq"][:, gap, h_star, c_star].mean(0).cpu().numpy()  # (G,) class-0 traj
            z1 = ig1[li]["Z_seq"][:, gap, h_star, c_star].mean(0).cpu().numpy()  # (G,) class-1 traj
            margin = np.abs(z1 - z0)                                            # state separation per pos
            tau = 1.0 / max(1e-3, 1 - g_c)
            lay = {"layer": li, "carrier_head": h_star, "carrier_chan": c_star,
                   "corr": round(corr_carrier, 3), "gamma_carrier_mean": round(g_c, 4),
                   "gamma_carrier_max": round(g_c_max, 4), "tau_carrier": round(tau, 1),
                   "alpha_carrier_gap": round(al_c, 4), "alpha_carrier_beacon": round(al_beacon, 4),
                   "a_carrier_rms_gap": round(a_rms, 4),
                   "margin_start": round(float(margin[0]), 4), "margin_end": round(float(margin[-1]), 4),
                   "margin_ratio_end_start": round(float(margin[-1] / (margin[0] + 1e-9)), 3)}
            # full per-position trajectories for the figure (carrier channel only)
            lay["traj_z0"] = [round(float(v), 4) for v in z0]
            lay["traj_z1"] = [round(float(v), 4) for v in z1]
            lay["traj_gamma"] = [round(float(v), 4) for v in
                                 ig_mix[li]["gamma"][:, gap, h_star, c_star].mean(0).cpu().numpy()]
            lay["traj_alpha"] = [round(float(v), 4) for v in
                                 ig_mix[li]["alpha"][:, gap, h_star, c_star].mean(0).cpu().numpy()]
            out["layers"].append(lay)
            print(f"   L{li}: carrier H{h_star}C{c_star} (corr {corr_carrier:+.2f}) | "
                  f"γ_carrier={g_c:.3f} (max {g_c_max:.3f}, τ≈{tau:.0f}) | "
                  f"α gap={al_c:.3f} beacon={al_beacon:.3f} | a_rms_gap={a_rms:.4f} | "
                  f"margin {margin[0]:.3f}→{margin[-1]:.3f} (×{margin[-1]/(margin[0]+1e-9):.2f})")

    # verdict: which hypothesis does the carrier of the most-correlated layer support?
    best = max(out["layers"], key=lambda L: abs(L["corr"]))
    if best["gamma_carrier_max"] > 0.95:
        verdict = "HIDDEN-LONG-HEAD: carrier channel runs at γ≈1 (near-lossless integrator); the head-mean γ hid it"
    elif best["alpha_carrier_gap"] < 0.1 and best["a_carrier_rms_gap"] < 0.02:
        verdict = "FROZEN-BY-α: input gate closes on fillers (α≈0) → state is FROZEN, not decaying; opens at beacon"
    elif best["margin_ratio_end_start"] < 0.7:
        verdict = "DECAY-BUT-ROBUST: state decays geometrically but margin stays above threshold to G*"
    else:
        verdict = "MIXED: carrier holds via a combination (γ moderate, α partial) — state effectively preserved"
    out["verdict"] = verdict
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n→ {args.out}")
    print(f"MECHANISM: {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="run equivalence + grad-exactness only")
    ap.add_argument("--train", action="store_true", help="run the constant-memory streaming train loop")
    ap.add_argument("--gamma-spectrum", action="store_true", help="(C) per-head γ spectrum, init vs trained")
    ap.add_argument("--idle", action="store_true", help="(D) idle-persistence beacon-through-the-gap")
    ap.add_argument("--carrier", action="store_true", help="(E) per-channel carrier probe — how the bit survives")
    ap.add_argument("--ckpt", default="results/streaming_train.pt", help="trained model for C/D probes")
    ap.add_argument("--iters", type=int, default=2000, help="(D) beacon training iters")
    ap.add_argument("--train-gap", type=int, default=32, help="(D) gap length during training")
    ap.add_argument("--gaps", default="0,8,32,64,128,256,512", help="(D) test gap lengths")
    ap.add_argument("--no-detach", action="store_true", help="CONTROL: don't detach state → graph grows")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--overlap", type=int, default=16)
    ap.add_argument("--target-tokens", type=int, default=2_000_000)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--mem-hard-gb", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/streaming_train.json")
    args = ap.parse_args()

    if args.check:
        V, MASK = 200, 199
        print("── §0 equivalence: streaming(states=None) == original NoPE forward ──")
        d = check_equivalence(V, MASK, seed=args.seed)
        print(f"   max |Δ| = {d:.2e}  {'OK (same operator)' if d < 1e-4 else 'MISMATCH'}")
        print(f"\n── grad-exactness: truncated+carry+overlap vs full-window BPTT "
              f"(chunk={args.chunk}, overlap={args.overlap}) ──")
        cos, rel = check_grad_exactness(V, MASK, chunk=args.chunk, overlap=args.overlap, seed=args.seed)
        print(f"   grad cosine = {cos:.4f}   rel-err = {rel:.4f}   "
              f"{'OK (truncation near-exact)' if cos > 0.95 else 'TOO LOSSY — receptive field longer than claimed'}")
    elif args.train:
        streaming_train(args)
    elif args.gamma_spectrum:
        run_gamma_spectrum(args)
    elif args.idle:
        run_idle_persistence(args)
    elif args.carrier:
        run_carrier_probe(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
