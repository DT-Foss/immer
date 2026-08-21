#!/usr/bin/env python3 -u
"""
POS FAMILY TRANSFER — the operating mode on the Mamba/S6 configuration (F5).
==============================================================================
ssm_family_reduction.py proves Mamba/S6 is a SWITCH-restriction of the same
affine operator that carries GSSM-Selective: real DIAGONAL A (not scalar),
INPUT-DEPENDENT via Delta_t, LINEAR drive Delta_t . Bbar . u_t (not the GSSM
nonlinear log(1-v^2) rapidity feature). This file asks whether the POS
OPERATING PRINCIPLE (surprise-gated backward, streaming_train's Z-carry) is a
property of the FAMILY (any switch setting) or an artifact of the GSSM house
architecture — by running the identical POS recipe from pos_run.py on BOTH.

P22 (registered prediction, MOONSHOTS.md F5): S6-POS-ratio >= 0.85 of the
GSSM-POS-ratio (family-generic gating benefit); GSSM-full <= S6-full + 0.1 nats
(architecture-competitive at matched pipeline/tokens -- the DD baseline).

Design
------
1. StreamingS6Layer: the Switch-1=(b)/Switch-2=selective/Switch-3=Mamba builder
   from ssm_family_reduction.build_mamba_s6, turned into a STATEFUL nn.Module
   layer (Z-carry across chunk boundaries, streaming_train.stateful_scan
   pattern) with the exact same recurrence semantics as
   ssm_family_reduction.affine_combine / sequential_diag_ssm:
       Abar_t = exp(Delta_t . A_log)          in (0,1)^N, real diagonal, input-dep
       drive  = Delta_t . Bbar_t . u_t         linear, input-scaled (Switch 3)
       h_t    = Abar_t * h_{t-1} + drive_t     (the SAME h_t = A_t h_{t-1} + B_t u_t)
       read   = C . h_t                        linear readout (S6's C-projection)
   Delta_t = softplus(W_delta x_t) > 0 (per head,chan, input-dependent -> A_t
   input-dependent, Switch 2 = selective). A_log < 0 fixed per (head,chan) so
   Abar_t in (0,1)^N always (stability), matching build_mamba_s6 exactly.

2. Parameter-matched to GSSM-Selective's scan layer at d_model=128/n_heads=4/
   d_head=32 (81,920 params: 5 square 128x128 projections W_v/W_gate/W_gamma/
   W_alpha/W_out). S6 layer: W_in, W_delta, W_B, W_gate, W_out (5 square
   128x128 projections, same shape family) + a small A_log (n_heads*d_head=128)
   -> 82,048 params, ratio 1.0016 (well inside +/-10%). Both counts logged.

3. Envelope: embed -> 2 blocks [scan + ln + ffn] -> head, NoPE (pos=Identity),
   IDENTICAL wrapper shape to streaming_train.StreamingNoPELM. Z-carry via the
   same stateful_scan .detach() truncated-BPTT pattern (streaming_train.py S0).

4. Equivalence mini-gate: full-sequence forward (Z0=None once) vs chunked with
   carried+detached Z must agree to < 1e-4 (proves the carry is the SAME
   operator as one full pass) -- the stateful-scan pattern correctly ported.

5. The experiment: 2x2 {GSSM-Selective, S6} x {full-gradient (A2 recipe),
   surprise-gated (A3 recipe, q=0.75, window=500, min_window=100,
   ignition_chunks=100)} on ONE C4 stream, cloned to all four arms per chunk
   (pos_run.py's clone-per-arm pattern), WT-2 held-out eval on a frozen slice.
   Seed 42 throughout.

Outputs: results/pos_family.json (or results/pos_family_<tag>.json under
--tag, e.g. --tag smoke) with per-arm curves, param counts, and BOTH verdicts:
  (a) POS-ratio within each architecture: (gated_improvement / full_improvement)
      -- is S6's ratio ~ GSSM's ratio? (family-generic claim)
  (b) architecture head-to-head at full-gradient, matched tokens/pipeline
      -- the DD baseline GSSM vs S6.

CPU only, torch.set_num_threads(1), os.nice(19) best-effort. Does not touch
any live pos_*.py output path (writes only to results/pos_family*.json).
"""
import argparse
import json
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.mps.is_available = lambda: False  # force CPU, matches streaming_train/pos_run
torch.set_num_threads(1)                          # bit-deterministic, matches pos_run.py
try:
    torch.set_num_interop_threads(1)              # linux: inter-op pool defaults to ncores/2
except RuntimeError:
    pass  # already started parallel work (e.g. on re-import); intra-op=1 is the critical one

try:
    os.nice(19)
except Exception:
    pass

import sys
sys.path.insert(0, "reference")
sys.path.insert(0, "src")

from moebius_scan_transformer_selective import SelectiveRapiditySqrtScanLayer
from streaming_train import StreamingScanLayer, stateful_scan
from length_extrap_v2 import load_wikitext2, build_vocab, tokenize


# ═══════════════════════════════════════════════════════════════════════════
# §1  StreamingS6Layer — Switch 1=(b) real diagonal A in (0,1)^N, Switch 2=
#     selective (Delta_t input-dep), Switch 3=Mamba linear drive Delta_t.Bbar.u_t.
#     Exactly ssm_family_reduction.build_mamba_s6's math, wired as a trainable
#     nn.Module with the streaming_train Z-carry pattern.
# ═══════════════════════════════════════════════════════════════════════════
class StreamingS6Layer(nn.Module):
    """S6 (Mamba selective-scan) layer, stateful across chunk boundaries.

    Abar_t = exp(Delta_t * A_log)   -- A_log < 0 fixed per (head,chan) => Abar_t
                                        in (0,1)^N always; Delta_t input-dep makes
                                        Abar_t input-dep (selective, Switch 2).
    drive  = Delta_t * Bbar_t * u_t -- linear, input-scaled (Switch 3, Mamba).
    h_t    = Abar_t * h_{t-1} + drive_t   (identical recurrence shape to GSSM's
                                            z_t = gamma_t*z_{t-1} + a_t; here A_t
                                            is a real DIAGONAL vector, not scalar).
    read   = C . h_t  (linear readout, S6's C-projection; GSSM instead reads
             sqrt(1 - exp(z)) -- Switch-3-adjacent choice, kept OUT of the state
             recurrence itself so the family-reduction claim is about h_t, not
             the readout nonlinearity).

    Param-matched to SelectiveRapiditySqrtScanLayer (5 square d_model x d_model
    projections): W_in, W_delta, W_B, W_gate, W_out here play that role, plus a
    small A_log (n_heads*d_head). See module docstring for the count.
    """

    def __init__(self, d_model, d_head=32, n_heads=4, causal=True, dropout=0.0):
        super().__init__()
        assert causal, "family-transfer experiment is causal-LM only"
        self.d_model, self.d_head, self.n_heads = d_model, d_head, n_heads
        proj = n_heads * d_head
        self.W_in = nn.Linear(d_model, proj, bias=False)     # u_t projection (drive input)
        self.W_delta = nn.Linear(d_model, proj, bias=False)  # Delta_t = softplus(.) > 0
        self.W_B = nn.Linear(d_model, proj, bias=False)      # Bbar_t (input-scaled B-map)
        self.W_gate = nn.Linear(d_model, proj, bias=False)   # gate on u_t (mirrors GSSM's W_gate)
        self.W_out = nn.Linear(proj, d_model, bias=False)    # C-projection (linear readout)
        # A_log < 0 fixed per (head,chan): Abar_t = exp(Delta_t * A_log) in (0,1)^N.
        # Init in [-2.05, -0.05] -- same range as ssm_family_reduction.build_mamba_s6.
        self.A_log = nn.Parameter(-torch.rand(n_heads, d_head) * 2.0 - 0.05)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x, Z_in=None, return_internals=False):
        B, T, D = x.shape
        u = torch.tanh(self.W_in(x))
        gate = torch.sigmoid(self.W_gate(x))
        u_gated = u * gate
        if self.dropout is not None:
            u_gated = self.dropout(u_gated)
        u_gated = u_gated.view(B, T, self.n_heads, self.d_head)
        delta = F.softplus(self.W_delta(x)).view(B, T, self.n_heads, self.d_head)  # > 0
        Bbar = self.W_B(x).view(B, T, self.n_heads, self.d_head)

        Abar = torch.exp(delta * self.A_log.view(1, 1, self.n_heads, self.d_head))  # (0,1)^N, input-dep
        drive = delta * Bbar * u_gated                                             # Switch-3 Mamba drive

        Z_seq, Z_fin = stateful_scan(drive, Abar, Z_in)      # SAME carry pattern as GSSM StreamingScanLayer
        state = self.W_out(Z_seq.reshape(B, T, self.n_heads * self.d_head))         # linear C-readout
        if return_internals:
            return state, Z_fin, {"Abar": Abar, "delta": delta, "drive": drive, "Z_seq": Z_seq}
        return state, Z_fin

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════════════════
# §2  Streaming LM envelopes — embed -> 2x[scan+ln+ffn] -> head, NoPE.
#     Identical shape to streaming_train.StreamingNoPELM; one instantiated
#     from GSSM-Selective scan layers, one from StreamingS6Layer.
# ═══════════════════════════════════════════════════════════════════════════
class FamilyStreamingLM(nn.Module):
    """LM envelope generic over the scan-layer class (GSSM StreamingScanLayer or
    StreamingS6Layer). embed -> n_layers x [scan(Z-carry) + ln + ffn] -> head.
    NoPE by construction (no positional module at all -- not even nn.Identity
    wired through a parent's assert, since neither scan layer needs it)."""

    def __init__(self, layer_cls, vocab_size, mask_idx, d_model=128, n_layers=2,
                 n_heads=4, d_head=32, dropout=0.0, causal=True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 2, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            blk = nn.Module()
            blk.scan = layer_cls(d_model, d_head=d_head, n_heads=n_heads,
                                 causal=causal, dropout=dropout)
            blk.ln1 = nn.LayerNorm(d_model)
            blk.ln2 = nn.LayerNorm(d_model)
            blk.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                    nn.Linear(4 * d_model, d_model))
            self.layers.append(blk)
        self.head = nn.Linear(d_model, vocab_size + 1)

    def forward(self, x, states=None):
        h = self.embed(x)                                    # NoPE: plain embedding, no PE term
        si = states if states is not None else [None] * len(self.layers)
        new_states = []
        for layer, z in zip(self.layers, si):
            y, z_out = layer.scan(h, z)
            h = layer.ln1(h + y)
            h = layer.ln2(h + layer.ffn(h))
            new_states.append(z_out)
        return self.head(h), new_states

    def scan_param_count(self):
        return sum(layer.scan.param_count() if hasattr(layer.scan, "param_count")
                   else sum(p.numel() for p in layer.scan.parameters())
                   for layer in self.layers)


def build_gssm_lm(vocab_size, mask_idx, d_model=128, n_layers=2, n_heads=4, d_head=32):
    return FamilyStreamingLM(StreamingScanLayer, vocab_size, mask_idx, d_model=d_model,
                             n_layers=n_layers, n_heads=n_heads, d_head=d_head,
                             dropout=0.0, causal=True)


def build_s6_lm(vocab_size, mask_idx, d_model=128, n_layers=2, n_heads=4, d_head=32):
    return FamilyStreamingLM(StreamingS6Layer, vocab_size, mask_idx, d_model=d_model,
                             n_layers=n_layers, n_heads=n_heads, d_head=d_head,
                             dropout=0.0, causal=True)


# ═══════════════════════════════════════════════════════════════════════════
# §3  Equivalence mini-gate: full-seq forward vs chunked+carried(+detached).
#     Proves the S6 stateful-scan carry is the SAME operator as one full pass,
#     the exact property streaming_train.check_equivalence certifies for GSSM.
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def check_equivalence(build_fn, vocab_size, mask_idx, T=96, chunk=32, seed=0):
    torch.manual_seed(seed)
    model = build_fn(vocab_size, mask_idx, d_model=64, n_layers=2, n_heads=4, d_head=16).eval()
    x = torch.randint(0, vocab_size, (2, T))

    out_full, _ = model(x, None)

    states, outs = None, []
    pos = 0
    while pos < T:
        hi = min(T, pos + chunk)
        logits, states = model(x[:, pos:hi], states)
        outs.append(logits)
        pos = hi
    out_chunked = torch.cat(outs, dim=1)

    return float((out_full - out_chunked).abs().max())


# ═══════════════════════════════════════════════════════════════════════════
# §4  C4 stream (shared block source cloned to all 4 arms, pos_run.py pattern)
# ═══════════════════════════════════════════════════════════════════════════
class C4Stream:
    def __init__(self, stoi, unk, block=65536):
        self.stoi, self.unk, self.block = stoi, unk, block
        self.pending = []
        self._it = None

    def _connect(self):
        from datasets import load_dataset
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        self._it = iter(ds)

    def next_block(self):
        while len(self.pending) < self.block:
            if self._it is None:
                self._connect()
            try:
                row = next(self._it)
            except StopIteration:
                self._it = None
                continue
            t = row.get("text", "") if isinstance(row, dict) else ""
            if t.strip():
                self.pending.extend(tokenize(t, self.stoi, self.unk))
        out, self.pending = self.pending[:self.block], self.pending[self.block:]
        return out


# ═══════════════════════════════════════════════════════════════════════════
# §5  POS recipe (verbatim structure from pos_run.py step_a2 / step_gated)
# ═══════════════════════════════════════════════════════════════════════════
def build_arm(tag, build_fn, V, mask, args, lr):
    torch.manual_seed(args.seed)
    m = build_fn(V, mask, d_model=args.d_model, n_layers=2, n_heads=4, d_head=args.d_model // 4)
    m.train()
    return {"tag": tag, "model": m, "opt": torch.optim.Adam(m.parameters(), lr=lr),
            "states": None, "grad_tokens": 0, "n_chunks": 0, "n_bwd": 0,
            "window": deque(maxlen=args.window)}


def _detach(states):
    return [s.detach() for s in states]


def _fwd_nograd(arm, x, y):
    with torch.no_grad():
        logits, st = arm["model"](x, arm["states"])
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                              reduction="none").view(x.shape)
    return nll, st


def step_full(arm, x, y, clip=5.0):
    """A2-equivalent: full-gradient, backward on every chunk."""
    logits, st = arm["model"](x, arm["states"])
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    arm["opt"].zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(arm["model"].parameters(), clip)
    arm["opt"].step()
    arm["states"] = _detach(st)
    arm["grad_tokens"] += x.numel()
    arm["n_chunks"] += 1
    arm["n_bwd"] += 1
    return float(loss.detach())


def step_gated(arm, x, y, args, clip=5.0):
    """A3-equivalent: surprise-gated backward (pos_run.py step_gated, verbatim recipe)."""
    nll, st_ng = _fwd_nograd(arm, x, y)
    s = float(nll.mean())
    if arm["n_chunks"] < args.ignition_chunks:
        gated = True
    elif len(arm["window"]) >= args.min_window:
        thresh = float(np.quantile(np.fromiter(arm["window"], dtype=np.float64), args.q))
        gated = s > thresh
    else:
        gated = True
    if gated:
        logits, st = arm["model"](x, arm["states"])
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
    arm["window"].append(s)
    arm["n_chunks"] += 1
    return s, gated


# ═══════════════════════════════════════════════════════════════════════════
# §6  Frozen WT-2 held-out eval (build_eval_set / heldout, pos_run.py pattern)
# ═══════════════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════════════
# §7  Main experiment: 2x2 {GSSM, S6} x {full, gated}, one shared C4 source
#     cloned per-chunk to all four arms (identical tokens/pipeline to all).
# ═══════════════════════════════════════════════════════════════════════════
def run_experiment(args):
    out_path = (os.path.join("results", f"pos_family_{args.tag}.json") if args.tag
                else os.path.join("results", "pos_family.json"))
    os.makedirs("results", exist_ok=True)

    train_text, val_text = load_wikitext2()
    vocab, stoi, unk, mask = build_vocab(train_text)
    V = len(vocab)
    val_ids = tokenize(val_text, stoi, unk)
    evX, evY = build_eval_set(val_ids, args.eval_tokens, args.chunk)
    print(f"[data] vocab={V} | frozen WT-2 held-out {evY.numel():,} tokens", flush=True)

    # ── equivalence mini-gate (both architectures) ──
    eq_gssm = check_equivalence(build_gssm_lm, V, mask, seed=args.seed)
    eq_s6 = check_equivalence(build_s6_lm, V, mask, seed=args.seed)
    print(f"[equivalence] GSSM full-seq vs chunked+carried max|delta| = {eq_gssm:.3e} "
          f"{'PASS' if eq_gssm < 1e-4 else 'FAIL'}", flush=True)
    print(f"[equivalence] S6   full-seq vs chunked+carried max|delta| = {eq_s6:.3e} "
          f"{'PASS' if eq_s6 < 1e-4 else 'FAIL'}", flush=True)
    assert eq_gssm < 1e-4, f"GSSM equivalence gate FAILED: {eq_gssm:.3e}"
    assert eq_s6 < 1e-4, f"S6 equivalence gate FAILED: {eq_s6:.3e}"

    # ── param counts (both scan layers, both full models) ──
    torch.manual_seed(args.seed)
    probe_gssm = build_gssm_lm(V, mask, d_model=args.d_model, n_layers=2, n_heads=4, d_head=args.d_model // 4)
    torch.manual_seed(args.seed)
    probe_s6 = build_s6_lm(V, mask, d_model=args.d_model, n_layers=2, n_heads=4, d_head=args.d_model // 4)
    scan_params_gssm = probe_gssm.scan_param_count()
    scan_params_s6 = probe_s6.scan_param_count()
    total_params_gssm = sum(p.numel() for p in probe_gssm.parameters())
    total_params_s6 = sum(p.numel() for p in probe_s6.parameters())
    ratio = scan_params_s6 / scan_params_gssm
    print(f"[params] GSSM scan-layer params (both layers) = {scan_params_gssm:,} | "
          f"total = {total_params_gssm:,}", flush=True)
    print(f"[params] S6   scan-layer params (both layers) = {scan_params_s6:,} | "
          f"total = {total_params_s6:,}", flush=True)
    print(f"[params] S6/GSSM scan-param ratio = {ratio:.4f} "
          f"{'PASS (within +/-10%)' if 0.9 <= ratio <= 1.1 else 'OUT OF +/-10% BAND'}", flush=True)
    del probe_gssm, probe_s6

    # ── build the 2x2 arms, identical init recipe per architecture ──
    arms = {
        "GSSM_full": build_arm("GSSM_full", build_gssm_lm, V, mask, args, args.lr),
        "GSSM_gated": build_arm("GSSM_gated", build_gssm_lm, V, mask, args, args.lr),
        "S6_full": build_arm("S6_full", build_s6_lm, V, mask, args, args.lr),
        "S6_gated": build_arm("S6_gated", build_s6_lm, V, mask, args, args.lr),
    }
    # same-architecture arms must start from identical weights (fair full-vs-gated compare)
    with torch.no_grad():
        d_gssm = sum(float((p1 - p2).abs().sum()) for p1, p2 in
                    zip(arms["GSSM_full"]["model"].parameters(), arms["GSSM_gated"]["model"].parameters()))
        d_s6 = sum(float((p1 - p2).abs().sum()) for p1, p2 in
                  zip(arms["S6_full"]["model"].parameters(), arms["S6_gated"]["model"].parameters()))
    assert d_gssm == 0.0, f"GSSM_full/gated init mismatch: {d_gssm}"
    assert d_s6 == 0.0, f"S6_full/gated init mismatch: {d_s6}"
    print(f"[arms] GSSM_full/GSSM_gated identical init | S6_full/S6_gated identical init", flush=True)

    stream = C4Stream(stoi, unk)
    B, K = args.batch, args.chunk
    bufs = [[] for _ in range(B)]
    for b in range(B):
        bufs[b].extend(stream.next_block())

    n_tok = 0
    curves = {k: [] for k in arms}
    t0 = time.time()

    def run_eval():
        rec = {"n_streamed": n_tok, "wall_s": round(time.time() - t0, 1)}
        for k, a in arms.items():
            hl = heldout(a["model"], evX, evY)
            rec[k] = {"heldout": round(hl, 6), "grad_tokens": a["grad_tokens"],
                      "n_chunks": a["n_chunks"], "n_bwd": a["n_bwd"],
                      "gate_frac": round(a["n_bwd"] / max(1, a["n_chunks"]), 4)}
            curves[k].append(rec[k] | {"n_streamed": n_tok})
        line = " | ".join(f"{k} {rec[k]['heldout']:.4f}" for k in arms)
        print(f"[eval] n={n_tok:>10,} wall={time.time()-t0:6.1f}s | {line}", flush=True)
        return rec

    base = run_eval()
    step = 0
    print(f"[pos-family] streaming... target={args.target_tokens:,} tokens/arm-family "
          f"(B={B} K={K})", flush=True)
    while n_tok < args.target_tokens:
        for b in range(B):
            while len(bufs[b]) < K + 1:
                bufs[b].extend(stream.next_block())
        x = torch.tensor([bufs[b][:K] for b in range(B)], dtype=torch.long)
        y = torch.tensor([bufs[b][1:K + 1] for b in range(B)], dtype=torch.long)
        for b in range(B):
            del bufs[b][:K]
        n_tok += B * K

        # identical cloned chunk to all four arms (pos_run.py's x.clone()/y.clone() pattern)
        step_full(arms["GSSM_full"], x.clone(), y.clone())
        step_gated(arms["GSSM_gated"], x.clone(), y.clone(), args)
        step_full(arms["S6_full"], x.clone(), y.clone())
        step_gated(arms["S6_gated"], x.clone(), y.clone(), args)

        step += 1
        if n_tok % args.eval_every_tokens < B * K:
            run_eval()

    final = run_eval()

    # ── verdicts ──
    base_gssm_full = base["GSSM_full"]["heldout"]
    base_gssm_gated = base["GSSM_gated"]["heldout"]
    base_s6_full = base["S6_full"]["heldout"]
    base_s6_gated = base["S6_gated"]["heldout"]
    fin_gssm_full = final["GSSM_full"]["heldout"]
    fin_gssm_gated = final["GSSM_gated"]["heldout"]
    fin_s6_full = final["S6_full"]["heldout"]
    fin_s6_gated = final["S6_gated"]["heldout"]

    imp_gssm_full = base_gssm_full - fin_gssm_full
    imp_gssm_gated = base_gssm_gated - fin_gssm_gated
    imp_s6_full = base_s6_full - fin_s6_full
    imp_s6_gated = base_s6_gated - fin_s6_gated

    pos_ratio_gssm = imp_gssm_gated / imp_gssm_full if imp_gssm_full != 0 else float("nan")
    pos_ratio_s6 = imp_s6_gated / imp_s6_full if imp_s6_full != 0 else float("nan")
    ratio_of_ratios = pos_ratio_s6 / pos_ratio_gssm if pos_ratio_gssm not in (0, float("nan")) else float("nan")

    arch_gap_nats = fin_s6_full - fin_gssm_full  # GSSM - S6 at full-gradient (positive = GSSM worse)

    verdict_a_pass = ratio_of_ratios >= 0.85 if ratio_of_ratios == ratio_of_ratios else False
    verdict_b_pass = (fin_gssm_full - fin_s6_full) <= 0.1

    print("\n" + "=" * 78)
    print("VERDICT (a) — POS-ratio family-generic?")
    print(f"  GSSM: full-improvement {imp_gssm_full:+.4f} nats | gated-improvement {imp_gssm_gated:+.4f} nats "
          f"| POS-ratio {pos_ratio_gssm:.4f}")
    print(f"  S6:   full-improvement {imp_s6_full:+.4f} nats | gated-improvement {imp_s6_gated:+.4f} nats "
          f"| POS-ratio {pos_ratio_s6:.4f}")
    print(f"  S6-ratio / GSSM-ratio = {ratio_of_ratios:.4f}  (P22 threshold >= 0.85) "
          f"{'PASS -- family-generic' if verdict_a_pass else 'FAIL -- ratio does not transfer'}")
    print("-" * 78)
    print("VERDICT (b) — architecture head-to-head at full-gradient (DD baseline)")
    print(f"  GSSM-full held-out {fin_gssm_full:.4f} nats | S6-full held-out {fin_s6_full:.4f} nats "
          f"| gap (GSSM-S6) {fin_gssm_full - fin_s6_full:+.4f} nats")
    print(f"  P22 threshold: GSSM-full <= S6-full + 0.1 nats "
          f"{'PASS -- competitive' if verdict_b_pass else 'FAIL -- GSSM behind by more than 0.1 nats'}")
    print("=" * 78)

    out = {
        "prediction_id": "P22",
        "prediction_text": "S6-POS-ratio >= 0.85 x GSSM-POS-ratio (family-generic); "
                           "GSSM-full <= S6-full + 0.1 nats (architecture-competitive)",
        "config": vars(args),
        "equivalence_gate": {"GSSM_max_abs_err": eq_gssm, "S6_max_abs_err": eq_s6,
                             "tol": 1e-4, "pass": eq_gssm < 1e-4 and eq_s6 < 1e-4},
        "param_counts": {
            "GSSM_scan_params_both_layers": scan_params_gssm,
            "S6_scan_params_both_layers": scan_params_s6,
            "GSSM_total_params": total_params_gssm,
            "S6_total_params": total_params_s6,
            "s6_over_gssm_scan_ratio": ratio,
            "within_10pct_band": bool(0.9 <= ratio <= 1.1),
        },
        "baseline": base,
        "final": final,
        "curves": curves,
        "improvements_nats": {
            "GSSM_full": imp_gssm_full, "GSSM_gated": imp_gssm_gated,
            "S6_full": imp_s6_full, "S6_gated": imp_s6_gated,
        },
        "verdict_a_pos_ratio_family_generic": {
            "pos_ratio_gssm": pos_ratio_gssm, "pos_ratio_s6": pos_ratio_s6,
            "ratio_of_ratios_s6_over_gssm": ratio_of_ratios,
            "threshold": 0.85, "pass": verdict_a_pass,
        },
        "verdict_b_architecture_competitive": {
            "gssm_full_heldout": fin_gssm_full, "s6_full_heldout": fin_s6_full,
            "gap_gssm_minus_s6_nats": fin_gssm_full - fin_s6_full,
            "threshold_nats": 0.1, "pass": verdict_b_pass,
        },
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {out_path}")
    return out


def main():
    ap = argparse.ArgumentParser(description="F5: POS on the Mamba/S6 configuration (family-generic test)")
    ap.add_argument("--smoke", action="store_true", help="4 arms x 800k tokens, held-out 20k, eval every 200k")
    ap.add_argument("--full", action="store_true", help="4 arms x 6M tokens, held-out 50k, eval every 500k")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--q", type=float, default=0.75, help="gating quantile")
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--min-window", type=int, default=100)
    ap.add_argument("--ignition-chunks", type=int, default=100)
    ap.add_argument("--target-tokens", type=int, default=0)
    ap.add_argument("--eval-tokens", type=int, default=0)
    ap.add_argument("--eval-every-tokens", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--check", action="store_true", help="equivalence gate only, no training")
    args = ap.parse_args()

    if args.smoke:
        args.target_tokens = args.target_tokens or 800_000
        args.eval_tokens = args.eval_tokens or 20_000
        args.eval_every_tokens = args.eval_every_tokens or 200_000
        args.tag = args.tag or "smoke"
    elif args.full:
        args.target_tokens = args.target_tokens or 6_000_000
        args.eval_tokens = args.eval_tokens or 50_000
        args.eval_every_tokens = args.eval_every_tokens or 500_000
    else:
        args.target_tokens = args.target_tokens or 800_000
        args.eval_tokens = args.eval_tokens or 20_000
        args.eval_every_tokens = args.eval_every_tokens or 200_000

    if args.check:
        V, MASK = 200, 199
        eq_gssm = check_equivalence(build_gssm_lm, V, MASK, seed=args.seed)
        eq_s6 = check_equivalence(build_s6_lm, V, MASK, seed=args.seed)
        print(f"GSSM equivalence max|delta| = {eq_gssm:.3e} {'OK' if eq_gssm < 1e-4 else 'FAIL'}")
        print(f"S6   equivalence max|delta| = {eq_s6:.3e} {'OK' if eq_s6 < 1e-4 else 'FAIL'}")
        return

    run_experiment(args)


if __name__ == "__main__":
    main()
