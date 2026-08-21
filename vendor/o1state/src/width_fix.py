#!/usr/bin/env python3 -u
"""
Phase-2 Width-Fix Runner — d=512 collapse — by Opus 4.8
=======================================================

FULL-DATA BASELINES (results/phase1_fulldata.json, 8 ep + early-stop — the
numbers any ablation row must beat, at MATCHED budget):
  selective_d512_T32: best 342, acc 0.1643 FROZEN from epoch 1  ← collapsed
  pure_d512_T32:      best 280, acc 0.164→0.197 (breaks through ep4)  ← NOT collapsed
  selective_d128_T32: best 161 (working width reference)

REVISED hypothesis (the collapse is SELECTIVE-ONLY): Pure and Selective share the
same unbounded W_v, yet Pure recovers (280) while Selective is dead-on-arrival
(acc frozen from epoch 1). So pure W_v velocity saturation is NOT the sole cause —
the DIFFERENCE between the two is the gating (W_gamma, W_alpha). The frozen-from-
epoch-1 signature points to an init / symmetry-breaking failure in the gated log-
complement recurrence at width. diagnose_d512.py measures velocity AND gates to
attribute it. This runner ablates which lever (warmup / pre-LN / muP / gate-bias)
breaks the deadlock. Pure rows test "does the fix HURT a healthy model", not rescue.

This runner deploys the MASTER_BRIEF §3 levers, each INDEPENDENTLY toggleable,
on top of the FROZEN scan layers (reference/ is read-only, never edited):

  #1  --warmup    LR warmup + cosine decay
  #2  --preln     Pre-LN block (src re-wrap of the LN/FFN boilerplate)
  #3  --mup       muP-lite per-fan-in LR scaling on the unbounded projections
  #4  --clip      grad clip knob (1.0 = fixed, 5.0 = reproduce baseline)
  #5  --gate-bias forget-open gate nudge (weakest lever, OFF by default)

THE CONTRIBUTION IS IMPORTED, NOT COPIED. The novel part — the scan layer
(SelectiveRapiditySqrtScanLayer / SqrtCouplingMoebiusScanLayer): the
data-dependent gates, the log-complement recurrence, the sqrt bounding — is
imported UNMODIFIED from reference/. Only the LN→residual→FFN wrapper,
boilerplate present in every transformer, is re-implemented here so it can be
flipped from post-LN to pre-LN. Re-wrapping boilerplate while importing the
contribution does not modify the contribution.

Pre-LN approach (option (c), argued in the spec):
  Post-LN (frozen ref):  x = ln1(x + scan(x));   x = ln2(x + ffn(x))
  Pre-LN  (this file):   x = x + scan(ln1(x));    x = x + ffn(ln2(x))
  + a final ln_f before the head (standard Pre-LN convention).
When --preln is OFF we reuse the ORIGINAL post-LN TransformerLayer (imported,
not rebuilt), so the baseline lever is byte-faithful to results_v2 / phase1.

muP-lite (#3): two Adam param groups, classified by module name. PROJECTION
(W_v, W_gate, W_out, ffn.0, ffn.3, head) get lr = base_lr * BASE_D/d_model.
BASE (embed, all LN, all biases, AND the gates W_gamma/W_alpha) keep base_lr.
Gates stay at base lr ON PURPOSE: they are small-init (gain 0.1), sigmoid-
bounded → the muP variance argument does not apply, and shrinking their lr
would slow selectivity learning precisely at the width where escaping the
frozen-acc basin is hardest. At d=128 the scale is exactly 1.0, so a --mup
d=128 run is numerically identical to the no-mup run (verifiable sanity).

Hardware: 16GB Mac Mini M4, MPS. Data: full WikiText-2 MLM. Code stays English.
Matches the style of instrumented_runner.py (same data pipeline, train_epoch/
evaluate, MPS device, incremental results JSON). diagnose_d512's
GateVelocityProbe is reused so every run REPORTS whether the fix actually
reduced |v| saturation.

DO NOT run heavily — GPU busy with Phase 1. __main__ default is a TINY smoke
(d=128, capped data, 1 epoch, all flags on) to prove it runs; --matrix is the
real sweep.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import os
import re
import math
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Import ORIGINAL, UNMODIFIED models from the read-only reference dir ──
# reference/ is chmod 444 — imported, never edited. The scan layers below are
# THE CONTRIBUTION and are used as-is; only their LN/FFN wrapper is re-written.
sys.path.insert(0, str(Path(__file__).resolve().parent))   # src/ — for diagnose_d512
REF = Path(__file__).resolve().parent.parent / "reference"
sys.path.insert(0, str(REF))
from moebius_attention import SinusoidalPositionalEncoding
from moebius_scan_transformer_selective import (
    SelectiveRapiditySqrtTransformerLM,
    SelectiveRapiditySqrtTransformerLayer,
    SelectiveRapiditySqrtScanLayer,
)
from moebius_scan_transformer_sqrt import (
    SqrtCouplingMoebiusScanTransformerLM,
    SqrtCouplingMoebiusScanTransformerLayer,
    SqrtCouplingMoebiusScanLayer,
)

# Reuse the velocity/gate probe from the diagnosis script if present, so the
# saturation metric in this runner is IDENTICAL to the one that produced the
# attribution verdict. Fall back to a minimal inline copy otherwise.
try:
    from diagnose_d512 import GateVelocityProbe  # noqa: F401
    _PROBE_SOURCE = "diagnose_d512.GateVelocityProbe"
except Exception:                                # pragma: no cover - fallback
    _PROBE_SOURCE = "inline (diagnose_d512 unavailable)"

    class GateVelocityProbe:
        """Minimal inline copy: velocity saturation + gate health off the
        ORIGINAL selective scan layer's own weights. Mirrors diagnose_d512."""

        def __init__(self, model):
            self.h = []
            self.reset()
            for m in model.modules():
                if isinstance(m, SelectiveRapiditySqrtScanLayer):
                    self.h.append(m.register_forward_hook(self._hook))

        def reset(self):
            self.n = 0
            self.vel_sat = 0.0
            self.vel_abs_mean = 0.0
            self.gamma_mean = 0.0
            self.alpha_mean = 0.0
            self.gamma_frozen = 0.0
            self.alpha_dead = 0.0
            self.wv_preact_absmax = 0.0

        @torch.no_grad()
        def _hook(self, module, inp, out):
            x = inp[0]
            wv_pre = module.W_v(x)
            v = torch.tanh(wv_pre)
            gate = torch.sigmoid(module.W_gate(x))
            vg = v * gate
            gamma = torch.sigmoid(module.W_gamma(x))
            alpha = torch.sigmoid(module.W_alpha(x))
            self.vel_sat += (vg.abs() > 0.95).float().mean().item()
            self.vel_abs_mean += vg.abs().mean().item()
            self.gamma_mean += gamma.mean().item()
            self.alpha_mean += alpha.mean().item()
            self.gamma_frozen += (gamma > 0.95).float().mean().item()
            self.alpha_dead += (alpha < 0.05).float().mean().item()
            self.wv_preact_absmax = max(self.wv_preact_absmax,
                                        wv_pre.abs().max().item())
            self.n += 1

        def summary(self):
            n = max(1, self.n)
            return {
                "velocity_sat_gt95": round(self.vel_sat / n, 4),
                "velocity_abs_mean": round(self.vel_abs_mean / n, 4),
                "gamma_mean": round(self.gamma_mean / n, 4),
                "alpha_mean": round(self.alpha_mean / n, 4),
                "gamma_frozen_gt95": round(self.gamma_frozen / n, 4),
                "alpha_dead_lt05": round(self.alpha_dead / n, 4),
                "wv_preact_absmax": round(self.wv_preact_absmax, 2),
            }

        def remove(self):
            for h in self.h:
                h.remove()


DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# ── Fixed protocol (matches instrumented_runner.py / the original benchmark) ──
SEED = 42
BATCH_SIZE = 32
VOCAB_MAX = 5000
MASK_PROB = 0.15
LR = 3e-3
N_LAYERS = 2
DROPOUT = 0.1
EPS = 1e-6
BASE_D = 128          # width where lr=3e-3 transfers; muP base width
LOG_COMPLEMENT_CLAMP = 0.999

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ===========================================================================
# Data  (full WikiText-2 MLM protocol — identical to instrumented_runner.py)
# ===========================================================================

def load_wikitext2():
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    return ("\n\n".join(ds["train"]["text"]),
            "\n\n".join(ds["validation"]["text"]))


def build_vocab(text):
    words = re.findall(r"[a-zA-Z]+", text.lower())
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    vocab = [w for w, _ in sorted(freq.items(), key=lambda kv: -kv[1])][:VOCAB_MAX]
    stoi = {w: i for i, w in enumerate(vocab)}
    return vocab, stoi, len(vocab), len(vocab) + 1


def tokenize(text, stoi, unk_idx):
    return [stoi.get(w, unk_idx) for w in re.findall(r"[a-zA-Z]+", text.lower())]


def make_mlm_batches(ids, seq_len, batch_size, mask_idx, mask_prob=0.15,
                     max_tokens=None):
    if max_tokens is not None:
        ids = ids[:max_tokens + 1]
    total = len(ids) // seq_len
    X, Y, M = [], [], []
    for i in range(total):
        seq = ids[i * seq_len:(i + 1) * seq_len]
        if len(seq) < seq_len:
            continue
        y = seq.copy()
        mask = (torch.rand(seq_len) < mask_prob).long()
        for j in range(seq_len):
            if mask[j]:
                r = torch.rand(1).item()
                if r < 0.8:
                    seq[j] = mask_idx
                elif r < 0.9:
                    seq[j] = torch.randint(0, mask_idx, (1,)).item()
        X.append(seq)
        Y.append(y)
        M.append(mask)
    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    M = torch.stack(M)
    n = (len(X) // batch_size) * batch_size
    return X[:n], Y[:n], M[:n]


# ===========================================================================
# #2 Pre-LN — re-wrap the BOILERPLATE; import the CONTRIBUTION unmodified
# ===========================================================================

class PreLNBlock(nn.Module):
    """Pre-LN wrapper around an ALREADY-CONSTRUCTED scan layer instance.

    Post-LN (frozen ref):  x = ln1(x + scan(x));   x = ln2(x + ffn(x))
    Pre-LN  (this):        x = x + scan(ln1(x));    x = x + ffn(ln2(x))

    The scan instance is passed in, so this class is agnostic to WHICH
    contribution (selective vs pure) it wraps — it never touches the scan math.
    """

    def __init__(self, scan_layer: nn.Module, d_model: int,
                 ffn_dim: int = None, dropout: float = 0.0):
        super().__init__()
        self.scan = scan_layer                      # <-- the contribution, unmodified
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        ffn_dim = ffn_dim or 4 * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.scan(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# scan-class -> (post-LN reference block, full reference LM) lookup.
_SCAN_TABLE = {
    "selective": (SelectiveRapiditySqrtScanLayer,
                  SelectiveRapiditySqrtTransformerLayer,
                  SelectiveRapiditySqrtTransformerLM),
    "pure": (SqrtCouplingMoebiusScanLayer,
             SqrtCouplingMoebiusScanTransformerLayer,
             SqrtCouplingMoebiusScanTransformerLM),
}


class FixableLM(nn.Module):
    """LM that can run in EITHER topology with the SAME parameters elsewhere.

    use_preln=False : reuse the ORIGINAL post-LN reference TransformerLayer
                      (imported, not rebuilt) -> baseline is byte-faithful.
    use_preln=True  : stack PreLNBlocks (this file's re-wrap) + a final ln_f
                      before the head (standard Pre-LN convention).

    Constructor mirrors the reference LM signature so the runner swaps it in
    with zero call-site changes. `model_kind` selects the contribution.
    """

    def __init__(self, vocab_size, mask_idx, model_kind="selective",
                 d_model=128, n_layers=2, n_heads=4, d_head=32,
                 seq_len=32, dropout=0.1, causal=True, use_preln=False):
        super().__init__()
        if model_kind not in _SCAN_TABLE:
            raise ValueError(f"unknown model_kind {model_kind!r}")
        scan_cls, ref_block_cls, _ = _SCAN_TABLE[model_kind]
        self.model_kind = model_kind
        self.use_preln = use_preln
        self.mask_idx = mask_idx
        self.embed = nn.Embedding(vocab_size + 2, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)

        if use_preln:
            # Pre-LN: build fresh scan instances (the contribution) and wrap
            # them in this file's boilerplate. SAME scan ctor args the
            # reference TransformerLayer would pass.
            self.layers = nn.ModuleList([
                PreLNBlock(
                    scan_cls(d_model, d_head=d_head, n_heads=n_heads,
                             causal=causal, dropout=dropout),
                    d_model, ffn_dim=4 * d_model, dropout=dropout,
                )
                for _ in range(n_layers)
            ])
            self.ln_f = nn.LayerNorm(d_model)       # final pre-head LN
        else:
            # Post-LN: reuse the ORIGINAL reference block verbatim.
            self.layers = nn.ModuleList([
                ref_block_cls(d_model, d_head=d_head, n_heads=n_heads,
                              ffn_dim=4 * d_model, dropout=dropout, causal=causal)
                for _ in range(n_layers)
            ])
            self.ln_f = nn.Identity()               # post-LN already normalizes

        self.head = nn.Linear(d_model, vocab_size + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pos(self.embed(x))
        for layer in self.layers:
            h = layer(h)
        return self.head(self.ln_f(h))


# ===========================================================================
# #3 muP-lite — per-fan-in LR scaling on the unbounded matmul projections
# ===========================================================================

# substrings that mark an UNBOUNDED projection matmul (weight, dim >= 2)
_PROJ_KEYS = ("scan.W_v", "scan.W_gate", "scan.W_out",
              "ffn.0.weight", "ffn.3.weight", "head.weight")
# gates kept at base lr ON PURPOSE (small-init, sigmoid-bounded)
_GATE_KEYS = ("scan.W_gamma", "scan.W_alpha")


def is_projection(name: str, param) -> bool:
    if param.dim() < 2:                              # biases / LN -> base
        return False
    if any(g in name for g in _GATE_KEYS):           # gates -> base (argued)
        return False
    return any(k in name for k in _PROJ_KEYS)


def build_mup_param_groups(model, base_lr: float, d_model: int,
                           base_d: int = BASE_D):
    """Two Adam param groups. Projection lr = base_lr * base_d / d_model;
    everything else (incl. gates) keeps base_lr. No-op at d_model==base_d."""
    scale = base_d / d_model
    proj, base = [], []
    proj_names, base_names = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if is_projection(name, p):
            proj.append(p)
            proj_names.append(name)
        else:
            base.append(p)
            base_names.append(name)
    groups = [
        {"params": base, "lr": base_lr,             "name": "base"},
        {"params": proj, "lr": base_lr * scale,     "name": "projection"},
    ]
    meta = {"proj_lr": base_lr * scale, "base_lr": base_lr, "scale": scale,
            "n_proj": len(proj), "n_base": len(base),
            "proj_names": proj_names, "base_names": base_names}
    return groups, meta


def build_optimizer(model, base_lr, d_model, use_mup, n_layers=N_LAYERS,
                    verbose=True):
    """Adam with muP-lite groups when use_mup, else plain Adam. Asserts the
    projection count to catch silent gate misclassification."""
    if use_mup:
        groups, meta = build_mup_param_groups(model, base_lr, d_model)
        # per layer: W_v, W_gate, W_out, ffn.0.weight, ffn.3.weight = 5
        # + head.weight = 1  ->  n_proj must be 5*n_layers + 1
        expected = 5 * n_layers + 1
        assert meta["n_proj"] == expected, (
            f"muP misclassification: n_proj={meta['n_proj']} != {expected}. "
            f"proj_names={meta['proj_names']}")
        # eyeball guard: NO gate must ever land in the projection group.
        for nm in meta["proj_names"]:
            assert "W_gamma" not in nm and "W_alpha" not in nm, \
                f"gate leaked into projection group: {nm}"
        if verbose:
            print(f"  muP: scale={meta['scale']:.4f} "
                  f"base_lr={meta['base_lr']:.2e} proj_lr={meta['proj_lr']:.2e} "
                  f"| n_proj={meta['n_proj']} n_base={meta['n_base']}")
            print(f"  muP projection params: {meta['proj_names']}")
        opt = torch.optim.Adam(groups, betas=(0.9, 0.999))
        return opt, meta
    opt = torch.optim.Adam(model.parameters(), lr=base_lr)
    return opt, {"scale": 1.0, "proj_lr": base_lr, "base_lr": base_lr,
                 "n_proj": 0, "n_base": sum(1 for _ in model.parameters())}


# ===========================================================================
# #1 warmup + cosine schedule
# ===========================================================================

def warmup_cosine(optimizer, warmup_steps: int, total_steps: int,
                  min_lr_ratio: float = 0.1):
    """Linear warmup 0->1 over warmup_steps, then cosine decay 1->min_lr_ratio.
    Multiplicative factor on EACH group's own initial lr, so muP's base /
    projection lrs are both scheduled proportionally."""
    from torch.optim.lr_scheduler import LambdaLR
    warmup_steps = max(1, warmup_steps)

    def fn(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        prog = min(1.0, prog)
        cos = 0.5 * (1.0 + math.cos(math.pi * prog))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos

    return LambdaLR(optimizer, fn)


# ===========================================================================
# #5 gate-bias — forget-open weight nudge (frozen layers use bias=False, so
# there is no bias to init; we nudge the gate input-projection instead).
# Weakest lever, OFF by default — attribution says gates are healthy.
# ===========================================================================

def apply_gate_bias(model, amount=0.05):
    applied = 0
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, SelectiveRapiditySqrtScanLayer):
                # push gamma pre-activation positive -> sigmoid -> "remember"
                mod.W_gamma.weight.add_(amount)
                applied += 1
    return applied


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===========================================================================
# Train / eval  (identical loop to instrumented_runner.py, two knobs added:
# args.clip replaces the literal 5.0, and sched.step() after opt.step())
# ===========================================================================

def train_epoch(model, X, Y, M, opt, clip=5.0, sched=None):
    model.train()
    perm = torch.randperm(len(X))
    total, nb = 0.0, 0
    for i in range(0, len(X), BATCH_SIZE):
        idx = perm[i:i + BATCH_SIZE]
        xb, yb, mb = X[idx].to(DEVICE), Y[idx].to(DEVICE), M[idx].to(DEVICE)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               yb.reshape(-1), reduction='none')
        loss = (loss * mb.reshape(-1).float()).sum() / (mb.sum() + 1e-6)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)   # #4 knob
        opt.step()
        if sched is not None:                                       # #1 per-step
            sched.step()
        total += loss.item()
        nb += 1
    return total / nb


@torch.no_grad()
def evaluate(model, X, Y, M):
    model.eval()
    total, correct, masked, nb = 0.0, 0, 0, 0
    for i in range(0, len(X), BATCH_SIZE):
        xb = X[i:i + BATCH_SIZE].to(DEVICE)
        yb = Y[i:i + BATCH_SIZE].to(DEVICE)
        mb = M[i:i + BATCH_SIZE].to(DEVICE)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               yb.reshape(-1), reduction='none')
        ml = (loss * mb.reshape(-1).float()).sum() / (mb.sum() + 1e-6)
        total += ml.item()
        preds = logits.argmax(dim=-1)
        correct += ((preds == yb) & mb.bool()).sum().item()
        masked += mb.sum().item()
        nb += 1
    avg = total / nb
    return avg, math.exp(avg), correct / masked


@torch.no_grad()
def probe_velocity(model, X, max_batches=10):
    """Run a few forward passes through a GateVelocityProbe to measure |v|
    saturation / gate health. Selective only (probe keys off the gate layer);
    returns {} for pure (no gates to read)."""
    has_gate = any(isinstance(m, SelectiveRapiditySqrtScanLayer)
                   for m in model.modules())
    if not has_gate:
        return {}
    probe = GateVelocityProbe(model)
    probe.reset()
    model.eval()
    for i in range(0, min(len(X), max_batches * BATCH_SIZE), BATCH_SIZE):
        model(X[i:i + BATCH_SIZE].to(DEVICE))
    s = probe.summary()
    probe.remove()
    return s


# ===========================================================================
# One configured run
# ===========================================================================

def run_one(cfg, vocab_size, mask_idx, X_tr, Y_tr, M_tr, X_val, Y_val, M_val,
            verbose=True):
    """cfg: dict with keys model, d_model, seq_len, epochs, lr, preln, mup,
    warmup, warmup_steps, clip, gate_bias, tag."""
    torch.manual_seed(SEED)
    d_model = cfg["d_model"]
    n_heads = max(1, d_model // 32)
    d_head = d_model // n_heads

    model = FixableLM(
        vocab_size, mask_idx, model_kind=cfg["model"], d_model=d_model,
        n_layers=N_LAYERS, n_heads=n_heads, d_head=d_head,
        seq_len=cfg["seq_len"], dropout=DROPOUT, causal=True,
        use_preln=cfg["preln"],
    ).to(DEVICE)

    if cfg["gate_bias"]:
        n = apply_gate_bias(model)
        if verbose:
            print(f"  gate-bias applied to {n} scan layer(s)")

    opt, opt_meta = build_optimizer(model, cfg["lr"], d_model, cfg["mup"],
                                    n_layers=N_LAYERS, verbose=verbose)

    total_steps = cfg["epochs"] * math.ceil(len(X_tr) / BATCH_SIZE)
    sched = (warmup_cosine(opt, cfg["warmup_steps"], total_steps)
             if cfg["warmup"] else None)

    n_params = count_params(model)
    if verbose:
        print(f"  params {n_params:,} ({n_params/1e6:.3f}M) | "
              f"preln={cfg['preln']} mup={cfg['mup']} warmup={cfg['warmup']} "
              f"clip={cfg['clip']} gate_bias={cfg['gate_bias']}")

    t0 = time.time()
    epoch_ppls, epoch_accs = [], []
    # Match phase1 baseline protocol so best_ppl is comparable: 8 epochs + early-stop.
    best_es, since_es, patience = float("inf"), 0, 2
    for ep in range(cfg["epochs"]):
        tl = train_epoch(model, X_tr, Y_tr, M_tr, opt,
                         clip=cfg["clip"], sched=sched)
        vl, vppl, vacc = evaluate(model, X_val, Y_val, M_val)
        epoch_ppls.append(vppl)
        epoch_accs.append(vacc)
        lrs = [f"{g['lr']:.2e}" for g in opt.param_groups]  # base + (muP) projection
        if verbose:
            print(f"  [{cfg['tag']}] {cfg['model']} d{d_model}/T{cfg['seq_len']} "
                  f"ep {ep+1}/{cfg['epochs']} | train {tl:.4f} | "
                  f"ppl {vppl:.2f} | acc {vacc:.4f} | lr {','.join(lrs)}")
        if vppl < best_es - 0.5:
            best_es, since_es = vppl, 0
        else:
            since_es += 1
            if since_es >= patience:
                if verbose:
                    print(f"    early-stop (no improvement {patience} epochs)")
                break
    elapsed = time.time() - t0

    vel = probe_velocity(model, X_val)
    if verbose and vel:
        print(f"  velocity-sat |v|>0.95 = {vel.get('velocity_sat_gt95')} "
              f"| |v|̄ = {vel.get('velocity_abs_mean')} "
              f"| γ̄ = {vel.get('gamma_mean')} ᾱ = {vel.get('alpha_mean')}")

    del model
    if DEVICE.type == "mps":
        torch.mps.empty_cache()

    return {
        "tag": cfg["tag"],
        "model": cfg["model"],
        "d_model": d_model,
        "seq_len": cfg["seq_len"],
        "flags": {k: cfg[k] for k in
                  ("preln", "mup", "warmup", "warmup_steps", "clip", "gate_bias")},
        "lr": cfg["lr"],
        "params_M": round(n_params / 1e6, 3),
        "opt": {k: opt_meta[k] for k in ("scale", "proj_lr", "base_lr",
                                         "n_proj", "n_base") if k in opt_meta},
        "best_ppl": round(min(epoch_ppls), 3),
        "final_ppl": round(epoch_ppls[-1], 3),
        "final_acc": round(epoch_accs[-1], 4),
        "epoch_ppls": [round(p, 3) for p in epoch_ppls],
        "epoch_accs": [round(a, 4) for a in epoch_accs],
        "velocity_metrics": vel,
        "time_s": round(elapsed, 1),
    }


# ===========================================================================
# Ablation matrix (§5)
# ===========================================================================

def _cfg(model, d_model, seq_len, epochs, lr, tag,
         preln=False, mup=False, warmup=False, warmup_steps=1000,
         clip=5.0, gate_bias=False):
    return dict(model=model, d_model=d_model, seq_len=seq_len, epochs=epochs,
                lr=lr, tag=tag, preln=preln, mup=mup, warmup=warmup,
                warmup_steps=warmup_steps, clip=clip, gate_bias=gate_bias)


def matrix_configs(seq_len, epochs, lr, warmup_steps):
    """The §5 experiment matrix: baseline + each lever alone + full stack at
    d=512, + d=128 full-stack sanity, for selective; then baseline/mup/stack
    for pure. 10 runs total."""
    cfgs = []
    # --- selective, d=512 ---
    cfgs.append(_cfg("selective", 512, seq_len, epochs, lr, "baseline", clip=5.0))
    cfgs.append(_cfg("selective", 512, seq_len, epochs, lr, "warmup",
                     warmup=True, warmup_steps=warmup_steps, clip=5.0))
    cfgs.append(_cfg("selective", 512, seq_len, epochs, lr, "preln",
                     preln=True, clip=5.0))
    cfgs.append(_cfg("selective", 512, seq_len, epochs, lr, "mup",
                     mup=True, clip=5.0))
    cfgs.append(_cfg("selective", 512, seq_len, epochs, lr, "clip", clip=1.0))
    cfgs.append(_cfg("selective", 512, seq_len, epochs, lr, "stack",
                     preln=True, mup=True, warmup=True,
                     warmup_steps=warmup_steps, clip=1.0))
    # --- selective, d=128 sanity (fixes must not hurt the working width) ---
    cfgs.append(_cfg("selective", 128, seq_len, epochs, lr, "stack_d128",
                     preln=True, mup=True, warmup=True,
                     warmup_steps=warmup_steps, clip=1.0))
    # --- pure, d=512: baseline / mup / stack (gate-free family check) ---
    cfgs.append(_cfg("pure", 512, seq_len, epochs, lr, "pure_baseline", clip=5.0))
    cfgs.append(_cfg("pure", 512, seq_len, epochs, lr, "pure_mup",
                     mup=True, clip=5.0))
    cfgs.append(_cfg("pure", 512, seq_len, epochs, lr, "pure_stack",
                     preln=True, mup=True, warmup=True,
                     warmup_steps=warmup_steps, clip=1.0))
    return cfgs


def run_matrix(seq_len, epochs, lr, warmup_steps, max_tokens=None):
    print("=" * 72)
    print("PHASE-2 WIDTH-FIX ABLATION MATRIX  (full data, instrumented)  by Opus 4.8")
    print(f"velocity probe: {_PROBE_SOURCE}")
    print("=" * 72)
    print("Loading WikiText-2 (full corpus)...")
    train_text, val_text = load_wikitext2()
    vocab, stoi, unk_idx, mask_idx = build_vocab(train_text)
    vocab_size = len(vocab)
    train_ids = tokenize(train_text, stoi, unk_idx)
    val_ids = tokenize(val_text, stoi, unk_idx)
    print(f"Vocab: {vocab_size} | tokens: train {len(train_ids):,}, "
          f"val {len(val_ids):,}")

    cfgs = matrix_configs(seq_len, epochs, lr, warmup_steps)
    out_file = RESULTS_DIR / "width_fix.json"
    results = {}

    # Batches depend only on seq_len here -> build once per seq_len.
    X_tr, Y_tr, M_tr = make_mlm_batches(train_ids, seq_len, BATCH_SIZE,
                                        mask_idx, MASK_PROB, max_tokens=max_tokens)
    X_val, Y_val, M_val = make_mlm_batches(val_ids, seq_len, BATCH_SIZE,
                                           mask_idx, MASK_PROB)
    print(f"train batches: {len(X_tr)//BATCH_SIZE}, "
          f"val batches: {len(X_val)//BATCH_SIZE}\n")

    for i, cfg in enumerate(cfgs, 1):
        key = f"{cfg['model']}_{cfg['tag']}_d{cfg['d_model']}_T{cfg['seq_len']}"
        print(f"── [{i}/{len(cfgs)}] {key} ──")
        try:
            res = run_one(cfg, vocab_size, mask_idx, X_tr, Y_tr, M_tr,
                          X_val, Y_val, M_val)
            results[key] = res
            print(f"  -> best PPL {res['best_ppl']:.2f} | "
                  f"final acc {res['final_acc']:.4f} | "
                  f"vel-sat {res['velocity_metrics'].get('velocity_sat_gt95','—')} "
                  f"| {res['time_s']}s\n")
        except Exception as e:                       # keep the sweep alive
            print(f"  -> ERROR: {e}\n")
            results[key] = {"error": str(e)[:300], "config": cfg}
        with open(out_file, "w") as f:               # incremental dump
            json.dump(results, f, indent=2)

    print(f"Results written to {out_file}")
    _print_matrix_table(results)
    return results


def _print_matrix_table(results):
    print("\n" + "=" * 72)
    print("ABLATION SUMMARY")
    print("=" * 72)
    hdr = f"{'run':<28} {'PPL':>8} {'acc':>7} {'vel-sat':>8} {'params':>8}"
    print(hdr)
    print("-" * len(hdr))
    for key, r in results.items():
        if "error" in r:
            print(f"{key:<28} {'ERR':>8}")
            continue
        vs = r["velocity_metrics"].get("velocity_sat_gt95", "—")
        print(f"{key:<28} {r['best_ppl']:>8.2f} {r['final_acc']:>7.4f} "
              f"{str(vs):>8} {r['params_M']:>7.3f}M")
    print("\nReading (baselines: sel d512=342 collapsed, pure d512=280 healthy, "
          "sel d128=161):")
    print("  • A Selective d512 row that breaks acc 0.1643 and lands well below 342 "
          "= that lever broke the deadlock.")
    print("  • If `gate-bias` or `mup` alone rescues it → init/gate symmetry was the "
          "cause (matches the Selective-only signature). If `warmup`/`clip` alone → "
          "grad-spike. If only `stack` → complementary.")
    print("  • Pure d512 rows test HARM, not rescue (Pure already reaches 280 unaided): "
          "a fix that pushes Pure much above 280 is hurting a healthy model.")
    print("  • `stack_d128` must NOT regress vs ref d128 (~161).")


# ===========================================================================
# CLI
# ===========================================================================

def build_argparser():
    p = argparse.ArgumentParser(
        description="Phase-2 Width-Fix Runner — d=512 collapse — by Opus 4.8")
    p.add_argument("--model", choices=["selective", "pure"], default="selective")
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--epochs", type=int, default=8)  # match phase1 (8 ep + early-stop)
    p.add_argument("--lr", type=float, default=LR)
    # ---- the five levers, each independently toggleable ----
    p.add_argument("--preln", action="store_true",
                   help="#2 Pre-LN block (src re-wrap of the LN/FFN boilerplate)")
    p.add_argument("--mup", action="store_true",
                   help="#3 muP-lite per-fan-in lr on the unbounded projections")
    p.add_argument("--warmup", action="store_true",
                   help="#1 warmup+cosine schedule")
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--clip", type=float, default=5.0,
                   help="#4 grad clip (1.0 = fixed, 5.0 = reproduce baseline)")
    p.add_argument("--gate-bias", action="store_true",
                   help="#5 forget-open gate nudge (weakest lever, OFF unless "
                        "attribution shows frozen gates)")
    p.add_argument("--tag", default="run", help="suffix for results json")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="cap training tokens (for smoke runs)")
    # ---- modes ----
    p.add_argument("--matrix", action="store_true",
                   help="run the full §5 ablation matrix instead of one config")
    p.add_argument("--smoke", action="store_true",
                   help="tiny smoke: d=128, capped data, 1 epoch, all flags on")
    return p


def main():
    args = build_argparser().parse_args()
    print(f"Device: {DEVICE} | velocity probe: {_PROBE_SOURCE}")

    if args.matrix:
        run_matrix(args.seq_len, args.epochs, args.lr, args.warmup_steps,
                   max_tokens=args.max_tokens)
        return

    # ---- single configured run ----
    print("Loading WikiText-2...")
    train_text, val_text = load_wikitext2()
    vocab, stoi, unk_idx, mask_idx = build_vocab(train_text)
    vocab_size = len(vocab)
    train_ids = tokenize(train_text, stoi, unk_idx)
    val_ids = tokenize(val_text, stoi, unk_idx)
    print(f"Vocab: {vocab_size} | tokens: train {len(train_ids):,}, "
          f"val {len(val_ids):,}")

    if args.smoke:
        # TINY smoke: prove the whole machine runs end to end, cheaply.
        cfg = _cfg(args.model, 128, args.seq_len, 1, args.lr, "smoke",
                   preln=True, mup=True, warmup=True, warmup_steps=20,
                   clip=1.0, gate_bias=args.gate_bias)
        max_tokens = 30_000
        print("SMOKE: d=128, 1 epoch, ~30K tokens, preln+mup+warmup+clip1.0")
    else:
        cfg = _cfg(args.model, args.d_model, args.seq_len, args.epochs, args.lr,
                   args.tag, preln=args.preln, mup=args.mup, warmup=args.warmup,
                   warmup_steps=args.warmup_steps, clip=args.clip,
                   gate_bias=args.gate_bias)
        max_tokens = args.max_tokens

    X_tr, Y_tr, M_tr = make_mlm_batches(train_ids, cfg["seq_len"], BATCH_SIZE,
                                        mask_idx, MASK_PROB, max_tokens=max_tokens)
    X_val, Y_val, M_val = make_mlm_batches(val_ids, cfg["seq_len"], BATCH_SIZE,
                                           mask_idx, MASK_PROB,
                                           max_tokens=max_tokens)
    print(f"train batches: {len(X_tr)//BATCH_SIZE}, "
          f"val batches: {len(X_val)//BATCH_SIZE}\n")

    res = run_one(cfg, vocab_size, mask_idx, X_tr, Y_tr, M_tr,
                  X_val, Y_val, M_val)
    out_file = RESULTS_DIR / f"width_fix_{cfg['tag']}.json"
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nbest PPL {res['best_ppl']:.2f} | final acc {res['final_acc']:.4f}")
    print(f"Saved {out_file}")


if __name__ == "__main__":
    main()
