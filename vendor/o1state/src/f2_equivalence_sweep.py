#!/usr/bin/env python3
"""F2 LOCKING EXPERIMENT — the exactness license, swept.

FOUNDATIONS.md F2 claims layout decoupling holds "for the selective scalar
scan, the complex holographic scan, and biased-gamma variants", and cites
results/streaming_check.json. That artifact measures ONE operator at ONE
(chunk, overlap) point. The claim is broader than its anchor.

This sweep closes that gap. For each operator x each (chunk, overlap) point
it measures the two exactness limbs F2 actually asserts:

  (1) LAYOUT EQUIVALENCE — full-sequence forward from zero state vs
      chunked-carried forward. Same operator or not.
  (2) GRADIENT EXACTNESS — truncated BPTT with detached carry vs
      full-window BPTT: cosine and relative error over all parameters.

F2's own precondition is that the chunk must exceed the receptive field r.
The sweep therefore includes chunks BELOW the stated horizon on purpose: a
license with no measured failure mode is a claim, not a law. If small
chunks degrade, that is the boundary F2 predicts and it belongs in the
table; if they do not, F2 is stronger than stated and that belongs there too.

Runs on CPU in well under a minute. No training, no data — pure operator
identity.
"""
import json
import os
import sys

import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from streaming_train import StreamingNoPELM
from holo_stream_recall import StreamingHolographicLM

torch.set_num_threads(1)

VOCAB = 256
MASK = 0
FULL_T = 512
SEED = 0

# (chunk, overlap) grid. overlap=0 is the pure detach-carry case F2 states;
# nonzero overlap is the warmup variant the existing check used.
GRID = [(64, 16), (64, 0), (128, 16), (128, 0), (32, 8), (16, 0)]

# The operators under test. Each is a distinct class, not a flag: the
# holographic scan lives in holo_stream_recall (the streaming subclass —
# plain HolographicLM.forward takes no state and is therefore not a chunked
# operator at all).
#
# NOTE on F2's wording: its text names "biased-gamma variants" as a third
# measured operator. No such streaming operator exists in this checkout —
# gamma bias is an initialisation of the persistence channels inside the
# complex scan, not a separate scan. The sweep substitutes the two read
# paths of the complex scan plus a phase-off control, which is what can
# honestly be measured here; F2's wording is corrected to match.
OPERATORS = [
    {"tag": "selective_scalar", "cls": StreamingNoPELM, "kwargs": {},
     "desc": "the selective scalar scan (F2's primary anchor)"},
    {"tag": "holographic_complex", "cls": StreamingHolographicLM,
     "kwargs": {"use_phase": True},
     "desc": "the complex holographic scan"},
    {"tag": "holographic_tanh_m", "cls": StreamingHolographicLM,
     "kwargs": {"use_phase": True, "readout": "tanh_m"},
     "desc": "the complex scan under the tanh_m readout (the second read path)"},
    {"tag": "phase_off_control", "cls": StreamingHolographicLM,
     "kwargs": {"use_phase": False},
     "desc": "phase-off control: same container, no complex binding — "
             "isolates whether the exactness comes from the scan or the phase"},
]


def build(op):
    """Build a streaming model, tolerating operators this checkout does not
    expose. Returns (model, None) or (None, reason)."""
    torch.manual_seed(SEED)
    try:
        return op["cls"](VOCAB, MASK, d_model=128, n_layers=2, n_heads=4,
                         d_head=32, seq_len=32, dropout=0.0, causal=True,
                         **op["kwargs"]), None
    except (TypeError, ValueError) as e:
        return None, f"{type(e).__name__}: {e}"


def _detach(obj):
    """Detach every tensor in a nested state structure, preserving its shape.
    The operators under test carry different containers (list of tensors vs
    list of dicts), and the sweep must not assume either."""
    if torch.is_tensor(obj):
        return obj.detach()
    if isinstance(obj, dict):
        return {k: _detach(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_detach(v) for v in obj)
    return obj


def layout_equivalence(model, chunk, overlap):
    """Limb 1: full-sequence forward vs chunked-carried forward, same weights,
    same input, no gradient. Returns max |delta| over all logits."""
    model.eval()
    x = torch.randint(0, VOCAB, (2, FULL_T))
    with torch.no_grad():
        full, _ = model(x, None)
        outs, states, pos = [], None, 0
        while pos < FULL_T:
            lo = max(0, pos - overlap)
            hi = min(FULL_T, pos + chunk)
            lg, states = model(x[:, lo:hi], states)
            outs.append(lg[:, pos - lo:, :])
            pos = hi
        chunked = torch.cat(outs, dim=1)
    return float((full - chunked).abs().max())


def gradient_exactness(model, chunk, overlap):
    """Limb 2: truncated BPTT with detached carry vs full-window BPTT.
    Returns (cosine, relative L2 error)."""
    model.train()
    x = torch.randint(0, VOCAB, (1, FULL_T))
    y = torch.randint(0, VOCAB, (1, FULL_T))
    lossf = nn.CrossEntropyLoss()

    # float64 for the comparison itself: in float32 the cosine of two nearly
    # identical high-dimensional vectors rounds ABOVE 1.0 (observed
    # 1.000186), which is not a number a cosine can take and would put the
    # measurement at its own resolution limit. The forward/backward stay in
    # the model's native dtype; only the accumulated gradient vectors are
    # promoted before the dot products.
    model.zero_grad()
    logits, _ = model(x, None)
    lossf(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).backward()
    g_full = torch.cat([p.grad.reshape(-1) for p in model.parameters()
                        if p.grad is not None]).double()

    model.zero_grad()
    states, pos = None, 0
    while pos < FULL_T:
        lo = max(0, pos - overlap)
        hi = min(FULL_T, pos + chunk)
        logits, states = model(x[:, lo:hi], states)
        score_from = pos - lo
        lg = logits[:, score_from:, :]
        tg = y[:, pos:hi]
        (lossf(lg.reshape(-1, lg.size(-1)), tg.reshape(-1)) * (hi - pos) / FULL_T).backward()
        # detach the carried state so the next chunk's graph starts fresh —
        # this IS the truncation under test. Structure-agnostic because the
        # operators disagree on shape: the scalar scan carries a list of
        # tensors, the holographic scan a list of {"S_re","S_im",...} dicts.
        states = _detach(states)
        pos = hi
    g_trunc = torch.cat([p.grad.reshape(-1) for p in model.parameters()
                         if p.grad is not None]).double()

    cos = float(torch.nn.functional.cosine_similarity(g_full, g_trunc, dim=0))
    rel = float((g_full - g_trunc).norm() / g_full.norm().clamp_min(1e-30))
    return cos, rel


rows = []
for op in OPERATORS:
    model, reason = build(op)
    if model is None:
        print(f"[{op['tag']}] NOT AVAILABLE — {reason}", flush=True)
        rows.append({"operator": op["tag"], "desc": op["desc"],
                     "available": False, "reason": reason, "points": []})
        continue
    print(f"\n[{op['tag']}] {op['desc']}", flush=True)
    pts = []
    for chunk, overlap in GRID:
        m, _ = build(op)                     # fresh weights per point
        fwd = layout_equivalence(m, chunk, overlap)
        cos, rel = gradient_exactness(m, chunk, overlap)
        pts.append({"chunk": chunk, "overlap": overlap,
                    "layout_max_abs_delta": fwd,
                    "grad_cosine": round(cos, 12),
                    "grad_rel_err": round(rel, 12)})
        print(f"  chunk={chunk:>4} overlap={overlap:>3} | layout |Δ| {fwd:.3e} | "
              f"grad cos {cos:.12f} | rel err {rel:.3e}", flush=True)
    rows.append({"operator": op["tag"], "desc": op["desc"],
                 "available": True, "points": pts})

avail = [r for r in rows if r["available"]]
allpts = [p for r in avail for p in r["points"]]
worst_layout = max((p["layout_max_abs_delta"] for p in allpts), default=None)
worst_cos = min((p["grad_cosine"] for p in allpts), default=None)

out = {
    "check": "F2 exactness license, swept over operators x (chunk, overlap)",
    "full_T": FULL_T, "seed": SEED, "torch_threads": 1,
    "operators_claimed_by_F2": [o["tag"] for o in OPERATORS],
    "operators_measured": [r["operator"] for r in avail],
    "operators_unavailable": [{"operator": r["operator"], "reason": r["reason"]}
                              for r in rows if not r["available"]],
    "rows": rows,
    "worst_layout_max_abs_delta": worst_layout,
    "worst_grad_cosine": worst_cos,
    "note": ("Supersedes results/streaming_check.json as F2's anchor: that file "
             "measured one operator at one (chunk, overlap) point while F2's text "
             "claims three operators. Chunks below the stated receptive field are "
             "included deliberately — a license with no measured failure mode is a "
             "claim, not a law."),
}
path = os.path.join(REPO_ROOT, "results", "f2_equivalence_sweep.json")
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nworst layout |Δ| = {worst_layout:.3e} | worst grad cosine = {worst_cos:.12f}")
print(f"-> {path}")
