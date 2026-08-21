#!/usr/bin/env python3 -u
"""
MQAR associative-recall harness — by Opus 4.8   (v2, canonical layout)
=====================================================================

Multi-Query Associative Recall (Zoology/Based line). Separates models that can
do EXACT key-value recall from those that can't. GSSM is predicted to LOSE here
(bounded scalar state can't bind key→value); hybrids (GSSM+attention) and pure
attention are predicted to WIN. This is the honest-limitation benchmark.

CANONICAL LAYOUT (v2 — fixes the v1 SEP-slot bug that capped attention at 1/n_pairs).
A sequence is a stream of (key, value) pairs, then queries that REPEAT a key; the
target is the paired value, predicted as the NEXT TOKEN at the query-key position:

    K3 V41 K1 V52 K7 V44 ... K1 [predict→V52] ... K7 [predict→V44] ...
                                ^score here, target = the value bound to K1

Why canonical works: the query KEY carries identity, so a single induction head
(or a model that bound k→v) solves it by looking back to the matching key. The v1
design scored at a content-free SEP slot, giving the model no key to route on —
attention itself capped at 1/n_pairs, which would have framed GSSM for a cliff the
HARNESS manufactured. Verified fix: canonical layout → attention reaches ~1.0.

Queries are SPREAD across the sequence (not packed at the end) so gap distances
cover [1, seq_len] and the predicted recall cliff (~gap 9-10 for bounded scalar
state) actually gets sampled in every bin.

Committed prediction: pure-GSSM recall collapses past gap ~9-10; attention flat
~100%; GSSM+attention hybrid recovers attention-level recall.

MPS-safe: all tensors real long, built on CPU, .to(device) at the boundary.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Gap bins for the cliff analysis (fine near the predicted cliff at ~9-10).
GAP_BINS = [(1, 2), (3, 4), (5, 8), (9, 12), (13, 16), (17, 24),
            (25, 32), (33, 48), (49, 64), (65, 128), (129, 256), (257, 1024)]


# ===========================================================================
# 1. DATA GENERATION — canonical MQAR, queries spread, score at query-key pos
# ===========================================================================

def make_mqar_batch(batch_size, seq_len, n_pairs, n_queries,
                    n_keys=64, n_values=64, device="cpu", generator=None):
    """One MQAR batch, canonical layout.

    Layout per sequence:
      - n_pairs (key, value) tokens laid down first, at the front.
      - n_queries query-KEY tokens spread through the remaining positions.
      - At each query-key position p (input = the key id), the TARGET is the
        value bound to that key, scored as the next-token prediction AT p.
        (Causal LM: logits[:, p] predict the token that should follow — the value.)

    Returns
    -------
    tokens     (B, T) long  -- input ids in [0, SEP_ID]
    targets    (B, T) long  -- value id at query-key positions, 0 elsewhere
    loss_mask  (B, T) bool  -- True only at query-key positions (scored slots)
    gap        (B, T) long  -- distance from query-key pos back to its key's pos
    """
    SEP_ID = n_keys + n_values            # padding / filler token id
    vocab_size = SEP_ID + 1
    g = generator

    tokens    = torch.full((batch_size, seq_len), SEP_ID, dtype=torch.long)
    targets   = torch.zeros((batch_size, seq_len), dtype=torch.long)
    loss_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    gap       = torch.zeros((batch_size, seq_len), dtype=torch.long)

    # KV block occupies 2*n_pairs positions at the front; queries live after it.
    kv_len = 2 * n_pairs
    assert kv_len + n_queries <= seq_len, \
        f"sequence too short: 2*{n_pairs}+{n_queries} > {seq_len}"
    assert n_queries <= n_pairs <= n_keys, "need n_queries <= n_pairs <= n_keys"

    for b in range(batch_size):
        # --- KV region: distinct keys, values (with replacement) ---
        keys = torch.randperm(n_keys, generator=g)[:n_pairs]
        vals = torch.randint(0, n_values, (n_pairs,), generator=g) + n_keys
        key_value, key_pos = {}, {}
        for i in range(n_pairs):
            kp, vp = 2 * i, 2 * i + 1
            tokens[b, kp] = keys[i]
            tokens[b, vp] = vals[i]
            key_value[int(keys[i])] = int(vals[i])
            key_pos[int(keys[i])] = kp        # gap measured to the KEY position

        # --- QUERY region: spread queries across [kv_len, seq_len) ---
        q_keys = keys[torch.randperm(n_pairs, generator=g)[:n_queries]]
        slots = torch.arange(kv_len, seq_len)
        # choose n_queries distinct positions spread across the available slots
        perm = torch.randperm(len(slots), generator=g)[:n_queries]
        q_positions = slots[perm.sort().values]   # sorted for stable gap stats
        for j in range(n_queries):
            qp = int(q_positions[j])
            k = int(q_keys[j])
            tokens[b, qp]     = k                 # query repeats the key
            targets[b, qp]    = key_value[k]      # next-token target = bound value
            loss_mask[b, qp]  = True
            gap[b, qp]        = qp - key_pos[k]

    return (tokens.to(device), targets.to(device),
            loss_mask.to(device), gap.to(device))


def decode_example(tokens, targets, loss_mask, n_keys=64, n_values=64):
    """Human-readable single-sequence dump."""
    SEP_ID = n_keys + n_values
    toks, tgts, msk = tokens.tolist(), targets.tolist(), loss_mask.tolist()
    out = []
    for pos, t in enumerate(toks):
        if t < n_keys:
            sym = f"K{t}"
        elif t < SEP_ID:
            sym = f"V{t}"
        else:
            sym = "·"
        if msk[pos]:
            sym = f"[{sym}→V{tgts[pos]}]"       # query-key, predicting its value
        out.append(sym)
    return " ".join(out)


# ===========================================================================
# 2. EVAL — exact-match accuracy at scored slots, binned by gap
# ===========================================================================

@torch.no_grad()
def mqar_accuracy(model, cfg, n_batches, seed, device=DEVICE):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    bin_correct = {b: 0 for b in GAP_BINS}
    bin_total = {b: 0 for b in GAP_BINS}
    tot_correct = tot = 0
    for _ in range(n_batches):
        tokens, targets, mask, gap = make_mqar_batch(generator=gen, device=device, **cfg)
        logits = model(tokens)
        preds = logits.argmax(-1)               # (B,T) next-token prediction
        hit = (preds == targets) & mask
        tot_correct += hit.sum().item()
        tot += mask.sum().item()
        # bin by gap
        gvals = gap[mask]
        hvals = hit[mask]
        for (lo, hi) in GAP_BINS:
            sel = (gvals >= lo) & (gvals <= hi)
            bin_total[(lo, hi)] += sel.sum().item()
            bin_correct[(lo, hi)] += (hvals & sel).sum().item()
    overall = tot_correct / max(1, tot)
    by_gap = {f"{lo}-{hi}": (bin_correct[(lo, hi)] / bin_total[(lo, hi)]
                             if bin_total[(lo, hi)] else None)
              for (lo, hi) in GAP_BINS}
    return overall, by_gap, {f"{lo}-{hi}": bin_total[(lo, hi)] for (lo, hi) in GAP_BINS}


# ===========================================================================
# 3. TRAIN
# ===========================================================================

def mqar_train(model, cfg, steps, lr=3e-3, seed=0, device=DEVICE, log_every=200):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    for step in range(steps):
        tokens, targets, mask, _ = make_mqar_batch(generator=gen, device=device, **cfg)
        logits = model(tokens)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1), reduction='none')
        loss = (loss * mask.reshape(-1).float()).sum() / (mask.sum() + 1e-6)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if log_every and (step + 1) % log_every == 0:
            print(f"    step {step+1}/{steps} | loss {loss.item():.4f}")
    return model


def run_mqar(model, train_cfg, test_cfg, steps, lr=3e-3, seed=42, device=DEVICE):
    """Train at train_cfg length, freeze, eval at train AND test length."""
    mqar_train(model, train_cfg, steps, lr=lr, seed=seed, device=device)
    tr_overall, tr_gap, tr_n = mqar_accuracy(model, train_cfg, 8, seed + 1, device)
    te_overall, te_gap, te_n = mqar_accuracy(model, test_cfg, 8, seed + 2, device)
    return {
        "train_len": {"overall": round(tr_overall, 4), "by_gap": tr_gap, "n": tr_n},
        "test_len": {"overall": round(te_overall, 4), "by_gap": te_gap, "n": te_n},
    }


# ===========================================================================
# 4. Self-contained baseline (tiny causal transformer) — file is testable alone
# ===========================================================================

class TinyCausalTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=2, n_heads=4,
                 max_len=1024, dropout=0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        T = x.size(1)
        h = self.embed(x) + self.pos[:, :T]
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        h = self.enc(h, mask=mask)
        return self.head(h)


def build_project_model(arch, vocab_size, mask_idx, d_model=128, n_layers=2,
                        n_heads=4, d_head=32, seq_len=64):
    """Lazy factory for the project's GSSM/transformer LMs (imported on demand)."""
    REF = Path(__file__).resolve().parent.parent / "reference"
    if str(REF) not in sys.path:
        sys.path.insert(0, str(REF))
    if arch == "selective":
        from moebius_scan_transformer_selective import SelectiveRapiditySqrtTransformerLM as C
    elif arch == "pure":
        from moebius_scan_transformer_sqrt import SqrtCouplingMoebiusScanTransformerLM as C
    elif arch == "transformer":
        return TinyCausalTransformerLM(vocab_size, d_model, n_layers, n_heads)
    else:
        raise ValueError(f"unknown arch {arch}")
    return C(vocab_size, mask_idx, d_model=d_model, n_layers=n_layers,
             n_heads=n_heads, d_head=d_head, seq_len=seq_len, dropout=0.0, causal=True)


# ===========================================================================
# 5. Smoke test — WITH the validity-gate assertion (spec sanity #4)
# ===========================================================================

def _smoke():
    print("=== MQAR harness smoke test (canonical v2) ===")
    n_keys = n_values = 16
    SEP_ID = n_keys + n_values
    vocab_size = SEP_ID + 1
    cfg = dict(batch_size=32, seq_len=32, n_pairs=4, n_queries=4,
               n_keys=n_keys, n_values=n_values)

    gen = torch.Generator().manual_seed(0)
    tokens, targets, mask, gap = make_mqar_batch(generator=gen, device="cpu", **cfg)
    print(f"shapes: tokens {tuple(tokens.shape)} targets {tuple(targets.shape)} "
          f"mask {tuple(mask.shape)} | scored slots/seq = {mask[0].sum().item()}")
    print("example:", decode_example(tokens[0], targets[0], mask[0], n_keys, n_values))
    # sanity: every scored slot's target is a value id, input is a key id
    assert (tokens[mask] < n_keys).all(), "query-key inputs must be key ids"
    assert (targets[mask] >= n_keys).all() and (targets[mask] < SEP_ID).all(), \
        "targets must be value ids"

    print("\ntraining tiny attention baseline (must reach ~100% — validity gate)...")
    model = TinyCausalTransformerLM(vocab_size, d_model=64, n_layers=2, n_heads=4, max_len=64)
    mqar_train(model, cfg, steps=800, lr=3e-3, seed=1, device=DEVICE, log_every=400)
    overall, by_gap, n = mqar_accuracy(model, cfg, 8, seed=2, device=DEVICE)
    print(f"attention baseline train-len acc: {overall:.3f}")
    print("by gap:", {k: (round(v, 2) if v is not None else None)
                      for k, v in by_gap.items() if n[k] > 0})
    # THE GATE: a correct harness lets attention solve MQAR. If not, harness is broken.
    assert overall >= 0.90, (
        f"HARNESS BROKEN: attention baseline only {overall:.3f} at train len "
        f"(spec sanity#4 wants ~100%). Do NOT trust any GSSM numbers from this harness.")
    print("\n✓ validity gate PASSED — harness measures recall correctly.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=None,
                    choices=["selective", "pure", "transformer"],
                    help="run a full MQAR experiment with this arch (else smoke test)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--train-len", type=int, default=64)
    ap.add_argument("--test-len", type=int, default=256)
    args = ap.parse_args()

    if args.arch is None:
        _smoke()
    else:
        n_keys = n_values = 64
        vocab_size = n_keys + n_values + 1
        mask_idx = vocab_size            # never collides with a real id
        train_cfg = dict(batch_size=32, seq_len=args.train_len, n_pairs=args.n_pairs,
                         n_queries=args.n_pairs, n_keys=n_keys, n_values=n_values)
        test_cfg = dict(batch_size=32, seq_len=args.test_len, n_pairs=args.n_pairs,
                        n_queries=args.n_pairs, n_keys=n_keys, n_values=n_values)
        print(f"MQAR {args.arch}: train len {args.train_len}, test len {args.test_len}, "
              f"{args.n_pairs} pairs")
        model = build_project_model(args.arch, vocab_size, mask_idx, seq_len=args.train_len)
        res = run_mqar(model, train_cfg, test_cfg, args.steps, device=DEVICE)
        print(f"\n{args.arch} train-len overall: {res['train_len']['overall']}")
        print(f"{args.arch} test-len  overall: {res['test_len']['overall']}")
        print("test-len by gap:", {k: v for k, v in res['test_len']['by_gap'].items()
                                    if v is not None})
