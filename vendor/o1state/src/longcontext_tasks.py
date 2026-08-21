#!/usr/bin/env python3 -u
"""
Long-context CAPABILITY tasks — where bounded-state wins and attention crashes — 2026-06-25
==========================================================================================
The length-invariance PPL result shows bounded-state holds to 256×. These tasks turn that
into a capability boundary: "GSSM does X at T=8192; attention's forward pass crashes."

Two tasks, both playing to bounded-state STRENGTHS (single-thread state, aggregation) —
NOT multi-key recall (that's the mapped ~13% ceiling). Both require info from far back.

TASK 1 — FLIP-FLOP STATE TRACKING (primary):
  A single register. Stream of IGNORE filler + sparse SET_v writes (overwrite) + QUERY reads.
  At each QUERY, target = the most-recent SET value. Bounded O(1) state is the NATURAL
  representation (one value at a time → no binding ceiling). Gap = query_pos − last_set_pos.

TASK 2 — RUNNING PARITY (secondary):
  A 1-bit counter mod 2. Sparse MARKs in NOISE; at each QUERY, target = parity of MARKs so far.
  Aggregation over an unbounded prefix — the canonical hard case for attention. chance=50%.

Why attention fails at long T: (a) fixed PE buffer crashes past its max_len (2048 sinusoidal /
1024 learned); (b) even below it, no length generalization (OOD positions). NoPE-GSSM has no
positional buffer — it carries state through ordering alone, O(1) memory, runs to any length.

Validity gate: the attention baseline MUST solve the task at TRAIN length, else the harness is
broken (a 1-register flip-flop is trivially attention-solvable at short T). Long-T transformer
crashes are LOGGED AS RESULTS ("CRASH @ T"), not failures — the crash IS the finding.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import math, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

GAP_BINS = [(1, 8), (9, 32), (33, 128), (129, 512), (513, 2048),
            (2049, 8192), (8193, 32768)]


# ===========================================================================
# TASK 1 — Flip-flop state tracking
# ===========================================================================

def make_flipflop_batch(batch_size, seq_len, n_vals=8, p_set=0.10, p_query=0.10,
                        device="cpu", generator=None):
    """Single-register flip-flop.
    vocab layout:  [0..n_vals-1] = SET_v values (also the value targets),
                   n_vals = IGNORE filler,  n_vals+1 = QUERY.
    At each QUERY position, next-token target = the most-recent SET value seen.
    Returns tokens, targets, loss_mask, gap (all (B,T)).
    """
    g = generator
    IGNORE = n_vals
    QUERY = n_vals + 1
    vocab_size = n_vals + 2

    tokens = torch.full((batch_size, seq_len), IGNORE, dtype=torch.long)
    targets = torch.zeros((batch_size, seq_len), dtype=torch.long)
    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    gap = torch.zeros((batch_size, seq_len), dtype=torch.long)

    for b in range(batch_size):
        # draw action per position: SET (write), QUERY (read), or IGNORE
        r = torch.rand(seq_len, generator=g)
        cur_val = -1
        cur_pos = -1
        for t in range(seq_len):
            if r[t] < p_set:
                v = int(torch.randint(0, n_vals, (1,), generator=g))
                tokens[b, t] = v
                cur_val = v
                cur_pos = t
            elif r[t] < p_set + p_query and cur_val >= 0 and t > 0:
                tokens[b, t] = QUERY
                targets[b, t] = cur_val          # next-token = current register value
                mask[b, t] = True
                gap[b, t] = t - cur_pos
            # else IGNORE (already filled)

    return (tokens.to(device), targets.to(device),
            mask.to(device), gap.to(device), vocab_size)


# ===========================================================================
# TASK 2 — Running parity
# ===========================================================================

def make_parity_batch(batch_size, seq_len, p_mark=0.12, p_query=0.10,
                      device="cpu", generator=None):
    """Parity of MARKs-so-far.
    vocab: 0=NOISE, 1=MARK, 2=QUERY, 3=EVEN(target), 4=ODD(target). vocab_size=5.
    At each QUERY, next-token target = EVEN if even # of MARKs so far else ODD.
    """
    g = generator
    NOISE, MARK, QUERY, EVEN, ODD = 0, 1, 2, 3, 4
    vocab_size = 5
    tokens = torch.full((batch_size, seq_len), NOISE, dtype=torch.long)
    targets = torch.zeros((batch_size, seq_len), dtype=torch.long)
    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)

    for b in range(batch_size):
        r = torch.rand(seq_len, generator=g)
        parity = 0
        for t in range(seq_len):
            if r[t] < p_mark:
                tokens[b, t] = MARK
                parity ^= 1
            elif r[t] < p_mark + p_query and t > 0:
                tokens[b, t] = QUERY
                targets[b, t] = EVEN if parity == 0 else ODD
                mask[b, t] = True
    return (tokens.to(device), targets.to(device), mask.to(device),
            torch.zeros_like(targets), vocab_size)


# ===========================================================================
# Eval (gap-binned) + train
# ===========================================================================

@torch.no_grad()
def task_accuracy(model, make_batch, cfg, n_batches, seed, device):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    tot_c = tot = 0
    bin_c = {b: 0 for b in GAP_BINS}; bin_n = {b: 0 for b in GAP_BINS}
    for _ in range(n_batches):
        tok, tgt, m, gap, _ = make_batch(generator=gen, device=device, **cfg)
        logits = model(tok)
        pred = logits.argmax(-1)
        hit = (pred == tgt) & m
        tot_c += hit.sum().item(); tot += m.sum().item()
        gv = gap[m]; hv = hit[m]
        for (lo, hi) in GAP_BINS:
            sel = (gv >= lo) & (gv <= hi)
            bin_n[(lo, hi)] += sel.sum().item()
            bin_c[(lo, hi)] += (hv & sel).sum().item()
    overall = tot_c / max(1, tot)
    by_gap = {f"{lo}-{hi}": (bin_c[(lo, hi)] / bin_n[(lo, hi)] if bin_n[(lo, hi)] else None)
              for (lo, hi) in GAP_BINS}
    return overall, by_gap


def task_train(model, make_batch, cfg, steps, lr, seed, device, log_every=200):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    for s in range(steps):
        tok, tgt, m, _, _ = make_batch(generator=gen, device=device, **cfg)
        logits = model(tok)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               tgt.reshape(-1), reduction="none")
        loss = (loss * m.reshape(-1).float()).sum() / (m.sum() + 1e-6)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if log_every and (s + 1) % log_every == 0:
            print(f"    step {s+1}/{steps} loss {loss.item():.4f}", flush=True)
    return model


if __name__ == "__main__":
    # smoke: shape-check both generators (no training)
    g = torch.Generator().manual_seed(0)
    tok, tgt, m, gap, vs = make_flipflop_batch(4, 64, generator=g)
    print(f"[flipflop] tokens{tuple(tok.shape)} vocab={vs} scored/seq={m[0].sum().item()} "
          f"max_gap={gap[m].max().item() if m.any() else 0}")
    tok, tgt, m, _, vs = make_parity_batch(4, 64, generator=g)
    print(f"[parity]   tokens{tuple(tok.shape)} vocab={vs} scored/seq={m[0].sum().item()}")
    print("smoke ok.")
