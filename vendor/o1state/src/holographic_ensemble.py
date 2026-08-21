#!/usr/bin/env python3 -u
"""
Holographic ENSEMBLE — David's idea: N bounded states, each ~13%, DIFFERENT pairs → vote → high recall
=====================================================================================================
A single bounded holographic channel caps at ~13% on MQAR n_pairs=8 (interference-bound).
David: "if the plateau is 13%, run N arms so you get ~100% — still bounded, still fast."

THIS IS NOT the dead multi-slot partition (DEADENDS entry 10). Multi-slot FORCED each pair
into one slot (router → destroyed coherence). The ENSEMBLE lets EVERY member see ALL pairs;
they differ only in INIT / key-projection, so they fail on DIFFERENT pairs → uncorrelated
errors → a learned combine / vote at the output recovers far more than any single member.

BOUNDEDNESS: N fixed holographic states = N × O(1) = still O(1) in T and vocab. No KV-cache,
no alphabet index, no growth with sequence length. N is a fixed architectural constant.

Design:
  - N independent HolographicScanLayer "members", each its own W_key/W_v/W_out, diverse init.
  - Each member reads the residual stream and produces a d_model output.
  - COMBINE: concatenate member outputs → learned linear → d_model  (the "vote").
    (Learned combine, not plain mean, so the model can learn which member to trust per token.)
  - Diversity driver: each member gets a different init seed (decorrelated key phases).
"""
import sys, math, argparse, time
sys.path.insert(0, "reference"); sys.path.insert(0, "src")
import torch
import torch.nn as nn
import torch.nn.functional as F
from mqar import make_mqar_batch, mqar_accuracy, TinyCausalTransformerLM
from holographic_gssm import HolographicScanLayer


class EnsembleHoloLayer(nn.Module):
    """N independent holographic members + learned combine. Bounded: N fixed states."""
    def __init__(self, d_model, n_members=6, d_head=32, n_heads=4, causal=True,
                 phase_scale=math.pi, readout="rms", member_seed_base=1000):
        super().__init__()
        self.n_members = n_members
        self.members = nn.ModuleList()
        for i in range(n_members):
            # diverse init: seed each member differently so key phases decorrelate
            torch.manual_seed(member_seed_base + 17 * i)
            m = HolographicScanLayer(
                d_model, d_head=d_head, n_heads=n_heads, causal=causal,
                phase_scale=phase_scale, use_phase=True, readout=readout)
            self.members.append(m)
        # learned combine ("vote"): N·d_model → d_model
        self.combine = nn.Linear(n_members * d_model, d_model, bias=False)

    def forward(self, x):
        outs = [m(x) for m in self.members]          # each (B,T,d_model)
        cat = torch.cat(outs, dim=-1)                 # (B,T,N·d_model)
        return self.combine(cat)


class EnsembleHoloBlock(nn.Module):
    def __init__(self, d_model, n_members=6, d_head=32, n_heads=4, causal=True,
                 phase_scale=math.pi, readout="rms", ffn_dim=None):
        super().__init__()
        self.scan = EnsembleHoloLayer(d_model, n_members=n_members, d_head=d_head,
                                      n_heads=n_heads, causal=causal,
                                      phase_scale=phase_scale, readout=readout)
        self.ln1 = nn.LayerNorm(d_model)
        ffn_dim = ffn_dim or 4 * d_model
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(),
                                 nn.Linear(ffn_dim, d_model))
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ln1(x + self.scan(x))
        x = self.ln2(x + self.ffn(x))
        return x


class EnsembleHoloLM(nn.Module):
    def __init__(self, vocab_size, mask_idx, d_model=128, n_layers=2, n_members=6,
                 n_heads=4, d_head=32, seq_len=64, causal=True,
                 phase_scale=math.pi, readout="rms"):
        super().__init__()
        from moebius_attention import SinusoidalPositionalEncoding
        self.embed = nn.Embedding(vocab_size + 2, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            EnsembleHoloBlock(d_model, n_members=n_members, d_head=d_head,
                              n_heads=n_heads, causal=causal, phase_scale=phase_scale,
                              readout=readout)
            for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size + 1)

    def forward(self, x):
        h = self.pos(self.embed(x))
        for l in self.layers:
            h = l(h)
        return self.head(h)


def bounded_report(model, n_members, n_heads, d_head, n_layers):
    # state = per layer, per member, n_heads complex accumulators of size d_head
    states = n_layers * n_members * n_heads * d_head * 2  # ×2 for re+im
    print(f"  [BOUNDED] state = {n_layers}L × {n_members}members × {n_heads}heads × "
          f"{d_head}d × 2(re,im) = {states} reals/sample — O(1) in T and vocab", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=6)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--seeds", default="1,7,42")
    ap.add_argument("--readout", default="rms")
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    dev = torch.device("cpu")
    nk = nv = 64; vocab = nk + nv + 1; mask = vocab; chance = 1/nv
    tr = dict(batch_size=32, seq_len=64, n_pairs=8, n_queries=8, n_keys=nk, n_values=nv)
    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 78)
    print(f"HOLOGRAPHIC ENSEMBLE — {args.members} bounded members, learned combine")
    print(f"steps={args.steps} seeds={seeds} readout={args.readout} chance={chance:.4f}")
    print("  vs single-channel holographic plateau ~7-9%. Question: does the ensemble")
    print("  of weak (~13%) members with diverse errors VOTE its way higher?")
    print("=" * 78)

    recalls = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = EnsembleHoloLM(vocab, mask, d_model=128, n_layers=2,
                               n_members=args.members, n_heads=4, d_head=32,
                               seq_len=64, causal=True, phase_scale=math.pi,
                               readout=args.readout)
        np_ = sum(p.numel() for p in model.parameters())
        print(f"\n=== seed {seed} === params={np_:,}")
        bounded_report(model, args.members, 4, 32, 2)
        model.to(dev).train()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        g = torch.Generator(device="cpu").manual_seed(seed)
        t0 = time.time()
        for s in range(args.steps):
            tok, tgt, m, _ = make_mqar_batch(generator=g, device=dev, **tr)
            lo = model(tok)
            l = F.cross_entropy(lo.reshape(-1, lo.size(-1)), tgt.reshape(-1), reduction="none")
            l = (l * m.reshape(-1).float()).sum() / (m.sum() + 1e-6)
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            if (s + 1) % 500 == 0:
                model.eval()
                with torch.no_grad():
                    acc, _, _ = mqar_accuracy(model, tr, 8, seed + 1, dev)
                model.train()
                print(f"  step {s+1}/{args.steps} loss {l.item():.3f} recall {acc*100:.2f}% "
                      f"({time.time()-t0:.0f}s)", flush=True)
        model.eval()
        with torch.no_grad():
            acc, _, _ = mqar_accuracy(model, tr, 8, seed + 1, dev)
        recalls.append(acc * 100)
        print(f"  >>> seed {seed} FINAL recall {acc*100:.2f}%", flush=True)

    mu = sum(recalls) / len(recalls)
    sd = (sum((x - mu) ** 2 for x in recalls) / len(recalls)) ** 0.5
    print(f"\n{'='*78}")
    print(f"ENSEMBLE {args.members}-member recall: {mu:.2f} ± {sd:.2f}%  {recalls}")
    print(f"  vs single-channel ~7-9%. Ignition rate: {sum(r>4 for r in recalls)}/{len(recalls)} above floor.")


if __name__ == "__main__":
    main()
