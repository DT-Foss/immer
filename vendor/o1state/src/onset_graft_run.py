#!/usr/bin/env python3 -u
"""
ONSET campaign — Lever 2: Ignite-then-graft.
============================================
Dead-end entry 20 KILLED the COLD 6-member holographic ensemble: 1.87M params
start in the ln(64) uniform fixpoint and NEVER ignite (more params = harder
onset). But it left the WARM graft explicitly open:
    "To rescue the ensemble you'd have to ignite ONE member first (cheap config /
     curriculum) then graft it; a cold 6-member start cannot ignite."

Protocol:
  PHASE 1 — ignite a SINGLE holographic member (HolographicLM, 1 layer of scan
            per block, tanh_m, 2500 steps). Confirm it reaches ~7-9% (else abort:
            no warm seed to graft).
  PHASE 2 — build a SMALL ensemble (n_members=3) whose members' scan weights are
            SEEDED from the ignited member, and continue-train. Two graft modes:
       warm_seeded : all 3 members init = ignited weights + small perturbation
                     (diversity), everything trainable.
       warm_frozen : member 0 = ignited weights FROZEN; members 1..2 cold; train
                     combine + cold members only (the ignited member anchors).
  Baseline (same run): COLD 3-member ensemble (entry-20 style, no graft).

Gate (pre-registered):
  CONFIRM : a graft mode mean > single-member ignition mean by >1σ (the ensemble
            vote beats its own seed) AND > 8.89% wall.
  PARTIAL : graft mode ignites (>4% = above floor) where COLD ensemble stays at
            chance — proves warm-start rescues onset even if it doesn't beat the wall.
  KILL    : graft mode ≈ chance like the cold ensemble → warm-start does not
            transfer; onset wall holds even with a pre-ignited seed.

CPU-deterministic, 4 threads. Output → results/onset_graft.json + logfile.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import os
import json
import math
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "reference"))

from mqar import make_mqar_batch, mqar_accuracy, TinyCausalTransformerLM  # noqa: E402
from holographic_gssm import HolographicLM, HolographicScanLayer  # noqa: E402

DEVICE = torch.device("cpu")
N_KEYS = N_VALUES = 64
VOCAB = N_KEYS + N_VALUES + 1
MASK_IDX = VOCAB
CHANCE = 1.0 / N_VALUES
D_MODEL, N_HEADS, D_HEAD, N_LAYERS = 128, 4, 32, 2
READOUT = "tanh_m"


# ── small ensemble whose members can be seeded from an ignited single member ──
class GraftEnsembleLayer(nn.Module):
    def __init__(self, d_model, n_members=3, d_head=D_HEAD, n_heads=N_HEADS,
                 phase_scale=math.pi, readout=READOUT):
        super().__init__()
        self.n_members = n_members
        self.members = nn.ModuleList([
            HolographicScanLayer(d_model, d_head=d_head, n_heads=n_heads,
                                 causal=True, phase_scale=phase_scale,
                                 use_phase=True, readout=readout)
            for _ in range(n_members)])
        self.combine = nn.Linear(n_members * d_model, d_model, bias=False)

    def forward(self, x):
        outs = [m(x) for m in self.members]
        return self.combine(torch.cat(outs, dim=-1))


class GraftEnsembleBlock(nn.Module):
    def __init__(self, d_model, n_members=3, d_head=D_HEAD, n_heads=N_HEADS,
                 phase_scale=math.pi, readout=READOUT):
        super().__init__()
        self.scan = GraftEnsembleLayer(d_model, n_members=n_members, d_head=d_head,
                                       n_heads=n_heads, phase_scale=phase_scale,
                                       readout=readout)
        self.ln1 = nn.LayerNorm(d_model)
        # FFN structurally identical to HolographicTransformerLayer (Linear,GELU,
        # Identity,Linear) so grafting its state_dict from a single member matches.
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Identity(),
            nn.Linear(4 * d_model, d_model))
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ln1(x + self.scan(x))
        x = self.ln2(x + self.ffn(x))
        return x


class GraftEnsembleLM(nn.Module):
    def __init__(self, vocab_size, mask_idx, d_model=D_MODEL, n_layers=N_LAYERS,
                 n_members=3, n_heads=N_HEADS, d_head=D_HEAD, seq_len=64,
                 phase_scale=math.pi, readout=READOUT):
        super().__init__()
        from moebius_attention import SinusoidalPositionalEncoding
        self.embed = nn.Embedding(vocab_size + 2, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            GraftEnsembleBlock(d_model, n_members=n_members, d_head=d_head,
                               n_heads=n_heads, phase_scale=phase_scale,
                               readout=readout)
            for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size + 1)

    def forward(self, x):
        h = self.pos(self.embed(x))
        for l in self.layers:
            h = l(h)
        return self.head(h)


def train_model(model, cfg, steps, lr, seed, eval_every=0):
    model.to(DEVICE).train()
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for step in range(steps):
        tok, tgt, mask, _ = make_mqar_batch(generator=gen, device=DEVICE, **cfg)
        logits = model(tok)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               tgt.reshape(-1), reduction="none")
        loss = (loss * mask.reshape(-1).float()).sum() / (mask.sum() + 1e-6)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 5.0)
        opt.step()
        if eval_every and (step + 1) % eval_every == 0:
            model.eval()
            with torch.no_grad():
                acc, _, _ = mqar_accuracy(model, cfg, 6, seed + 1, DEVICE)
            model.train()
            print(f"      step {step+1}/{steps} loss {loss.item():.3f} recall {acc*100:.2f}%",
                  flush=True)
    return model


def graft_from_single(ens_lm, single_lm, mode, perturb=0.02):
    """Copy the ignited single member's scan weights into each ensemble member.

    single_lm.layers[L].scan is ONE HolographicScanLayer.
    ens_lm.layers[L].scan.members[j] are the ensemble's members at layer L.
    We also copy the single model's embed/pos/head/ln/ffn so the warm context is
    transferred, not just the scan.
    """
    with torch.no_grad():
        # shared non-member weights: embed, pos(none), head
        ens_lm.embed.weight.copy_(single_lm.embed.weight)
        ens_lm.head.weight.copy_(single_lm.head.weight)
        if ens_lm.head.bias is not None and single_lm.head.bias is not None:
            ens_lm.head.bias.copy_(single_lm.head.bias)
        for L in range(len(ens_lm.layers)):
            eb = ens_lm.layers[L]
            sb = single_lm.layers[L]
            # copy block LN + FFN (warm context)
            eb.ln1.load_state_dict(sb.ln1.state_dict())
            eb.ln2.load_state_dict(sb.ln2.state_dict())
            eb.ffn.load_state_dict(sb.ffn.state_dict())
            src_scan = sb.scan          # HolographicScanLayer
            for j, mem in enumerate(eb.scan.members):
                mem.load_state_dict(src_scan.state_dict())
                if mode == "warm_seeded" and j > 0:
                    # perturb members 1.. for diversity (decorrelated errors)
                    for p in mem.parameters():
                        p.add_(perturb * torch.randn_like(p))
                if mode == "warm_frozen":
                    if j == 0:
                        for p in mem.parameters():
                            p.requires_grad_(False)   # anchor member frozen
                    else:
                        # members 1.. get a FRESH cold init (diverse)
                        for p in mem.parameters():
                            if p.dim() >= 2:
                                nn.init.xavier_uniform_(p, gain=0.3)
                            else:
                                nn.init.zeros_(p)
            # init combine to AVERAGE the members (so start ≈ member output)
            n = eb.scan.n_members
            W = eb.scan.combine.weight   # (d_model, n*d_model)
            W.zero_()
            for j in range(n):
                W[:, j * ens_lm.embed.embedding_dim:(j + 1) * ens_lm.embed.embedding_dim] += \
                    torch.eye(ens_lm.embed.embedding_dim) / n
    return ens_lm


def mean_std(xs):
    mu = sum(xs) / len(xs)
    return mu, (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ignite-steps", type=int, default=2500)
    ap.add_argument("--graft-steps", type=int, default=1500)
    ap.add_argument("--members", type=int, default=3)
    ap.add_argument("--seeds", default="1,7,42")
    ap.add_argument("--out", default=str(REPO / "results" / "onset_graft.json"))
    args = ap.parse_args()

    cfg = dict(batch_size=32, seq_len=64, n_pairs=8, n_queries=8,
               n_keys=N_KEYS, n_values=N_VALUES)
    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 74)
    print(f"ONSET Lever 2: Ignite-then-graft  ignite={args.ignite_steps} "
          f"graft={args.graft_steps} members={args.members} seeds={seeds}")
    print("=" * 74)

    # attn validity gate (once)
    print("\n── attn validity gate ──", flush=True)
    torch.manual_seed(0)
    attn = TinyCausalTransformerLM(VOCAB, d_model=64, n_layers=2, n_heads=4, max_len=64)
    train_model(attn, cfg, 1000, 3e-3, 0)
    attn.eval()
    attn_acc, _, _ = mqar_accuracy(attn, cfg, 8, 1, DEVICE)
    print(f"  attn recall {attn_acc:.4f}  {'PASS' if attn_acc>=0.9 else 'FAIL — VOID'}",
          flush=True)

    modes = ["single", "cold_ensemble", "warm_seeded", "warm_frozen"]
    acc = {m: [] for m in modes}
    t0 = time.time()

    for seed in seeds:
        print(f"\n{'='*60}\n--- seed {seed} ---", flush=True)

        # PHASE 1: ignite single member
        print("  [phase1] igniting single holographic member...", flush=True)
        torch.manual_seed(seed)
        single = HolographicLM(VOCAB, MASK_IDX, d_model=D_MODEL, n_layers=N_LAYERS,
                               n_heads=N_HEADS, d_head=D_HEAD, seq_len=64,
                               use_phase=True, readout=READOUT)
        train_model(single, cfg, args.ignite_steps, 3e-3, seed, eval_every=500)
        single.eval()
        s_acc, _, _ = mqar_accuracy(single, cfg, 8, seed + 1, DEVICE)
        acc["single"].append(s_acc)
        print(f"  [phase1] single ignition recall {s_acc*100:.2f}%", flush=True)

        # COLD ensemble baseline (entry-20 style, 3 members, no graft)
        print("  [cold] cold 3-member ensemble baseline...", flush=True)
        torch.manual_seed(seed + 500)
        cold = GraftEnsembleLM(VOCAB, MASK_IDX, n_members=args.members, seq_len=64)
        train_model(cold, cfg, args.ignite_steps, 3e-3, seed, eval_every=0)
        cold.eval()
        c_acc, _, _ = mqar_accuracy(cold, cfg, 8, seed + 1, DEVICE)
        acc["cold_ensemble"].append(c_acc)
        print(f"  [cold] cold ensemble recall {c_acc*100:.2f}%", flush=True)

        # PHASE 2: graft into small ensemble, two modes
        for mode in ("warm_seeded", "warm_frozen"):
            print(f"  [phase2:{mode}] grafting + continue-train...", flush=True)
            torch.manual_seed(seed + 900)
            ens = GraftEnsembleLM(VOCAB, MASK_IDX, n_members=args.members, seq_len=64)
            ens = graft_from_single(ens, single, mode)
            # measure graft recall BEFORE continue-train (does the warm seed carry?)
            ens.eval()
            g0, _, _ = mqar_accuracy(ens, cfg, 8, seed + 1, DEVICE)
            print(f"    graft@0steps recall {g0*100:.2f}%", flush=True)
            train_model(ens, cfg, args.graft_steps, 3e-3, seed + 7, eval_every=500)
            ens.eval()
            g_acc, _, _ = mqar_accuracy(ens, cfg, 8, seed + 1, DEVICE)
            acc[mode].append(g_acc)
            print(f"  [phase2:{mode}] recall {g_acc*100:.2f}%", flush=True)

    print("\n" + "=" * 74)
    print("AGGREGATE (mean ± std)")
    summ = {}
    for m in modes:
        mu, sd = mean_std(acc[m])
        summ[m] = {"mean": mu, "std": sd, "per_seed": acc[m]}
        print(f"  {m:16s} {mu:.4f} ± {sd:.4f}")

    single_mean = summ["single"]["mean"]
    cold_mean = summ["cold_ensemble"]["mean"]
    verdicts = {}
    for mode in ("warm_seeded", "warm_frozen"):
        mu = summ[mode]["mean"]
        diffs = [acc[mode][i] - acc["single"][i] for i in range(len(seeds))]
        dmu, dsd = mean_std(diffs)
        if mu > single_mean + dsd and mu > 0.0889 and dmu > 0:
            v = f"CONFIRM (vote beats seed, Δ={dmu*100:+.2f}pp, >8.89% wall)"
        elif mu > 0.04 and cold_mean < 0.04:
            v = f"PARTIAL (warm graft ignites {mu*100:.2f}% where cold stays {cold_mean*100:.2f}%)"
        else:
            v = f"KILL (≈cold/chance, Δvs-single={dmu*100:+.2f}pp)"
        verdicts[mode] = {"verdict": v, "delta_vs_single_pp": round(dmu * 100, 3),
                          "delta_std_pp": round(dsd * 100, 3)}
        print(f"  {mode:16s} → {v}")

    payload = {
        "config": {"lever": "2_ignite_then_graft", "ignite_steps": args.ignite_steps,
                   "graft_steps": args.graft_steps, "members": args.members,
                   "seeds": ",".join(str(s) for s in seeds), "readout": READOUT,
                   "out": args.out},
        "chance": CHANCE,
        "attn_gate": round(attn_acc, 4),
        "summary": summ,
        "verdicts": verdicts,
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWritten {args.out}  ({payload['elapsed_s']}s)")


if __name__ == "__main__":
    main()
