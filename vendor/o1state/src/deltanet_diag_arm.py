#!/usr/bin/env python3 -u
"""
DeltaNet diagnostic single-arm — why 12%% not 80%%? — 2026-06-25
=================================================================
Trains ONE DeltaNet config on MQAR n_pairs=8, prints a recall trajectory
(every EVAL_EVERY steps) plus a final beta/key-orthogonality diagnostic.

The point: isolate WHICH knob lifts DeltaNet off the ~12%% floor.
Run as:  python deltanet_diag_arm.py --tag d_k64 --d-k 64 --d-v 64 --steps 6000

Diagnostics dumped at the end:
  - recall trajectory (overfit signal vs still-climbing)
  - mean/max beta at KV-key positions  (is the erase-then-write firing? beta→1?)
  - key Gram off-diagonal RMS           (are distinct keys near-orthogonal?)
All bounded: state is n_heads*d_k*d_v reals, independent of T and vocab.
"""
import os, sys, math, time, argparse, json

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "reference"))
sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F
from mqar import make_mqar_batch, mqar_accuracy
from deltanet_gssm import DeltaNetLM


def key_orthogonality(model, tokens, device):
    """RMS of off-diagonal key-Gram for the DISTINCT key tokens in a batch.
    Near 0 = keys near-orthogonal (good for erase-then-write);
    near 1 = keys collapse onto each other (crosstalk returns)."""
    layer = model.layers[0].scan
    with torch.no_grad():
        h = model.pos(model.embed(tokens))
        k_raw = layer.W_k(h).view(h.shape[0], h.shape[1], layer.n_heads, layer.d_k)
        k = k_raw / (k_raw.norm(dim=-1, keepdim=True) + 1e-6)
        # head 0, first sample; take the first 8 unique key positions (even idx 0..14)
        kv = k[0, 0:16:2, 0]          # (8, d_k) — the 8 KV keys, head 0
        G = kv @ kv.t()               # (8,8) gram
        off = G - torch.diag(torch.diag(G))
        n = off.numel() - off.shape[0]
        return (off.pow(2).sum() / max(1, n)).sqrt().item()


def beta_at_kv(model, tokens, device):
    """Mean & max sigmoid(beta) at the KV-key positions (even indices 0..2*np-2)."""
    layer = model.layers[0].scan
    with torch.no_grad():
        h = model.pos(model.embed(tokens))
        beta = torch.sigmoid(layer.W_beta(h))    # (B,T,H)
        kv_beta = beta[:, 0:16:2]                  # KV-key positions
        return kv_beta.mean().item(), kv_beta.max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--d-k", type=int, default=32)
    ap.add_argument("--d-v", type=int, default=32)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--use-gate", action="store_true")
    ap.add_argument("--beta-bias", type=float, default=None,
                    help="override W_beta bias init (logit); e.g. 2.0 → sigmoid 0.88")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = torch.device("cpu")
    nk = nv = 64
    vocab = nk + nv + 1
    mask = vocab
    chance = 1.0 / nv
    tr = dict(batch_size=32, seq_len=64, n_pairs=8, n_queries=8,
              n_keys=nk, n_values=nv)

    torch.manual_seed(args.seed)
    model = DeltaNetLM(vocab, mask, d_model=args.d_model, n_layers=args.n_layers,
                       n_heads=args.n_heads, d_k=args.d_k, d_v=args.d_v,
                       seq_len=64, dropout=0.0, use_gate=args.use_gate)
    if args.beta_bias is not None:
        for lyr in model.layers:
            torch.nn.init.constant_(lyr.scan.W_beta.bias, args.beta_bias)

    n_params = sum(p.numel() for p in model.parameters())
    state_reals = args.n_heads * args.d_k * args.d_v
    print(f"[{args.tag}] steps={args.steps} seed={args.seed} d_k={args.d_k} "
          f"d_v={args.d_v} H={args.n_heads} L={args.n_layers} gate={args.use_gate} "
          f"beta_bias={args.beta_bias}", flush=True)
    print(f"[{args.tag}] params={n_params:,}  bounded_state={state_reals} reals "
          f"(O(1) in T, vocab)", flush=True)

    model.to(dev).train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    t0 = time.time()
    traj = []
    for s in range(args.steps):
        tok, tgt, m, _ = make_mqar_batch(generator=g, device=dev, **tr)
        lo = model(tok)
        l = F.cross_entropy(lo.reshape(-1, lo.size(-1)), tgt.reshape(-1),
                            reduction="none")
        l = (l * m.reshape(-1).float()).sum() / (m.sum() + 1e-6)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if (s + 1) % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                acc, _, _ = mqar_accuracy(model, tr, 8, args.seed + 1, dev)
            model.train()
            traj.append((s + 1, round(l.item(), 3), round(acc * 100, 2)))
            print(f"[{args.tag}] step {s+1}/{args.steps}  loss {l.item():.3f}  "
                  f"recall {acc*100:.2f}%  ({time.time()-t0:.0f}s)", flush=True)

    # final diagnostics on a fresh eval batch
    model.eval()
    dg = torch.Generator(device="cpu").manual_seed(args.seed + 100)
    tok, _, _, _ = make_mqar_batch(generator=dg, device=dev, **tr)
    ortho = key_orthogonality(model, tok, dev)
    bmean, bmax = beta_at_kv(model, tok, dev)
    final_recall = traj[-1][2] if traj else 0.0
    print(f"[{args.tag}] DONE  final_recall={final_recall:.2f}%  "
          f"key_offdiag_rms={ortho:.3f}  beta_kv mean={bmean:.3f} max={bmax:.3f}",
          flush=True)

    result = {
        "tag": args.tag, "final_recall": final_recall,
        "trajectory": traj, "key_offdiag_rms": round(ortho, 4),
        "beta_kv_mean": round(bmean, 4), "beta_kv_max": round(bmax, 4),
        "params": n_params, "bounded_state_reals": state_reals,
        "config": {"d_k": args.d_k, "d_v": args.d_v, "n_heads": args.n_heads,
                   "n_layers": args.n_layers, "steps": args.steps,
                   "use_gate": args.use_gate, "beta_bias": args.beta_bias,
                   "lr": args.lr, "seed": args.seed},
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[{args.tag}] → {args.out}", flush=True)
    print(f"RESULT_JSON {json.dumps(result)}", flush=True)


if __name__ == "__main__":
    main()
