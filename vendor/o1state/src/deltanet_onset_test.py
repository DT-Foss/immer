#!/usr/bin/env python3 -u
"""
DeltaNet ignition-rate test — is the 12%% floor an OPTIMIZATION-ONSET problem? — 2026-06-25
============================================================================================
THE QUESTION (sharpened):  plain DeltaNet on MQAR ignites on SOME seeds (loss falls below
ln(64)=4.159, recall climbs) and stays in the trivial uniform fixpoint on OTHERS
(loss frozen at ln(64), recall = chance).  Seed 1 ignited (12%%); seed 42 did NOT (4500 steps).

Does a structural change RAISE THE IGNITION RATE across seeds?
We sweep N seeds × two configs and count: how many ignited, and the recall of those that did.

CONFIGS compared (one structural axis at a time):
  d_k32      — the baseline (offdiag_rms ~0.55, keys NOT orthogonal)
  d_k64      — more key-space → lower offdiag_rms → stronger gradient to the erase mechanism?
  beta_hi    — start beta~0.88 (logit 2.0) → sharper erase from step 0, easier to find?

IGNITED := final loss < ln(64) - 0.05  (clearly off the uniform fixpoint)
We report ignition_rate, mean recall of ignited seeds, and per-seed detail.
Bounded: state is n_heads*d_k*d_v reals, O(1) in T and vocab (printed).
"""
import os, sys, math, time, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "reference"))
sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F
from mqar import make_mqar_batch, mqar_accuracy
from deltanet_gssm import DeltaNetLM

LN64 = math.log(64)            # 4.1589 — the uniform-over-64-values fixpoint
IGNITE_MARGIN = 0.05


def train_one(seed, d_k, beta_bias, steps, lr, dev, tr):
    torch.manual_seed(seed)
    model = DeltaNetLM(129, 128, d_model=128, n_layers=2, n_heads=4,
                       d_k=d_k, d_v=d_k, seq_len=64, dropout=0.0, use_gate=False)
    if beta_bias is not None:
        for lyr in model.layers:
            torch.nn.init.constant_(lyr.scan.W_beta.bias, beta_bias)
    model.to(dev).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator(device="cpu").manual_seed(seed)
    last_loss = None
    for s in range(steps):
        tok, tgt, m, _ = make_mqar_batch(generator=g, device=dev, **tr)
        lo = model(tok)
        l = F.cross_entropy(lo.reshape(-1, lo.size(-1)), tgt.reshape(-1),
                            reduction="none")
        l = (l * m.reshape(-1).float()).sum() / (m.sum() + 1e-6)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        last_loss = l.item()
    model.eval()
    with torch.no_grad():
        acc, _, _ = mqar_accuracy(model, tr, 8, seed + 1, dev)
    ignited = last_loss < (LN64 - IGNITE_MARGIN)
    return round(last_loss, 3), round(acc * 100, 2), ignited


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=["d_k32", "d_k64", "beta_hi"])
    ap.add_argument("--seeds", default="1,7,42,123,2024,11,99,7777")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = torch.device("cpu")
    nk = nv = 64
    tr = dict(batch_size=32, seq_len=64, n_pairs=8, n_queries=8,
              n_keys=nk, n_values=nv)
    seeds = [int(s) for s in args.seeds.split(",")]

    cfg = {"d_k32": (32, None), "d_k64": (64, None),
           "beta_hi": (32, 2.0)}[args.config]
    d_k, beta_bias = cfg
    state_reals = 4 * d_k * d_k

    print(f"[{args.config}] d_k={d_k} beta_bias={beta_bias} steps={args.steps} "
          f"seeds={seeds}", flush=True)
    print(f"[{args.config}] bounded_state={state_reals} reals (O(1) in T, vocab)  "
          f"IGNITE if loss < {LN64-IGNITE_MARGIN:.3f}", flush=True)

    t0 = time.time()
    per_seed = []
    for sd in seeds:
        loss, recall, ignited = train_one(sd, d_k, beta_bias, args.steps,
                                          args.lr, dev, tr)
        per_seed.append({"seed": sd, "loss": loss, "recall": recall,
                         "ignited": ignited})
        print(f"[{args.config}] seed {sd:>4}  loss {loss:.3f}  recall {recall:5.2f}%  "
              f"{'IGNITED' if ignited else 'dead(uniform)'}  ({time.time()-t0:.0f}s)",
              flush=True)

    n_ign = sum(p["ignited"] for p in per_seed)
    rate = n_ign / len(seeds)
    ign_recalls = [p["recall"] for p in per_seed if p["ignited"]]
    mean_ign = round(sum(ign_recalls) / len(ign_recalls), 2) if ign_recalls else 0.0
    mean_all = round(sum(p["recall"] for p in per_seed) / len(per_seed), 2)

    result = {
        "config": args.config, "d_k": d_k, "beta_bias": beta_bias,
        "steps": args.steps, "seeds": seeds,
        "ignition_rate": round(rate, 3), "n_ignited": n_ign, "n_seeds": len(seeds),
        "mean_recall_ignited": mean_ign, "mean_recall_all": mean_all,
        "bounded_state_reals": state_reals, "per_seed": per_seed,
    }
    print(f"\n[{args.config}] IGNITION RATE {n_ign}/{len(seeds)} = {rate*100:.0f}%  "
          f"| mean recall (ignited) {mean_ign:.2f}%  | mean recall (all) {mean_all:.2f}%",
          flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    print(f"RESULT_JSON {json.dumps(result)}", flush=True)


if __name__ == "__main__":
    main()
