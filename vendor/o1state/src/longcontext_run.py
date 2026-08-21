#!/usr/bin/env python3 -u
"""
Long-context capability run — NoPE-GSSM vs Transformer, train short, eval to T=8192.
The capability boundary: attention's forward pass CRASHES past its PE buffer; GSSM holds.
Crashes are logged as results ("CRASH @ T"), not failures — the crash IS the finding.
"""
import os, sys, json, argparse, traceback
sys.path.insert(0, "reference"); sys.path.insert(0, "src")
import torch

from longcontext_tasks import (make_flipflop_batch, make_parity_batch,
                               task_accuracy, task_train)
from moebius_scan_transformer_selective import SelectiveRapiditySqrtTransformerLM
from mqar import TinyCausalTransformerLM

# NoPE Selective subclass (identity positional encoding) — reuse the published one
from length_extrap_v2 import SelectiveNoPETransformerLM

TASKS = {
    "flipflop": (make_flipflop_batch, dict(n_vals=8, p_set=0.10, p_query=0.10)),
    "parity":   (make_parity_batch,   dict(p_mark=0.12, p_query=0.10)),
}


def build(arm, vocab_size, mask_idx, seq_len, device):
    if arm == "nope_gssm":
        return SelectiveNoPETransformerLM(
            vocab_size, mask_idx, d_model=128, n_layers=2, n_heads=4, d_head=32,
            seq_len=seq_len, dropout=0.0, causal=True).to(device)
    if arm == "transformer":
        return TinyCausalTransformerLM(
            vocab_size, d_model=128, n_layers=2, n_heads=4, max_len=1024).to(device)
    raise ValueError(arm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="flipflop", choices=list(TASKS))
    ap.add_argument("--train-len", type=int, default=64)
    ap.add_argument("--eval-lens", default="64,256,1024,2048,4096,8192")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = torch.device("cpu")
    make_batch, task_cfg = TASKS[args.task]
    # discover vocab from a sample batch
    g0 = torch.Generator().manual_seed(0)
    *_, vocab_size = make_batch(2, args.train_len, generator=g0, **task_cfg)
    mask_idx = vocab_size
    eval_lens = [int(x) for x in args.eval_lens.split(",")]

    print("=" * 74)
    print(f"LONG-CONTEXT CAPABILITY — task={args.task}")
    print(f"train_len={args.train_len}  eval_lens={eval_lens}  steps={args.steps}  seed={args.seed}")
    print(f"vocab={vocab_size}  arms=[nope_gssm, transformer]")
    print("=" * 74)

    results = {"task": args.task, "train_len": args.train_len, "eval_lens": eval_lens,
               "seed": args.seed, "arms": {}}

    for arm in ["nope_gssm", "transformer"]:
        print(f"\n── {arm} ──")
        torch.manual_seed(args.seed)
        model = build(arm, vocab_size, mask_idx, args.train_len, dev)
        train_cfg = dict(batch_size=32, seq_len=args.train_len, **task_cfg)
        task_train(model, make_batch, train_cfg, args.steps, args.lr, args.seed, dev)

        curve = {}
        for T in eval_lens:
            eval_cfg = dict(batch_size=8 if T >= 512 else 32, seq_len=T, **task_cfg)
            try:
                acc, by_gap = task_accuracy(model, make_batch, eval_cfg, 8,
                                            args.seed + 1, dev)
                curve[T] = {"acc": round(acc, 4)}
                print(f"  T={T:>5} ({T//args.train_len:>3}×): acc {acc*100:5.1f}%", flush=True)
            except Exception as e:
                # the crash IS the finding — log it, keep going
                msg = f"{type(e).__name__}: {str(e)[:80]}"
                curve[T] = {"acc": None, "crash": msg}
                print(f"  T={T:>5} ({T//args.train_len:>3}×): CRASH — {msg}", flush=True)
        results["arms"][arm] = curve

    # validity gate: transformer must solve the task at train length
    tf_train = results["arms"]["transformer"].get(args.train_len, {}).get("acc")
    gate_ok = tf_train is not None and tf_train >= 0.80
    results["validity_gate"] = {"transformer_train_len_acc": tf_train, "passed": gate_ok}

    print("\n" + "=" * 74)
    print("CAPABILITY BOUNDARY")
    nope = results["arms"]["nope_gssm"]; tf = results["arms"]["transformer"]
    for T in eval_lens:
        n = nope[T].get("acc"); t = tf[T].get("acc")
        ns = f"{n*100:.0f}%" if n is not None else "—"
        ts = f"{t*100:.0f}%" if t is not None else ("CRASH" if "crash" in tf[T] else "—")
        print(f"  T={T:>5} ({T//args.train_len:>3}×):  NoPE-GSSM {ns:>6}   Transformer {ts:>6}")
    print(f"\n  validity gate (transformer @ train len ≥80%): "
          f"{tf_train*100 if tf_train else 0:.0f}%  {'PASS' if gate_ok else 'FAIL→harness suspect'}")

    out = args.out or os.path.join("results", f"longcontext_{args.task}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
