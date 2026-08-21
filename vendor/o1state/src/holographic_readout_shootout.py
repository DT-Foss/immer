"""
Holographic readout shootout — does lifting the tanh·m damping unlock recall? — by Opus 4.8
===========================================================================================

The smoke run showed the key-conditioned write helps (+5.6pp) but stayed at ~7% with the
original tanh_m readout (read std ~0.018 — heavily damped). Hypothesis: the m·tanh(read)
squashing is the bottleneck, not the binding mechanism. Test it directly: same MQAR harness,
same seeds, three readout modes head-to-head:
  tanh_m    : m·tanh(read)            (original, damped)
  layernorm : raw read, post-LN norms (medium)
  rms       : read / rms(read)        (full signal, unit-scale)
Plus holo_off (== Selective) as the floor and attn as the validity gate.

If rms/layernorm clearly beat tanh_m, the damping was the cap and the next lever is scale.
If all three plateau together, the cap is the mechanism (write/read key sharing) and the
next lever is separate write/read projections. Either way it's a clean datapoint.
"""
import os, sys, math, json, time, argparse
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "reference"))
sys.path.insert(0, HERE)

from mqar import make_mqar_batch, mqar_accuracy, TinyCausalTransformerLM  # noqa: E402
from holographic_gssm import HolographicLM  # noqa: E402


def build(arm, vocab_size, mask_idx, dm, nl, nh, dh, sl):
    if arm == "attn":
        return TinyCausalTransformerLM(vocab_size, d_model=dm, n_layers=nl,
                                       n_heads=nh, max_len=max(sl, 1024))
    if arm == "holo_off":
        return HolographicLM(vocab_size, mask_idx, d_model=dm, n_layers=nl, n_heads=nh,
                             d_head=dh, seq_len=sl, use_phase=False)
    # holo_<readout>
    ro = arm.split("_", 1)[1]
    return HolographicLM(vocab_size, mask_idx, d_model=dm, n_layers=nl, n_heads=nh,
                         d_head=dh, seq_len=sl, use_phase=True, readout=ro)


def train(model, cfg, steps, lr, seed, device):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(steps):
        tok, tgt, mask, _ = make_mqar_batch(generator=gen, device=device, **cfg)
        logits = model(tok)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               tgt.reshape(-1), reduction="none")
        loss = (loss * mask.reshape(-1).float()).sum() / (mask.sum() + 1e-6)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    return model


def mean_std(xs):
    mu = sum(xs) / len(xs)
    return mu, (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--seeds", default="1,7,42")
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--train-len", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--d-head", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(REPO, "results", "holo_readout_shootout.json"))
    args = ap.parse_args()

    device = torch.device("cpu")
    nk = nv = 64
    vocab = nk + nv + 1
    mask_idx = vocab
    chance = 1.0 / nv
    cfg = dict(batch_size=32, seq_len=args.train_len, n_pairs=args.n_pairs,
               n_queries=args.n_pairs, n_keys=nk, n_values=nv)
    seeds = [int(s) for s in args.seeds.split(",")]
    arms = ["attn", "holo_off", "holo_tanh_m", "holo_layernorm", "holo_rms"]

    print("=" * 70)
    print(f"Holographic readout shootout  steps={args.steps} seeds={seeds} chance={chance:.4f}")
    print("=" * 70)

    acc = {a: [] for a in arms}
    t0 = time.time()
    for seed in seeds:
        print(f"\n--- seed {seed} ---")
        for a in arms:
            torch.manual_seed(seed)
            m = build(a, vocab, mask_idx, args.d_model, 2, 4, args.d_head, args.train_len)
            train(m, cfg, args.steps, 3e-3, seed, device)
            m.eval()
            ov, _, _ = mqar_accuracy(m, cfg, 8, seed + 1, device)
            acc[a].append(ov)
            print(f"  {a:16s} {ov:.4f}")

    print("\n" + "=" * 70)
    print("AGGREGATE (mean ± std)")
    summ = {}
    for a in arms:
        mu, sd = mean_std(acc[a])
        summ[a] = {"mean": mu, "std": sd, "per_seed": acc[a]}
        print(f"  {a:16s} {mu:.4f} ± {sd:.4f}")
    off = summ["holo_off"]["mean"]
    print(f"\n  contributions over holo_off ({off:.4f}):")
    for a in ["holo_tanh_m", "holo_layernorm", "holo_rms"]:
        print(f"    {a:16s} {100*(summ[a]['mean']-off):+.2f} pp")
    print(f"  validity (attn): {summ['attn']['mean']:.4f} "
          f"{'PASS' if summ['attn']['mean']>=0.9 else 'FAIL'}")

    json.dump({"config": vars(args), "chance": chance, "summary": summ,
               "elapsed_s": round(time.time()-t0, 1)}, open(args.out, "w"), indent=2)
    print(f"\nWritten {args.out}  ({round(time.time()-t0,1)}s)")


if __name__ == "__main__":
    main()
