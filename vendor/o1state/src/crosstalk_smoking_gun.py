"""
Crosstalk smoking-gun: recall vs n_pairs, tested against the 1/sqrt(N) law — by Opus 4.8
========================================================================================

Three capacity levers are now dead (d_head flat, more heads worse, more layers worse,
separate-QK worse). All "more resources" levers fail. That is the signature of CROSSTALK,
not capacity. The holographic read is

    read = Σ_k γ_{k→t} u_k cos(φ_k − φ_q)

Matched key: cos(0)=1. The other N−1 keys contribute cos(φ_k−φ_q); for keys spread on the
circle their sum is a random walk of magnitude ~sqrt(N−1). So signal-to-interference falls
like 1/sqrt(N) — the classic HRR/VSA holographic-memory capacity law.

FALSIFIABLE PREDICTION (committed before running): holographic recall rises as pairs drop,
and the recall-above-floor should track ~ C/sqrt(n_pairs). If recall is HIGH at n_pairs=1-2
and decays ~1/sqrt(N) toward the floor as N grows → crosstalk IS the cap, proven. If recall
is flat in n_pairs → crosstalk is NOT the cap and the hypothesis is wrong (report it straight).

Single focused experiment, multi-seed, CPU-deterministic. No competing sweeps.
"""
import os, sys, math, json, time
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "reference"))
sys.path.insert(0, HERE)

from mqar import make_mqar_batch, mqar_accuracy, TinyCausalTransformerLM  # noqa: E402
from holographic_gssm import HolographicLM  # noqa: E402
from moebius_scan_transformer_selective import SelectiveRapiditySqrtTransformerLM  # noqa: E402


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
    device = torch.device("cpu")
    nk = nv = 64
    vocab = nk + nv + 1
    mask_idx = vocab
    chance = 1.0 / nv
    seeds = [1, 7, 42]
    steps = 1500
    train_len = 64
    n_pairs_list = [1, 2, 3, 4, 6, 8]

    print("=" * 74)
    print("CROSSTALK SMOKING GUN — holographic recall vs n_pairs (1/sqrt(N) law)")
    print(f"seeds={seeds} steps={steps} train_len={train_len} chance={chance:.4f}")
    print("prediction: recall_above_floor ~ C/sqrt(n_pairs) if crosstalk-limited")
    print("=" * 74)

    rows = []
    t0 = time.time()
    for npairs in n_pairs_list:
        cfg = dict(batch_size=32, seq_len=train_len, n_pairs=npairs,
                   n_queries=npairs, n_keys=nk, n_values=nv)
        on, off, at = [], [], []
        for seed in seeds:
            torch.manual_seed(seed)
            m_on = HolographicLM(vocab, mask_idx, d_model=128, n_layers=2, n_heads=4,
                                 d_head=32, seq_len=train_len, use_phase=True, readout="tanh_m")
            train(m_on, cfg, steps, 3e-3, seed, device); m_on.eval()
            on.append(mqar_accuracy(m_on, cfg, 8, seed + 1, device)[0])

            torch.manual_seed(seed)
            m_off = SelectiveRapiditySqrtTransformerLM(vocab, mask_idx, d_model=128,
                                                       n_layers=2, n_heads=4, d_head=32,
                                                       seq_len=train_len, dropout=0.0, causal=True)
            train(m_off, cfg, steps, 3e-3, seed, device); m_off.eval()
            off.append(mqar_accuracy(m_off, cfg, 8, seed + 1, device)[0])

            torch.manual_seed(seed)
            m_at = TinyCausalTransformerLM(vocab, d_model=128, n_layers=2, n_heads=4,
                                           max_len=max(train_len, 1024))
            train(m_at, cfg, steps, 3e-3, seed, device); m_at.eval()
            at.append(mqar_accuracy(m_at, cfg, 8, seed + 1, device)[0])

        on_mu, on_sd = mean_std(on)
        off_mu, _ = mean_std(off)
        at_mu, _ = mean_std(at)
        above = on_mu - off_mu
        rows.append({"n_pairs": npairs, "holo_on": on_mu, "holo_on_std": on_sd,
                     "holo_off": off_mu, "attn": at_mu, "above_floor": above,
                     "above_x_sqrtN": above * math.sqrt(npairs)})
        print(f"  n_pairs={npairs:2d}  holo_on {on_mu:.4f}±{on_sd:.4f}  "
              f"floor {off_mu:.4f}  attn {at_mu:.4f}  above {above:+.4f}  "
              f"above·sqrt(N) {above*math.sqrt(npairs):.4f}")

    # The 1/sqrt(N) test: if crosstalk-limited, above_floor * sqrt(N) is ~CONSTANT.
    consts = [r["above_x_sqrtN"] for r in rows if r["above_floor"] > 0]
    cmu, csd = mean_std(consts) if consts else (0, 0)
    cv = (csd / cmu) if cmu else float("inf")
    print("\n" + "=" * 74)
    print("1/sqrt(N) LAW TEST")
    print(f"  above_floor·sqrt(N) across n_pairs: mean {cmu:.4f}  std {csd:.4f}  CV {cv:.3f}")
    if cv < 0.35 and rows[0]["above_floor"] > rows[-1]["above_floor"]:
        verdict = "CROSSTALK CONFIRMED — recall decays ~1/sqrt(N), above·sqrt(N) ~ const"
    elif rows[0]["above_floor"] > 2 * rows[-1]["above_floor"]:
        verdict = "CAPACITY-LIMITED (recall falls with N) but not cleanly 1/sqrt(N)"
    else:
        verdict = "NOT crosstalk — recall ~flat in n_pairs; hypothesis WRONG"
    print(f"  >>> {verdict}")

    out = {"config": {"seeds": seeds, "steps": steps, "train_len": train_len,
                      "n_pairs_list": n_pairs_list, "chance": chance},
           "rows": rows, "sqrtN_const_mean": cmu, "sqrtN_const_cv": cv,
           "verdict": verdict, "elapsed_s": round(time.time() - t0, 1)}
    json.dump(out, open(os.path.join(REPO, "results", "crosstalk_smoking_gun.json"), "w"), indent=2)
    print(f"\nWritten results/crosstalk_smoking_gun.json  ({out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
