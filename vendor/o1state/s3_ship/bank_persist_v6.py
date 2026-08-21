"""bank_persist_v6.py — Die Bank wird vollständig kalt: mod-kreis und
dezimal als Digest-Artefakte mit Gruppen-Label und TRÄGER-REGISTER.

Nach 8.119 (Träger-Regel) und 8.122-Korrektur (Atomizität) speichert
jedes Artefakt: state_dict, Digest, Gruppe, Kristall-Parameter UND
das Gültigkeits-Intervall seiner Invariante. Ship v6 ist damit
komplett kalt-ladbar (4 Organe):

    organ_dual_donor.pt   (R,+)   Träger 1..9    e6931667f93eff05
    organ_mul_donor.pt    (R+,·)  Träger 1..8    a8e293ed7675ba63
    organ_mod_kreis.pt    (Z_3)   Träger 1..9    <neu>
    organ_dezimal.pt      Parser  Träger 0..9    <neu>

    OMP_NUM_THREADS=6 python3 bank_persist_v6.py
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from w12_mod_organ import ModOrgan  # noqa: E402
from w14_dezimal_parser import DezimalOrgan, ZIFFERN, make_tasks  # noqa: E402
from w3_komposition import O1, SEED, W  # noqa: E402
from length_extrap_v2 import build_vocab, load_wikitext2  # noqa: E402
from streaming_train import StreamingNoPELM  # noqa: E402


def _digest(sd):
    h = hashlib.sha256()
    for k, v in sd.items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def main() -> None:
    t0 = time.time()
    torch.manual_seed(SEED)
    train_text, _ = load_wikitext2()
    vocab, stoi, unk, mask = build_vocab(train_text)
    ck = torch.load(O1 / "results/pos_ckpt.pt", map_location="cpu",
                    weights_only=False)
    host = StreamingNoPELM(len(vocab), mask, d_model=128, n_layers=2,
                           n_heads=4, d_head=32, seq_len=64, dropout=0.0,
                           causal=True)
    host.load_state_dict(ck["arms"]["A1"]["model"])
    host.eval()
    for p in host.parameters():
        p.requires_grad_(False)
    E = host.embed.weight.detach()
    registry = {}

    # --- mod-kreis (Brücken-Rezept, w12_bruecke) ----------------------
    npz = np.load("donor_data/donor_targets_mod3_27b.npz",
                  allow_pickle=True)
    lg = np.concatenate([np.asarray(npz["logits"][:, 0, :], np.float32),
                         np.asarray(npz["logits_eval"][:, 0, :],
                                    np.float32)])
    P = torch.zeros(3, 3)
    for p, pr in zip([str(x) for x in list(npz["prompts"])
                      + list(npz["prompts_eval"])],
                     torch.softmax(torch.tensor(lg), 1)):
        m = re.search(r"mod3\((\d+)\)=", p.strip().split("\n")[-1])
        if m:
            P[int(m.group(1)) % 3] += pr[:3] / pr[:3].sum()
    P = P / P.sum(1, keepdim=True)
    P = (P ** 2) / (P ** 2).sum(1, keepdim=True)
    alle = [(a, b) for a in range(1, 10) for b in range(1, 10)]
    tr_ids = torch.tensor([[stoi[W[a]], stoi["remainder"], stoi[W[b]],
                            stoi["is"]] for a, b in alle])
    tr_P = torch.stack([P[(a + b) % 3] for a, b in alle])
    torch.manual_seed(SEED)
    mod = ModOrgan(E[stoi["remainder"]])
    opt = torch.optim.AdamW(mod.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(SEED)
    for _ in range(3000):
        idx = torch.randperm(len(tr_ids), generator=g)[:32]
        with torch.no_grad():
            emb = host.embed(tr_ids[idx])
        delta, _ = mod(emb)
        loss = -(tr_P[idx] * F.log_softmax(delta, 1)).sum(1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    # Eingangskontrolle: zirkulaere Resultante (Kreis-Verifier, 8.118)
    import math
    with torch.no_grad():
        vs = {n: float(mod.phi(torch.tanh(mod.P(E[stoi[W[n]]]))))
              for n in range(1, 10)}
    zentren, res_min = {}, 1.0
    for k in range(3):
        xs = [vs[n] for n in range(1, 10) if n % 3 == k]
        cx = sum(math.cos(2 * math.pi * x / 3) for x in xs) / len(xs)
        sx = sum(math.sin(2 * math.pi * x / 3) for x in xs) / len(xs)
        zentren[k] = 3 * math.atan2(sx, cx) / (2 * math.pi) % 3
        res_min = min(res_min, (cx ** 2 + sx ** 2) ** 0.5)
    assert res_min >= 0.95, f"Kreis-Verifier: Resultante {res_min:.3f}"
    sd = mod.state_dict()
    dig = _digest(sd)
    torch.save({"state_dict": sd, "digest": dig,
                "meta": {"gruppe": "Z_3", "quelle": "Format-Bruecke aus "
                         "mod3-Ziffern-Ernte (0 Labels, 0 neue Aufrufe)",
                         "kristall": {"zentren": zentren,
                                      "resultante_min": round(res_min, 5)},
                         "traeger": "Zahlwoerter one..nine",
                         "seed": SEED}}, "organ_mod_kreis.pt")
    registry["mod-kreis"] = {"digest": dig, "gruppe": "Z_3",
                             "resultante": round(res_min, 5)}
    print(f"organ_mod_kreis.pt | digest {dig} | Resultante {res_min:.5f}",
          flush=True)

    # --- dezimal (w14-Parser) -----------------------------------------
    rng0 = random.Random(SEED)
    spec_tr = ([(a, b, op) for a in range(1, 10) for b in range(1, 10)
                for op in ("plus", "less")]
               + [(a, b, op) for a in (10, 11, 12) for b in range(1, 5)
                  for op in ("plus", "less")])
    dtrain = make_tasks(spec_tr, rng0, stoi)
    torch.manual_seed(SEED)
    dez = DezimalOrgan(E[stoi["plus"]], E[stoi["less"]], E[stoi["is"]],
                       stellen=True)
    dopt = torch.optim.AdamW(dez.parameters(), lr=3e-3)
    g2 = torch.Generator().manual_seed(SEED)
    by_l = collections.defaultdict(list)
    for i, t in enumerate(dtrain):
        by_l[len(t[0])].append(i)
    keys = list(by_l)
    tgt_all = torch.tensor([t[1] - 1 for t in dtrain])
    cnt = collections.Counter(tgt_all.tolist())
    wts = torch.tensor([1.0 / cnt[int(t)] for t in tgt_all])
    for _ in range(3000):
        L = keys[int(torch.randint(len(keys), (1,), generator=g2))]
        pool = by_l[L]
        idx = [pool[i] for i in
               torch.randperm(len(pool), generator=g2)[:32].tolist()]
        ids = torch.tensor([dtrain[i][0] for i in idx])
        with torch.no_grad():
            emb = host.embed(ids)
        li = F.cross_entropy(dez(emb), tgt_all[idx], reduction="none")
        loss = (li * wts[idx]).sum() / wts[idx].sum()
        dopt.zero_grad(set_to_none=True)
        loss.backward()
        dopt.step()
    with torch.no_grad():
        vs2 = torch.tensor([float(dez.phi(torch.tanh(
            dez.P(E[stoi[z]])))) for z in ZIFFERN])
    ts = torch.arange(0, 10).float()
    alpha = float(((ts - ts.mean()) * (vs2 - vs2.mean())).sum()
                  / ((ts - ts.mean()) ** 2).sum())
    gamma = float(vs2.mean() - alpha * ts.mean())
    r2 = 1 - float(((vs2 - (alpha * ts + gamma)) ** 2).sum()
                   / ((vs2 - vs2.mean()) ** 2).sum())
    assert r2 >= 0.95, f"Eingangskontrolle: r² {r2:.3f}"
    sd2 = dez.state_dict()
    dig2 = _digest(sd2)
    torch.save({"state_dict": sd2, "digest": dig2,
                "meta": {"gruppe": "(R,+) mit Dezimal-Parser (h<-10h+v)",
                         "kristall": {"alpha": alpha, "gamma": gamma,
                                      "r2": round(r2, 5)},
                         "atomar": "Deployment nutzt NUR den vollen "
                                   "Kristall-Pfad (Parse->exakt->"
                                   "Emission); das gelernte Readout ist "
                                   "Trainings-Geruest (8.122-Korrektur)",
                         "traeger": "Ziffernwoerter zero..nine",
                         "seed": SEED}}, "organ_dezimal.pt")
    registry["dezimal"] = {"digest": dig2, "r2": round(r2, 5)}
    print(f"organ_dezimal.pt | digest {dig2} | r² {r2:.5f}", flush=True)

    json.dump({"registry": registry,
               "bank_komplett": ["organ_dual_donor.pt",
                                 "organ_mul_donor.pt",
                                 "organ_mod_kreis.pt", "organ_dezimal.pt"],
               "runtime_s": round(time.time() - t0, 1)},
              open("bank_persist_v6.json", "w"), indent=1)
    print(f"({time.time()-t0:.0f}s) -> bank_persist_v6.json", flush=True)


if __name__ == "__main__":
    main()
