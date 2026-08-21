"""dual_organ_lade_api.py — Persistenz + Lade-API fuer das Dual-Organ.

Speichert das beste Donor-KL-Dual-Organ (w3_dual_organ.py, Teil B:
sub 1,000/1,000 + add 1,000/1,000, 0 Labels) als s3/organ_dual_donor.pt
und stellt die Lade-API fuer wesen_demo_v4 bereit:

    from dual_organ_lade_api import lade_dual_organ
    api = lade_dual_organ(host, stoi)
    kandidat, margin = api.antworte(["nine", "less", "four", "is"])

antworte() arbeitet mount-frei als Shell (Wirt unberuehrt), nutzt den
harten Operator-Router (t-2-Token 'less'/'plus' waehlt Kopf UND
Kandidatenraum) und gibt die Margin p_top1 - p_top2 der Softmax auf dem
Kandidaten-Slice zurueck (das w2-Margin-Verhalten). Ohne erkannten
Operator: (None, 0.0).

Erzeugen/erneuern des Artefakts (trainiert Seed 7, prueft held-out,
schreibt .pt + Digest):

    OMP_NUM_THREADS=6 python3 dual_organ_lade_api.py
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from w3_dual_organ import (  # noqa: E402
    ADD_NPZ, SUB_NPZ, DualOrgan, DualShell, W, load_donor, sharpen_bal,
)

PT_PATH = Path(__file__).parent / "organ_dual_donor.pt"
SEED = 7


def _digest(state_dict) -> str:
    """sha256-hex16 ueber die Parameter-Bytes (feste state_dict-Ordnung)."""
    h = hashlib.sha256()
    for k, v in state_dict.items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


class DualOrganAPI:
    """Mount-freie Antwort-Schnittstelle um einen frozen Wirt."""

    def __init__(self, host, stoi, organ, sub_cids, add_cids):
        self.shell = DualShell(host, organ, sub_cids, add_cids)
        self.stoi = stoi
        self.sub_cids = sub_cids
        self.add_cids = add_cids
        self.itos = {i: w for w, i in stoi.items()}

    @torch.no_grad()
    def antworte(self, words: list[str]):
        """words z.B. ['nine','less','four','is'] -> (kandidat|None, margin)."""
        if "less" in words:
            cids = self.sub_cids
        elif "plus" in words:
            cids = self.add_cids
        else:
            return None, 0.0
        unk = self.stoi.get("<unk>", 0)
        ids = torch.tensor([[self.stoi.get(w, unk) for w in words]])
        logits, _ = self.shell(ids, None)
        p = F.softmax(logits[0, -1, cids], 0)
        top2 = p.topk(2)
        return (self.itos[cids[int(top2.indices[0])]],
                float(top2.values[0] - top2.values[1]))


def lade_dual_organ(host, stoi, pt_path=PT_PATH) -> DualOrganAPI:
    """Laedt organ_dual_donor.pt, verifiziert den Digest, baut die API."""
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    e_less = host.embed.weight[stoi["less"]].detach().clone()
    e_plus = host.embed.weight[stoi["plus"]].detach().clone()
    organ = DualOrgan(e_less, e_plus)
    organ.load_state_dict(ck["state_dict"])
    organ.eval()
    got = _digest(organ.state_dict())
    if got != ck["digest"]:
        raise ValueError(f"Digest-Mismatch: {got} != {ck['digest']}")
    return DualOrganAPI(host, stoi, organ,
                        ck["meta"]["sub_cand_ids"], ck["meta"]["add_cand_ids"])


def main() -> None:
    """Trainiert das Donor-KL-Dual-Organ (Seed 7) und persistiert es."""
    from w3_komposition import O1, lm_nll  # noqa: E402
    from length_extrap_v2 import build_vocab, load_wikitext2, tokenize  # noqa: E402
    from streaming_train import StreamingNoPELM  # noqa: E402

    t0 = time.time()
    train_text, val_text = load_wikitext2()
    vocab, stoi, unk, mask = build_vocab(train_text)
    val_ids = tokenize(val_text, stoi, unk)
    wt2_ids = torch.tensor(tokenize(train_text, stoi, unk)[:200000])
    ck = torch.load(O1 / "results/pos_ckpt.pt", map_location="cpu",
                    weights_only=False)
    host = StreamingNoPELM(len(vocab), mask, d_model=128, n_layers=2,
                           n_heads=4, d_head=32, seq_len=64, dropout=0.0,
                           causal=True)
    host.load_state_dict(ck["arms"]["A1"]["model"])
    host.eval()
    for p in host.parameters():
        p.requires_grad_(False)

    sub_probs, sub_cw = load_donor(SUB_NPZ)
    add_probs, add_cw = load_donor(ADD_NPZ)
    sub_cids = [stoi[w] for w in sub_cw]
    add_cids = [stoi[w] for w in add_cw]
    sub_all, add_all = sorted(sub_probs), sorted(add_probs)
    sub_held = sorted(random.Random(49).sample(sub_all, 4))
    add_held = sorted(random.Random(49).sample(add_all, 4))
    sub_tr = [p for p in sub_all if p not in sub_held]
    add_tr = [p for p in add_all if p not in add_held]

    def mk(ps, op):
        return torch.tensor([[stoi[W[a]], stoi[op], stoi[W[b]], stoi["is"]]
                             for a, b in ps])

    sub_ids, add_ids = mk(sub_tr, "less"), mk(add_tr, "plus")
    subP, subW_ = sharpen_bal(sub_probs, sub_tr)
    addP, addW_ = sharpen_bal(add_probs, add_tr)

    torch.manual_seed(SEED)
    e_less = host.embed.weight[stoi["less"]].detach().clone()
    e_plus = host.embed.weight[stoi["plus"]].detach().clone()
    organ = DualOrgan(e_less, e_plus)
    shell = DualShell(host, organ, sub_cids, add_cids)
    g = torch.Generator().manual_seed(SEED)

    def kl_loss(ids, cid, P, Wb):
        lg, _ = shell(ids, None)
        lp = F.log_softmax(lg[:, -1, torch.tensor(cid)], 1)
        return ((-(P * lp).sum(1)) * Wb).mean()

    def wt2_gates():
        starts = torch.randint(0, len(wt2_ids) - 33, (8,), generator=g)
        chunks = torch.stack([wt2_ids[s:s + 32] for s in starts])
        with torch.no_grad():
            emb = host.embed(chunks)
        _, gtx_s, _, gtx_a = organ(emb)
        return gtx_s.mean() + gtx_a.mean()

    # 3-Phasen-Rezept (identisch zu w3_dual_organ.train_dual)
    p1 = (list(organ.phi.parameters()) + list(organ.Rs.parameters())
          + list(organ.Pg.parameters()) + list(organ.gs.parameters()))
    opt = torch.optim.AdamW(p1, lr=3e-3)
    for _ in range(3600):
        loss = kl_loss(sub_ids, sub_cids, subP, subW_) + wt2_gates()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        vs = torch.tensor([float(organ.phi(host.embed.weight[stoi[W[n]]]))
                           for n in range(1, 10)])
        ts = torch.arange(1, 10).float()
        alpha = float(((ts - ts.mean()) * (vs - vs.mean())).sum()
                      / ((ts - ts.mean()) ** 2).sum())
        gamma = float(vs.mean() - alpha * ts.mean())
        organ.Ra.weight.copy_(torch.tensor(
            [[0.5 * c / alpha] for c in range(4, 11)]))
        organ.Ra.bias.copy_(torch.tensor(
            [-0.5 * c * c / 2 - 2 * gamma * 0.5 * c / alpha
             for c in range(4, 11)]))
    opt2 = torch.optim.AdamW(list(organ.Ra.parameters())
                             + list(organ.ga.parameters()), lr=3e-3)
    for _ in range(2400):
        loss = kl_loss(add_ids, add_cids, addP, addW_) + wt2_gates()
        opt2.zero_grad(set_to_none=True)
        loss.backward()
        opt2.step()
    opt3 = torch.optim.AdamW(organ.parameters(), lr=1e-3)
    for _ in range(1200):
        loss = (kl_loss(sub_ids, sub_cids, subP, subW_)
                + kl_loss(add_ids, add_cids, addP, addW_) + wt2_gates())
        opt3.zero_grad(set_to_none=True)
        loss.backward()
        opt3.step()

    organ.eval()
    sd = organ.state_dict()
    dig = _digest(sd)
    n_params = sum(p.numel() for p in organ.parameters())
    torch.save({
        "state_dict": sd,
        "digest": dig,
        "meta": {
            "quelle": "Qwen3.8-27B Donor-KL tau=2, 0 Labels",
            "params": n_params,
            "seeds": [SEED],
            "rezept": "3 Phasen: sub formt Gerade -> Selbst-Kalibrierung "
                      "Ra -> add-Kopf -> joint-Finetuning (w3_dual_organ)",
            "sub_cand_ids": sub_cids,
            "add_cand_ids": add_cids,
            "sub_cand_words": sub_cw,
            "add_cand_words": add_cw,
            "sub_held": sub_held,
            "add_held": add_held,
        },
    }, PT_PATH)
    print(f"gespeichert: {PT_PATH} | digest {dig} | {n_params:,} Params",
          flush=True)

    # Selbsttest ueber die Lade-API (frisches Objekt, Digest-Check inklusive)
    api = lade_dual_organ(host, stoi)
    ok = 0
    tests = ([([W[a], "less", W[b], "is"], W[a - b]) for a, b in sub_held]
             + [([W[a], "plus", W[b], "is"], W[a + b]) for a, b in add_held])
    for words, want in tests:
        kand, marg = api.antworte(words)
        ok += kand == want
        print(f"  {' '.join(words)} -> {kand} (margin {marg:.3f}, "
              f"wahr {want})", flush=True)
    kand, marg = api.antworte(["three", "times", "two", "is"])
    print(f"  Router-Negativtest 'times': {(kand, round(marg, 3))}",
          flush=True)
    nll = lm_nll(api.shell, val_ids)
    print(f"held-out via API: {ok}/8 | NLL {nll:.4f} | "
          f"({time.time() - t0:.0f}s)", flush=True)
    json.dump({"digest": dig, "params": n_params, "held_api": f"{ok}/8",
               "nll": round(nll, 4)},
              open(Path(__file__).parent / "organ_dual_donor_check.json",
                   "w"), indent=1)


if __name__ == "__main__":
    main()
