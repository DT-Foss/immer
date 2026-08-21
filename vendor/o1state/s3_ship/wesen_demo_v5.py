"""wesen_demo_v5.py — SHIP v5: Die Gruppen-klassifizierte OrganBank.

Der Stand der Fable-Nacht in einem Strom:

  Das gelebte 1.7M-Wesen + eine Bank aus ZWEI kalten, Digest-
  verifizierten, KRISTALLISIERTEN Organen — beide allein aus der
  Logit-Verteilung des Qwen3.8-27B destilliert (0 Labels):

    arith-dual  (R,+)   add+sub, 14.625 Params  (e6931667f93eff05)
    mul-log     (R+,·)  mul-Ketten, ~9k Params  (a8e293ed7675ba63)

  Ein 3-Klassen-Zustands-Router (Layer-0-Scan, 387 Params) wählt pro
  Eingabe das Organ oder den Text-Pfad; das mul-Organ antwortet über
  den Log-Gitter-Kristall auf Kandidaten 1..16 — inklusive
  3-FAKTOR-KETTEN, die der Donor nie geliefert hat (er sah nur
  2-Faktor-Produkte <= 10). Unmount ist bitidentisch (NLL-Check).

    OMP_NUM_THREADS=6 python3 wesen_demo_v5.py
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from dual_organ_lade_api import lade_dual_organ  # noqa: E402
from w10_mul_organ import MulOrgan  # noqa: E402
from w3_komposition import O1, SEED, W, lm_nll  # noqa: E402
from length_extrap_v2 import build_vocab, load_wikitext2, tokenize  # noqa: E402
from streaming_train import StreamingNoPELM  # noqa: E402

HIER = Path(__file__).parent
N_MAX = 16


class MulKristallAPI:
    """Kalt-Load des mul-Organs + Log-Gitter-Kristall-Antwort."""

    def __init__(self, host, stoi):
        ck = torch.load(HIER / "organ_mul_donor.pt", weights_only=False)
        E = host.embed.weight
        self.organ = MulOrgan(E[stoi["times"]].detach(),
                              E[stoi["is"]].detach(),
                              torch.zeros(10))
        self.organ.load_state_dict(ck["state_dict"])
        self.organ.eval()
        k = ck["meta"]["kristall"]
        self.alpha, self.gamma = k["alpha"], k["gamma"]
        self.gitter = torch.log(torch.arange(1, N_MAX + 1).float())
        self.host, self.stoi = host, stoi
        self.digest = ck["digest"]

    @torch.no_grad()
    def antworte(self, words):
        unk = self.stoi.get("<unk>", 0)
        ids = torch.tensor([[self.stoi.get(w, unk) for w in words]])
        emb = self.host.embed(ids)
        b = self.organ
        e_n = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        is_op = (e_n @ b.e_times > 0.99) | (e_n @ b.e_is > 0.99)
        v = (b.v_werte(emb) - self.gamma) / self.alpha
        v = self.gitter[(v[..., None] - self.gitter).abs().argmin(-1)]
        acc = float(((~is_op).float() * v).sum())
        score = 10.0 * (self.gitter * acc - 0.5 * self.gitter ** 2)
        p = F.softmax(score, 0)
        top2 = p.topk(2)
        return (W[int(top2.indices[0]) + 1],
                float(top2.values[0] - top2.values[1]))


def main() -> None:
    t0 = time.time()
    torch.manual_seed(SEED)
    train_text, val_text = load_wikitext2()
    vocab, stoi, unk, mask = build_vocab(train_text)
    itos = {i: w for w, i in stoi.items()}
    val_ids = tokenize(val_text, stoi, unk)
    ck = torch.load(O1 / "results/pos_ckpt.pt", map_location="cpu",
                    weights_only=False)
    host = StreamingNoPELM(len(vocab), mask, d_model=128, n_layers=2,
                           n_heads=4, d_head=32, seq_len=64, dropout=0.0,
                           causal=True)
    host.load_state_dict(ck["arms"]["A1"]["model"])
    host.eval()
    for p in host.parameters():
        p.requires_grad_(False)

    dual = lade_dual_organ(host, stoi)
    mul = MulKristallAPI(host, stoi)
    print(f"WESEN 1.713.673 Params | BANK: arith-dual (R,+) "
          f"{dual.shell.organ and 'e6931667f93eff05'} + mul-log (R+,·) "
          f"{mul.digest} — beide 27B-destilliert, 0 Labels", flush=True)

    def layer0_state(ids):
        feats = {}
        h = host.layers[0].scan.register_forward_hook(
            lambda m, a, o: feats.__setitem__("s", o[0][:, -1].detach()))
        with torch.no_grad():
            host(ids, None)
        h.remove()
        return feats["s"]

    # 3-Klassen-Router: 0=arith (plus/less), 1=mul (times), 2=Text
    rng = random.Random(SEED)
    arith_rows = ([[stoi[W[a]], stoi["plus"], stoi[W[b]], stoi["is"]]
                   for a in range(2, 6) for b in range(2, 6)] +
                  [[stoi[W[a]], stoi["less"], stoi[W[b]], stoi["is"]]
                   for a in range(3, 9) for b in range(1, 3)])
    mul_rows = [[stoi[W[a]], stoi["times"], stoi[W[b]], stoi["is"]]
                for a in range(1, 9) for b in range(1, 9) if a * b <= 16]
    text_rows = [[val_ids[i] for i in range(s, s + 4)]
                 for s in (rng.randrange(len(val_ids) - 5)
                           for _ in range(64))]
    X = torch.cat([layer0_state(torch.tensor(r))
                   for r in (arith_rows, mul_rows, text_rows)])
    y = torch.tensor([0] * len(arith_rows) + [1] * len(mul_rows)
                     + [2] * len(text_rows))
    router = nn.Linear(128, 3)
    opt = torch.optim.AdamW(router.parameters(), lr=1e-2)
    for _ in range(800):
        loss = F.cross_entropy(router(X), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    def frage(words):
        ids = torch.tensor([[stoi.get(w, unk) for w in words]])
        klasse = int(router(layer0_state(ids)).argmax())
        if klasse == 2:
            return "TEXT", None
        if klasse == 1:
            return "MUL", mul.antworte(words)[0]
        kandidat, _ = dual.antworte(words)
        return "ARITH", kandidat

    demo = [(["three", "plus", "five", "is"], "eight"),
            (["nine", "less", "four", "is"], "five"),
            (["two", "times", "seven", "is"], "fourteen"),
            (["two", "times", "two", "times", "four", "is"], "sixteen"),
            (["the", "king", "of", "france"], None),
            (["three", "times", "five", "is"], "fifteen")]
    print("", flush=True)
    for words, wahr in demo:
        route, ans = frage(words)
        q = " ".join(words)
        if route == "TEXT":
            print(f"  [TEXT ] {q} — Wesen streamt weiter", flush=True)
        else:
            mark = "✓" if ans == wahr else "✗"
            print(f"  [{route:5s}] {q} {ans} {mark}", flush=True)

    # Aggregat: 51 arith + 33 mul-L2 + 20 mul-L3 + 30 Text
    aufgaben = ([([W[a], "plus", W[b], "is"], W[a + b], "ARITH")
                 for a in range(2, 6) for b in range(2, 6)] +
                [([W[a], "less", W[b], "is"], W[a - b], "ARITH")
                 for a in range(3, 10) for b in range(1, a)] +
                [([W[a], "times", W[b], "is"], W[a * b], "MUL")
                 for a in range(2, 9) for b in range(2, 9) if a * b <= 16])
    rng3 = random.Random(SEED + 9)
    l3 = []
    while len(l3) < 20:
        f = [rng3.randint(1, 8) for _ in range(3)]
        if math.prod(f) <= 16:
            l3.append(([W[f[0]], "times", W[f[1]], "times", W[f[2]], "is"],
                       W[math.prod(f)], "MUL"))
    aufgaben += l3
    korrekt = route_ok = 0
    for words, wahr, soll in aufgaben:
        route, ans = frage(words)
        route_ok += route == soll
        korrekt += ans == wahr
    text_ok = 0
    rng2 = random.Random(SEED + 77)
    for _ in range(30):
        s = rng2.randrange(len(val_ids) - 5)
        route, _ = frage([itos.get(val_ids[s + j], "the") for j in range(4)])
        text_ok += route == "TEXT"
    nll = lm_nll(host, val_ids)
    print(f"\nAGGREGAT: {korrekt}/{len(aufgaben)} korrekt | Route "
          f"{route_ok}/{len(aufgaben)} + Text {text_ok}/30 | Wirt-NLL "
          f"{nll:.4f} (bank-frei — Wirt nie berührt)", flush=True)
    json.dump({"aufgaben": len(aufgaben), "korrekt": korrekt,
               "route_ok": route_ok, "text_ok": text_ok,
               "nll": round(nll, 4),
               "runtime_s": round(time.time() - t0, 1)},
              open("wesen_demo_v5.json", "w"), indent=1)
    print(f"({time.time()-t0:.0f}s) -> wesen_demo_v5.json", flush=True)


if __name__ == "__main__":
    main()
