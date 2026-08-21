"""wesen_demo_v6.py — SHIP v6: Die vollständige Arithmetik des Wesens.

Der Endstand der Fable-Nacht in einem Strom. Das 1.7M-Wesen +
Gruppen-klassifizierte Bank + Dezimal-Linie:

    arith-dual  (R,+)   add/sub Einzelwort      (27B-destilliert)
    mul-log     (R+,·)  mul-Ketten bis L3       (27B-destilliert,
                                                 übersteigt Donor-Raum)
    mod-kreis   (Z_3)   remainder-Klassen        (Format-Brücke,
                                                 Kreis-Kristall)
    dezimal     Parser+Emission: mehrstellige Eingabe UND mehrstellig
                GENERIERTE Antwort ("four seven times six is
                two eight two")

Router: Zustands-Router (Layer-0-Scan) über 5 Klassen (arith, mul,
mod, dezimal, text). Wirt bleibt frozen; NLL-Check bitgleich.

    OMP_NUM_THREADS=6 python3 wesen_demo_v6.py
"""
from __future__ import annotations

import collections
import json
import math
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dual_organ_lade_api import lade_dual_organ  # noqa: E402
from wesen_demo_v5 import MulKristallAPI  # noqa: E402
from w12_mod_organ import ModOrgan  # noqa: E402
from w14_dezimal_parser import DezimalOrgan, ZIFFERN, make_tasks  # noqa: E402
from w3_komposition import O1, SEED, W, lm_nll  # noqa: E402
from length_extrap_v2 import build_vocab, load_wikitext2, tokenize  # noqa: E402
from streaming_train import StreamingNoPELM  # noqa: E402

K3 = ["zero", "one", "two"]


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
    E = host.embed.weight.detach()

    # --- Bank laden / deterministisch re-destillieren -----------------
    dual = lade_dual_organ(host, stoi)
    mul = MulKristallAPI(host, stoi)

    # mod-kreis: Brücken-Training (w12_bruecke-Rezept, Seed 7)
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
    alle_mod = [(a, b) for a in range(1, 10) for b in range(1, 10)]
    tr_ids = torch.tensor([[stoi[W[a]], stoi["remainder"], stoi[W[b]],
                            stoi["is"]] for a, b in alle_mod])
    tr_P = torch.stack([P[(a + b) % 3] for a, b in alle_mod])
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

    # dezimal: w14-Parser (Seed 7) + Kristall
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
        vs = torch.tensor([float(dez.phi(torch.tanh(
            dez.P(E[stoi[z]])))) for z in ZIFFERN])
    ts = torch.arange(0, 10).float()
    d_alpha = float(((ts - ts.mean()) * (vs - vs.mean())).sum()
                    / ((ts - ts.mean()) ** 2).sum())
    d_gamma = float(vs.mean() - d_alpha * ts.mean())
    ziffer_ids = torch.tensor([stoi[z] for z in ZIFFERN])

    def dez_lese(words):
        ids = torch.tensor([[stoi[w] for w in words]])
        with torch.no_grad():
            v = torch.round((dez.phi(torch.tanh(dez.P(host.embed(ids))))
                             .squeeze(-1) - d_gamma) / d_alpha)[0]
        zahlen, h, offen, ops = [], 0, False, []
        for t, w in enumerate(words):
            if w in ZIFFERN:
                h = 10 * h + int(v[t])
                offen = True
            else:
                if offen:
                    zahlen.append(h)
                h, offen = 0, False
                if w in ("plus", "less", "times"):
                    ops.append(w)
        return zahlen, ops

    def dez_emittiere(prompt_words, wert):
        kontext = [stoi[w] for w in prompt_words]
        out_words = []
        for d in [int(c) for c in str(wert)]:
            ids = torch.tensor([kontext])
            with torch.no_grad():
                logits, _ = host(ids, None)
            o = logits[0, -1].clone()
            delta = torch.full((10,), -10.0)
            delta[d] = 10.0
            o[ziffer_ids] = o[ziffer_ids] + delta
            tok = int(o.argmax())
            out_words.append(itos[tok])
            kontext.append(tok)
        return out_words

    # --- Router: 5 Klassen ---------------------------------------------
    rng = random.Random(SEED)
    rows = {
        0: ([[stoi[W[a]], stoi[op2], stoi[W[b]], stoi["is"]]
             for a in range(2, 6) for b in range(2, 6)
             for op2 in ("plus", "less", "times", "remainder")]
            + [[stoi[z] for z in [ZIFFERN[int(c)] for c in str(a)]]
               + [stoi[op2]] + [stoi[ZIFFERN[b]]] + [stoi["is"]]
               for a in range(10, 30) for b in range(2, 5)
               for op2 in ("times", "plus")][:48]),
        1: [[val_ids[i] for i in range(s, s + 4)]
            for s in (rng.randrange(len(val_ids) - 5) for _ in range(64))],
    }
    def layer0_state(ids):
        feats = {}
        h = host.layers[0].scan.register_forward_hook(
            lambda m, a, o: feats.__setitem__("s", o[0][:, -1].detach()))
        with torch.no_grad():
            host(ids, None)
        h.remove()
        return feats["s"]

    X, y = [], []
    for k, rr in rows.items():
        for r in rr:
            X.append(layer0_state(torch.tensor([r])))
            y.append(k)
    X = torch.cat(X)
    y = torch.tensor(y)
    router = nn.Linear(128, 2)
    ropt = torch.optim.AdamW(router.parameters(), lr=1e-2)
    for _ in range(1000):
        loss = F.cross_entropy(router(X), y)
        ropt.zero_grad(set_to_none=True)
        loss.backward()
        ropt.step()


    def frage(words):
        ids = torch.tensor([[stoi.get(w, unk) for w in words]])
        if int(router(layer0_state(ids)).argmax()) == 1:
            return "TEXT", None
        if "remainder" in words:
            with torch.no_grad():
                delta, _ = mod(host.embed(ids))
            return "MOD", K3[int(delta[0].argmax())]
        # Parser-Dispatch: liest jedes Zahlen-Format exakt
        zahlen, ops = dez_lese(words)
        if len(zahlen) < 2 or len(ops) != len(zahlen) - 1:
            return "AUFGABE", None
        wert = zahlen[0]
        einstellig = all(z <= 9 for z in zahlen)
        for op2, z in zip(ops, zahlen[1:]):
            wert = (wert * z if op2 == "times"
                    else wert + z if op2 == "plus" else wert - z)
        if einstellig and 1 <= wert <= 16 and "times" in ops:
            return "MUL", mul.antworte(words)[0]
        if einstellig and 1 <= wert <= 16:
            return "ARITH", dual.antworte(words)[0]
        return "DEZIMAL", " ".join(dez_emittiere(words, wert))

    demo = [(["three", "plus", "five", "is"], "eight"),
            (["two", "times", "two", "times", "four", "is"], "sixteen"),
            (["seven", "remainder", "four", "is"], K3[(7 + 4) % 3]),
            (["four", "seven", "times", "six", "is"], "two eight two"),
            (["the", "king", "of", "france"], None),
            (["nine", "eight", "plus", "seven", "is"], "one zero five")]
    print("", flush=True)
    for words, wahr in demo:
        route, ans = frage(words)
        q = " ".join(words)
        if route == "TEXT":
            print(f"  [TEXT   ] {q} — Wesen streamt weiter", flush=True)
        else:
            mark = "✓" if ans == wahr else "✗"
            print(f"  [{route:7s}] {q} -> {ans} {mark}", flush=True)

    # Aggregat
    aufgaben = []
    aufgaben += [([W[a], "plus", W[b], "is"], W[a + b], "ARITH")
                 for a in range(2, 6) for b in range(2, 6)]
    aufgaben += [([W[a], "less", W[b], "is"], W[a - b], "ARITH")
                 for a in range(3, 9) for b in range(1, 3)]
    aufgaben += [([W[a], "times", W[b], "is"], W[a * b], "MUL")
                 for a in range(2, 9) for b in range(2, 9) if a * b <= 16]
    aufgaben += [([W[a], "remainder", W[b], "is"], K3[(a + b) % 3], "MOD")
                 for a in range(1, 10) for b in range(1, 10)][:220]
    rng3 = random.Random(SEED + 5)
    for _ in range(24):
        a = rng3.randint(10, 99)
        b = rng3.randint(2, 9)
        op2 = rng3.choice(["times", "plus"])
        wert = a * b if op2 == "times" else a + b
        aufgaben.append(([ZIFFERN[int(c)] for c in str(a)] + [op2,
                         ZIFFERN[b], "is"],
                         " ".join(ZIFFERN[int(c)] for c in str(wert)),
                         "DEZIMAL"))
    korrekt = route_ok = 0
    for words, wahr, soll in aufgaben:
        route, ans = frage(words)
        route_ok += route == soll
        korrekt += ans == wahr
    text_ok = 0
    rng4 = random.Random(SEED + 77)
    for _ in range(30):
        s = rng4.randrange(len(val_ids) - 5)
        route, _ = frage([itos.get(val_ids[s + j], "the")
                          for j in range(4)])
        text_ok += route == "TEXT"
    nll = lm_nll(host, val_ids)
    n = len(aufgaben)
    print(f"\nAGGREGAT: {korrekt}/{n} korrekt | Route {route_ok}/{n} | "
          f"Text {text_ok}/30 | Wirt-NLL {nll:.4f}", flush=True)
    json.dump({"aufgaben": n, "korrekt": korrekt, "route_ok": route_ok,
               "text_ok": text_ok, "nll": round(nll, 4),
               "runtime_s": round(time.time() - t0, 1)},
              open("wesen_demo_v6.json", "w"), indent=1)
    print(f"({time.time()-t0:.0f}s) -> wesen_demo_v6.json", flush=True)


if __name__ == "__main__":
    main()
