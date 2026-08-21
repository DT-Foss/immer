"""w12_mod_organ.py — Das STRIKTE REZEPT (R23) end-to-end am dritten Organ.

mod3 in Wortform ("a remainder b is" -> zero/one/two, Klasse (a+b)%3),
label-frei vom 27B destilliert, nach dem vollständigen Rezept:

  STRUKTURFORM  Zyklus-Hart (Gruppe Z_3): delta_c = s0·cos(2π/3·(u−c))
  KARTE         id auf die Zahlengerade: u = v(a) + v(b)
  ROUTE-GATE    parameterlos: cos-Match des t-2-Tokens auf 'remainder'
  KRISTALL      v -> round((v−gamma)/alpha); bei ganzzahligem u ist der
                Kreis-Readout EXAKT maximal bei c ≡ u (mod 3)

Kein freier Rechen-Kanal: kein MLP-Gate, keine Klassen-DOF, nur
phi (Zahlengerade) und s0 lernen. Training: Donor-KL tau=2 + Balance
auf ~2/3 der 81 Zellen; held-out: 27 Zellen inkl. ganzer Orbits.

Prognose (festgelegt, 20.08. vor dem Lauf): Donor-KL roh held >= 0,8;
nach Ganzzahl-Kristall ALLE 81 Zellen 1,000; gate_txt = 0, NLL-Pfad
unberührt (Route-Gate feuert auf WT2 nie).

    OMP_NUM_THREADS=6 python3 w12_mod_organ.py
"""
from __future__ import annotations

import collections
import glob
import json
import math
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from w3_komposition import O1, SEED, W  # noqa: E402
from length_extrap_v2 import build_vocab, load_wikitext2, tokenize  # noqa: E402
from streaming_train import StreamingNoPELM  # noqa: E402

K = ["zero", "one", "two"]
CHUNK_GLOB = "donor_data/donor_chunk_modw_train_*.npz"


class ModOrgan(nn.Module):
    """Zyklus-Hart-Form + Route-Gate, EINE Zahlengerade, s0 lernbar."""

    def __init__(self, e_rem, d_model=128, d_e=32):
        super().__init__()
        self.P = nn.Linear(d_model, d_e, bias=False)
        self.phi = nn.Sequential(nn.Linear(d_e, 32), nn.Tanh(),
                                 nn.Linear(32, 1))
        self.register_buffer("e_rem", e_rem / e_rem.norm())
        self.s0 = nn.Parameter(torch.tensor(3.0))
        self.krist = None                      # (alpha, gamma) nach Snap

    def u_of(self, emb):
        v = self.phi(torch.tanh(self.P(emb))).squeeze(-1)      # [B,T]
        if self.krist is not None:
            a, g = self.krist
            v = torch.round((v - g) / a)
        return v[:, 0] + v[:, 2]               # v(a) + v(b) (a rem b is)

    def route(self, emb):
        e_n = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return (e_n[:, -3] @ self.e_rem > 0.9).float()          # t-2

    def forward(self, emb):
        u = self.u_of(emb)
        cs = torch.arange(3).float()
        delta = self.s0 * torch.cos(2 * math.pi / 3
                                    * (u[:, None] - cs[None]))
        return delta, self.route(emb)


def main() -> None:
    t0 = time.time()
    chunks = sorted(glob.glob(CHUNK_GLOB))
    assert chunks, f"keine Chunks unter {CHUNK_GLOB}"
    prompts, logits = [], []
    for c in chunks:
        npz = np.load(c, allow_pickle=True)
        prompts += [str(p) for p in npz["prompts"]]
        logits.append(np.asarray(npz["logits"][:, 0, :], dtype=np.float32))
        cand_words = [str(w) for w in npz["cand"]]
    assert cand_words == K
    logits = np.concatenate(logits)

    train_text, val_text = load_wikitext2()
    vocab, stoi, unk, mask = build_vocab(train_text)
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

    # Donor-Zeilen -> Paare; held-out 27 Zellen (Zell-Ebene, nicht Zeile)
    def paar_of(prompt):
        q = prompt.strip().split("\n")[-1]
        ws = re.findall(r"[a-zA-Z]+", q.lower())
        return (W.index(ws[0]), W.index(ws[2]))

    paare = [paar_of(p) for p in prompts]
    alle = sorted(set(paare))
    held = set(random.Random(49).sample(alle, 27))
    keep = [i for i, pr in enumerate(paare) if pr not in held]
    tr_ids = torch.tensor([[stoi[W[a]], stoi["remainder"], stoi[W[b]],
                            stoi["is"]] for a, b in
                           [paare[i] for i in keep]])
    probs = torch.softmax(torch.tensor(logits[keep]), 1)
    probs = (probs ** 2) / (probs ** 2).sum(1, keepdim=True)
    donor_arg = probs.argmax(1)
    cnt = collections.Counter(donor_arg.tolist())
    wts = torch.tensor([1.0 / cnt[int(i)] for i in donor_arg])
    donor_acc = float(np.mean(
        [K[int(i)] == K[(a + b) % 3] for i, (a, b) in
         zip(donor_arg, [paare[j] for j in keep])]))
    print(f"{len(keep)} Zeilen (held {len(held)} Zellen) | donor acc "
          f"{donor_acc:.3f} | 0 Labels", flush=True)

    torch.manual_seed(SEED)
    organ = ModOrgan(E[stoi["remainder"]])
    opt = torch.optim.AdamW(organ.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(SEED)
    for step in range(3000):
        idx = torch.randperm(len(tr_ids), generator=g)[:32]
        with torch.no_grad():
            emb = host.embed(tr_ids[idx])
        delta, _ = organ(emb)
        logp = F.log_softmax(delta, 1)
        li = -(probs[idx] * logp).sum(1)
        loss = (li * wts[idx]).sum() / wts[idx].sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    def messen(tag):
        ok_tr = ok_h = n_tr = n_h = 0
        for a, b in alle:
            ids = torch.tensor([[stoi[W[a]], stoi["remainder"], stoi[W[b]],
                                 stoi["is"]]])
            with torch.no_grad():
                delta, _ = organ(host.embed(ids))
            hit = int(delta[0].argmax()) == (a + b) % 3
            if (a, b) in held:
                ok_h += hit
                n_h += 1
            else:
                ok_tr += hit
                n_tr += 1
        r = {"train": round(ok_tr / n_tr, 4), "held": round(ok_h / n_h, 4)}
        print(f"{tag}: {r}", flush=True)
        return r

    res = {"donor_acc": round(donor_acc, 4), "roh": messen("roh (Donor-KL)")}

    # KRISTALL: Zahlengerade snappen
    with torch.no_grad():
        vs = torch.tensor([float(organ.phi(torch.tanh(
            organ.P(E[stoi[W[n]]])))) for n in range(1, 10)])
    ts = torch.arange(1, 10).float()
    alpha = float(((ts - ts.mean()) * (vs - vs.mean())).sum()
                  / ((ts - ts.mean()) ** 2).sum())
    gamma = float(vs.mean() - alpha * ts.mean())
    r2 = 1 - float(((vs - (alpha * ts + gamma)) ** 2).sum()
                   / ((vs - vs.mean()) ** 2).sum())
    res["invariante"] = {"alpha": round(alpha, 4), "gamma": round(gamma, 4),
                         "r2": round(r2, 5)}
    print(f"INVARIANTE: {res['invariante']}", flush=True)
    organ.krist = (alpha, gamma)
    res["kristall"] = messen("kristall (Ganzzahl-Snap + exakter Kreis)")

    # Route-Gate auf Text: feuert nie
    rng = random.Random(SEED)
    gtx = []
    for _ in range(64):
        s = rng.randrange(len(val_ids) - 5)
        ids = torch.tensor([val_ids[s:s + 4]])
        with torch.no_grad():
            _, gt = organ(host.embed(ids))
        gtx.append(float(gt))
    res["gate_txt"] = round(sum(gtx) / len(gtx), 5)
    print(f"gate_txt: {res['gate_txt']}", flush=True)

    json.dump({"results": res,
               "prognose": "roh held >= 0,8; kristall 81/81 = 1,000; "
                           "gate_txt = 0",
               "rezept": "R23: Zyklus-Hart + id-Karte + Route-Gate + "
                         "Ganzzahl-Kristall; 0 Labels",
               "runtime_s": round(time.time() - t0, 1)},
              open("w12_mod_organ.json", "w"), indent=1)
    print(f"({time.time()-t0:.0f}s) -> w12_mod_organ.json", flush=True)


if __name__ == "__main__":
    main()
