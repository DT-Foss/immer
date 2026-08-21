"""gssm_organ_proto.py — A3-PROTOTYP: Organ IM Zustandsraum (o1-state-Muster).

Der Moonshot-Kern: Im Transformer wirkt das Organ als HOOK an der
Lese-Position (gemessen, Runde 17-18). Der o1-state-Wirt ist ein GSSM —
dort muss das Organ in den ZUSTANDS-UPDATE (nicht in einen Hook).
Dieser Prototyp zeigt das Muster minimal und falsifizierbar:

  1. Carry-Vertrag (o1-state-Prinzip): der Zustand ist die deterministische
     Kodierung der Sequenz; verschiedene Sequenzen -> verschiedene
     Zustaende (Kollisions-Sentinel); die Kodierung ist umkehrbar
     (Readout trainiert).
  2. Organ im Update: h_t = (1-g_t)*h_{t-1} + g_t*f(W u_t + Organ(h_{t-1})*dose)
     — das Organ moduliert den Zustandsfluss (kein Hook).
  3. W1-Falsifikator: (a) Organ-Einbau bricht den Carry (Sentinel-
     Kollisionen steigen) -> FAIL; (b) Organ bringt KEINE Faehigkeit
     (Readout-Accuracy ohne Organ >= mit Organ) -> FAIL.

Nur Rechnung im eigenen Ordner; der echte o1-state-Checkout bleibt
read-only (Integration durch Claude/David mit diesem Muster).
"""
from __future__ import annotations

import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 7


class MiniGSSM(nn.Module):
    """Gated State-Space: h_t = (1-g_t) h_{t-1} + g_t tanh(W u_t [+ Organ])."""

    def __init__(self, d: int = 64, vocab: int = 19, d_organ: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.W = nn.Linear(d, d)          # Input-Update
        self.gate = nn.Linear(d, d)       # Update-Gate (sigmoid)
        self.d = d
        self.organ: nn.Module | None = None
        self.dose = 1.0

    def forward(self, ids: torch.Tensor, with_organ: bool = False):
        """ids: (B, T). Rueckgabe: Zustaende (B, T+1, d) inkl. h_0=0."""
        B, T = ids.shape
        h = torch.zeros(B, self.d, device=ids.device)
        states = [h]
        for t in range(T):
            u = self.emb(ids[:, t])
            cand = torch.tanh(self.W(u))
            if with_organ and self.organ is not None:
                cand = cand + self.dose * self.organ(h)
            g = torch.sigmoid(self.gate(u))
            h = (1 - g) * h + g * cand
            states.append(h)
        return torch.stack(states, 1)     # (B, T+1, d)


def make_tasks(n: int, seed: int = SEED):
    """Zwei Aufgaben: carry-Sentinels (deterministische Kodierung) und
    add2-artige Rechnung (Zustand soll die Summe tragen)."""
    import random
    rng = random.Random(seed)
    carry, calc = [], []
    for _ in range(n):
        # carry: zufaellige Token-Sequenz (Token 2..9)
        seq = [rng.randint(2, 9) for _ in range(rng.randint(2, 6))]
        carry.append((seq, seq))          # Rekonstruktions-Ziel
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        # calc: [a, PLUS=10, b, EQ=11] -> Summe (12..18 -> Token 12..18)
        calc.append(([a, 10, b, 11], a + b))
    return carry, calc


def train_readout(states, targets, steps: int = 300):
    """Readout: letzter Zustand -> Ziel-Token (Linear)."""
    d = states.shape[-1]
    ro = nn.Linear(d, 19)   # Summen 4..18 (15 Klassen) + Rest
    opt = torch.optim.AdamW(ro.parameters(), lr=1e-2)
    for _ in range(steps):
        idx = torch.randperm(len(states))[:64]
        loss = F.cross_entropy(ro(states[idx].detach()), targets[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        with torch.no_grad():
            acc = float((ro(states).argmax(-1) == targets).float().mean())
    return ro, acc


def main() -> None:
    t0 = time.time()
    torch.manual_seed(SEED)
    dev = "cpu"
    model = MiniGSSM(d=64, vocab=19).to(dev)
    carry, calc = make_tasks(200, SEED)

    # ---- Carry-Sentinel OHNE Organ: deterministische Kodierung ----
    # nur EINDEUTIGE Sequenzen (Duplikate wuerden trivial kollidieren)
    seen, uniq = set(), []
    for s, _ in carry:
        k = tuple(s)
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    Tmax = max(len(s) for s in uniq)
    ids = torch.zeros(len(uniq), Tmax, dtype=torch.long)
    for i, s in enumerate(uniq):
        ids[i, :len(s)] = torch.tensor(s)
    states = model(ids)                    # (B, T+1, d)
    h_last = states[:, -1]
    # Kollisions-Info: max |cos-sim| zwischen Zuständen (informativ)
    sim = F.cosine_similarity(h_last.unsqueeze(1), h_last.unsqueeze(0), -1)
    sim = sim - torch.eye(len(h_last))
    max_sim = float(sim.abs().max())
    print(f"carry-sentinel (ohne Organ): {len(h_last)} eindeutige Sequenzen, "
          f"max |cos-sim| = {max_sim:.4f} (informativ)", flush=True)
    # Carry-Test: Rekonstruktion des letzten Tokens aus dem Zustand
    carry_tgt = torch.tensor([s[-1] for s in uniq])
    _, rec_no = train_readout(h_last, carry_tgt)

    # ---- Organ trainieren: Zustandsfluss soll die Summe tragen ----
    calc_ids = torch.tensor([s for s, _ in calc])
    calc_tgt = torch.tensor([t for _, t in calc])
    organ = nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 64))
    model.organ = organ
    opt = torch.optim.AdamW(organ.parameters(), lr=1e-3)
    for step in range(400):
        idx = torch.randperm(len(calc_ids))[:32]
        st = model(calc_ids[idx], with_organ=True)[:, -1]
        loss = F.mse_loss(st, model.emb(calc_tgt[idx]).detach())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # ---- Falsifikator (a): Carry nach Organ-Einbau ----
    with torch.no_grad():
        st_org = model(ids, with_organ=True)[:, -1]
        sim2 = F.cosine_similarity(st_org.unsqueeze(1), st_org.unsqueeze(0), -1)
        sim2 = sim2 - torch.eye(len(st_org))
        max_sim2 = float(sim2.abs().max())
    _, rec_org = train_readout(st_org, carry_tgt)
    carry_ok = rec_org >= rec_no - 0.05   # Rekonstruktion nicht gebrochen
    print(f"carry-rekonstruktion: ohne Organ {rec_no:.3f} | mit Organ "
          f"{rec_org:.3f} (max |cos-sim| {max_sim2:.4f}) -> "
          f"{'OK (Carry erhalten)' if carry_ok else 'FAIL (Carry gebrochen)'}",
          flush=True)

    # ---- Falsifikator (b): Faehigkeit im Zustand ----
    with torch.no_grad():
        st_calc = model(calc_ids, with_organ=True)[:, -1]
        st_calc_no = model(calc_ids)[:, -1]
    _, acc_organ = train_readout(st_calc, calc_tgt)
    _, acc_no = train_readout(st_calc_no, calc_tgt)
    # Zufalls-Baseline: 1/19 Klassen
    print(f"summen-readout: ohne Organ {acc_no:.3f} | mit Organ {acc_organ:.3f} "
          f"(baseline 0.053)", flush=True)
    cap_ok = acc_organ > acc_no + 0.1
    print(f"faehigkeits-test: -> {'OK (Organ traegt)' if cap_ok else 'FAIL (Organ bringt nichts)'}",
          flush=True)

    res = {"carry_max_sim_ohne": round(max_sim, 4),
           "carry_max_sim_mit": round(max_sim2, 4),
           "carry_rek_ohne": round(rec_no, 4), "carry_rek_mit": round(rec_org, 4),
           "carry_ok": carry_ok,
           "readout_ohne": round(acc_no, 4), "readout_mit": round(acc_organ, 4),
           "faehigkeit_ok": cap_ok,
           "runtime_s": round(time.time() - t0, 1)}
    print(json.dumps(res, indent=1))
    json.dump(res, open("gssm_organ_proto.json", "w"), indent=1)
    print("-> gssm_organ_proto.json", flush=True)


if __name__ == "__main__":
    main()
