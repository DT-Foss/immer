"""wirt_gate.py — W4-PROTOTYP: Graft-Organ als VERIFIER-Gate im Wirt.

Kapselt die System-2-Mechanik (verifier_loop, Runde 18: twostep beam
1,000 / KL 0,000) als aufrufbares Gate fuer die Wirt-Integration
(o1-state / WIRT-ANSCHLUSS W4). Zwei Ebenen:

  1. Organ-Ebene (hier): train_organ() trainiert f_host (digkl,
     erweitertes Set) -> Organ mit Digest/Snapshot (OrganRegistry-
     Muster). verify() rankt Kandidaten per Lese-Positions-Beam.
  2. Wirt-Ebene (Claude/David): WirtGate an die utterance-Kette
     haengen — VOR dem Sprechen; Falsifikator W4: Halluzinationsrate
     sinkt nicht >= 5x.

Nur Rechnung im eigenen Ordner; keine Schreibrechte anderswo.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import redistill as R  # make_tasks, train_f_host, GraftHook, SEED
from organ_registry import Organ, OrganRegistry


def make_gate(host, tok, donor_npz: str, steps: int = 300, hidden: int = 512,
              dev: str = "mps", seed: int = 7) -> "WirtGate":
    """Kompletter Aufbau: f_host trainieren + WirtGate zurueckgeben.

    host: transformers-Modell (frozen). tok: zugehoeriger Tokenizer.
    donor_npz: 27B-Targets (Ziffern-Raum). Gibt ein einsatzbereites
    WirtGate zurueck (organ gemountet, digest gesetzt).
    """
    for p in host.parameters():
        p.requires_grad_(False)
    data = np.load(donor_npz, allow_pickle=True)
    lg = data["logits"]
    if lg.ndim == 2:
        lg = lg[:, None, :]
        mask = np.ones((lg.shape[0], 1), dtype=bool)
    else:
        mask = data["mask"]
    pr, an = list(data["prompts"]), list(data["ans"])
    pairs, rows = [], []
    for i, (p_, a_) in enumerate(zip(pr, an)):
        for p in range(lg.shape[1]):
            if not bool(mask[i, p]):
                break
            pairs.append((p_ + a_[:p],
                          tok.encode(a_[p], add_special_tokens=False)[0]))
            rows.append(lg[i, p])
    dl = torch.tensor(np.stack(rows), dtype=torch.float32, device=dev)
    dig_idx = torch.tensor(
        [tok.encode(c, add_special_tokens=False)[0] for c in "0,1,2,3,4,5,6,7,8,9"],
        device=dev)
    torch.manual_seed(seed)
    f_host = R.train_f_host(host, tok, pairs, dev, dl, steps, 1e-3, seed,
                            hidden=hidden, loss_mode="digkl", dig_idx=dig_idx)
    organ = Organ(name="organ-27b-" + donor_npz.split("/")[-1].split(".")[0],
                  module=f_host)
    gate = WirtGate(host, tok, dev, dig_idx)
    gate.registry.mount(organ, dose=1.0)
    return gate


class WirtGate:
    """Verifier-Gate: rankt Kandidaten mit der gepfropften
    Lese-Positions-Ziffernverteilung (Beam-Scoring, System-2).

    verify(problem, candidates, max_len) -> {"ranking": [...],
    "scores": [...], "logp": [...]} — der beste Kandidat ist ranking[0].
    """

    def __init__(self, host, tok, dev: str = "mps", dig_idx=None,
                 cands: list[str] | None = None):
        self.host = host
        self.tok = tok
        self.dev = dev
        self.cands = cands or [str(c) for c in "0,1,2,3,4,5,6,7,8,9"]
        if dig_idx is None:
            dig_idx = torch.tensor(
                [tok.encode(c, add_special_tokens=False)[0]
                 for c in self.cands], device=dev)
        self.dig_idx = dig_idx
        self.registry = OrganRegistry(host)
        self.hook = R.GraftHook(host)

    # ---- Verifier-Kern (wie verifier_loop.Verifier, aber ueber das
    #      Registry-Organ gemountet statt direktem f) ----
    def _digits(self, prompt: str) -> torch.Tensor:
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.dev)
        rm = torch.zeros(1, ids.shape[1], dtype=torch.bool, device=self.dev)
        rm[0, -1] = True
        # Registry-Hooks feuern (Ablation + gemountete Organe mit Dosis)
        self.registry._rm_cur = rm
        self.registry._sync_hooks()
        with torch.no_grad():
            out = self.host(input_ids=ids)
        self.registry._rm_cur = None
        self.registry._sync_hooks()
        return F.softmax(out.logits[0, -1][self.dig_idx], -1)

    def verify(self, problem: str, candidates: list[str],
               max_len: int = 4, k: int = 3) -> dict:
        """Rankt Kandidaten: p(cand) = prod_p p_graft(digit_p | prefix).

        Kandidaten sind Ziffern-Strings (ggf. unterschiedlicher Laenge);
        die Laenge L wird aus dem Kandidaten uebernommen (Format-Wissen
        des Wirts), das Scoring nutzt den Beam-Pfad.
        """
        scores = {}
        for cand in candidates:
            lp = 0.0
            pref = ""
            for ch in cand:
                if not ch.isdigit():
                    break
                d = self._digits(problem + pref)
                lp += float(torch.log(d[int(ch)] + 1e-9))
                pref += ch
            scores[cand] = lp
        order = sorted(scores, key=lambda c: -scores[c])
        return {"ranking": order, "scores": {c: round(s, 4)
                                             for c, s in scores.items()},
                "logp": {c: round(scores[c], 4) for c in scores}}

    # ---- OrganRegistry-Muster (Snapshot/Digest) ----
    def snapshot(self) -> bytes:
        organ = next(iter(self.registry.organs.values()))
        return organ.snapshot()

    def digest(self) -> str:
        organ = next(iter(self.registry.organs.values()))
        organ.compute_digest()
        return organ.digest


if __name__ == "__main__":
    # Selbsttest: twostep-min, 8 eval-Prompts — die Wahrheit muss in
    # den Top-3 des Rankings landen (Prototyp-Nachweis).
    import sys
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    host = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float32).to(dev).eval()
    npz = sys.argv[1] if len(sys.argv) > 1 else "donor_data/donor_targets_twostep_pos_27b.npz"
    gate = make_gate(host, tok, npz, steps=150, hidden=512, dev=dev)
    print("digest:", gate.digest())
    evals = R.make_tasks(tok, 8, R.SEED + 3, task="twostep-min")
    ok = 0
    for prompt, ans in evals:
        truth = tok.decode([ans]).strip()
        cands = [str(d) for d in range(10)] + [truth]
        res = gate.verify(prompt, cands, max_len=2)
        top3 = res["ranking"][:3]
        hit = truth in top3
        ok += int(hit)
        print(f"  {prompt[:36]!r} truth={truth} top3={top3} hit={hit}")
    print(f"W4-Gate-Selbsttest: {ok}/{len(evals)} Wahrheit in Top-3")
