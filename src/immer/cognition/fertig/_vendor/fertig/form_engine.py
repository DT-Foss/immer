"""
fertig.form_engine — HSSLM-C als Form-Engine für das Utterance-IR.

MOONSHOOT B: flüssige Prosa, die gegen den Plan verifiziert wird.

  Plan (belegte Kanten) -> HSSLM generiert Satz-Varianten (flüssige Form)
  -> Utterance-IR verifiziert jede Variante gegen den Plan (Beleg)
  -> nur belegte Varianten werden freigegeben — nie "hoffen"

HSSLM-C (6.3M, Mamba/S6-Familie — F5-Operator-Klasse) liefert die
Oberflächen-Flüssigkeit; FERTIG liefert den Plan und die Verifikation.
Das Modell wird auf dem Formen-Korpus trainiert (verbalisierte Kausal-
Prosa + Faraday) — scripts/train_form.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "hsslm_form.pt"


class FormEngine:
    """HSSLM-C-Wrapper: generiert Satz-Varianten, deterministisch seedbar.

    Tokenizer: BPE (data/hsslm_bpe.json) — identisch mit dem Training
    (train_form_bpe.py). Ohne BPE-Datei: HierarchicalTokenizer-Fallback.
    """

    def __init__(self, model_path: str = str(_MODEL_PATH),
                 bpe_path: str = ""):
        import json
        import torch
        from fertig.hsslm.model import HSSLMC
        bpe_path = bpe_path or str(Path(__file__).resolve().parent.parent /
                                   "data" / "hsslm_bpe.json")
        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tk = None
        self.bpe = None
        if Path(bpe_path).exists():
            from fertig.hsslm.bpe import BPETokenizer
            data = json.loads(Path(bpe_path).read_text())
            self.bpe = BPETokenizer(400)
            self.bpe.merges = [tuple(m) for m in data["merges"]]
            self.bpe.vocab = {str(k): int(v) for k, v in
                              data["vocab"].items()}
            self.bpe.itos = {int(k): v for k, v in data["itos"].items()}
            self.bpe._char_vocab = {c: i for i, c in
                                    enumerate(sorted(set(
                                        "".join(self.bpe.itos.values()))))}
            vocab_size = self.bpe.VOCAB_SIZE
        else:
            from fertig.hsslm.tokenizer import HierarchicalTokenizer
            self.tk = HierarchicalTokenizer()
            vocab_size = self.tk.VOCAB_SIZE
        self.model = HSSLMC(config={
            "d_model": 256, "vocab_size": vocab_size,
            "n_layers": 4, "hierarchical": False,
        })
        self.ready = False
        if Path(model_path).exists():
            # Auf CPU laden, DANN aufs Geraet: ein in-place copy_ von CPU-
            # nach MPS-Tensoren (load_state_dict auf .to(device)-Modell)
            # erzeugt MPS-Placeholder-Storage (korrupte Gewichte).
            state = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(state)
            self.model.to(self.device)
            self.model.eval()
            self.ready = True

    def _encode(self, text: str):
        if self.bpe is not None:
            return self.torch.tensor([self.bpe.encode(text)],
                                     dtype=self.torch.long)
        return self.tk.encode(text, add_bos=True, add_eos=False)[
            "input_ids"].unsqueeze(0)

    def _decode(self, ids):
        if self.bpe is not None:
            return self.bpe.decode(ids.tolist())
        return self.tk.decode(ids)

    def variants(self, prompt: str, n: int = 3,
                 max_new_tokens: int = 60) -> List[str]:
        """n Form-Varianten zu einem Prompt generieren (flüssig, variabel).

        VARIANZ: jedes n bekommt ein anderes τ (0.45/0.6/0.75) — das
        Modell ist hochkonfident (PPL 5.4), identische Seeds erzeugen
        fast identische Texte. τ-Staffel erzeugt echte Form-Varianten,
        die IR-Verifikation wählt die belegten.
        """
        if not self.ready:
            return []
        import torch
        taus = [0.45, 0.60, 0.75, 0.55, 0.70, 0.50, 0.65, 0.80]
        outs = []
        for i in range(n):
            torch.manual_seed(i)
            pids = self._encode(prompt).to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    pids, max_new_tokens=max_new_tokens,
                    tau=taus[i % len(taus)],
                    use_zeno=False, use_foss_gate=False)
            outs.append(self._decode(out[0].cpu()))
        return outs


def speak_with_engine(graph_path: str, entity: str, engine: Optional[
        FormEngine] = None, n: int = 3) -> dict:
    """Der Kreis mit HSSLM: Plan -> HSSLM-Varianten -> IR-Verifikation.

    Liefert nur freigegebene Varianten — die flüssigste belegte Aussage.
    Ohne trainierte Gewichte: deterministischer Fallback (verbalize).
    """
    from .utterance import speak, plan_from_graph
    from .pipeline import load_graph, verbalize, walk_chain
    from . import state_init

    res = speak(graph_path, entity, n=5)
    if not res.all_verified:
        return {"ok": False, "reason": "Plan-Verifikation fehlgeschlagen"}

    plan = plan_from_graph(graph_path, entity, n=5)
    if engine is None or not engine.ready:
        return {"ok": True, "engine": "fallback",
                "prose": res.prose, "verified": res.all_verified,
                "variants": [res.prose]}

    # Prosa-Kette als Prompt-Basis
    vocab, stoi, adj, mech = load_graph(graph_path)
    SM = state_init.initialize_symbol_state(len(vocab))
    hops = []
    cur = stoi.get(entity.lower())
    seen = set()
    for _ in range(5):
        if cur is None or cur in seen:
            break
        seen.add(cur)
        nbrs = adj.get(cur, {})
        if not nbrs:
            break
        nxt = max(nbrs, key=nbrs.get)
        hops.append((cur, nxt))
        cur = nxt
    base = verbalize(hops, vocab, mech, seed=0)
    prompt = base.split(".")[0] + "."

    variants = engine.variants(prompt, n=n)
    # IR-Verifikation jeder Variante: der Plan muss belegt bleiben
    verified = []
    for v in variants:
        ok = all(u.obj.lower() in v.lower() for u in res.utterances)
        verified.append({"text": v, "verified": ok})
    freigegeben = [v["text"] for v in verified if v["verified"]]
    return {"ok": True, "engine": "hsslm",
            "prompt": prompt,
            "variants": verified,
            "freigegeben": freigegeben,
            "fallback": res.prose}
