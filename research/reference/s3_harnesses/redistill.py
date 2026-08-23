"""redistill.py — Organ-Re-Destillation im Wirt-Kontext (S3-Bruecke v2).

cross_graft-Befund (19.8.): lineare Projektionen ueber Modellgrenzen
scheitern kontrolliert (graft 0,016 vs. placebo_graft 0,438). Hier die
Alternative: der SPENDER ist nur noch TEACHER — seine Verteilung am
Lese-Punkt ist das Ziel, die Organ-Funktion wird IM WIRT trainiert:

    f_host(x_wirt)  trainiert mit  KL( p_host@Lese || p_donor@Lese )

  - Spender-Forward einmal (p_donor pro Prompt, frozen).
  - Wirt-Forward mit Ablation (MLP L16-23 @Lese = 0) + Graft-Hook
    (f_host an L23-MLP-Ausgang), nur f_host trainiert (Modell frozen).
  - Keine Raum-Projektion: f_host arbeitet auf der Wirt-Repraesentation.

Kontrollen: intact / ablated / graft(f_host) / placebo (f_host gegen
permutierte Donor-Verteilungen trainiert) / placebo2 (Gewicht-Shuffle).
Falsifikator: graft - placebo < 3x Abstand -> Organ-Verhaltens-Destillation
traegt nicht; dann M6-Hidden-Stitch als letzter Weg.

Nur Rechnung im eigenen Ordner; schreibt JSON nach --out.
"""
from __future__ import annotations

import argparse
import json
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 7
FEWSHOT = "17+25=42\n38+14=52\n"
SENTINEL = "The quick brown fox jumps over the lazy dog. "

# Aufgaben-Registry: (fewshot, generator) — gleiche Sonden wie gap_probe.py
def is_prime(n):
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return n >= 2


TASKS = {
    "add2": ("17+25=42\n38+14=52\n",
             lambda r: (lambda a, b: (f"{a}+{b}=", a + b))(r.randint(12, 88), r.randint(11, 87))),
    "add3": ("123+456=579\n234+567=801\n",
             lambda r: (lambda a, b: (f"{a}+{b}=", a + b))(r.randint(100, 499), r.randint(100, 499))),
    "mul2x1": ("12*3=36\n23*4=92\n",
               lambda r: (lambda a, b: (f"{a}*{b}=", a * b))(r.randint(12, 98), r.randint(2, 9))),
    "mul2x2": ("12*13=156\n23*14=322\n",
               lambda r: (lambda a, b: (f"{a}*{b}=", a * b))(r.randint(12, 49), r.randint(12, 49))),
    "add3t": ("17+25+31=73\n38+14+22=74\n",
              lambda r: (lambda a, b, c: (f"{a}+{b}+{c}=", a + b + c))(
                  r.randint(12, 88), r.randint(11, 87), r.randint(11, 87))),
    "twostep": ("Example: a=3, b=a+2, c=b*2 -> c=10\n"
                "Example: a=5, b=a+1, c=b*3 -> c=18\n",
                lambda r: (lambda a: (f"a={a}, b=a+2, c=b*2 -> c=", str((a + 2) * 2)))(
                    r.randint(1, 9))),
    "wordlen": ("Example: length of 'hello' is 5\n"
                "Example: length of 'world' is 5\n",
                lambda r: (lambda w: (f"length of '{w}' is ", str(len(w))))(
                    "".join(random.Random(r.randrange(999)).choice("abcdefghij")
                            for _ in range(r.randint(3, 8))))),
    # Minimal-Varianten (OHNE Fewshot) — 27B-Basis reagiert auf
    # Pfeil-Fewshot mit Meta-Kommentar (S31-KARTE-Format-Lektion);
    # gemessen (19.8.): Host-Lese-Punkt twostep-min 0,312, wordlen-min
    # 0,010 — die echten Fähigkeits-Lücken am Lese-Punkt.
    "twostep-min": ("",
                    lambda r: (lambda a: (f"a={a}, b=a+2, c=b*2 -> c=",
                                          str((a + 2) * 2)))(r.randint(1, 9))),
    "div3": ("",
             lambda r: (lambda a: (f"is {a} divisible by 3? ",
                                   "yes" if a % 3 == 0 else "no"))(
                 r.randint(2, 999))),
    "prime": ("",
              lambda r: (lambda a: (f"is {a} prime? ", "yes" if is_prime(a)
                                    else "no"))(r.randint(2, 199))),
    # 0/1-Ziffernraum-Varianten — MUSS exakt make_27b_prompts.TASKS matchen:
    "prime01": ("prime(11)=1\nprime(12)=0\nprime(29)=1\nprime(91)=0\n",
                lambda r: (lambda a: (f"prime({a})=", "1" if is_prime(a)
                                      else "0"))(r.randint(2, 199))),
    "div301": ("div3(12)=1\ndiv3(13)=0\ndiv3(81)=1\ndiv3(92)=0\n",
               lambda r: (lambda a: (f"div3({a})=", "1" if a % 3 == 0
                                     else "0"))(r.randint(2, 999))),
    "mod3": ("mod3(12)=0\nmod3(13)=1\nmod3(81)=0\nmod3(92)=2\n",
             lambda r: (lambda a: (f"mod3({a})=", str(a % 3)))(
                 r.randint(2, 999))),
    "wordlen-min": ("",
                    lambda r: (lambda w: (f"length of '{w}' is ", str(len(w))))(
                        "".join(random.Random(r.randrange(999)).choice("abcdefghij")
                                for _ in range(r.randint(3, 8))))),
}


def make_tasks(tok, n, seed, task="add2"):
    fewshot, gen = TASKS[task]
    rng = random.Random(seed)
    tasks = []
    while len(tasks) < n:
        q, ans = gen(rng)
        tasks.append((fewshot + q, tok.encode(str(ans), add_special_tokens=False)[0]))
    return tasks


def encode(tok, tasks, dev):
    enc = tok([p for p, _ in tasks], return_tensors="pt", padding=True)
    ids = enc["input_ids"].to(dev)
    attn = enc["attention_mask"].to(dev)
    ans = torch.tensor([t for _, t in tasks], device=dev)
    return ids, attn, ans, attn.sum(1) - 1


def donor_targets(donor, tok, tasks, dev):
    """p_donor am Lese-Punkt pro Prompt (frozen, einmal berechnet)."""
    ids, attn, ans, last = encode(tok, tasks, dev)
    with torch.no_grad():
        out = donor(input_ids=ids, attention_mask=attn)
    logits = out.logits[torch.arange(len(last)), last]
    return logits.detach(), ans


def mlx_donor_targets(mlx_model, tok, tasks, dev):
    """7B-Donor via MLX: Lese-Positions-Logits (B, V)."""
    import mlx.core as mx
    import numpy as np
    ids, attn, ans, last = encode(tok, tasks, dev)
    lg = mlx_model(mx.array(ids.cpu().tolist()))
    lg = torch.tensor(np.array(lg), device=dev)
    return lg[torch.arange(len(last)), last].detach(), ans


class GraftHook:
    """Ablation (MLP L16-23 @Lese = 0) + f_host an L23-MLP-Ausgang."""

    def __init__(self, model, lo=16, hi=24):
        self.layers = model.model.layers
        self.lo, self.hi = lo, hi
        self.donor_x = {}
        self.f = None
        self.rm = None

    def attach(self, f_host):
        self.f = f_host
        self.hooks = []
        for L in range(self.lo, self.hi):
            self.hooks.append(
                self.layers[L].mlp.register_forward_pre_hook(self.zero_mlp))
        self.hooks.append(
            self.layers[self.hi - 1].mlp.register_forward_hook(self.add_graft))

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.f = None

    def _active(self, x):
        # Hook feuert auch waehrend der Generierung pro Token (B,1,D):
        # nur beim vollen Prompt (erster Schritt) anwenden.
        return self.rm is not None and x.shape[1] == self.rm.shape[1]

    def zero_mlp(self, module, hargs):
        x = hargs[0]
        if self._active(x):
            if module is self.layers[self.lo].mlp:
                self.donor_x["x"] = x[self.rm]   # NICHT detachen (Training!)
            x = x.clone()
            x[self.rm] = 0.0
        return (x,)

    def add_graft(self, module, hargs, hout):
        out = hout.clone()
        if self.f is not None and self._active(out) and self.donor_x:
            out[self.rm] = out[self.rm] + self.f(self.donor_x["x"])
        return out


@torch.no_grad()
def run_eval_gen(model, tok, tasks, dev, mode, hook=None, f_host=None,
                 max_new=8):
    """Voll-Antwort-Eval per Greedy-Generierung (Hooks aktiv)."""
    ok, n = 0, len(tasks)
    for prompt, _ in tasks:
        ans = prompt.rsplit("=", 1)[1] if False else None  # Platzhalter
        break
    # Antworten aus den Tasks extrahieren: (prompt, ans) Format pruefen
    answers = [t[1] if isinstance(t[1], str) else str(t[1]) for t in tasks]
    for i, (prompt, _) in enumerate(tasks):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        if mode != "intact":
            rm = torch.zeros(1, ids.shape[1], dtype=torch.bool, device=dev)
            rm[0, -1] = True
            hook.rm = rm
            hook.attach(f_host)
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        if mode != "intact":
            hook.detach()
            hook.rm = None
        gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        first = gen.split("\n")[0]
        digits = "".join(ch for ch in first if ch.isdigit())
        ok += int(digits == answers[i])
    return ok / n


def run_eval(model, tok, tasks, dev, mode, hook=None, f_host=None,
             donor_dig_eval=None, dig_idx=None, cands=None):
    ids, attn, ans, last = encode(tok, tasks, dev)
    rm = torch.arange(attn.shape[1], device=dev).unsqueeze(0).expand(
        attn.shape[0], attn.shape[1]) == last.unsqueeze(1)
    if mode != "intact":
        hook.rm = rm
        hook.attach(f_host)      # f=None: nur Ablation (add_graft no-op)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=attn)
    if mode != "intact":
        hook.detach()
        hook.rm = None
    rl = out.logits[torch.arange(len(ans)), last]
    p = F.softmax(rl, -1)
    acc = float((rl.argmax(-1) == ans).float().mean())
    cand_acc = None
    if cands is not None and dig_idx is not None:
        # Kandidaten-Raum (A5): argmax ueber die Kandidaten-Ids
        pos = torch.tensor(
            [cands.index(tok.decode([a]).strip()) for a in ans],
            device=dev)
        cl = rl[:, dig_idx]
        cand_acc = float((cl.argmax(-1) == pos).float().mean())
    st = tok(SENTINEL, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        ppl = float(F.cross_entropy(model(input_ids=st).logits[0, :-1], st[0, 1:]))
    kl_dig = None
    if donor_dig_eval is not None and dig_idx is not None:
        # Kalibrierung: KL zwischen Wirt- und Donor-Ziffernverteilung
        # am Lese-Punkt (kleiner = Wirt folgt dem Donor)
        hd = rl[:, dig_idx]
        kl_dig = float(F.kl_div(F.log_softmax(hd / 2.0, -1),
                                F.softmax(donor_dig_eval / 2.0, -1),
                                reduction="batchmean") * 4.0)
    return acc, float(p[torch.arange(len(ans)), ans].mean()), ppl, kl_dig, cand_acc


def train_f_host(host, tok, tasks, dev, donor_logits, steps, lr, seed,
                 permute=False, hidden=512, loss_mode="kl", dig_idx=None,
                 lo=16, hi=24):
    """f_host trainieren (nur f_host). loss_mode:
    kl   — Voll-Vokabular-KL gegen Donor-Verteilung (verduennt das
           Aufgaben-Signal unter dem LM-Prior — gemessen: graft 0,000)
    ce   — Task-fokussiert: CE auf das Antwort-Token (Donor-p als
           weiches Ziel), kein LM-Prior im Loss
    digkl— Ziffern-Raum-KL (27B-Donor, V=248320 != Wirt 151936):
           KL(p_wirt@Lese/Ziffern || p_donor@Lese/Ziffern), tau=2"""
    torch.manual_seed(seed)
    d = host.config.hidden_size
    f = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d)).to(dev)
    opt = torch.optim.AdamW(f.parameters(), lr=lr)
    hook = GraftHook(host, lo=lo, hi=hi)
    ids, attn, ans, last = encode(tok, tasks, dev)
    rm = torch.arange(attn.shape[1], device=dev).unsqueeze(0).expand(
        attn.shape[0], attn.shape[1]) == last.unsqueeze(1)
    if permute:
        donor_logits = donor_logits[torch.randperm(len(donor_logits))]
    B = min(32, len(tasks))
    n = len(tasks)
    for step in range(steps):
        idx = torch.randperm(n)[:B]
        hook.rm = rm[idx]
        hook.attach(f)
        out = host(input_ids=ids[idx], attention_mask=attn[idx])
        hook.detach()
        hook.rm = None
        logits = out.logits[torch.arange(B), last[idx]]
        if loss_mode == "ce":
            # weiches Ziel: Donor-p(ans) als Gewicht, hartes Ziel: ans
            p_d = F.softmax(donor_logits[idx], -1)
            w = p_d[torch.arange(B), ans[idx]].clamp_min(1e-3)
            loss = (F.cross_entropy(logits, ans[idx], reduction="none") * w).mean()
        elif loss_mode == "digkl":
            hd = logits[:, dig_idx]              # (B,10) Wirt-Lese-Ziffern
            dd = donor_logits[idx]               # (B,10) Donor-Ziffern
            loss = F.kl_div(F.log_softmax(hd / 2.0, -1),
                            F.softmax(dd / 2.0, -1),
                            reduction="batchmean") * 4.0
        else:
            loss = F.kl_div(F.log_softmax(logits / 2.0, -1),
                            F.softmax(donor_logits[idx] / 2.0, -1),
                            reduction="batchmean") * 4.0
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"  step {step}: loss={float(loss.detach()):.4f}", flush=True)
    return f


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--donor", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--host", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--n-pairs", type=int, default=256)
    ap.add_argument("--n-eval", type=int, default=96)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--task", default="add2",
                    choices=sorted(TASKS), help="Aufgaben-Sonde (gap_probe)")
    ap.add_argument("--donor-npz", default="",
                    help="27B donor targets (NPZ read-position logits)")
    ap.add_argument("--donor-mlx", default="",
                    help="MLX-Donor (4-bit, z. B. mlx-community/Qwen2.5-7B-Instruct-4bit)")
    ap.add_argument("--lo", type=int, default=16,
                    help="Ablations-Region: erster Layer (Lokalisierungs-"
                         "Befund: kritische Region ~L18-21, L16 schaedlich)")
    ap.add_argument("--hi", type=int, default=24,
                    help="Ablations-Region: exklusives Ende")
    ap.add_argument("--cand", default="0,1,2,3,4,5,6,7,8,9",
                    help="Kandidaten-Strings (A5; semantisch aligniert)")
    ap.add_argument("--loss", default="ce", choices=["kl", "ce", "digkl"],
                    help="ce=Antwort-CE (Standard), kl=Voll-Vokabular-KL, "
                         "digkl=Ziffern-KL (27B-npz)")
    ap.add_argument("--out", default="redistill.json")
    args = ap.parse_args()

    t0 = time.time()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(SEED); random.seed(SEED)
    tok = AutoTokenizer.from_pretrained(args.host)
    host = AutoModelForCausalLM.from_pretrained(args.host, torch_dtype=torch.float32).to(dev).eval()
    # Nur f_host trainiert: Modell komplett einfrieren
    for p in host.parameters():
        p.requires_grad_(False)
    dig_idx = None
    donor_dig_eval = None
    if args.donor_npz:
        import numpy as _np
        data = _np.load(args.donor_npz, allow_pickle=True)
        lg = data["logits"]                       # v3: (B, L, 10) | v2: (B, 10)
        if lg.ndim == 2:
            lg = lg[:, None, :]                   # v2 -> (B, 1, 10)
            mask = _np.ones((lg.shape[0], 1), dtype=bool)
        else:
            mask = data["mask"]
        cands = [c.strip() for c in args.cand.split(",") if c.strip()]
        if lg.shape[-1] != len(cands):
            raise SystemExit(f"npz-Format: erwarte Kandidaten-Logits "
                             f"(…,{len(cands)}), habe {lg.shape}")
        if args.loss != "digkl":
            raise SystemExit("--donor-npz (27B, V=248320) nur mit "
                             "--loss digkl (Ziffern-Raum)")
        pr = list(data["prompts"])
        an = list(data["ans"])
        # erweitertes Trainings-Set: jede echte Antwort-Position ist ein
        # Paar (prompt + Ansatz-Praefix, Donor-Ziffern-Logits an der
        # Lese-Position) — System-2-Modus (v3)
        pairs, rows = [], []
        for i, (p_, a_) in enumerate(zip(pr, an)):
            for p in range(lg.shape[1]):
                if not bool(mask[i, p]):
                    break
                pairs.append((p_ + a_[:p],
                              tok.encode(a_[p], add_special_tokens=False)[0]))
                rows.append(lg[i, p])
        dl = torch.tensor(_np.stack(rows), dtype=torch.float32, device=dev)
        donor_name = args.donor_npz
        # Host-Kandidaten-Ids (semantisch aligniert, eigene Ids je Modell)
        dig_idx = torch.tensor(
            [tok.encode(c, add_special_tokens=False)[0] for c in cands],
            device=dev)
        donor_dig_eval = None
        if "logits_eval" in data:
            le = data["logits_eval"]
            if le.ndim == 3:
                le = le[:, 0, :]                  # v3: nur Lese-Position
            donor_dig_eval = torch.tensor(le, dtype=torch.float32, device=dev)
        if "ids0" in data:
            ids0 = list(data["ids0"])
            h0 = tok.encode(pr[0], add_special_tokens=False)
            print(f"  tokenizer-check 27B==host: {ids0 == h0} "
                  f"(V_donor=248320, V_host={host.config.vocab_size})",
                  flush=True)
        print(f"27B-donor-targets: {dl.shape} ({len(pairs)} paare inkl. "
              f"Antwort-Positionen, Ziffern-Raum, {args.donor_npz})",
              flush=True)
    elif args.donor_mlx:
        from mlx_lm import load as mlx_load
        donor_mlx, _ = mlx_load(args.donor_mlx)
        donor_name = args.donor_mlx
        print(f"host: {args.host} d={host.config.hidden_size}, "
              f"donor(mlx): {args.donor_mlx} ({dev})", flush=True)
        pairs = make_tasks(tok, args.n_pairs, SEED + 1, task=args.task)
        dl, _ = mlx_donor_targets(donor_mlx, tok, pairs, dev)
        print(f"donor-targets: {dl.shape} (frozen)", flush=True)
    else:
        donor = AutoModelForCausalLM.from_pretrained(args.donor,
                                                     torch_dtype=torch.float32).to(dev).eval()
        donor_name = args.donor
        print(f"host: {args.host} d={host.config.hidden_size}, "
              f"donor: {args.donor} d={donor.config.hidden_size} ({dev})",
              flush=True)
        pairs = make_tasks(tok, args.n_pairs, SEED + 1, task=args.task)
        dl, _ = donor_targets(donor, tok, pairs, dev)
        print(f"donor-targets: {dl.shape} (frozen)", flush=True)

    print(f"trainiere f_host ({args.loss}) …", flush=True)
    f_host = train_f_host(host, tok, pairs, dev, dl, args.steps, 1e-3, SEED,
                          hidden=args.hidden, loss_mode=args.loss,
                          dig_idx=dig_idx, lo=args.lo, hi=args.hi)
    print("trainiere placebo (permutierte Donor-Targets) …", flush=True)
    f_pl = train_f_host(host, tok, pairs, dev, dl, args.steps, 1e-3, SEED + 2,
                        permute=True, hidden=args.hidden, loss_mode=args.loss,
                        dig_idx=dig_idx, lo=args.lo, hi=args.hi)
    import copy
    f_sh = copy.deepcopy(f_host)   # NIE f_host in-place shufflen (Bug v1:
    with torch.no_grad():          # der echte Graft wurde nie evaluiert!)
        for p in f_sh.parameters():
            flat = p.flatten()
            perm = torch.randperm(flat.numel(), device=flat.device)
            p.copy_(flat[perm].reshape(p.shape))

    hook = GraftHook(host, lo=args.lo, hi=args.hi)
    evals = make_tasks(tok, args.n_eval, SEED + 3, task=args.task)
    if donor_dig_eval is not None:
        npz_ev = list(data["prompts_eval"])
        int_ev = [p for p, _ in evals]
        if int_ev != npz_ev:
            raise SystemExit("eval-prompt mismatch: redistill intern != npz "
                             "(Seeds/Generator abweichend!)")
        print(f"  eval-set-check: {len(int_ev)} prompts identisch "
              f"(npz-eval-logits {tuple(donor_dig_eval.shape)})", flush=True)
    res = {"donor": donor_name, "host": args.host, "task": args.task, "cfg": {}}
    print("\nkonfigurationen:", flush=True)
    for mode, f in (("intact", None), ("ablated", None), ("graft", f_host),
                    ("placebo", f_pl), ("placebo2", f_sh)):
        acc, p, ppl, kl_dig, cand_acc = run_eval(
            host, tok, evals, dev, mode, hook, f, donor_dig_eval, dig_idx,
            cands if args.donor_npz else None)
        gacc = run_eval_gen(host, tok, evals, dev, mode, hook, f)
        res["cfg"][mode] = {"acc": round(acc, 4), "p": round(p, 4),
                            "ppl": round(ppl, 4), "gen_acc": round(gacc, 4)}
        kl_s = ""
        if kl_dig is not None:
            res["cfg"][mode]["dig_kl"] = round(kl_dig, 4)
            kl_s = f"  dig_kl {kl_dig:.3f}"
        if cand_acc is not None:
            res["cfg"][mode]["cand_acc"] = round(cand_acc, 4)
            kl_s += f"  cand_acc {cand_acc:.3f}"
        print(f"  {mode:8s}: acc {acc:.3f}  p {p:.4f}  gen {gacc:.3f}  "
              f"ppl {ppl:.3f}{kl_s}", flush=True)

    res["meta"] = {"runtime_s": round(time.time() - t0, 1), "device": dev,
                   "steps": args.steps, "hidden": args.hidden}
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"\n-> {args.out} | gesamt {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
