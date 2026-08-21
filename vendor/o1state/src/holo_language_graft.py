#!/usr/bin/env python3 -u
"""
HOLO-LANGUAGE-GRAFT (MS5) — holographic recall leaves the lab, grafted onto the
REAL 400M-token language stream.
=============================================================================
Every prior recall result in this repo (the knee, the carrier, the streaming-gap
survival in holo_stream_recall.py) ran on synthetic MQAR: disjoint integer id
ranges for keys/values/fillers, never real language. MS5 asks the harder
question: can a holographic layer, GRAFTED onto the live 430M-token POS
language model (results/pos_ckpt.pt, arm A3), learn to recall a named fact
("<key> means <value>") stated once in a REAL C4 text stream, after real text
— not synthetic fillers — has flowed through the carried state?

THE GRAFT.
  Load A3 read-only (StreamingNoPELM pattern from pos_sleep.load_snapshot):
  2 frozen StreamingScanLayer blocks (d_model=128, requires_grad=False on
  EVERYTHING old, including the embedding). On top: ONE new third block —
  StreamingHolographicScanLayer (d_model=128, n_heads=4, d_head=32,
  use_phase=True, readout="rms", imported unmodified from
  holo_stream_recall.py so the math is byte-identical to the validated
  streaming-gap layer) plus its own ln1/ln2/ffn (same shapes as the old
  blocks) — and a NEW head (Linear d_model -> V). Only the new block + new
  head are trainable; the old two blocks and the embedding are frozen.
  Forward: frozen block0 -> frozen block1 -> new holo block -> new head.
  State threading: [Z0_frozen, Z1_frozen, holo_state_dict] chunked-carried
  with .detach() at every boundary, exactly streaming_train.py Sec.0 /
  holo_stream_recall.py Sec.3's convention. Equivalence (full-sequence
  forward == chunked+carried forward for the WHOLE 3-block stack) is the
  mandatory gate before any training — everything below is void without it.

THE TASK — a fact riding real text.
  Carrier = real C4 validation-split text (C4ValStream, pos_sleep.py's
  read-only pattern — never touches the live run's training stream). A fact
  sentence is injected into the stream: "<key_i> means <value_j>" where
  key_i/value_j are REAL WikiText-2-vocabulary words at frequency rank
  3000-5000 (rare enough to carry little prior, common enough that the
  embedding is actually trained — see build_vocab: vocab is frequency-sorted,
  so this rank band is a deliberate, principled slice, not an arbitrary one).
  32 keys / 16 values, seed-fixed lists (see FACT_KEY_SEED/FACT_VAL_SEED).
  Per-trial layout: [~16 tokens real C4 text][key "means" value][G tokens
  real C4 text][key "means"?] -> predict value at the LAST token. P=1 fact
  per trial. Loss/eval scored ONLY at the last position, over the 16 value
  words (a 16-way classification read off the shared vocabulary logits).

TRAINING REGIME (the M2/M3 lesson, not the naive one). FULL-SEQUENCE
  training (full graph, no chunk-boundary truncation) with a gap curriculum
  (patience=25, bar=0.8, grows toward G_train=64) — truncated BPTT would
  give the query loss zero gradient into the write the moment G exceeds one
  chunk, exactly the v1 collapse diagnosed in analysis/HOLO_STREAM_VERDICT.md.
  Only EVAL is chunked (chunk=32), because the frozen 2-block underbelly
  makes a chunked forward cheap and it is eval that must match the deployed
  streaming regime. Only the new block backprops, so full-sequence training
  is affordable even though C4 tokenization is the real IO cost.

ARMS (identical graft init/seed):
  (a) holo         — use_phase=True, state carried across the gap.
  (b) ctrl         — use_phase=False (Selective-equivalent ablation), same
                      parameter-count order (no complex projections, but the
                      same magnitude scan + ln/ffn/head skeleton).
  (c) null         — arm (a)'s trained weights, but the NEW holo block's
                      state is ZEROED at the gap boundary (the two FROZEN
                      language blocks' Z keep running through the gap
                      untouched). This isolates whether the fact rides in
                      the new holographic state specifically, not in the
                      frozen language model's own recurrent state.

EVAL: recall accuracy vs G in {8,32,128,256,512}, eval_batch=100, 2 seeds.
Plus a gamma-spectrum of the holo layer on real-text fillers (per-channel
max gamma, gamma_spectrum_fillers idiom adapted from holo_stream_recall's
phi-margin probe / streaming_train's gamma_spectrum).

PREDICTION (P21, registered here): holo-graft clears >= 3x chance
(chance=1/16=0.0625 -> >=0.19) at G=128 on real text; ctrl-graft stays
below that; null decays to chance. Honest reporting either way — a miss is
a data point for the next attack, not a verdict on the project.

CPU-only, threads=1, os.nice(19) best-effort. Live POS files
(results/pos_ckpt.pt) are opened READ-ONLY, never written. All outputs go
to results/holo_graft*.json only.

Usage:
  python src/holo_language_graft.py --smoke   # 1 seed, G<=32, 600 iters, <25min
  python src/holo_language_graft.py --full    # 2 seeds, G up to 512, 2500 iters
"""
import os
import sys
import json
import math
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference"))

try:
    os.nice(19)
except Exception:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.mps.is_available = lambda: False   # force CPU (repo convention)
torch.set_num_threads(1)

from streaming_train import StreamingNoPELM, StreamingScanLayer               # noqa: E402
from holo_stream_recall import StreamingHolographicScanLayer                  # noqa: E402
from length_extrap_v2 import load_wikitext2, build_vocab, tokenize            # noqa: E402
from pos_sleep import C4ValStream                                             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(REPO, "results")
CKPT = os.path.join(RESULTS, "pos_ckpt.pt")

FACT_KEY_SEED = 20260722          # fixed seed for the 32-key rare-word draw
FACT_VAL_SEED = 20260723          # fixed seed for the 16-value rare-word draw
RANK_LO, RANK_HI = 3000, 5000     # frequency-rank band the fact words are drawn from
N_KEYS, N_VALS = 32, 16


# ═══════════════════════════════════════════════════════════════════════════
# 1. Load the frozen A3 snapshot (pos_sleep.load_snapshot pattern) — READ ONLY.
# ═══════════════════════════════════════════════════════════════════════════
def load_frozen_a3(ckpt_path=CKPT):
    """Loads the live POS A3 checkpoint read-only, builds the matching
    StreamingNoPELM, and returns (frozen_model, cfg, vocab, stoi, unk, mask, V).
    Vocab is rebuilt from WT-2 train text exactly as pos_run.py/pos_sleep.py do
    (build_vocab(load_wikitext2()[0])) — deterministic, not stored in the ckpt."""
    ck = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    cfg = ck["config"]
    train_text, _ = load_wikitext2()
    vocab, stoi, unk, mask = build_vocab(train_text)
    V = len(vocab)
    model = StreamingNoPELM(V, mask, d_model=cfg["d_model"], n_layers=2, n_heads=4,
                            d_head=cfg["d_model"] // 4, seq_len=32, dropout=0.0, causal=True)
    model.load_state_dict(ck["arms"]["A3"]["model"])
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model, cfg, vocab, stoi, unk, mask, V, ck["n_streamed"]


# ═══════════════════════════════════════════════════════════════════════════
# 2. The grafted model — 2 frozen blocks + 1 new trainable holographic (or
#    control) block + a new head. State = [Z0, Z1, holo_state_dict].
# ═══════════════════════════════════════════════════════════════════════════
class GraftBlock(nn.Module):
    """The new third block: a scan (holographic or its use_phase=False control)
    + its own ln1/ln2/ffn, same shapes/pattern as StreamingNoPELM's layers."""

    def __init__(self, d_model, n_heads, d_head, use_phase, phase_scale=math.pi,
                 readout="rms"):
        super().__init__()
        self.scan = StreamingHolographicScanLayer(
            d_model, d_head=d_head, n_heads=n_heads, causal=True, dropout=0.0,
            phase_scale=phase_scale, use_phase=use_phase, readout=readout,
            separate_qk=False, n_slots=1)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, state_in=None):
        y, state_out = self.scan(x, state_in)
        h = self.ln1(x + y)
        h = self.ln2(h + self.ffn(h))
        return h, state_out


class HoloGraftLM(nn.Module):
    """embed(frozen) -> block0(frozen) -> block1(frozen) -> GraftBlock(NEW,
    trainable) -> head(NEW, trainable). Only GraftBlock + head carry grad.
    State threading: [Z0, Z1, holo_state] per forward call, chunked-carried
    by the caller exactly as StreamingNoPELM/StreamingHolographicLM do."""

    def __init__(self, frozen_model, vocab_size, d_model, n_heads, d_head,
                 use_phase, readout="rms"):
        super().__init__()
        self.frozen = frozen_model            # embed + 2 frozen StreamingScanLayer blocks
        for p in self.frozen.parameters():
            p.requires_grad = False
        self.graft = GraftBlock(d_model, n_heads, d_head, use_phase, readout=readout)
        self.head = nn.Linear(d_model, vocab_size + 1)   # NEW head, matches frozen.head shape

    def forward(self, x, states=None):
        assert isinstance(self.frozen.pos, nn.Identity), "NoPE required on the frozen underbelly"
        if states is None:
            z0 = z1 = zh = None
        else:
            z0, z1, zh = states
        h = self.frozen.embed(x)              # frozen embedding, NoPE (pos=Identity)
        y0, z0_out = self.frozen.layers[0].scan(h, z0)
        h = self.frozen.layers[0].ln1(h + y0)
        h = self.frozen.layers[0].ln2(h + self.frozen.layers[0].ffn(h))
        y1, z1_out = self.frozen.layers[1].scan(h, z1)
        h = self.frozen.layers[1].ln1(h + y1)
        h = self.frozen.layers[1].ln2(h + self.frozen.layers[1].ffn(h))
        h, zh_out = self.graft(h, zh)
        logits = self.head(h)
        return logits, [z0_out, z1_out, zh_out]

    def zero_holo_state(self, states):
        """Decisive null: zero ONLY the new graft's holo state; the two FROZEN
        language blocks' Z keep whatever they carried (documented design —
        isolates whether the FACT rides in the new holo state, not in the
        frozen language model's own recurrent state)."""
        z0, z1, zh = states
        zh_zeroed = {k: torch.zeros_like(v) for k, v in zh.items()}
        return [z0, z1, zh_zeroed]

    def trainable_parameters(self):
        return list(self.graft.parameters()) + list(self.head.parameters())


def build_graft(frozen_model, V, d_model, n_heads, d_head, use_phase, seed):
    torch.manual_seed(seed)
    return HoloGraftLM(frozen_model, V, d_model, n_heads, d_head, use_phase)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Equivalence — full-sequence forward (states=None) == chunked+carried
#    forward, for the WHOLE 3-block stack. HIGHEST PRIORITY, mandatory gate.
# ═══════════════════════════════════════════════════════════════════════════
def chunked_forward(model, x, chunk):
    B, T = x.shape
    states = None
    outs = []
    pos = 0
    while pos < T:
        hi = min(T, pos + chunk)
        logits_c, states = model(x[:, pos:hi], states)
        z0, z1, zh = states
        states = [z0.detach(), z1.detach(), {k: v.detach() for k, v in zh.items()}]
        outs.append(logits_c)
        pos = hi
    return torch.cat(outs, dim=1), states


@torch.no_grad()
def check_equivalence(model, V, T=48, chunk=16, seed=0):
    torch.manual_seed(seed)
    x = torch.randint(0, V, (2, T))
    model.eval()
    out_full, _ = model(x, None)
    out_chunked, _ = chunked_forward(model, x, chunk)
    return float((out_full - out_chunked).abs().max())


# ═══════════════════════════════════════════════════════════════════════════
# 4. Fact vocabulary — 32 keys / 16 values, real WT-2 words at rank
#    [RANK_LO, RANK_HI), seed-fixed. "means" token id looked up from stoi.
# ═══════════════════════════════════════════════════════════════════════════
def build_fact_vocab(vocab, stoi):
    band = vocab[RANK_LO:RANK_HI]
    assert len(band) >= N_KEYS + N_VALS, f"rank band too small: {len(band)}"
    g_k = torch.Generator().manual_seed(FACT_KEY_SEED)
    g_v = torch.Generator().manual_seed(FACT_VAL_SEED)
    idx_k = torch.randperm(len(band), generator=g_k)[:N_KEYS].tolist()
    remaining = [i for i in range(len(band)) if i not in set(idx_k)]
    perm_v = torch.randperm(len(remaining), generator=g_v)[:N_VALS].tolist()
    idx_v = [remaining[i] for i in perm_v]
    key_words = [band[i] for i in idx_k]
    val_words = [band[i] for i in idx_v]
    key_ids = torch.tensor([stoi[w] for w in key_words], dtype=torch.long)
    val_ids = torch.tensor([stoi[w] for w in val_words], dtype=torch.long)
    means_id = stoi["means"]
    return key_words, val_words, key_ids, val_ids, means_id


# ═══════════════════════════════════════════════════════════════════════════
# 5. Real-text carrier — C4 validation stream, tokenized with the SAME
#    stoi/unk as the frozen model (pos_sleep.C4ValStream, read-only).
# ═══════════════════════════════════════════════════════════════════════════
class FactStreamBatcher:
    """Produces (x, y_target, target_pos) trials: [~pre tokens C4][key means
    value][G tokens C4][key] -> predict value at the LAST token. G is drawn
    per-trial in the batch from the current curriculum gap (fixed for a
    given call, varied by the caller across calls)."""

    def __init__(self, stoi, unk, key_ids, val_ids, means_id, seed, split="validation",
                 pre_len=16, skip_tokens=0):
        self.key_ids, self.val_ids, self.means_id = key_ids, val_ids, means_id
        self.pre_len = pre_len
        self.gen = torch.Generator().manual_seed(seed)
        self.c4 = C4ValStream(stoi, unk, split=split)
        if skip_tokens:
            self.c4.next_block(skip_tokens)

    def make_batch(self, B, G):
        P = self.pre_len
        trial_len = P + 3 + G + 1   # pre + [key,means,value] + G fillers + query-key
        xs = torch.empty(B, trial_len, dtype=torch.long)
        y = torch.empty(B, dtype=torch.long)
        n_keys, n_vals = len(self.key_ids), len(self.val_ids)
        ki = torch.randint(0, n_keys, (B,), generator=self.gen)
        vi = torch.randint(0, n_vals, (B,), generator=self.gen)
        for b in range(B):
            pre = self.c4.next_block(P)
            fillers = self.c4.next_block(G)
            k_id = int(self.key_ids[ki[b]])
            v_id = int(self.val_ids[vi[b]])
            seq = pre + [k_id, self.means_id, v_id] + fillers + [k_id]
            xs[b] = torch.tensor(seq, dtype=torch.long)
            y[b] = v_id
        return xs, y, vi   # vi = target value INDEX in [0, N_VALS)


def eval_target_logits(logits, val_ids):
    """Restrict the shared-vocab logits to the N_VALS fact-value columns only —
    the 16-way classification read-off described in the module docstring."""
    return logits[:, -1, :].index_select(1, val_ids)   # (B, N_VALS)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Training — FULL-SEQUENCE (full graph), gap curriculum patience=25 bar=0.8
#    up to G_train=64. Only graft.parameters()+head.parameters() get grad.
# ═══════════════════════════════════════════════════════════════════════════
def train_gap_curriculum(model, batcher, iters, lr, batch, g_start, g_train_max,
                         patience=25, bar=0.8, log_every=0):
    opt = torch.optim.Adam(model.trainable_parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    model.train()
    # keep the frozen underbelly in eval() (dropout=0 everywhere anyway, but explicit)
    model.frozen.eval()
    Gcur = min(g_start, g_train_max) if g_train_max > 0 else 0
    acc = 0.0
    good = 0
    for it in range(iters):
        x, y_idx, val_target_idx = batcher.make_batch(batch, Gcur)
        logits, _ = model(x, None)     # FULL-SEQUENCE forward, full graph
        pred = eval_target_logits(logits, model.graft_val_ids)
        loss = lossf(pred, val_target_idx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 5.0)
        opt.step()
        acc = float((pred.argmax(-1) == val_target_idx).float().mean())
        good = good + 1 if acc > bar else 0
        if good >= patience and Gcur < g_train_max:
            Gcur = min(g_train_max, int(Gcur * 1.5) + 1)
            good = 0
        if log_every and (it + 1) % log_every == 0:
            print(f"    it {it+1:>4}/{iters}: loss {float(loss):.3f} acc {acc:.3f} (train-gap {Gcur})")
    return {"final_train_gap": Gcur, "final_train_acc": round(acc, 4)}


# ═══════════════════════════════════════════════════════════════════════════
# 7. Eval — CHUNKED forward (chunk=32), accuracy vs G, incl. zeroed-at-gap null.
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def eval_gap_recall(model, batcher, G, NB, chunk, val_ids, zero_at_gap=False, pre_len=16):
    model.eval()
    x, _, val_target_idx = batcher.make_batch(NB, G)
    kv_len = pre_len + 3   # pre + key + means + value

    if not zero_at_gap:
        logits, _ = chunked_forward(model, x, chunk)
    else:
        logits_kv, states = chunked_forward(model, x[:, :kv_len], chunk)
        states = model.zero_holo_state(states)
        logits_rest, states = chunked_forward_from_state(model, x[:, kv_len:], chunk, states)
        logits = torch.cat([logits_kv, logits_rest], dim=1)

    pred = eval_target_logits(logits, val_ids)
    acc = float((pred.argmax(-1) == val_target_idx).float().mean())
    return acc


def chunked_forward_from_state(model, x, chunk, states):
    B, T = x.shape
    outs = []
    pos = 0
    while pos < T:
        hi = min(T, pos + chunk)
        logits_c, states = model(x[:, pos:hi], states)
        z0, z1, zh = states
        states = [z0.detach(), z1.detach(), {k: v.detach() for k, v in zh.items()}]
        outs.append(logits_c)
        pos = hi
    return torch.cat(outs, dim=1), states


# ═══════════════════════════════════════════════════════════════════════════
# 8. gamma-spectrum of the holo layer on real-text fillers (per-channel max),
#    adapted from streaming_train.gamma_spectrum / holo_stream_recall's probe.
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def gamma_spectrum_fillers(model, stoi, unk, n_tok=2000, seed=999):
    c4 = C4ValStream(stoi, unk, split="validation")
    ids = c4.next_block(n_tok)
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    h = model.frozen.embed(x)
    y0, _ = model.frozen.layers[0].scan(h, None)
    h = model.frozen.layers[0].ln1(h + y0)
    h = model.frozen.layers[0].ln2(h + model.frozen.layers[0].ffn(h))
    y1, _ = model.frozen.layers[1].scan(h, None)
    h = model.frozen.layers[1].ln1(h + y1)
    h = model.frozen.layers[1].ln2(h + model.frozen.layers[1].ffn(h))
    if not model.graft.scan.use_phase:
        return {"use_phase": False, "note": "control arm has no gamma-conditioned complex write"}
    a, gamma = model.graft.scan._drive_and_gamma(h)   # (1,T,H,D)
    per_channel_max = gamma.amax(dim=(0, 1)).cpu().numpy()   # (H,D)
    per_channel_mean = gamma.mean(dim=(0, 1)).cpu().numpy()
    return {
        "use_phase": True,
        "per_head_max_over_channels": [round(float(per_channel_max[h_].max()), 4)
                                       for h_ in range(per_channel_max.shape[0])],
        "per_head_mean_over_channels": [round(float(per_channel_mean[h_].mean()), 4)
                                        for h_ in range(per_channel_mean.shape[0])],
        "global_max_gamma": round(float(per_channel_max.max()), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 9. Orchestration
# ═══════════════════════════════════════════════════════════════════════════
def run(args):
    t0 = time.time()
    print("=" * 78)
    print("HOLO-LANGUAGE-GRAFT (MS5) — key-conditioned recall grafted onto the "
          "real 400M-token POS language stream")
    print("=" * 78)

    print("\n── loading frozen A3 snapshot (read-only) ──")
    frozen_model, cfg, vocab, stoi, unk, mask, V, n_streamed = load_frozen_a3()
    print(f"   A3: n_streamed={n_streamed:,}  d_model={cfg['d_model']}  V={V}")

    key_words, val_words, key_ids, val_ids, means_id = build_fact_vocab(vocab, stoi)
    print(f"   fact vocab: {N_KEYS} keys / {N_VALS} values, rank band "
          f"[{RANK_LO},{RANK_HI}) — e.g. keys[:5]={key_words[:5]} vals[:5]={val_words[:5]}")
    chance = 1.0 / N_VALS

    d_model = cfg["d_model"]
    n_heads, d_head = 4, d_model // 4

    # ── mandatory equivalence gate (whole 3-block stack) ──
    print("\n── equivalence: full-sequence forward == chunked+carried forward (3-block stack) ──")
    probe_model = build_graft(frozen_model, V, d_model, n_heads, d_head, use_phase=True, seed=0)
    eq_delta = check_equivalence(probe_model, V, T=48, chunk=16, seed=0)
    print(f"   max|Δ| = {eq_delta:.3e}")
    eq_ok = eq_delta < 1e-4
    print(f"   {'PASS — chunked carry is the SAME operator' if eq_ok else 'FAIL — do not trust anything below'}")
    if not eq_ok:
        out = {"config": vars(args), "equivalence": {"max_abs_delta": eq_delta, "passed": False},
               "verdict": "VOID — equivalence check failed"}
        os.makedirs(RESULTS, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n→ {args.out}")
        return

    Gs = [int(g) for g in args.gaps.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"\nGaps(G)={Gs}  seeds={seeds}  chunk={args.chunk}  g_train_max={args.g_train_max}")

    arms = {"holo": True, "ctrl": False}
    results = {"config": vars(args),
               "fact_vocab": {"key_words": key_words, "val_words": val_words,
                             "rank_band": [RANK_LO, RANK_HI], "n_keys": N_KEYS, "n_vals": N_VALS},
               "equivalence": {"max_abs_delta": eq_delta, "passed": True},
               "chance": chance, "sweep": {}, "curriculum": {}, "gamma_spectrum": {}}

    for seed in seeds:
        print(f"\n{'='*78}\nseed={seed}\n{'='*78}")
        trained = {}
        for arm_name, use_phase in arms.items():
            model = build_graft(frozen_model, V, d_model, n_heads, d_head,
                                use_phase=use_phase, seed=seed)
            model.graft_val_ids = val_ids   # stashed for the target-logit readoff
            train_batcher = FactStreamBatcher(stoi, unk, key_ids, val_ids, means_id,
                                              seed=seed, split="validation", pre_len=16)
            curr = train_gap_curriculum(
                model, train_batcher, args.iters, args.lr, args.batch,
                g_start=args.g_start, g_train_max=args.g_train_max,
                patience=args.patience, bar=args.bar, log_every=args.log_every)
            key = f"seed{seed}_{arm_name}"
            results["curriculum"][key] = curr
            print(f"  [{arm_name:6s}] curriculum done: final_train_gap={curr['final_train_gap']} "
                  f"final_train_acc={curr['final_train_acc']:.3f}")
            trained[arm_name] = model

            if arm_name == "holo":
                gs = gamma_spectrum_fillers(model, stoi, unk)
                results["gamma_spectrum"][f"seed{seed}"] = gs
                print(f"    gamma-spectrum (real-text fillers): {gs}")

        # fresh eval-only C4 stream slice per (seed,G) to avoid overlap with training draws
        for G in Gs:
            for arm_name, model in trained.items():
                model.graft_val_ids = val_ids
                t_b = time.time()
                eval_batcher = FactStreamBatcher(stoi, unk, key_ids, val_ids, means_id,
                                                 seed=seed + 5000 + G, split="validation",
                                                 pre_len=16, skip_tokens=args.eval_skip)
                print(f"    [eval  ] batcher ready ({arm_name} G={G}, skip={args.eval_skip:,}, "
                      f"{time.time()-t_b:.0f}s)", flush=True)
                acc = eval_gap_recall(model, eval_batcher, G, args.eval_batch, args.chunk,
                                      val_ids, zero_at_gap=False)
                sk = f"{arm_name}|G{G}|seed{seed}"
                results["sweep"][sk] = {"seed": seed, "arm": arm_name, "G": G,
                                        "accuracy": round(acc, 4), "chance": round(chance, 4),
                                        "beats_chance_3x": bool(acc > 3 * chance)}
                print(f"    [{arm_name:6s}] G={G:>4}: acc={acc:.4f} (chance {chance:.4f}, "
                      f"3x-chance {'YES' if acc > 3*chance else 'no'})")

            # zeroed-at-gap null, holo arm only
            t_b = time.time()
            null_batcher = FactStreamBatcher(stoi, unk, key_ids, val_ids, means_id,
                                             seed=seed + 9000 + G, split="validation",
                                             pre_len=16, skip_tokens=2 * args.eval_skip)
            print(f"    [eval  ] null batcher ready (G={G}, skip={2*args.eval_skip:,}, "
                  f"{time.time()-t_b:.0f}s)", flush=True)
            holo_model = trained["holo"]
            holo_model.graft_val_ids = val_ids
            acc_null = eval_gap_recall(holo_model, null_batcher, G, args.eval_batch, args.chunk,
                                       val_ids, zero_at_gap=True)
            sk = f"null|G{G}|seed{seed}"
            results["sweep"][sk] = {"seed": seed, "arm": "null", "G": G,
                                    "accuracy": round(acc_null, 4), "chance": round(chance, 4),
                                    "beats_chance_3x": bool(acc_null > 3 * chance)}
            print(f"    [null  ] G={G:>4}: acc={acc_null:.4f} (chance {chance:.4f}, "
                  f"3x-chance {'YES' if acc_null > 3*chance else 'no'})")

    # ── verdict ──
    def _mean_at(arm, G):
        vals = [v["accuracy"] for k, v in results["sweep"].items()
                if v["arm"] == arm and v["G"] == G]
        return sum(vals) / len(vals) if vals else None

    lines = []
    p21_g128_holo = _mean_at("holo", 128)
    for G in Gs:
        holo = _mean_at("holo", G)
        ctrl = _mean_at("ctrl", G)
        null = _mean_at("null", G)
        if holo is None:
            continue
        survives = holo > 3 * chance and (null is None or holo > null + 0.10)
        lines.append(f"G={G}: holo={holo:.3f} ctrl={ctrl:.3f} null={null:.3f} "
                     f"chance={chance:.3f} -> {'SURVIVES' if survives else 'no'}")

    any_survives = any("SURVIVES" in ln for ln in lines)
    p21_pass = p21_g128_holo is not None and p21_g128_holo >= 3 * chance
    verdict = (
        f"language-stream keyed recall SURVIVES real text at at least one G "
        f"(holo beats zeroed-at-gap null and clears 3x chance) — see per-G detail"
        if any_survives else
        f"NEGATIVE at every G probed — holographic graft did not clear the "
        f"zeroed-at-gap null / 3x-chance bar on real language-stream fillers; "
        f"see 'detail' for the per-cell numbers"
    )
    results["verdict"] = verdict
    results["detail"] = lines
    results["p21_prediction"] = {
        "statement": "holo-graft >= 3x chance (>=0.1875) at G=128 on real text; "
                    "ctrl-graft below that; null at chance",
        "holo_acc_at_g128_mean": round(p21_g128_holo, 4) if p21_g128_holo is not None else None,
        "threshold": round(3 * chance, 4),
        "passed": p21_pass,
    }
    results["elapsed_s"] = round(time.time() - t0, 1)

    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "=" * 78)
    print("VERDICT")
    for ln in lines:
        print("  " + ln)
    print(f">>> {verdict}")
    print(f">>> P21 (holo>=3x chance at G=128): "
          f"{'PASS' if p21_pass else 'MISS'} (holo={p21_g128_holo})")
    print(f"\n→ {args.out}  ({results['elapsed_s']}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 seed, G<=32, 600 iters (<25min)")
    ap.add_argument("--full", action="store_true", help="2 seeds, G up to 512, 2500 iters")
    ap.add_argument("--gaps", default="8,32", help="comma list of gap lengths G to sweep")
    ap.add_argument("--seeds", default="0", help="comma list of seeds")
    ap.add_argument("--chunk", type=int, default=32, help="EVAL streaming chunk length")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval-batch", type=int, default=100)
    ap.add_argument("--g-start", type=int, default=2, help="curriculum starting gap")
    ap.add_argument("--g-train-max", type=int, default=64, help="max TRAINING gap (full-sequence)")
    ap.add_argument("--patience", type=int, default=25,
                    help="consecutive acc>bar iters required before curriculum grows the gap")
    ap.add_argument("--bar", type=float, default=0.8, help="curriculum accuracy bar")
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--eval-skip", type=int, default=200_000,
                    help="C4-val tokens to skip before drawing eval episodes (contamination guard; "
                         "the null batcher skips 2x this). Streaming-tokenizing this skip is the "
                         "dominant eval cost -- the smoke uses 20k.")
    ap.add_argument("--out", default=os.path.join(RESULTS, "holo_graft.json"))
    args = ap.parse_args()

    if args.full:
        args.gaps = "8,32,128,256,512"
        args.seeds = "0,1"
        args.iters = max(args.iters, 2500)
        if args.out == os.path.join(RESULTS, "holo_graft.json"):
            pass  # keep default full-run filename
    elif args.smoke or not args.full:
        args.gaps = args.gaps if args.gaps != "8,32" else "8,32"
        args.seeds = "0"
        args.iters = min(args.iters, 600)
        if args.out == os.path.join(RESULTS, "holo_graft.json"):
            args.out = os.path.join(RESULTS, "holo_graft_smoke.json")
        if args.eval_skip == 200_000:
            args.eval_skip = 20_000

    run(args)


if __name__ == "__main__":
    main()
