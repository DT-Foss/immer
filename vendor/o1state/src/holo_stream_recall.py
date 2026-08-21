#!/usr/bin/env python3 -u
"""
HOLO-STREAM-RECALL — holographic key-conditioned recall on the LIVE streaming state.
======================================================================================
Marries two previously separate lines of this repo for the first time:

  (a) the holographic key-conditioned COMPLEX write (src/holographic_gssm.py) —
      per-channel accumulator S_t = gamma_t * S_{t-1} + u_t * e^{i*phi_t}, read by
      de-rotating with a query's own key angle: read = Re(S * e^{-i*phi_q}).
      MQAR headline: holo_on cleared holo_off (Selective) by a wide margin at n_pairs=8
      canonical layout (src/results/holographic_mqar.json); at n_pairs=2, 25.8% was
      measured in exploration (analysis/RECALL_DEADENDS_LOG.md) — NOT treated as a hard
      ceiling here, per instruction: those numbers are one day of one agent's exploration.

  (b) O(1) streaming PERSISTENCE with detach-carried state across chunk boundaries
      (src/streaming_train.py Sec. 0/A/D) — a stateful scan that accepts Z_in, returns
      Z_out, and truncates the backward graph at chunk edges (near-exact vs full BPTT,
      measured via grad-cosine, not assumed).

QUESTION: does content-addressable (key -> value) recall survive an input GAP that
crosses real chunk boundaries in the CARRIED complex state? This is a new regime vs
both source lines: MQAR was never run chunked-with-carry, and idle-persistence was
never tested on a KEYED multi-pair memory (only a 1-bit 2-way beacon).

WHAT'S BUILT (exactly following the streaming_train.py Sec.0 pattern):
  1. StreamingHolographicScanLayer(HolographicScanLayer) — same math, but S_re/S_im
     (the complex accumulator) are threaded as state_in/state_out instead of always
     starting from zero. n_slots>1 folds slots into the head dim exactly as the parent
     does; the carried state is simply whatever shape the parent's scan produces.
  2. StreamingHolographicLM(HolographicLM) — NoPE (self.pos -> nn.Identity, same
     ablation StreamingNoPELM performs on SelectiveNoPETransformerLM), threads a
     per-layer list[state] through forward.
  3. check_equivalence — full-sequence forward (state=None) must equal chunked forward
     (chunks of 16, detach-carried) bit-for-bit. HIGHEST PRIORITY gate; everything else
     is void without it.
  4. MQAR-across-gap task: [k1,v1,...,kP,vP] (write phase) + [G filler tokens] +
     [query key kq] -> predict the paired value at the last token. Disjoint id ranges
     for keys/values/fillers (mirrors _make_beacon_batch's disjoint-range idiom and the
     mqar.py canonical layout). Gap-curriculum training (start G small, grow on
     acc > 0.9), the onset lesson from run_idle_persistence: from-scratch at a large
     gap never ignites.
  5. Three arms, same init/seed: (a) holo ON, state carried; (b) holo ON, state ZEROED
     at gap onset (decisive null, exactly run_idle_persistence's "zeroed_at_gap"); (c)
     holo OFF (use_phase=False, the Selective ablation baseline).
  6. Accuracy vs gap G in {0,8,32,128} x n_pairs P in {1,2,4}, chance = 1/V_max, plus a
     phi-carrier probe: at the query position, Re(S . e^{-i*phi_matched}) vs the mean
     over mismatched keys, as a function of G (does the PHASE survive the gap, the
     holographic analogue of streaming_train.py Sec. E's gamma-carrier analysis).

CPU-only (mps disabled), single-thread, torch/numpy/stdlib only. Results ->
results/holo_stream_recall*.json only.
"""
import os
import sys
import json
import math
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference"))

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.mps.is_available = lambda: False   # force CPU (repo convention)
torch.set_num_threads(1)

from holographic_gssm import HolographicScanLayer, HolographicLM, sequential_linear_scan  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(REPO, "results")


# ═══════════════════════════════════════════════════════════════════════════
# 1. StreamingHolographicScanLayer — state_in/state_out, exact HolographicScanLayer math.
# ═══════════════════════════════════════════════════════════════════════════
def stateful_linear_scan(a: torch.Tensor, gamma: torch.Tensor, Z0):
    """z_t = gamma_t*z_{t-1} + a_t, carried. a,gamma:(B,T,H,D); Z0:(B,H,D)|None.
    Identical recurrence to holographic_gssm.sequential_linear_scan when Z0 is None
    (zeros) — this is the SAME one load-bearing edit streaming_train.py Sec.0 makes to
    the Selective scan, applied here to the two real scans that make up the complex write."""
    B, T, H, D = a.shape
    Z = torch.zeros(B, H, D, device=a.device, dtype=a.dtype) if Z0 is None else Z0
    out = []
    for t in range(T):
        Z = gamma[:, t] * Z + a[:, t]
        out.append(Z)
    return torch.stack(out, dim=1), Z


class StreamingHolographicScanLayer(HolographicScanLayer):
    """Causal-only streaming forward for the holographic layer.

    RECURRENT STATE CARRIED ACROSS CHUNKS (design decision, read from the parent's
    forward() line by line):
      - use_phase=False (ablation / Selective-equivalent path): the ONLY recurrent
        state is the magnitude scan's accumulator Z_mag (B,H,D) — the same real leaky
        scan Selective/StreamingScanLayer carries. No complex state exists on this path.
      - use_phase=True, n_slots==1: the recurrent state is the complex accumulator,
        carried as a PAIR of real tensors (S_re, S_im), each (B,H,D) — exactly the two
        real leaky scans the class docstring says the complex write decomposes into.
        The magnitude scan Z_mag is ALSO computed on this path only when
        readout=="tanh_m" (it gates the read by m = sqrt(1-exp(Z_mag))); for the
        default readout="rms" (chosen here so the streaming LM avoids a redundant
        state channel) Z_mag is never touched, so it need not be carried.
      - n_slots>1: S_re/S_im simply carry the slot-folded shape (B, H*n_slots, D) that
        the parent's own scan already produces (slots live in the head axis, so the
        carry code below is agnostic to n_slots — it only ever sees "the head axis
        the parent's scan iterates over").
    We therefore carry a single state OBJECT per layer: a dict with keys "S_re",
    "S_im" (present iff use_phase and readout influences them) and "Z_mag" (present
    iff that scan is actually consumed downstream, i.e. use_phase=False or
    readout=="tanh_m"). Anything not carried is simply absent from the dict so no
    dead state ever crosses a chunk boundary.
    """

    def forward(self, x: torch.Tensor, state_in=None, return_internals: bool = False):
        B, T, _ = x.shape
        state_in = state_in or {}

        if not self.use_phase:
            # Ablation path: exact GSSM-Selective magnitude recurrence, carried.
            a, gamma = self._drive_and_gamma(x)
            if self.causal:
                Z_seq, Z_fin = stateful_linear_scan(a, gamma, state_in.get("Z_mag"))
            else:
                # non-causal streaming makes no sense (no well-defined chunk boundary);
                # the task harness below always uses causal=True.
                Z_seq, Z_fin = stateful_linear_scan(a, gamma, state_in.get("Z_mag"))
            s_sq = torch.clamp(1.0 - torch.exp(Z_seq), min=0.0)
            m = torch.sqrt(s_sq + self._eps())
            out = self.W_out(m.view(B, T, self.n_heads * self.d_head))
            state_out = {"Z_mag": Z_fin}
            if return_internals:
                return out, state_out, {"gamma": gamma, "a": a, "Z_seq": Z_seq}
            return out, state_out

        # ── key-conditioned holographic write, carried ──
        a, gamma = self._drive_and_gamma(x)
        phi_w = self.phase_scale * torch.tanh(self.W_key(x))
        phi_w = phi_w.view(B, T, self.n_heads, self.d_head)
        if self.separate_qk:
            phi_r = self.phase_scale * torch.tanh(self.W_read_key(x))
            phi_r = phi_r.view(B, T, self.n_heads, self.d_head)
        else:
            phi_r = phi_w

        drive_re = a * torch.cos(phi_w)
        drive_im = a * torch.sin(phi_w)

        if self.n_slots > 1:
            slot_logits = self.W_slot(x).view(B, T, self.n_heads, self.n_slots)
            slot_soft = torch.softmax(slot_logits, dim=-1)
            slot_hard = torch.zeros_like(slot_soft).scatter_(
                -1, slot_soft.argmax(dim=-1, keepdim=True), 1.0)
            slot_oh = slot_hard + (slot_soft - slot_soft.detach())

            wmask = slot_oh.unsqueeze(-1)
            dre = (drive_re.unsqueeze(3) * wmask).reshape(
                B, T, self.n_heads * self.n_slots, self.d_head)
            dim = (drive_im.unsqueeze(3) * wmask).reshape(
                B, T, self.n_heads * self.n_slots, self.d_head)
            gslot = gamma.unsqueeze(3).expand(
                B, T, self.n_heads, self.n_slots, self.d_head).reshape(
                B, T, self.n_heads * self.n_slots, self.d_head)

            Sre_all, Sre_fin = stateful_linear_scan(dre, gslot, state_in.get("S_re"))
            Sim_all, Sim_fin = stateful_linear_scan(dim, gslot, state_in.get("S_im"))

            Sre_all = Sre_all.view(B, T, self.n_heads, self.n_slots, self.d_head)
            Sim_all = Sim_all.view(B, T, self.n_heads, self.n_slots, self.d_head)
            rmask = slot_oh.unsqueeze(-1)
            S_re = (Sre_all * rmask).sum(dim=3)
            S_im = (Sim_all * rmask).sum(dim=3)
        else:
            Sre_all, Sre_fin = stateful_linear_scan(drive_re, gamma, state_in.get("S_re"))
            Sim_all, Sim_fin = stateful_linear_scan(drive_im, gamma, state_in.get("S_im"))
            S_re, S_im = Sre_all, Sim_all

        read_re = S_re * torch.cos(phi_r) + S_im * torch.sin(phi_r)
        read_im = S_im * torch.cos(phi_r) - S_re * torch.sin(phi_r)

        state_out = {"S_re": Sre_fin, "S_im": Sim_fin}

        if self.readout == "tanh_m":
            a_m, gamma_m = a, gamma
            Z_seq, Z_fin = stateful_linear_scan(a_m, gamma_m, state_in.get("Z_mag"))
            s_sq = torch.clamp(1.0 - torch.exp(Z_seq), min=0.0)
            m = torch.sqrt(s_sq + self._eps())
            read_re = m * torch.tanh(read_re)
            read_im = m * torch.tanh(read_im)
            state_out["Z_mag"] = Z_fin
        elif self.readout == "rms":
            rms_re = read_re.pow(2).mean(dim=-1, keepdim=True).add(self._eps()).sqrt()
            rms_im = read_im.pow(2).mean(dim=-1, keepdim=True).add(self._eps()).sqrt()
            read_re = read_re / rms_re
            read_im = read_im / rms_im
        # "layernorm": pass raw read through, no extra state.

        read_re = read_re.view(B, T, self.n_heads * self.d_head)
        read_im = read_im.view(B, T, self.n_heads * self.d_head)
        out = self.W_out(read_re) + self.W_im(read_im)

        if return_internals:
            internals = {"S_re": S_re, "S_im": S_im, "phi_w": phi_w, "phi_r": phi_r}
            return out, state_out, internals
        return out, state_out

    @staticmethod
    def _eps():
        from holographic_gssm import EPS
        return EPS


# ═══════════════════════════════════════════════════════════════════════════
# 2. StreamingHolographicLM — per-layer state threading, NoPE (same ablation as
#    StreamingNoPELM over SelectiveNoPETransformerLM: absolute PE has no support
#    past its trained length and no meaning across a carried-state chunk boundary).
# ═══════════════════════════════════════════════════════════════════════════
class StreamingHolographicLM(HolographicLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # NoPE: forward becomes h = self.embed(x), matching StreamingNoPELM's guard.
        self.pos = nn.Identity()
        for layer in self.layers:
            old = layer.scan
            new = StreamingHolographicScanLayer(
                old.d_model, d_head=old.d_head, n_heads=old.n_heads, causal=old.causal,
                dropout=(old.dropout.p if old.dropout is not None else 0.0),
                phase_scale=old.phase_scale, use_phase=old.use_phase,
                readout=old.readout, separate_qk=old.separate_qk, n_slots=old.n_slots)
            new.load_state_dict(old.state_dict())
            layer.scan = new

    def forward(self, x, states=None):
        assert isinstance(self.pos, nn.Identity)
        h = self.embed(x)
        si = states if states is not None else [None] * len(self.layers)
        new_states = []
        for layer, st in zip(self.layers, si):
            y, st_out = layer.scan(h, st)
            h = layer.ln1(h + y)
            h = layer.ln2(h + layer.ffn(h))
            new_states.append(st_out)
        return self.head(h), new_states

    def zero_states(self, states):
        """Decisive null: zero every carried tensor in every layer's state dict,
        preserving shapes (mirrors run_idle_persistence's [torch.zeros_like(s) for s in st])."""
        return [{k: torch.zeros_like(v) for k, v in st.items()} for st in states]


# ═══════════════════════════════════════════════════════════════════════════
# 3. Equivalence check — full-sequence forward (state=None) == chunked forward
#    (chunk=16, detach-carried). HIGHEST PRIORITY: everything else is void without this.
# ═══════════════════════════════════════════════════════════════════════════
def _build_lm(vocab_size, mask_idx, use_phase=True, n_slots=1, separate_qk=False,
              readout="rms", d_model=64, n_layers=2, n_heads=4, d_head=16, seq_len=32):
    return StreamingHolographicLM(
        vocab_size, mask_idx, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        d_head=d_head, seq_len=seq_len, dropout=0.0, causal=True,
        phase_scale=math.pi, use_phase=use_phase, readout=readout,
        separate_qk=separate_qk, n_slots=n_slots)


@torch.no_grad()
def check_equivalence(vocab_size, mask_idx, seed=0, T=48, chunk=16, use_phase=True,
                       n_slots=1, separate_qk=False):
    torch.manual_seed(seed)
    model = _build_lm(vocab_size, mask_idx, use_phase=use_phase, n_slots=n_slots,
                      separate_qk=separate_qk).eval()
    x = torch.randint(0, vocab_size, (2, T))

    # (a) full-sequence forward, state=None
    out_full, _ = model(x, None)

    # (b) chunked forward, state carried + detached across chunk boundaries
    states = None
    outs = []
    pos = 0
    while pos < T:
        hi = min(T, pos + chunk)
        xc = x[:, pos:hi]
        logits_c, states = model(xc, states)
        states = [{k: v.detach() for k, v in st.items()} for st in states]
        outs.append(logits_c)
        pos = hi
    out_chunked = torch.cat(outs, dim=1)

    delta = float((out_full - out_chunked).abs().max())
    return delta


# ═══════════════════════════════════════════════════════════════════════════
# 4. MQAR-across-gap task. Disjoint id ranges: keys [0,P_max), values
#    [P_max,P_max+V_max), fillers [P_max+V_max, P_max+V_max+F). Layout:
#    [k1,v1,...,kP,vP][G fillers][query-key kq] -> predict v(kq) at the LAST token.
# ═══════════════════════════════════════════════════════════════════════════
def _gap_vocab(P_max, V_max, F):
    key_lo = 0
    val_lo = P_max
    fill_lo = P_max + V_max
    vocab_size = P_max + V_max + F
    return key_lo, val_lo, fill_lo, vocab_size


def make_gap_mqar_batch(B, P, G, gen, P_max, V_max, F, key_lo, val_lo, fill_lo):
    """One batch of the gap-recall task.

    Sequence layout (length 2P + G + 1):
      [k_1, v_1, k_2, v_2, ..., k_P, v_P, fill_1, ..., fill_G, query_key]
    Target: the value bound to query_key, scored ONLY at the last position.
    Keys are distinct per trial (sampled without replacement from P_max); the
    query key is one of the P written keys (uniformly chosen) so recall must
    disambiguate among the P currently-open pairs.
    """
    keys = torch.stack([torch.randperm(P_max, generator=gen)[:P] for _ in range(B)]) + key_lo
    vals = torch.randint(0, V_max, (B, P), generator=gen) + val_lo
    fillers = torch.randint(0, F, (B, max(G, 0)), generator=gen) + fill_lo
    q_idx = torch.randint(0, P, (B,), generator=gen)
    q_keys = keys.gather(1, q_idx.unsqueeze(1)).squeeze(1)
    q_vals = vals.gather(1, q_idx.unsqueeze(1)).squeeze(1)

    kv = torch.stack([keys, vals], dim=2).reshape(B, 2 * P)   # interleave k1,v1,k2,v2,...
    if G > 0:
        x = torch.cat([kv, fillers, q_keys.unsqueeze(1)], dim=1)
    else:
        x = torch.cat([kv, q_keys.unsqueeze(1)], dim=1)
    return x, q_vals - val_lo   # target = value INDEX in [0, V_max)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Training with gap-curriculum + chunked-carried forward (the point: the gap
#    crosses real chunk boundaries during training, exactly as it will at test time).
# ═══════════════════════════════════════════════════════════════════════════
def chunked_forward(model, x, chunk):
    """Run x through model chunk-by-chunk, carrying + detaching state across
    boundaries. Returns concatenated logits (grad flows within each chunk only,
    truncated BPTT — the streaming_train.py Sec.A pattern)."""
    B, T = x.shape
    states = None
    outs = []
    pos = 0
    while pos < T:
        hi = min(T, pos + chunk)
        logits_c, states = model(x[:, pos:hi], states)
        states = [{k: v.detach() for k, v in st.items()} for st in states]
        outs.append(logits_c)
        pos = hi
    return torch.cat(outs, dim=1), states


def train_gap_curriculum(model, P, Gmax, iters, lr, seed, batch, chunk,
                         P_max, V_max, F, log_every=0, g_start=2, patience=25):
    """patience: grow the gap only after acc>0.9 is SUSTAINED for `patience`
    consecutive iterations. Without it an easy task rockets G into the
    multi-chunk regime within ~10 iterations — where truncated BPTT gives the
    write phase zero gradient from the query loss — before the write has
    consolidated (the diagnosed v1 P=1 collapse, analysis/HOLO_STREAM_VERDICT.md)."""
    key_lo, val_lo, fill_lo, _ = _gap_vocab(P_max, V_max, F)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    model.train()
    Gcur = min(g_start, Gmax) if Gmax > 0 else 0
    acc = 0.0
    good = 0
    for it in range(iters):
        x, y = make_gap_mqar_batch(batch, P, Gcur, gen, P_max, V_max, F, key_lo, val_lo, fill_lo)
        logits, _ = chunked_forward(model, x, chunk)
        pred = logits[:, -1, :V_max]
        loss = lossf(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        acc = float((pred.argmax(-1) == y).float().mean())
        good = good + 1 if acc > 0.9 else 0
        if good >= patience and Gcur < Gmax:
            Gcur = min(Gmax, int(Gcur * 1.5) + 1)
            good = 0
        if log_every and (it + 1) % log_every == 0:
            print(f"    it {it+1:>4}/{iters}: loss {float(loss):.3f} acc {acc:.3f} (train-gap {Gcur})")
    return {"final_train_gap": Gcur, "final_train_acc": round(acc, 4)}


# ═══════════════════════════════════════════════════════════════════════════
# 6. Evaluation: accuracy vs gap G x n_pairs P, per arm; plus decisive-null and
#    phi-carrier margin probe.
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def eval_gap_recall(model, P, G, NB, seed, P_max, V_max, F, chunk, zero_at_gap=False):
    """Returns (accuracy, phi_margin). zero_at_gap: run [KV block] -> take carried
    state -> ZERO it -> run [fillers+query] with no carried info (the decisive null,
    exactly run_idle_persistence's 'zeroed_at_gap' control)."""
    key_lo, val_lo, fill_lo, _ = _gap_vocab(P_max, V_max, F)
    gen = torch.Generator().manual_seed(seed)
    x, y = make_gap_mqar_batch(NB, P, G, gen, P_max, V_max, F, key_lo, val_lo, fill_lo)
    model.eval()
    kv_len = 2 * P

    if not zero_at_gap:
        logits, states = chunked_forward(model, x, chunk)
    else:
        # KV block only (no detach needed, no_grad already active)
        logits_kv, states = chunked_forward(model, x[:, :kv_len], chunk)
        states = model.zero_states(states)
        logits_rest, states = chunked_forward(model, x[:, kv_len:], chunk)
        logits = torch.cat([logits_kv, logits_rest], dim=1)

    pred = logits[:, -1, :V_max]
    acc = float((pred.argmax(-1) == y).float().mean())

    # phi-carrier margin: only meaningful for use_phase=True layers. Re(S . e^{-i phi_q})
    # for the MATCHED key vs the mean over MISMATCHED keys, at the query position, layer 0.
    margin = None
    layer0 = model.layers[0].scan
    if layer0.use_phase:
        h = model.embed(x)
        _, _, internals = layer0(h, None, return_internals=True)
        S_re, S_im = internals["S_re"], internals["S_im"]   # (B,T,H,D)
        # query angle at the LAST position (the query-key token)
        phi_q = internals["phi_r"][:, -1]                    # (B,H,D)
        S_re_q = S_re[:, -1]                                  # (B,H,D) state AT query pos
        S_im_q = S_im[:, -1]
        read_matched = (S_re_q * torch.cos(phi_q) + S_im_q * torch.sin(phi_q)).mean().item()
        # mismatched: de-rotate the SAME state with each of the OTHER (P-1) written keys'
        # angles (recomputed by re-embedding those key tokens alone) and average |read|.
        if P > 1:
            key_positions = (torch.arange(P) * 2)
            other_reads = []
            for pi in range(P):
                kp = int(key_positions[pi])
                hk = model.embed(x[:, kp:kp + 1])
                phi_k = layer0.phase_scale * torch.tanh(
                    (layer0.W_read_key if layer0.separate_qk else layer0.W_key)(hk))
                phi_k = phi_k.view(x.size(0), layer0.n_heads, layer0.d_head)
                r = (S_re_q * torch.cos(phi_k) + S_im_q * torch.sin(phi_k)).mean().item()
                other_reads.append(r)
            # matched key is one of these P; separate it out by comparing to the true q_idx
            # (approx: use mean over ALL P as "mismatched proxy" incl. the matched one is
            # a slight underestimate of the true mismatch mean, but stays a stable, cheap probe)
            mismatched_mean = sum(other_reads) / len(other_reads)
        else:
            mismatched_mean = 0.0
        margin = round(read_matched - mismatched_mean, 5)

    return acc, margin


# ═══════════════════════════════════════════════════════════════════════════
# 7. Orchestration: build arms, curriculum-train, sweep (arm, P, G), write JSON.
# ═══════════════════════════════════════════════════════════════════════════
def run(args):
    dev = torch.device("cpu")
    P_max, V_max, F = args.p_max, args.v_max, args.f_fillers
    key_lo, val_lo, fill_lo, vocab_size = _gap_vocab(P_max, V_max, F)
    mask_idx = vocab_size
    chance = 1.0 / V_max

    Ps = [int(p) for p in args.pairs.split(",")]
    Gs = [int(g) for g in args.gaps.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 78)
    print("HOLO-STREAM-RECALL — key-conditioned MQAR across a carried-state gap")
    print(f"P_max={P_max} V_max={V_max} F={F} vocab={vocab_size}  chance=1/{V_max}={chance:.4f}")
    print(f"pairs(P)={Ps}  gaps(G)={Gs}  seeds={seeds}  chunk={args.chunk}")
    print("=" * 78)

    # ── (mandatory, highest priority) equivalence check ──
    print("\n── equivalence: full-sequence forward == chunked+carried forward ──")
    eq_delta_on = check_equivalence(vocab_size, mask_idx, seed=0, T=48, chunk=16, use_phase=True)
    eq_delta_off = check_equivalence(vocab_size, mask_idx, seed=0, T=48, chunk=16, use_phase=False)
    eq_delta_slots = check_equivalence(vocab_size, mask_idx, seed=0, T=48, chunk=16, use_phase=True, n_slots=4)
    print(f"   use_phase=True   max|Δ| = {eq_delta_on:.3e}")
    print(f"   use_phase=False  max|Δ| = {eq_delta_off:.3e}")
    print(f"   use_phase=True,n_slots=4 max|Δ| = {eq_delta_slots:.3e}")
    eq_ok = max(eq_delta_on, eq_delta_off, eq_delta_slots) < 1e-5
    print(f"   {'PASS — chunked carry is the SAME operator' if eq_ok else 'FAIL — do not trust anything below'}")
    if not eq_ok:
        out = {"config": vars(args), "equivalence": {
            "use_phase_true": eq_delta_on, "use_phase_false": eq_delta_off,
            "use_phase_true_n_slots4": eq_delta_slots}, "verdict": "VOID — equivalence check failed"}
        os.makedirs(RESULTS, exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"\n→ {args.out}")
        return

    arms = {
        "holo_carried": dict(use_phase=True, zero_at_gap=False),
        "holo_zeroed_at_gap": dict(use_phase=True, zero_at_gap=True),
        "holo_off": dict(use_phase=False, zero_at_gap=False),
    }

    t0 = time.time()
    results = {"config": vars(args),
               "equivalence": {"use_phase_true": eq_delta_on, "use_phase_false": eq_delta_off,
                               "use_phase_true_n_slots4": eq_delta_slots, "passed": eq_ok},
               "chance": chance, "sweep": {}, "curriculum": {}}

    for seed in seeds:
        for P in Ps:
            print(f"\n{'='*78}\nseed={seed}  n_pairs={P}\n{'='*78}")
            for arm_name, arm_cfg in arms.items():
                torch.manual_seed(seed)
                model = _build_lm(vocab_size, mask_idx, use_phase=arm_cfg["use_phase"],
                                  d_model=args.d_model, n_layers=args.n_layers,
                                  n_heads=args.n_heads, d_head=args.d_head)
                Gmax_eval = max(Gs) if Gs else 0
                # TRAIN-SHORT-EVAL-LONG (v3): cap the TRAINING gap at the single-chunk
                # maximum. Beyond one chunk the query loss is structurally blind to the
                # write phase (truncated BPTT detaches the KV block), so training there
                # teaches nothing and catastrophically forgets what consolidated (v1's
                # instant-growth collapse AND v2's patience collapse — see
                # analysis/HOLO_STREAM_VERDICT.md). Large gaps are pure EVAL
                # extrapolation — the repo's core recipe (train T=32, eval 131k).
                cap = args.train_gap_cap if args.train_gap_cap > 0 else args.chunk - 2 * P - 2
                Gmax_train = min(Gmax_eval, max(2, cap))
                curr = train_gap_curriculum(
                    model, P, Gmax_train, args.iters, args.lr, seed, args.batch, args.chunk,
                    P_max, V_max, F, log_every=args.log_every, g_start=args.g_start,
                    patience=args.patience)
                curr["train_gap_cap"] = Gmax_train
                key = f"seed{seed}_P{P}_{arm_name}"
                results["curriculum"][key] = curr
                print(f"  [{arm_name:20s}] curriculum done: final_train_gap={curr['final_train_gap']} "
                      f"final_train_acc={curr['final_train_acc']:.3f}")

                for G in Gs:
                    acc, margin = eval_gap_recall(
                        model, P, G, args.eval_batch, seed + 1000 + G, P_max, V_max, F,
                        args.chunk, zero_at_gap=arm_cfg["zero_at_gap"])
                    sk = f"{arm_name}|P{P}|G{G}"
                    results["sweep"][sk] = {"seed": seed, "arm": arm_name, "P": P, "G": G,
                                            "accuracy": round(acc, 4), "phi_margin": margin,
                                            "chance": round(chance, 4),
                                            "beats_chance_3x": bool(acc > 3 * chance)}
                    print(f"    G={G:>4}: acc={acc:.4f}  (chance {chance:.4f}, 3x-chance "
                          f"{'YES' if acc > 3*chance else 'no'})  phi_margin={margin}")

    # ── verdict ──
    def _mean_at(arm, P, G):
        vals = [v["accuracy"] for k, v in results["sweep"].items()
                if v["arm"] == arm and v["P"] == P and v["G"] == G]
        return sum(vals) / len(vals) if vals else None

    ignition_lines = []
    for P in Ps:
        for G in Gs:
            carried = _mean_at("holo_carried", P, G)
            zeroed = _mean_at("holo_zeroed_at_gap", P, G)
            off = _mean_at("holo_off", P, G)
            if carried is None:
                continue
            ignited = carried > 3 * chance and (zeroed is None or carried > zeroed + 0.10)
            ignition_lines.append(
                f"P={P} G={G}: carried={carried:.3f} zeroed={zeroed:.3f} off={off:.3f} "
                f"chance={chance:.3f} -> {'IGNITED' if ignited else 'no'}")

    any_ignited = any("IGNITED" in ln for ln in ignition_lines)
    verdict = ("CARRIED HOLOGRAPHIC RECALL SURVIVES THE STREAMING GAP for at least one "
               "(P,G) cell — carried beats zeroed-at-gap null and clears 3x chance"
               if any_ignited else
               "NEGATIVE at every (P,G) probed in this sweep — carried recall did not clear "
               "the zeroed-at-gap null / 3x-chance bar; see 'ignition_detail' for the per-cell numbers")
    results["verdict"] = verdict
    results["ignition_detail"] = ignition_lines
    results["elapsed_s"] = round(time.time() - t0, 1)

    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "=" * 78)
    print("VERDICT")
    for ln in ignition_lines:
        print("  " + ln)
    print(f">>> {verdict}")
    print(f"\n→ {args.out}  ({results['elapsed_s']}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="fast sanity sweep (P=1, G<=8, few hundred iters)")
    ap.add_argument("--full", action="store_true", help="the full sweep (P in 1,2,4 x G in 0,8,32,128, 2 seeds)")
    ap.add_argument("--p-max", type=int, default=16, help="distinct key ids available")
    ap.add_argument("--v-max", type=int, default=16, help="distinct value ids available")
    ap.add_argument("--f-fillers", type=int, default=16, help="distinct filler ids available")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-head", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=16, help="streaming chunk length (detach-carry boundary)")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-batch", type=int, default=200)
    ap.add_argument("--g-start", type=int, default=2, help="curriculum starting gap")
    ap.add_argument("--patience", type=int, default=25,
                    help="consecutive acc>0.9 iters required before the curriculum grows the gap")
    ap.add_argument("--train-gap-cap", type=int, default=0,
                    help="max TRAINING gap (0 = auto: single-chunk, chunk-2P-2); larger gaps are eval-only extrapolation")
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--pairs", default="1", help="comma list of n_pairs P to sweep")
    ap.add_argument("--gaps", default="0,2,4,8", help="comma list of gap lengths G to sweep")
    ap.add_argument("--seeds", default="0", help="comma list of seeds")
    ap.add_argument("--out", default=os.path.join(RESULTS, "holo_stream_recall.json"))
    args = ap.parse_args()

    if args.full:
        args.pairs = "1,2,4"
        args.gaps = "0,8,32,128"
        args.seeds = "0,1"
        args.iters = max(args.iters, 2000)
        if args.out == os.path.join(RESULTS, "holo_stream_recall.json"):
            pass  # keep default full-run filename
    elif args.smoke or not (args.full):
        args.pairs = args.pairs if args.pairs != "1" or True else "1"
        args.gaps = args.gaps if args.gaps != "0,2,4,8" or True else "0,2,4,8"
        if args.out == os.path.join(RESULTS, "holo_stream_recall.json"):
            args.out = os.path.join(RESULTS, "holo_stream_recall_smoke.json")

    run(args)


if __name__ == "__main__":
    main()
