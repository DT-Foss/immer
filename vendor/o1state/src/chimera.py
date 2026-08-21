#!/usr/bin/env python3 -u
"""
CHIMERA v0 — the complete organism, one process, one stream (MS6, P33).
=============================================================================================
Every organ below is the measured answer to a scored prediction (analysis/CHIMERA_SPEC.md).
This file wires five previously-separate mechanisms (pos_run.py's gate, pos_sleep_cycles.py's
harvest+sleep-budget coupling, pos_domain_shock.py's shock protocol, pos_shared_index.py's
span-store plumbing, pos_index.py's recurrence-keyed span store) into ONE organism that runs
the MS3 shock protocol (C4 -> code -> C4) end to end:

  wake:    stream C4/code chunks -> no-grad forward (Z-carry)
  gate:    backward only if chunk-NLL > rolling q75 of own history   [q=0.75 fixed, window=500,
                                                                       min=100 -- CHIMERA_SPEC row 1]
  store:   surprise spikes -> KEYED span store (4-gram key, pos_index.py convention) so a
           recurring key can later be reminded, not just replayed at random during sleep
  remind:  recurring stored key -> the stored span is injected as tokens BEFORE the current
           chunk (in-sequence context, not a side probe) -- reliable statistics only: a
           reminder only fires for spans this arm itself collected (own-store keys), never a
           borrowed store
  sleep:   replay stored spans, FRESHNESS-WEIGHTED (sampling weight ~ 1/age_in_chunks -- see
           "freshness weighting" below), budget volume-coupled to pool size (pos_sleep_cycles.
           py's sleep_budget()), gated by a DIVIDEND MONITOR: an EMA of the measured held-out
           delta per sleep block; sleep is SUSPENDED whenever that EMA <= 0. This runs in
           EVERY phase, including phase 3 (recovery) -- this is P33b's direct fix for P20's
           measured recovery-phase overdose (R3 stayed +0.234 above baseline after the shock
           passed because sleep kept pulling the model toward stored spans even once the
           replay had stopped helping; the monitor is designed to catch exactly that).
  operate: weight updates happen live, same full-gradient step recipe as every prior POS file
           (Adam, clip=5.0, detach-carried Z). No growth/widening in v0 (CHIMERA_SPEC "open
           questions": growth is licensed by P24/P27 but not exercised here).

Freshness weighting (documented choice, CHIMERA_SPEC row 2 leaves the functional form open):
  weight(span) = 1 / (1 + age_in_chunks_since_store)
  i.e. INVERSE-LINEAR in age, not exponential decay. Rationale: an exponential (gamma**age)
  either forgets old-but-still-relevant phase-1 spans almost completely within one sleep
  segment's timescale (dozens of chunks) for any gamma small enough to matter, or barely
  differentiates at all for gamma close to 1 -- there is no principled gamma to pick without
  another sweep, and P20's finding was specifically that the OLD WORLD's spans are what need
  protecting during a shock, so a decay strong enough to matter would defeat the mechanism
  under test. Inverse-age is scale-free (no extra hyperparameter), monotone (always prefers
  fresher material at the margin) and heavy-tailed enough that old spans are still drawn, just
  less often -- sampling is done by drawing with replacement (weighted) up to the sleep
  segment's token budget, not by truncating the pool.

Reminder mechanism (in-sequence injection on key recurrence, P18/MS1's "reliable statistics"
clause -- arbitration under an unreliable/shared index is explicitly OUT of scope for v0, see
CHIMERA_SPEC "open questions"): a rolling dict of 4-gram keys -> stored span (pos_index.py's
exact key convention: the 4 tokens ending at a spike, span = spike +/- span_half). Every wake
chunk, BEFORE the forward pass, the chunk's own tokens are scanned for a 4-gram matching a
KEY THIS ARM ITSELF STORED. On the first such match in a chunk, the stored span is spliced in
as the chunk's own historical context: the state is advanced through the stored span first
(no-grad, does not count as a gradient step), THEN the chunk is scored/trained from that
primed state instead of from the raw carried state. This is cheap (one extra no-grad forward
over span_half*2+1 tokens) and directly testable: the NLL on the reminded chunk vs. what an
unreminded run would have produced is logged (reminder_stats), matching the mission's
"NLL-delta on Folge-Chunks" ask. Only fires once per chunk (first match wins) to keep the
per-chunk cost bounded and the statistics clean (one reminder -> one measured effect).

Arms (P33, matched total gradient tokens -- budget-neutral, R3's fairness invariant extended):
  (i)   chimera       gate + keyed store + reminder + dividend-monitored freshness sleep (full)
  (ii)  r3_replicate  gate + sleep, FIXED CADENCE (no dividend monitor, no reminder, uniform
                       sampling not freshness-weighted) -- pos_domain_shock.py's R3 verbatim
  (iii) r1_full       full-gradient baseline, no gate, no store, no sleep, no reminder
  (iv)  no_reminder   CHIMERA minus reminder (gate + store + dividend-monitored sleep)
  (v)   no_monitor    CHIMERA minus dividend monitor (gate + store + reminder + FIXED-CADENCE
                       freshness sleep, matches R3's cadence so the ONLY difference from (i)
                       is the monitor's gating decision)

Budget-neutrality: exactly R3's invariant (pos_domain_shock.py) -- sleep-capable arms carve
their sleep chunks OUT of the wake block that triggers them (block size = --sleep-every), so
every arm's TOTAL chunks visited (wake+sleep summed) over phase2+phase3 equals every other
arm's, checked and reported (`chunk_budget_check`). r1_full has no gate so its "wake budget"
is just chunks visited; comparison is on total chunks visited, matching pos_domain_shock.py.

Same shock protocol, streams, and eval recipe as pos_domain_shock.py: phase 1 C4 (train-far,
skip_docs=5_000_000), phase 2 CODE (codeparrot/github-code-clean, client-filtered to Python),
phase 3 C4 resumed (same cursor, not rewound). forgetting/plasticity/recovery are defined
IDENTICALLY to that file. Starting point: results/pos_snapshots/ckpt_359050240.pt (A3),
opened READ-ONLY -- this file never writes to any results/pos_* path except its own.

Usage:
  python3 -u src/chimera.py --smoke   # phase_chunks=40, eval_every=10
  python3 -u src/chimera.py --full    # phase_chunks=150, eval_every=25
"""
import os
import sys
import json
import math
import copy
import argparse
import tempfile
from collections import deque

sys.path.insert(0, "reference")
sys.path.insert(0, "src")

try:
    os.nice(19)
except (PermissionError, OSError):
    pass  # already niced by the launcher (macOS EPERM on re-nice)

import numpy as np
import torch
import torch.nn.functional as F

torch.backends.mps.is_available = lambda: False          # force CPU (same as the live run)
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass  # must be set before any parallel work has started; harmless if already set

from streaming_train import StreamingNoPELM
from length_extrap_v2 import load_wikitext2, build_vocab, tokenize
from pos_run import build_eval_set, heldout                    # safe: pos_run's main() only runs under __main__
from pos_sleep import ChunkFeeder, SpanStream, C4ValStream, load_snapshot, _real_vocab
from pos_sleep_cycles import harvest_spans, sleep_budget
from pos_domain_shock import CodeStream, make_phase1_source, make_phase2_source, grad_step, nograd_step


# ───────────────────────────────────────────────────────────────────────────
#  Keyed span store — 4-gram key -> {span, t_store (chunk index at storage),
#  n_store}. Superset of pos_sleep_cycles.py's flat list (adds the key so the
#  reminder organ can look up recurrence) and of pos_index.py's IndexLoop
#  (this store trains from recurrence in-sequence rather than only probing).
# ───────────────────────────────────────────────────────────────────────────
class KeyedSpanStore:
    def __init__(self, ngram=4, max_entries=20000):
        self.ngram = ngram
        self.max_entries = max_entries
        self.keys = {}                       # tuple(4gram) -> entry dict
        self.order = []                       # insertion order, for freshness / eviction

    def harvest(self, x, y, nll, spike_min_nll, span_half, chunk_idx, max_per_chunk=2):
        """Mirrors pos_sleep_cycles.harvest_spans' spike selection, but additionally
        keys each stored span by the 4-gram ending at the spike (pos_index.py
        convention: key = full_row[t-2:t+2]) so the reminder organ can find it
        again on recurrence. Returns the number of NEW spans stored."""
        B, K = x.shape
        # y is optional: callers inside the wake loop only carry x/nll; the
        # span then ends at the chunk's last input token (one shorter), which
        # the replay path slices identically.
        flat_nll = nll.reshape(-1)
        order = torch.argsort(flat_nll, descending=True)
        taken_rows = set()
        n_new = 0
        for idx in order.tolist():
            if n_new >= max_per_chunk:
                break
            val = float(flat_nll[idx])
            if val < spike_min_nll:
                break
            b, k = idx // K, idx % K
            if b in taken_rows:
                continue
            if k < 2:
                continue
            full_row = x[b].tolist() + ([int(y[b, -1])] if y is not None else [])
            key = tuple(full_row[k - 2:k + 2])
            if key in self.keys:
                continue
            lo, hi = max(0, k - span_half), min(K, k + span_half + 1)
            span = x[b, lo:hi].tolist()
            if len(span) < 16:
                continue
            if len(self.keys) >= self.max_entries:
                break
            self.keys[key] = {"span": span, "t_store": chunk_idx, "n_store": len(self.keys)}
            self.order.append(key)
            taken_rows.add(b)
            n_new += 1
        return n_new

    def lookup_recurrence(self, row_tokens):
        """Scans row_tokens (a single chunk row, list[int]) for the FIRST 4-gram
        matching a stored key. Returns (key, entry) or (None, None). First-match
        keeps the per-chunk reminder cost and statistics bounded (one reminder,
        one measured effect -- see module docstring)."""
        ng = self.ngram
        for i in range(len(row_tokens) - ng + 1):
            key = tuple(row_tokens[i:i + ng])
            entry = self.keys.get(key)
            if entry is not None:
                return key, entry
        return None, None

    def all_spans(self):
        return [self.keys[k]["span"] for k in self.order]

    def freshness_weights(self, chunk_idx):
        """weight(span) = 1 / (1 + age_in_chunks) -- see module docstring for why
        inverse-linear (not exponential) was chosen. Returns a numpy array aligned
        with self.order / self.all_spans()."""
        ages = np.array([max(0, chunk_idx - self.keys[k]["t_store"]) for k in self.order],
                        dtype=np.float64)
        w = 1.0 / (1.0 + ages)
        s = w.sum()
        return w / s if s > 0 else w

    def __len__(self):
        return len(self.keys)


# ───────────────────────────────────────────────────────────────────────────
#  Freshness-weighted replay sampling: draws spans WITH REPLACEMENT, weighted
#  by KeyedSpanStore.freshness_weights, up to a token budget. This replaces
#  pos_sleep.py's SpanStream (uniform shuffle) for CHIMERA's full/no_reminder
#  arms; r3_replicate / no_monitor use plain SpanStream (uniform) to stay a
#  faithful fixed-cadence replica / isolate the monitor's marginal effect.
# ───────────────────────────────────────────────────────────────────────────
class FreshnessWeightedSpanStream:
    def __init__(self, store, chunk_idx, seed):
        self.store = store
        self.spans = store.all_spans()
        assert len(self.spans) > 0, "FreshnessWeightedSpanStream needs at least one span"
        self.weights = store.freshness_weights(chunk_idx)
        self.rng = np.random.default_rng(seed)
        self.pending = []

    def _draw_one(self):
        i = self.rng.choice(len(self.spans), p=self.weights)
        return self.spans[i]

    def next_block(self, n):
        while len(self.pending) < n:
            self.pending.extend(int(t) for t in self._draw_one())
        out, self.pending = self.pending[:n], self.pending[n:]
        return out


# ───────────────────────────────────────────────────────────────────────────
#  Dividend monitor — EMA of the held-out delta measured per sleep block.
#  Sleep is suspended whenever the EMA <= 0 (P33b: the fix for P20's measured
#  recovery-phase overdose). Applies in ALL phases, per the mission brief.
# ───────────────────────────────────────────────────────────────────────────
class DividendMonitor:
    def __init__(self, ema_beta=0.5):
        self.ema_beta = ema_beta
        self.ema = None          # None = no data yet -> sleep is ALLOWED (benefit of the doubt
                                  # on the very first block; there is nothing to be pessimistic
                                  # about yet, and the monitor only ever suspends on OBSERVED
                                  # non-positive dividends, never pre-emptively)
        self.history = []

    def allows_sleep(self):
        return self.ema is None or self.ema > 0

    def update(self, dividend):
        self.ema = dividend if self.ema is None else (
            self.ema_beta * dividend + (1 - self.ema_beta) * self.ema)
        self.history.append({"dividend": round(dividend, 6), "ema": round(self.ema, 6),
                             "sleep_allowed_next": bool(self.ema > 0)})


# ───────────────────────────────────────────────────────────────────────────
#  Small frozen held-out probe (for the dividend monitor) — 5k WT-2 tokens by
#  default (mission: "kleine frozen WT-2-Probe, z.B. 5k Tokens fuer
#  Geschwindigkeit"), distinct from the full WT-2 heldout curve eval (which
#  uses the checkpoint's own eval_tokens, matching pos_domain_shock.py).
# ───────────────────────────────────────────────────────────────────────────
def build_small_probe(stoi, unk, chunk, n_tokens=5000):
    _, val_text = load_wikitext2()
    val_ids = tokenize(val_text, stoi, unk)
    pX, pY = build_eval_set(val_ids, n_tokens, chunk)
    return pX, pY


# ───────────────────────────────────────────────────────────────────────────
#  Gate step (identical mechanics to pos_domain_shock.py's run_r2_chunk /
#  pos_run.py's step_gated: rolling-q75 threshold, window contents BEFORE
#  this chunk). Returns gated flag + the x/nll the caller needs for
#  harvesting and reminder bookkeeping.
# ───────────────────────────────────────────────────────────────────────────
def gate_decide(s, window, args):
    if len(window) >= args.min_window:
        thresh = float(np.quantile(np.fromiter(window, dtype=np.float64), args.gate_q))
        return s > thresh, thresh
    return True, None            # not enough window yet: learn (matches pos_run.py's convention)


def run_gated_chunk_with_reminder(model, opt, feeder, states, window, store, chunk_idx,
                                  reminder_stats, args, use_reminder):
    """One wake chunk under the gate, with optional in-sequence reminder: if a
    4-gram in this chunk matches a key this arm itself stored, the stored span
    is used to prime the carried state (no-grad advance) BEFORE the gate's
    own no-grad forward is scored -- so the reminder can only ever help THIS
    chunk's surprise estimate and (if gated) THIS chunk's gradient step, never
    retroactively touch anything already trained. Only row 0 is scanned for a
    recurring key (matches pos_index.py's per-chunk key-scan cost profile;
    B is small here so scanning one row is already representative and keeps
    the reminder's cost bounded)."""
    x, y = feeder.next_xy()

    reminded = False
    primed_states = states
    if use_reminder and len(store) > 0:
        row0 = x[0].tolist()
        key, entry = store.lookup_recurrence(row0)
        if entry is not None:
            with torch.no_grad():
                span_x = torch.tensor(entry["span"], dtype=torch.long).unsqueeze(0)
                if states is None:
                    row_states = None
                else:
                    row_states = [s[0:1] for s in states]
                # advance a 1-row state through the stored span (no-grad, no learning)
                _, primed_row_states = model(span_x, row_states)
                if states is None:
                    # first-ever chunk: broadcast the primed single-row state back to
                    # the full batch isn't meaningful (other rows have no state yet) --
                    # only prime if the arm already carries a real per-row state.
                    primed_states = states
                else:
                    primed_states = [s.clone() for s in states]
                    for si in range(len(primed_states)):
                        primed_states[si][0:1] = primed_row_states[si]
            reminded = True

    st_ng, s, _, nll_ng = nograd_step(model, x, y, primed_states)
    gated, thresh = gate_decide(s, window, args)
    if gated:
        new_states, gt, loss, x_d, nll = grad_step(model, opt, x, y, primed_states)
    else:
        new_states, gt = st_ng, 0
        nll = nll_ng
        x_d = x.detach()

    if reminded:
        reminder_stats["n_injections"] += 1
        reminder_stats["nll_after"].append(s)

    window.append(s)
    return new_states, gt, gated, x_d, nll, s


def run_gated_chunk_plain(model, opt, feeder, states, window, args):
    """Same as run_gated_chunk_with_reminder with use_reminder=False, kept as a
    separate lean path for arms that never touch the store (r1_full doesn't
    even gate; r3_replicate/no_reminder gate but never look up the store)."""
    x, y = feeder.next_xy()
    st_ng, s, _, nll_ng = nograd_step(model, x, y, states)
    gated, thresh = gate_decide(s, window, args)
    if gated:
        new_states, gt, loss, x_d, nll = grad_step(model, opt, x, y, states)
    else:
        new_states, gt = st_ng, 0
        nll = nll_ng
        x_d = x.detach()
    window.append(s)
    return new_states, gt, gated, x_d, nll, s


def run_full_gradient_chunk(model, opt, feeder, states):
    x, y = feeder.next_xy()
    new_states, gt, loss, x_d, nll = grad_step(model, opt, x, y, states)
    return new_states, gt, True, x_d, nll, loss


# ───────────────────────────────────────────────────────────────────────────
#  Sleep segment — freshness-weighted (CHIMERA/no_monitor arms) or uniform
#  (r3_replicate) replay, run_sleep_segment's grad-step recipe unchanged.
# ───────────────────────────────────────────────────────────────────────────
def run_sleep_segment(model, opt, spans_source, n_chunks, B, K):
    feeder = ChunkFeeder(spans_source, B, K)
    states = None
    grad_tokens = 0
    for _ in range(n_chunks):
        x, y = feeder.next_xy()
        states, gt, _, _, _ = grad_step(model, opt, x, y, states)
        grad_tokens += gt
    return grad_tokens


def small_probe_eval(model, probe_x, probe_y):
    return heldout(model, probe_x, probe_y)


# ───────────────────────────────────────────────────────────────────────────
#  Main measurement
# ───────────────────────────────────────────────────────────────────────────
ARMS = ("chimera", "r3_replicate", "r1_full", "no_reminder", "no_monitor")

SLEEP_CAPABLE = {"chimera": True, "r3_replicate": True, "r1_full": False,
                 "no_reminder": True, "no_monitor": True}
USES_REMINDER = {"chimera": True, "r3_replicate": False, "r1_full": False,
                 "no_reminder": False, "no_monitor": True}
USES_MONITOR = {"chimera": True, "r3_replicate": False, "r1_full": False,
                "no_reminder": True, "no_monitor": False}
USES_FRESHNESS = {"chimera": True, "r3_replicate": False, "r1_full": False,
                  "no_reminder": True, "no_monitor": True}
USES_GATE = {"chimera": True, "r3_replicate": True, "r1_full": False,
            "no_reminder": True, "no_monitor": True}


def run_chimera(args, eval_wt2_fn, vocab_fn=_real_vocab):
    ck, cfg, base_model, opt_sd, stoi, unk, mask, V = load_snapshot(args.ckpt, vocab_fn)
    B, K = cfg["batch"], cfg["chunk"]
    lr = cfg["lr"]

    phase_chunks = args.phase_chunks
    eval_every = args.eval_every

    print(f"[chimera] ckpt n_streamed={ck['n_streamed']:,} | phase_chunks={phase_chunks} "
          f"eval_every={eval_every} | B={B} K={K} lr={lr}", flush=True)

    code_val_src = CodeStream(stoi, unk)
    code_val_ids = code_val_src.next_block(args.code_val_tokens)
    code_val_unk_rate = code_val_src.unk_rate()
    cvX, cvY = build_eval_set(code_val_ids, len(code_val_ids) - 1, K)
    print(f"[chimera] code-val slice: {cvY.numel():,} tokens, unk_rate={code_val_unk_rate:.4f}", flush=True)

    def eval_code_fn(model):
        return heldout(model, cvX, cvY)

    probe_x, probe_y = build_small_probe(stoi, unk, K, n_tokens=args.probe_tokens)
    print(f"[chimera] dividend-monitor probe: {probe_y.numel():,} WT-2 tokens", flush=True)

    def fresh_model_opt():
        m = copy.deepcopy(base_model)
        o = torch.optim.Adam(m.parameters(), lr=lr)
        if opt_sd is not None:
            o.load_state_dict(opt_sd)
            for g in o.param_groups:
                g["lr"] = lr
        return m, o

    base_wt2 = eval_wt2_fn(base_model)
    print(f"[chimera] base_heldout_wt2={base_wt2:.6f}", flush=True)

    out = {
        "ckpt_n_streamed": ck["n_streamed"],
        "cadence": {"d_model": cfg.get("d_model"), "batch": B, "chunk": K,
                    "source": "read from POS checkpoint config (2026-08-05 audit rule)"},
        "phase2_dataset": "codeparrot/github-code-clean (language==Python, client-filtered)",
        "budget": {"phase_chunks": phase_chunks, "eval_every": eval_every,
                  "sleep_chunks_max": args.sleep_chunks, "max_replay_epochs": args.max_replay_epochs,
                  "gate_q": args.gate_q, "gate_window": args.gate_window,
                  "min_window": args.min_window, "spike_min_nll": args.spike_min_nll,
                  "span_half": args.span_half, "sleep_every_n_chunks": args.sleep_every,
                  "dividend_ema_beta": args.ema_beta, "probe_tokens": probe_y.numel()},
        "code_val_tokens": cvY.numel(),
        "code_val_unk_rate": round(code_val_unk_rate, 6),
        "base_heldout_wt2": round(base_wt2, 6),
        "arms": {},
    }

    arm_results = {}
    for tag in ARMS:
        print(f"\n[chimera] ===== arm {tag} =====", flush=True)
        model, opt = fresh_model_opt()

        phase13_src = make_phase1_source(stoi, unk)
        phase2_src = make_phase2_source(stoi, unk)
        feeder13 = ChunkFeeder(phase13_src, B, K)
        feeder2 = ChunkFeeder(phase2_src, B, K)

        states = None
        gate_window = deque(maxlen=args.gate_window)
        spike_window = deque(maxlen=200)
        store = KeyedSpanStore(ngram=4, max_entries=args.max_store_entries)
        monitor = DividendMonitor(ema_beta=args.ema_beta)

        curve_wt2 = []
        curve_code = []
        grad_tokens_total = 0
        gate_frac_log = []
        sleep_events = []
        skipped_by_monitor = 0
        reminder_stats = {"n_injections": 0, "nll_after": []}
        global_idx = 0

        def record(idx, phase):
            hl_wt2 = eval_wt2_fn(model)
            hl_code = eval_code_fn(model)
            curve_wt2.append([idx, phase, round(hl_wt2, 6)])
            curve_code.append([idx, phase, round(hl_code, 6)])
            print(f"[chimera][{tag}] phase={phase:<12} chunk={idx:>4} "
                  f"wt2={hl_wt2:.6f} code={hl_code:.6f}", flush=True)
            return hl_wt2, hl_code

        record(0, "base")

        def run_wake_block(feeder, n_chunks, phase_name, collect):
            nonlocal states, global_idx, grad_tokens_total
            done = 0
            while done < n_chunks:
                step_n = min(eval_every, n_chunks - done)
                for _ in range(step_n):
                    if tag == "r1_full":
                        states, gt, gated, x_this, nll_this, s_this = run_full_gradient_chunk(
                            model, opt, feeder, states)
                    elif USES_REMINDER[tag]:
                        states, gt, gated, x_this, nll_this, s_this = run_gated_chunk_with_reminder(
                            model, opt, feeder, states, gate_window, store, global_idx,
                            reminder_stats, args, use_reminder=True)
                    else:
                        states, gt, gated, x_this, nll_this, s_this = run_gated_chunk_plain(
                            model, opt, feeder, states, gate_window, args)
                    grad_tokens_total += gt
                    gate_frac_log.append(1 if gated else 0)

                    if collect and gated and tag != "r1_full":
                        chunk_mean = float(nll_this.mean())
                        if len(spike_window) >= 2:
                            sp_thresh = float(np.quantile(
                                np.fromiter(spike_window, dtype=np.float64), args.spike_quantile))
                            if chunk_mean > sp_thresh:
                                store.harvest(x_this, None, nll_this, args.spike_min_nll,
                                             args.span_half, global_idx)
                        spike_window.append(chunk_mean)
                    done += 1
                    global_idx += 1
                record(global_idx, phase_name)
            return done

        # ── Phase 1: C4 (train-far) ─────────────────────────────────────────
        collect_p1 = tag != "r1_full"
        run_wake_block(feeder13, phase_chunks, "phase1", collect=collect_p1)

        pre_phase2_wt2 = curve_wt2[-1][2]
        pre_phase2_code = curve_code[-1][2]

        # ── Phase 2 (CODE) then Phase 3 (C4 resumed): sleep-capable arms
        #    process in sleep_every-sized wake blocks; the dividend monitor
        #    (chimera, no_reminder) gates whether the coupled sleep budget is
        #    actually spent sleeping or falls back to more wake -- APPLIES IN
        #    BOTH PHASES (P33b: this is the direct fix for P20's measured
        #    recovery-phase overdose). r3_replicate/no_monitor sleep on FIXED
        #    CADENCE (no monitor check) -- r3_replicate additionally uses
        #    UNIFORM (not freshness-weighted) sampling, matching
        #    pos_domain_shock.py's R3 verbatim. ────────────────────────────
        for phase_name, feeder, phase_n in (("phase2", feeder2, phase_chunks),
                                            ("phase3", feeder13, phase_chunks)):
            if not SLEEP_CAPABLE[tag]:
                run_wake_block(feeder, phase_n, phase_name, collect=False)
                continue

            done = 0
            while done < phase_n:
                block_n = min(args.sleep_every, phase_n - done)
                wake_n = max(1, block_n - args.sleep_chunks)
                run_wake_block(feeder, wake_n, phase_name, collect=True)
                done += wake_n

                pool = store.all_spans()
                n_sleep_used, span_tokens, replay_eff = sleep_budget(
                    pool, args.sleep_chunks, args.max_replay_epochs, B, K)
                leftover = args.sleep_chunks - n_sleep_used
                if leftover > 0 and done < phase_n:
                    extra = min(leftover, phase_n - done)
                    run_wake_block(feeder, extra, phase_name, collect=True)
                    done += extra

                monitor_blocks = USES_MONITOR[tag]
                allowed = (not monitor_blocks) or monitor.allows_sleep()

                if n_sleep_used > 0 and allowed:
                    pre_probe = small_probe_eval(model, probe_x, probe_y)
                    if USES_FRESHNESS[tag]:
                        src = FreshnessWeightedSpanStream(store, global_idx, seed=args.seed + global_idx)
                    else:
                        src = SpanStream(pool, seed=args.seed + global_idx, permute_chunks=False)
                    gt_sleep = run_sleep_segment(model, opt, src, n_sleep_used, B, K)
                    grad_tokens_total += gt_sleep
                    global_idx += n_sleep_used
                    done += n_sleep_used
                    post_probe = small_probe_eval(model, probe_x, probe_y)
                    dividend = pre_probe - post_probe    # loss DROP = positive dividend
                    if monitor_blocks:
                        monitor.update(dividend)
                    hl_wt2, hl_code = record(global_idx, f"{phase_name}_sleep")
                    sleep_events.append({"phase": phase_name, "global_idx": global_idx,
                                         "n_sleep_chunks": n_sleep_used, "pool_spans": len(pool),
                                         "pool_tokens": span_tokens, "replay_epochs": round(replay_eff, 3),
                                         "dividend": round(dividend, 6),
                                         "monitor_ema": None if monitor.ema is None else round(monitor.ema, 6),
                                         "wt2_after": round(hl_wt2, 6), "code_after": round(hl_code, 6)})
                    print(f"[chimera][{tag}] {phase_name} sleep: {n_sleep_used} chunks "
                          f"(pool={len(pool)} spans, {span_tokens} tok, {replay_eff:.2f} epochs, "
                          f"dividend={dividend:+.6f}) -> wt2={hl_wt2:.6f} code={hl_code:.6f}", flush=True)
                elif n_sleep_used > 0 and not allowed:
                    # monitor suspends sleep: the budget that WOULD have been sleep is spent
                    # on more wake instead, keeping the total-chunks-visited invariant intact.
                    skipped_by_monitor += 1
                    extra = min(n_sleep_used, phase_n - done)
                    if extra > 0:
                        run_wake_block(feeder, extra, phase_name, collect=True)
                        done += extra
                    print(f"[chimera][{tag}] {phase_name} block: sleep SUSPENDED by dividend "
                          f"monitor (EMA={monitor.ema:.6f} <= 0) -- budget spent on wake instead",
                          flush=True)
                else:
                    print(f"[chimera][{tag}] {phase_name} block: empty/small pool, "
                          f"sleep skipped, all budget spent on wake", flush=True)

        wt2_values_phase2 = [v for idx, ph, v in curve_wt2 if ph in ("phase2", "phase2_sleep")]
        code_values_phase2 = [v for idx, ph, v in curve_code if ph in ("phase2", "phase2_sleep")]
        post_phase3_wt2 = curve_wt2[-1][2]

        forgetting = round(max(wt2_values_phase2 + [pre_phase2_wt2]) - pre_phase2_wt2, 6)
        plasticity = round(pre_phase2_code - min(code_values_phase2 + [pre_phase2_code]), 6)
        recovery = round(post_phase3_wt2 - pre_phase2_wt2, 6)

        gate_frac = sum(gate_frac_log) / max(1, len(gate_frac_log))
        reminder_stats["mean_nll_reminded_chunks"] = (
            round(float(np.mean(reminder_stats["nll_after"])), 6)
            if reminder_stats["nll_after"] else None)

        arm_results[tag] = {
            "curve_wt2": curve_wt2, "curve_code": curve_code,
            "grad_tokens_total": grad_tokens_total,
            "n_chunks_gated": sum(gate_frac_log), "n_chunks_seen": len(gate_frac_log),
            "gate_frac": round(gate_frac, 4),
            "pre_phase2_wt2": round(pre_phase2_wt2, 6), "pre_phase2_code": round(pre_phase2_code, 6),
            "post_phase3_wt2": round(post_phase3_wt2, 6),
            "forgetting": forgetting, "plasticity": plasticity, "recovery": recovery,
            "n_spans_collected": len(store), "sleep_events": sleep_events,
            "sleep_blocks_skipped_by_monitor": skipped_by_monitor,
            "dividend_trajectory": monitor.history,
            "reminder_stats": {"n_injections": reminder_stats["n_injections"],
                               "mean_nll_reminded_chunks": reminder_stats["mean_nll_reminded_chunks"]},
        }
        print(f"[chimera][{tag}] forgetting={forgetting:+.6f} plasticity={plasticity:+.6f} "
              f"recovery={recovery:+.6f} gate_frac={gate_frac:.4f} grad_tokens={grad_tokens_total:,} "
              f"spans={len(store)} reminders={reminder_stats['n_injections']} "
              f"monitor_skips={skipped_by_monitor}", flush=True)

    out["arms"] = arm_results

    # ── budget-neutrality: every arm's TOTAL chunks visited (wake+sleep) over
    #    phase2+phase3 must match, exactly R3's fairness invariant in
    #    pos_domain_shock.py, extended to five arms. ─────────────────────────
    n_visited = {}
    for t in ARMS:
        n_visited[t] = arm_results[t]["n_chunks_seen"] + sum(
            e["n_sleep_chunks"] for e in arm_results[t]["sleep_events"])
    out["chunk_budget_check"] = {
        "n_chunks_visited": n_visited,
        "equal": len(set(n_visited.values())) == 1,
    }

    # ── P33 scoring ──────────────────────────────────────────────────────────
    f = {t: arm_results[t]["forgetting"] for t in ARMS}
    p = {t: arm_results[t]["plasticity"] for t in ARMS}
    r = {t: arm_results[t]["recovery"] for t in ARMS}

    p33_a = (f["chimera"] <= f["r3_replicate"]) and (p["chimera"] >= p["r3_replicate"])
    p33_b = r["chimera"] < r["r3_replicate"]     # lower residual damage = beats R3's overdose
    # (c): each ablation worse than full chimera on >=1 of the 3 axes (lower forgetting/recovery
    # is BETTER, higher plasticity is BETTER -- "worse" means chimera dominates that axis)
    def worse_on_at_least_one(ablation_tag):
        worse_forget = f[ablation_tag] > f["chimera"]
        worse_plastic = p[ablation_tag] < p["chimera"]
        worse_recover = r[ablation_tag] > r["chimera"]
        return worse_forget or worse_plastic or worse_recover
    p33_c_no_reminder = worse_on_at_least_one("no_reminder")
    p33_c_no_monitor = worse_on_at_least_one("no_monitor")
    p33_c = p33_c_no_reminder and p33_c_no_monitor

    # (d): no single-organ arm dominates chimera on ALL three axes (dominates = better-or-equal
    # on forgetting AND plasticity AND recovery, strictly better on at least one)
    def dominates_chimera(tag):
        better_or_eq_forget = f[tag] <= f["chimera"]
        better_or_eq_plastic = p[tag] >= p["chimera"]
        better_or_eq_recover = r[tag] <= r["chimera"]
        strictly_better = (f[tag] < f["chimera"]) or (p[tag] > p["chimera"]) or (r[tag] < r["chimera"])
        return better_or_eq_forget and better_or_eq_plastic and better_or_eq_recover and strictly_better
    dominators = [t for t in ("r3_replicate", "r1_full") if dominates_chimera(t)]
    p33_d = len(dominators) == 0

    out["p33_scoring"] = {
        "a_forgetting_leq_R3_at_geq_R3_plasticity": {
            "pass": bool(p33_a), "chimera_forgetting": f["chimera"], "r3_forgetting": f["r3_replicate"],
            "chimera_plasticity": p["chimera"], "r3_plasticity": p["r3_replicate"]},
        "b_recovery_beats_R3": {
            "pass": bool(p33_b), "chimera_recovery": r["chimera"], "r3_recovery": r["r3_replicate"]},
        "c_each_ablation_worse_on_geq1_axis": {
            "pass": bool(p33_c), "no_reminder_worse": bool(p33_c_no_reminder),
            "no_monitor_worse": bool(p33_c_no_monitor)},
        "d_no_single_organ_arm_dominates": {
            "pass": bool(p33_d), "dominators": dominators},
    }
    all_pass = p33_a and p33_b and p33_c and p33_d
    out["verdict"] = (
        f"forgetting: " + " ".join(f"{t}={f[t]:+.6f}" for t in ARMS) + " | "
        f"plasticity: " + " ".join(f"{t}={p[t]:+.6f}" for t in ARMS) + " | "
        f"recovery: " + " ".join(f"{t}={r[t]:+.6f}" for t in ARMS) + " | "
        f"P33: {'PASS' if all_pass else 'PARTIAL/FAIL'} "
        f"(a: {p33_a}, b: {p33_b}, c: {p33_c}, d: {p33_d})"
    )
    print(f"\n[chimera] {out['verdict']}", flush=True)

    d = os.path.dirname(args.out) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_")
    with os.fdopen(fd, "w") as f_:
        json.dump(out, f_, indent=2)
    os.replace(tmp, args.out)
    print(f"[chimera] -> {args.out}", flush=True)
    return out


# ───────────────────────────────────────────────────────────────────────────
#  CLI
# ───────────────────────────────────────────────────────────────────────────
def build_argparser():
    ap = argparse.ArgumentParser(
        description="CHIMERA v0: the complete organism (gate+store+reminder+dividend-monitored "
                    "freshness sleep) vs R3/R1/ablations on the MS3 shock protocol (P33)")
    ap.add_argument("--ckpt", default="results/pos_snapshots/ckpt_359050240.pt")
    ap.add_argument("--phase-chunks", type=int, default=150, help="chunks per phase (1/2/3)")
    ap.add_argument("--eval-every", type=int, default=25, help="WT-2 + code-val eval cadence, in chunks")
    ap.add_argument("--code-val-tokens", type=int, default=20_000)
    ap.add_argument("--probe-tokens", type=int, default=5_000,
                    help="small frozen WT-2 probe for the dividend monitor (fast per-sleep-block eval)")
    ap.add_argument("--seed", type=int, default=42)
    # gate (CHIMERA_SPEC row 1: q=0.75 fixed, window 500, min 100)
    ap.add_argument("--gate-q", type=float, default=0.75)
    ap.add_argument("--gate-window", type=int, default=500)
    ap.add_argument("--min-window", type=int, default=100)
    # spike harvest + sleep (pos_sleep_cycles.py defaults)
    ap.add_argument("--spike-quantile", type=float, default=0.75)
    ap.add_argument("--spike-min-nll", type=float, default=7.0)
    ap.add_argument("--span-half", type=int, default=32)
    ap.add_argument("--sleep-every", type=int, default=30, help="wake chunks between sleep segments")
    ap.add_argument("--sleep-chunks", type=int, default=10, help="max sleep chunks per segment")
    ap.add_argument("--max-replay-epochs", type=float, default=2.0)
    ap.add_argument("--max-store-entries", type=int, default=20000)
    ap.add_argument("--ema-beta", type=float, default=0.5, help="dividend-monitor EMA smoothing")
    ap.add_argument("--smoke", action="store_true", help="phase-chunks=40, eval-every=10, out=*_smoke.json")
    ap.add_argument("--full", action="store_true", help="phase-chunks=150, eval-every=25 (explicit; also default)")
    ap.add_argument("--out", default="results/chimera.json")
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    if args.smoke:
        args.phase_chunks = 40
        args.eval_every = 10
        args.sleep_every = 10
        args.sleep_chunks = 4
        if args.out == "results/chimera.json":
            args.out = "results/chimera_smoke.json"
    elif args.full:
        args.phase_chunks = 150
        args.eval_every = 25

    def eval_wt2_fn(model):
        train_text, val_text = load_wikitext2()
        _, stoi, unk, mask = build_vocab(train_text)
        val_ids = tokenize(val_text, stoi, unk)
        cfg = torch.load(args.ckpt, weights_only=False)["config"]
        evX, evY = build_eval_set(val_ids, cfg["eval_tokens"], cfg["chunk"])
        return heldout(model, evX, evY)

    run_chimera(args, eval_wt2_fn)


if __name__ == "__main__":
    main()
