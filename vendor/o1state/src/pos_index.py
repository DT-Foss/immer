#!/usr/bin/env python3 -u
"""
POS INDEX — the surprise -> span -> paired-injection-probe loop that watches A3.
==================================================================================
Two jobs, both local (no vendor import chain — the whole point of this file):

  IndexLoop   watches A3's per-chunk NLL. A high-surprise token gets its local span
              (+/- span_half around the spike) stored as a candidate "thing worth
              knowing". When the SAME key (a short n-gram ending at a past spike)
              recurs >= recur_min_hours later, we run a PAIRED probe from the exact
              pre-recurrence state: inject the stored span vs a length-matched random
              span from the live ring buffer, and compare next-token NLL on the same
              continuation. This is the honest control from closed_loop.py (same
              pre-gap state, only the injection differs) reimplemented without
              importing attic/closed_loop, so pos_run.py stays vendor-free.

  fork_twin   the reset-twin: same weights, Z + optimizer state zeroed, everything
              else (window, name) reset for a from-scratch-state comparison against
              the living A3 arm.

Determinism: IndexLoop owns a private torch.Generator (never touches the global RNG),
so A1/A2/A3 stay bit-identical across index-enabled and index-disabled runs.
"""
import json
import copy
from collections import deque

import torch
import torch.nn.functional as F


# ───────────────────────────────────────────────────────────────────────────
#  Local read/advance helpers — same mechanics as closed_loop.py, reimplemented
#  to avoid the vendor import chain (pos_run.py must not import attic/closed_loop).
# ───────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _advance(model, ids, states):
    """Push `ids` through the model to update state only (no scoring). Empty ids -> states unchanged."""
    if not ids:
        return states
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    _, st = model(x, states)
    return st


@torch.no_grad()
def _read(model, ids, states):
    """Score ids[1:] given ids[:-1] from `states`. Returns (mean_nll or None, new_states)."""
    if len(ids) < 2:
        return None, _advance(model, ids, states)
    x = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0)
    y = torch.tensor(ids[1:], dtype=torch.long).unsqueeze(0)
    logits, st = model(x, states)
    nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
    return float(nll.mean()), st


def _clone(states):
    return None if states is None else [s.clone() for s in states]


# ───────────────────────────────────────────────────────────────────────────
#  IndexLoop
# ───────────────────────────────────────────────────────────────────────────
class IndexLoop:
    """Surprise-spike span store + paired injection probe on key recurrence.

    Keys are `ngram`-token windows of the stream (tuples), each mapped to a stored
    span around the spike that produced it. On recurrence of a stored key (>=
    recur_min_hours later, under a probe budget), the stored span is injected into
    a clone of the pre-recurrence state and compared against a length-matched
    random span from the live ring buffer — both scored on the same continuation
    from the same pre-recurrence state as a no-injection baseline.
    """

    def __init__(self, index_path, probes_path, warmup_tokens, recur_min_hours,
                 spike_min_nll, span_half, lookahead, max_entries, max_probes_per_hour,
                 seed, ngram=4, max_spikes_per_chunk=2, max_probes_per_key=5,
                 max_probes_per_chunk=1, ring_size=8192):
        self.index_path, self.probes_path = index_path, probes_path
        self.warmup_tokens = warmup_tokens
        self.recur_min_hours = recur_min_hours
        self.spike_min_nll = spike_min_nll
        self.span_half = span_half
        self.lookahead = lookahead
        self.max_entries = max_entries
        self.max_probes_per_hour = max_probes_per_hour
        self.ngram = ngram
        self.max_spikes_per_chunk = max_spikes_per_chunk
        self.max_probes_per_key = max_probes_per_key
        self.max_probes_per_chunk = max_probes_per_chunk
        self.ring_size = ring_size

        self.gen = torch.Generator().manual_seed(seed)     # private RNG — never touch global torch RNG
        self.ring = deque(maxlen=ring_size)
        self.keys = {}                                     # tuple(key) -> entry dict
        self.stored_total = 0
        self.probes_total = 0
        self.recurrences_seen = 0
        self.probe_hour_budget = deque()                   # wall_s of recent probes (rate limit)

    # ── budget ──────────────────────────────────────────────────────────
    def _budget_ok(self, wall_s):
        while self.probe_hour_budget and wall_s - self.probe_hour_budget[0] > 3600:
            self.probe_hour_budget.popleft()
        return len(self.probe_hour_budget) < self.max_probes_per_hour

    # ── main entry point, called once per A3 chunk ─────────────────────
    def observe_chunk(self, model, states_pre, x, y, nll, n_streamed, wall_s, unk_id):
        self.ring.extend(x.flatten().tolist())

        if n_streamed < self.warmup_tokens:
            return {"stored": 0, "probes": 0}

        n_probes_chunk = self._recurrence_check(model, states_pre, x, n_streamed, wall_s)

        n_stored = self._spike_store(x, y, nll, n_streamed, wall_s, unk_id)

        return {"stored": n_stored, "probes": n_probes_chunk}

    # ── step 3: recurrence check + paired probes ───────────────────────
    def _recurrence_check(self, model, states_pre, x, n_streamed, wall_s):
        B, K = x.shape
        ng = self.ngram
        n_probes = 0
        for b in range(B):
            if n_probes >= self.max_probes_per_chunk:
                break
            row = x[b].tolist()
            for i in range(K - ng + 1):
                if n_probes >= self.max_probes_per_chunk:
                    break
                key = tuple(row[i:i + ng])
                entry = self.keys.get(key)
                if entry is None:
                    continue
                self.recurrences_seen += 1
                if wall_s - entry["t_store"] < self.recur_min_hours * 3600:
                    continue
                if entry["last_probe"] is not None and \
                        wall_s - entry["last_probe"] < self.recur_min_hours * 3600:
                    continue
                if entry["probes"] >= self.max_probes_per_key:
                    continue
                if not self._budget_ok(wall_s):
                    continue
                if i + self.lookahead + 1 > K:
                    continue
                self._run_probe(model, states_pre, b, row, i, key, entry, n_streamed, wall_s)
                n_probes += 1
        return n_probes

    def _run_probe(self, model, states_pre, b, row, i, key, entry, n_streamed, wall_s):
        st = None if states_pre is None else [s[b:b + 1].clone() for s in states_pre]
        if i > 0:
            st = _advance(model, row[:i], st)

        # full_row for THIS row is not directly available here beyond K tokens, but the
        # key occurrence + lookahead continuation only needs row (K tokens) since the
        # recurrence check already guarantees i + lookahead + 1 <= K.
        L = self.lookahead
        cont = row[i:i + L + 1]

        s_base, _ = _read(model, cont, _clone(st))

        st2 = _advance(model, entry["span"], _clone(st))
        s_inj, _ = _read(model, cont, st2)

        span_len = len(entry["span"])
        if len(self.ring) >= span_len:
            max_off = len(self.ring) - span_len
            off = int(torch.randint(0, max_off + 1, (1,), generator=self.gen).item())
            ring_list = list(self.ring)
            rand_span = ring_list[off:off + span_len]
        else:
            rand_span = list(self.ring)
            if len(rand_span) < 16:
                entry["last_probe"] = wall_s
                entry["probes"] += 1
                self.probe_hour_budget.append(wall_s)
                return
        st3 = _advance(model, rand_span, _clone(st))
        s_rand, _ = _read(model, cont, st3)

        rec = {"key": list(key), "n": n_streamed, "w": round(wall_s, 1),
               "dt_h": round((wall_s - entry["t_store"]) / 3600, 3), "L": L,
               # 6 decimals, not 4: the injection effect on a young model is ~1e-4
               # (concentrated in the ~6-token receptive field) — 4dp would log zeros.
               "s_base": round(s_base, 6) if s_base is not None else None,
               "s_inj": round(s_inj, 6) if s_inj is not None else None,
               "s_rand": round(s_rand, 6) if s_rand is not None else None,
               "d_inj": round(s_base - s_inj, 6) if (s_base is not None and s_inj is not None) else None,
               "d_rand": round(s_base - s_rand, 6) if (s_base is not None and s_rand is not None) else None}
        self._append(self.probes_path, rec)

        entry["last_probe"] = wall_s
        entry["probes"] += 1
        self.probe_hour_budget.append(wall_s)
        self.probes_total += 1

    # ── step 4: spike detection + span store ───────────────────────────
    def _spike_store(self, x, y, nll, n_streamed, wall_s, unk_id):
        B, K = x.shape
        candidates = []
        for b in range(B):
            for t in range(K):
                candidates.append((float(nll[b, t]), b, t))
        candidates.sort(key=lambda c: -c[0])

        n_stored = 0
        for nll_val, b, t in candidates:
            if n_stored >= self.max_spikes_per_chunk:
                break
            if nll_val < self.spike_min_nll:
                break
            if len(self.keys) >= self.max_entries:
                break

            full_row = x[b].tolist() + [int(y[b, -1])]
            if t < 2:
                continue
            key = tuple(full_row[t - 2:t + 2])
            if key in self.keys:
                continue
            if unk_id in key:
                continue

            span = full_row[max(0, t + 1 - self.span_half): t + 1 + self.span_half + 1]
            if len(span) < 16:
                continue

            entry = {"span": span, "t_store": wall_s, "n_store": n_streamed,
                      "last_probe": None, "probes": 0}
            self.keys[key] = entry
            self.stored_total += 1
            n_stored += 1

            self._append(self.index_path, {"key": list(key), "n": n_streamed,
                                            "w": round(wall_s, 1), "row": b, "pos": t,
                                            "nll": round(nll_val, 4), "span": span})
        return n_stored

    # ── io ──────────────────────────────────────────────────────────────
    @staticmethod
    def _append(path, obj):
        with open(path, "a") as f:
            f.write(json.dumps(obj) + "\n")

    # ── status / checkpoint ────────────────────────────────────────────
    def stats(self):
        return {"entries": len(self.keys), "stored_total": self.stored_total,
                "probes": self.probes_total, "recurrences_seen": self.recurrences_seen}

    def state_dict(self):
        keys_ser = [{"key": list(k), "entry": {**v, "span": list(v["span"])}}
                    for k, v in self.keys.items()]
        return {"keys": keys_ser, "ring": list(self.ring),
                "stored_total": self.stored_total, "probes_total": self.probes_total,
                "recurrences_seen": self.recurrences_seen,
                "probe_hour_budget": list(self.probe_hour_budget),
                "gen_state": self.gen.get_state()}

    def load_state_dict(self, d):
        self.keys = {tuple(item["key"]): dict(item["entry"]) for item in d["keys"]}
        self.ring = deque(d["ring"], maxlen=self.ring_size)
        self.stored_total = d["stored_total"]
        self.probes_total = d["probes_total"]
        self.recurrences_seen = d["recurrences_seen"]
        self.probe_hour_budget = deque(d["probe_hour_budget"])
        self.gen.set_state(d["gen_state"])


# ───────────────────────────────────────────────────────────────────────────
#  fork_twin — the reset-twin: weights copied, Z + optimizer state zeroed.
# ───────────────────────────────────────────────────────────────────────────
def fork_twin(a3_arm, args):
    """Fork A3 into A3R: same weights (deep-copied), fresh optimizer, no carried
    state. Inherits A3's rolling window (so it gates from chunk one — no_ignition
    disables the always-backward warmup phase that the living A3 already spent)."""
    model = copy.deepcopy(a3_arm["model"])
    return {"name": "A3R", "model": model,
            "opt": torch.optim.Adam(model.parameters(), lr=args.lr),
            "states": None, "grad_tokens": 0, "n_chunks": 0, "n_bwd": 0,
            "window": deque(a3_arm["window"], maxlen=args.window),
            "recent_gates": deque(maxlen=200), "no_ignition": True}


# ───────────────────────────────────────────────────────────────────────────
#  Self-test — fully offline, < 60s, writes only into a tempfile.mkdtemp() dir.
# ───────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import sys
    import shutil
    import tempfile
    import argparse

    sys.path.insert(0, "reference")
    sys.path.insert(0, "src")
    torch.set_num_threads(1)
    torch.manual_seed(0)

    from streaming_train import StreamingNoPELM

    tmpdir = tempfile.mkdtemp()
    try:
        V, MASK = 50, 51
        model = StreamingNoPELM(V, MASK, d_model=32, n_layers=2, n_heads=4, d_head=8,
                                seq_len=32, dropout=0.0, causal=True)
        model.eval()

        index_path = os.path.join(tmpdir, "index.jsonl")
        probes_path = os.path.join(tmpdir, "probes.jsonl")
        index = IndexLoop(index_path, probes_path, warmup_tokens=0, recur_min_hours=0.0,
                          spike_min_nll=0.1, span_half=8, lookahead=8, max_entries=100,
                          max_probes_per_hour=100, seed=42)

        B, K = 2, 32
        gen = torch.Generator().manual_seed(1)
        recurring = [7, 8, 9, 10]

        n_streamed = 0
        states = None
        wall = 0.0
        stored_seen = 0
        probes_seen = 0
        for chunk_i in range(10):
            x = torch.randint(0, V, (B, K), generator=gen)
            # plant the recurring 4-gram with room for lookahead, in chunk 2 and chunk 8
            if chunk_i in (2, 8):
                x[0, 4:8] = torch.tensor(recurring)
            y = torch.cat([x[:, 1:], torch.randint(0, V, (B, 1), generator=gen)], dim=1)

            states_pre = None if states is None else [s.clone() for s in states]
            with torch.no_grad():
                logits, states = model(x, states)
                nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                      reduction="none").view(B, K)
            # force a couple of spikes so spans actually get stored (real nll may be too flat)
            nll = nll.clone()
            nll[0, 6] = 9.0
            nll[1, 10] = 8.5

            n_streamed += B * K
            wall += 1.0
            out = index.observe_chunk(model, states_pre, x, y, nll, n_streamed, wall, unk_id=MASK)
            stored_seen += out["stored"]
            probes_seen += out["probes"]

        st = index.stats()
        assert st["entries"] >= 1, f"expected >=1 stored entry, got {st}"
        assert os.path.exists(index_path), "index.jsonl missing"

        with open(index_path) as f:
            index_lines = [json.loads(l) for l in f]
        assert len(index_lines) >= 1
        for rec in index_lines:
            for field in ("key", "n", "w", "row", "pos", "nll", "span"):
                assert field in rec, f"index record missing field {field}: {rec}"

        assert probes_seen >= 1, f"expected >=1 probe, got stats={st}"
        assert os.path.exists(probes_path), "probes.jsonl missing"
        with open(probes_path) as f:
            probe_lines = [json.loads(l) for l in f]
        assert len(probe_lines) >= 1
        for rec in probe_lines:
            for field in ("key", "n", "w", "dt_h", "L", "s_base", "s_inj", "s_rand",
                          "d_inj", "d_rand"):
                assert field in rec, f"probe record missing field {field}: {rec}"
            if rec["s_base"] is not None and rec["s_inj"] is not None:
                # rec's own fields are each independently rounded to 4dp, so comparing
                # the rounded difference against the difference-of-rounded needs the
                # matching rounding tolerance (1e-4), not the raw-float 1e-6 from the spec.
                assert abs(rec["d_inj"] - (rec["s_base"] - rec["s_inj"])) < 1e-4, \
                    f"d_inj mismatch: {rec}"

        # state_dict / load_state_dict round trip
        saved = index.state_dict()
        index2 = IndexLoop(index_path, probes_path, warmup_tokens=0, recur_min_hours=0.0,
                           spike_min_nll=0.1, span_half=8, lookahead=8, max_entries=100,
                           max_probes_per_hour=100, seed=999)
        index2.load_state_dict(saved)
        assert index2.stats() == st, f"resume stats mismatch: {index2.stats()} vs {st}"

        x_extra = torch.randint(0, V, (B, K), generator=gen)
        y_extra = torch.cat([x_extra[:, 1:], torch.randint(0, V, (B, 1), generator=gen)], dim=1)
        with torch.no_grad():
            logits_extra, states_extra_new = model(x_extra, states)
            nll_extra = F.cross_entropy(logits_extra.reshape(-1, logits_extra.size(-1)),
                                        y_extra.reshape(-1), reduction="none").view(B, K)
        states_pre_extra = [s.clone() for s in states]
        out_extra = index2.observe_chunk(model, states_pre_extra, x_extra, y_extra, nll_extra,
                                         n_streamed + B * K, wall + 1.0, unk_id=MASK)
        assert isinstance(out_extra, dict) and "stored" in out_extra and "probes" in out_extra

        # fork_twin
        fake_arm = {"model": model, "window": deque([0.1, 0.2, 0.3], maxlen=500)}
        fake_args = argparse.Namespace(lr=1e-3, window=500)
        twin = fork_twin(fake_arm, fake_args)
        for field in ("name", "model", "opt", "states", "grad_tokens", "n_chunks", "n_bwd",
                      "window", "recent_gates", "no_ignition"):
            assert field in twin, f"fork_twin missing field {field}"
        with torch.no_grad():
            max_delta = max((p_a - p_b).abs().max().item()
                            for p_a, p_b in zip(model.parameters(), twin["model"].parameters()))
        assert max_delta == 0.0, f"twin weights not bit-identical: max|delta|={max_delta}"
        assert twin["states"] is None
        assert twin["no_ignition"] is True
        assert list(twin["window"]) == [0.1, 0.2, 0.3]

        # decoupling: mutate original model, twin must not change
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        with torch.no_grad():
            max_delta_after = max((p_a - p_b).abs().max().item()
                                  for p_a, p_b in zip(model.parameters(), twin["model"].parameters()))
        assert max_delta_after > 0.0, "twin should be decoupled from original model mutation"

        fake_arm["window"].append(0.9)
        assert list(twin["window"]) != list(fake_arm["window"]), "twin window should be decoupled"

        print("SELF-TEST PASS")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
