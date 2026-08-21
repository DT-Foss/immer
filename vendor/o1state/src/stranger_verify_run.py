#!/usr/bin/env python3
"""
P60 — stranger verification: a stranger on different hardware verifies a
knowledge file by pure recomputation, from the file + its coordinates alone
(no creator state beyond the file).
============================================================================
Registered clauses (analysis/PREDICTIONS.md, "P60 — stranger verification"):
  (a) CROSS-ISA VERIFICATION: a file harvested on one ISA verifies 10/10
      entries bit-exact on the other ISA from coordinates alone; the reverse
      direction likewise.
  (b) CONSENSUS FOR FREE: two independent replays of the same entries agree
      bit-exactly with each other — majority verification needs no trust,
      only determinism.
  Falsifier: any cross-ISA mismatch localises to a named layer (tokenizer,
  stream, arithmetic) — that layer becomes the spec's normative anchor.

This harness is the VERIFIER side. It recomputes each sampled entry twice
from independent fresh replay objects and checks:
  * verify   := recomputed fragment == the fragment stored in the file  (a)
  * consensus:= replay#1 fragment == replay#2 fragment                   (b)

Three substrates / directions:
  --direction 1 (default): the pixel-world harvest results/vizdoom_knowledge.jsonl
      (coordinates: lane_seed, episode, frame, offset). Replay mechanics are
      IMPORTED from src/vizdoom_run.py (make_game / action_for / tokenize_frame)
      so the stranger uses the identical engine factory the creator used —
      identical by construction, which is the whole point.
  --direction 2: the text keyed-file results/keyed_file*.json (P55; coordinates:
      doc_coord). Replay re-instantiates po.C4Stream(skip_docs=...) and searches
      for the stored span, mirroring src/knowledge_file_run.py:provenance_replay.
      Direction 2 is wired as a CLI path here; it runs on the other ISA later.
  --direction 3: the LIVE-CAUSAL graph (--file = a src/livecausal LiveStore
      directory; coordinates: segment_sha + record_idx, MVP-2's derivation
      shape). Two edge classes, sampled together up to --n total:
        (i)  BASE edges — verify := store.verify(segment_sha) (the sealed
             segment's own sha256 recheck) AND the record's trigger_key/
             outcome_key fields are present and consistent with the edge
             under test; consensus := two independent LiveStore mounts over
             the same dir agree on the loaded record content.
        (ii) INFERRED edges — verify := re-derive the chain from the CITED
             (segment_sha, idx) records alone (walk trigger_key ->
             outcome_key hop by hop, exactly as MVP-2's
             test_derivation_is_recomputable does) and check it reproduces
             the edge's from_key/to_key/depth; consensus := two independent
             re-derivation passes (fresh LiveStore mounts, fresh record
             loads) agree bit-for-bit. NEITHER check calls LiveGraph.query()
             or reads LiveGraph.inferred_edges() as ground truth for the
             chain itself — inferred_edges() is used only to SAMPLE which
             edges to test; the verifier does not trust the engine it is
             checking. An optional --with-corpus-replay flag additionally
             replays each base record's doc_coord against the source corpus
             (direction-2-style span replay); off by default, since real
             corpus coordinates only exist once P72 graphs land.

WHY NOT results/pos_index.jsonl (the 40h ARM POS harvest): inspected, and it
carries NO stream document coordinate — only a global token position `n`
(n_streamed), plus `row`/`pos` (in-batch/in-chunk position), `key`, and the
already-tokenised `span`. A stranger cannot re-instantiate the C4 stream from
`n` without tokenising every prior document (i.e. reconstructing the whole
stream), so `n` is not a verification coordinate. The missing field is an
explicit `doc_coord` (documents consumed at harvest time), which the text
harvest src/knowledge_file_run.py DOES record but the POS index does not. This
is recorded in results/STRANGER_VERIFY_NOTES.md; direction 1 therefore uses the
pixel harvest, which carries full coordinates.

Writes results/stranger_verify.json. Exit code always 0 (reports, not gates).

Usage:
  python src/stranger_verify_run.py --smoke                 # dir 1, N=2, local
  python src/stranger_verify_run.py --direction 1 --n 10
  python src/stranger_verify_run.py --direction 2 --file results/keyed_file.json --n 10
"""
import argparse
import hashlib
import json
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


# ─────────────────────────────────────────────────────────────────────────
#  helpers
# ─────────────────────────────────────────────────────────────────────────
def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────
#  Direction 1 — the pixel world (vizdoom), coordinates fully in the file
# ─────────────────────────────────────────────────────────────────────────
def vizdoom_replay_fragment(entry, skip=4):
    """Recompute the stored fragment for one entry from a FRESH engine, using
    the creator's own factory (imported from vizdoom_run). Returns the token
    list of length len(entry['tokens']), or None if the frame is unreachable.

    This is src/vizdoom_run.py:replay_entry, but it RETURNS THE FRAGMENT rather
    than a bool, so the caller can (a) compare to the file and (b) compare two
    independent replays to each other."""
    import vizdoom_run as vr  # noqa: E402  (engine factory + tokeniser)

    g = vr.make_game(entry["lane_seed"] * 10000 + entry["episode"])
    rng_cache = {}
    toks = None
    try:
        for fi in range(entry["frame"] + 1):
            if g.is_episode_finished():
                break
            st = g.get_state()
            if st is None:
                break
            if fi == entry["frame"]:
                toks = vr.tokenize_frame(st.screen_buffer)
                break
            a = vr.action_for(entry["lane_seed"], entry["episode"], fi, rng_cache)
            g.make_action(a, skip)
    finally:
        g.close()
    if toks is None:
        return None
    off, span = entry["offset"], entry["tokens"]
    ring = toks + toks  # frames wrap (offset+len may exceed one frame)
    return ring[off:off + len(span)]


def verify_direction1(path, n_samples, seed, skip=4):
    entries = read_jsonl(path)
    rng = random.Random(seed)
    picks = rng.sample(entries, min(n_samples, len(entries)))
    checks = []
    for e in picks:
        span = e["tokens"]
        frag1 = vizdoom_replay_fragment(e, skip=skip)
        frag2 = vizdoom_replay_fragment(e, skip=skip)   # independent fresh engine
        verify = (frag1 is not None) and (list(frag1) == list(span))
        consensus = (frag1 is not None) and (frag2 is not None) and (list(frag1) == list(frag2))
        coords = {"lane_seed": e["lane_seed"], "episode": e["episode"],
                  "frame": e["frame"], "offset": e["offset"]}
        checks.append({
            "coords": coords,
            "span_len": len(span),
            "found": bool(verify),
            "consensus": bool(consensus),
        })
        print(f"[dir1] {coords} span_len {len(span)} -> "
              f"{'VERIFIED' if verify else 'MISMATCH'} / "
              f"consensus {'OK' if consensus else 'FAIL'}", flush=True)
    return checks


# ─────────────────────────────────────────────────────────────────────────
#  Direction 2 — the text keyed-file (C4), coordinate = doc_coord
# ─────────────────────────────────────────────────────────────────────────
def c4_replay_fragment(entry, stoi, unk, back_docs=1500, fwd_docs=800):
    """Re-instantiate the C4 stream near the entry's doc_coord and search for
    the stored span, mirroring knowledge_file_run.provenance_replay. Returns
    (found_bool, matched_fragment_or_None). Two calls build two independent
    streams -> consensus."""
    import portable_organism as po  # noqa: E402

    coord = entry["doc_coord"]
    lo = max(0, coord - back_docs)
    stream = po.C4Stream(stoi, unk, skip_docs=lo)
    span = entry["tokens"]
    toks = []
    found_frag = None
    while stream.docs < coord + fwd_docs:
        toks.extend(stream.next_block())
        if len(toks) > 4_000_000:
            break
        window_lo = max(0, len(toks) - po.CHUNK * po.BATCH * 40)
        for i in range(window_lo, len(toks) - len(span) + 1):
            if toks[i:i + len(span)] == span:
                found_frag = toks[i:i + len(span)]
                break
        if found_frag is not None:
            break
    return (found_frag is not None), found_frag


def load_text_entries(path):
    """Resolve direction-2 entries from either a .jsonl of entries directly, or
    a .json scoring artifact that carries them inline ('entries') or names the
    producer .jsonl ('file_path'/'entries_path'). Mirrors the P47/P55 layout,
    where knowledge_file_run.freeze_file writes producer.knowledge.jsonl next to
    the scoring json. Returns (entries, resolved_path)."""
    if path.endswith(".jsonl"):
        return read_jsonl(path), path
    with open(path) as f:
        payload = json.load(f)
    entries = payload.get("entries")
    if entries is not None:
        return entries, path
    cand = payload.get("file_path") or payload.get("entries_path")
    # try the named path, then a same-dir *.knowledge.jsonl fallback
    tries = [cand] if cand else []
    tries += [os.path.join(os.path.dirname(path), b)
              for b in ("producer.knowledge.jsonl", "keyed_file.jsonl")]
    for t in tries:
        if t and os.path.exists(t):
            return read_jsonl(t), t
    raise FileNotFoundError(
        f"direction 2: {path} carries no inline 'entries' and its producer "
        f"jsonl (tried {tries}) is not on this host — run on the host that "
        f"holds the producer file.")


def verify_direction2(path, n_samples, seed):
    import portable_organism as po  # noqa: E402

    entries, resolved = load_text_entries(path)
    if resolved != path:
        print(f"[dir2] entries from {resolved}", flush=True)
    _, stoi, unk, _, _ = po.get_vocab()
    rng = random.Random(seed)
    picks = rng.sample(entries, min(n_samples, len(entries)))
    checks = []
    for e in picks:
        found1, frag1 = c4_replay_fragment(e, stoi, unk)
        found2, frag2 = c4_replay_fragment(e, stoi, unk)
        verify = found1 and (list(frag1) == list(e["tokens"]))
        consensus = found1 and found2 and (list(frag1) == list(frag2))
        coords = {"doc_coord": e["doc_coord"], "n_chunk": e.get("n_chunk")}
        checks.append({
            "coords": coords,
            "span_len": len(e["tokens"]),
            "found": bool(verify),
            "consensus": bool(consensus),
        })
        print(f"[dir2] coord {e['doc_coord']:,} span_len {len(e['tokens'])} -> "
              f"{'VERIFIED' if verify else 'MISMATCH'} / "
              f"consensus {'OK' if consensus else 'FAIL'}", flush=True)
    return checks


# ─────────────────────────────────────────────────────────────────────────
#  Direction 3 — the LIVE-CAUSAL graph (src/livecausal), coordinates =
#  (segment_sha, record_idx); engine-distrustful (never reads the engine's
#  own cache/query surface as ground truth for the thing being checked).
# ─────────────────────────────────────────────────────────────────────────
def _load_record(store, sha, idx):
    """One record by (segment_sha, idx), via LiveStore's public iter_records
    (a fresh scan of the segment file each call — no internal-index reuse)."""
    for seg_sha, rec_idx, record in store.iter_records(sha):
        if rec_idx == idx:
            return record
    return None


def _base_edge_fragment(store, from_key, to_key, citation):
    """Re-derive one base edge from its (segment_sha, idx) citation alone:
    load the record fresh, confirm it actually encodes from_key -> to_key.
    Returns the record dict, or None if unreadable / inconsistent."""
    sha, idx = citation
    record = _load_record(store, sha, idx)
    if record is None:
        return None
    if record.get("trigger_key") != from_key or record.get("outcome_key") != to_key:
        return None
    return record


def _inferred_edge_rederive(store, edge):
    """Re-derive an inferred edge's chain from ONLY its cited derivation —
    walks trigger_key -> outcome_key hop by hop over fresh record loads,
    exactly as MVP-2's test_derivation_is_recomputable does. Returns a
    canonical fragment dict {from_key, to_key, depth, derivation} if the
    cited records actually chain from edge['from_key'] to edge['to_key']
    with no repeats, or None if re-derivation fails (broken chain, cycle,
    unreadable record, or a landing point that disagrees with the edge).
    This never touches LiveGraph's inferred-edge cache or query() — only
    LiveStore.iter_records on the exact (sha, idx) pairs the edge cites."""
    derivation = edge["derivation"]
    if len(derivation) != edge["depth"]:
        return None
    cursor = edge["from_key"]
    seen_keys = {cursor}
    for sha, idx in derivation:
        record = _load_record(store, sha, idx)
        if record is None:
            return None
        if record.get("trigger_key") != cursor:
            return None
        nxt = record.get("outcome_key")
        if nxt is None or nxt in seen_keys:
            return None  # missing key field, or a cycle in the re-walked chain
        seen_keys.add(nxt)
        cursor = nxt
    if cursor != edge["to_key"]:
        return None
    return {
        "from_key": edge["from_key"],
        "to_key": edge["to_key"],
        "depth": edge["depth"],
        "derivation": [list(p) for p in derivation],
    }


def _sample_edges(store_dir, n_samples, seed):
    """Sample up to n_samples edges from the graph, split across the two
    classes (base, inferred). LiveGraph is mounted ONLY to enumerate
    candidates (base_edge_citations / inferred_edges) -- picking WHICH
    edges to test is not the thing under test; the re-derivation below
    never consults this mount again."""
    from livecausal.infer import LiveGraph  # noqa: E402
    from livecausal.store import LiveStore  # noqa: E402

    sampling_store = LiveStore(store_dir)
    graph = LiveGraph(store_dir)

    base_pool = []
    for from_key, targets in sorted(graph._base_edges.items()):
        for to_key, citations in sorted(targets.items()):
            for citation in citations:
                base_pool.append(("base", from_key, to_key, list(citation)))

    inferred_pool = [("inferred", e) for e in graph.inferred_edges()]

    rng = random.Random(seed)
    n_base_target = n_samples // 2
    n_inferred_target = n_samples - n_base_target
    base_picks = rng.sample(base_pool, min(n_base_target, len(base_pool)))
    inferred_picks = rng.sample(inferred_pool, min(n_inferred_target, len(inferred_pool)))

    # Backfill from whichever pool has slack, so a small population in one
    # class doesn't silently shrink the total below n_samples.
    shortfall = n_samples - len(base_picks) - len(inferred_picks)
    if shortfall > 0:
        remaining_base = [p for p in base_pool if p not in base_picks]
        extra_base = rng.sample(remaining_base, min(shortfall, len(remaining_base)))
        base_picks += extra_base
        shortfall -= len(extra_base)
    if shortfall > 0:
        remaining_inferred = [p for p in inferred_pool if p not in inferred_picks]
        extra_inferred = rng.sample(remaining_inferred, min(shortfall, len(remaining_inferred)))
        inferred_picks += extra_inferred

    return sampling_store, base_picks, inferred_picks


def _replay_doc_coord(store, record, back_docs=1500, fwd_docs=800):
    """Optional (--with-corpus-replay) span replay against the source
    corpus, direction-2 style, for a base record's doc_coord. Off by
    default: real corpus coordinates only exist once P72 builder-loop
    graphs land (synthetic/bench records carry doc_coord=0 placeholders,
    not real stream positions)."""
    import portable_organism as po  # noqa: E402

    coord = record.get("doc_coord")
    tokens = record.get("tokens")
    if coord is None or not tokens:
        return None  # nothing corpus-replayable on this record
    _, stoi, unk, _, _ = po.get_vocab()
    lo = max(0, coord - back_docs)
    stream = po.C4Stream(stoi, unk, skip_docs=lo)
    toks = []
    found_frag = None
    while stream.docs < coord + fwd_docs:
        toks.extend(stream.next_block())
        if len(toks) > 4_000_000:
            break
        window_lo = max(0, len(toks) - po.CHUNK * po.BATCH * 40)
        for i in range(window_lo, len(toks) - len(tokens) + 1):
            if toks[i:i + len(tokens)] == tokens:
                found_frag = toks[i:i + len(tokens)]
                break
        if found_frag is not None:
            break
    return (found_frag is not None) and (list(found_frag) == list(tokens))


def verify_direction3(store_dir, n_samples, seed, with_corpus_replay=False):
    from livecausal.store import LiveStore  # noqa: E402

    sampling_store, base_picks, inferred_picks = _sample_edges(store_dir, n_samples, seed)
    checks = []

    for kind, from_key, to_key, citation in base_picks:
        # Two independent LiveStore mounts over the same dir -> two
        # independent record loads -> consensus, mirroring dir1/dir2's
        # "two fresh replay objects" pattern.
        store1 = LiveStore(store_dir)
        store2 = LiveStore(store_dir)
        rec1 = _base_edge_fragment(store1, from_key, to_key, citation)
        rec2 = _base_edge_fragment(store2, from_key, to_key, citation)
        seg_sha = citation[0]
        segment_verified = store1.verify(seg_sha)
        verify = segment_verified and (rec1 is not None)
        consensus = (rec1 is not None) and (rec2 is not None) and (rec1 == rec2)
        corpus_replay = None
        if with_corpus_replay and rec1 is not None:
            corpus_replay = _replay_doc_coord(store1, rec1)
        coords = {"segment_sha": seg_sha, "record_idx": citation[1]}
        checks.append({
            "class": "base",
            "coords": coords,
            "edge": {"from_key": from_key, "to_key": to_key},
            "found": bool(verify),
            "consensus": bool(consensus),
            "segment_sha_verified": bool(segment_verified),
            "corpus_replay": corpus_replay,
        })
        print(f"[dir3/base] {from_key} -> {to_key} @ {coords} -> "
              f"{'VERIFIED' if verify else 'MISMATCH'} / "
              f"consensus {'OK' if consensus else 'FAIL'}", flush=True)

    for kind, edge in inferred_picks:
        store1 = LiveStore(store_dir)
        store2 = LiveStore(store_dir)
        frag1 = _inferred_edge_rederive(store1, edge)
        frag2 = _inferred_edge_rederive(store2, edge)
        verify = frag1 is not None
        consensus = (frag1 is not None) and (frag2 is not None) and (frag1 == frag2)
        coords = {
            "derivation": [list(p) for p in edge["derivation"]],
            "depth": edge["depth"],
        }
        checks.append({
            "class": "inferred",
            "coords": coords,
            "edge": {"from_key": edge["from_key"], "to_key": edge["to_key"], "depth": edge["depth"]},
            "found": bool(verify),
            "consensus": bool(consensus),
            "segment_sha_verified": None,
            "corpus_replay": None,
        })
        print(f"[dir3/inferred] {edge['from_key']} -> {edge['to_key']} "
              f"depth={edge['depth']} -> "
              f"{'VERIFIED' if verify else 'MISMATCH'} / "
              f"consensus {'OK' if consensus else 'FAIL'}", flush=True)

    return checks


# ─────────────────────────────────────────────────────────────────────────
#  scoring + runner
# ─────────────────────────────────────────────────────────────────────────
def score(checks, n_target):
    n = len(checks)
    n_found = sum(1 for c in checks if c["found"])
    n_cons = sum(1 for c in checks if c["consensus"])
    return {
        "n_sampled": n,
        "n_verified": n_found,
        "n_consensus": n_cons,
        # (a): the registered bar is 10/10 — pass iff every sampled entry
        # verified AND we sampled at least the target count.
        "p60a_verified_all": bool(n_found == n and n >= n_target),
        "p60a_fraction": f"{n_found}/{n}",
        # (b): two independent replays agree on every sampled entry.
        "p60b_consensus_all": bool(n_cons == n and n > 0),
        "p60b_fraction": f"{n_cons}/{n}",
    }


def main():
    ap = argparse.ArgumentParser(description="P60 stranger verification harness")
    ap.add_argument("--direction", type=int, default=1, choices=(1, 2, 3),
                    help="1 = pixel world (vizdoom_knowledge.jsonl); "
                         "2 = text keyed-file (keyed_file*.json); "
                         "3 = LIVE-CAUSAL graph (a src/livecausal LiveStore dir)")
    ap.add_argument("--file", default=None,
                    help="override the substrate file (direction 3: a LiveStore directory)")
    ap.add_argument("--n", type=int, default=10, help="entries to sample")
    ap.add_argument("--seed", type=int, default=60, help="sampling seed")
    ap.add_argument("--skip", type=int, default=4, help="dir1 frame skip")
    ap.add_argument("--with-corpus-replay", action="store_true",
                    help="direction 3 only: also replay each sampled base "
                         "record's doc_coord against the source corpus "
                         "(off by default -- real corpus coords need a P72 graph)")
    ap.add_argument("--smoke", action="store_true",
                    help="direction 1, N=2, local self-test")
    ap.add_argument("--out", default=os.path.join("results", "stranger_verify.json"))
    args = ap.parse_args()

    direction = 1 if args.smoke else args.direction
    n_samples = 2 if args.smoke else args.n
    # target count for the (a) bar: registered 10; smoke relaxes to its N.
    n_target = 2 if args.smoke else (max(10, args.n) if args.n >= 10 else args.n)

    def report_absent(path, reason):
        """A missing substrate is an expected state (a chained run has not
        landed yet), not a crash: write a valid, machine-readable report and
        exit 0 so the orchestrator can read status == substrate_absent."""
        payload = {
            "generated_by": "src/stranger_verify_run.py",
            "prediction": "P60 — stranger verification",
            "direction": direction,
            "status": "substrate_absent",
            "substrate_file": os.path.basename(path),
            "reason": reason,
        }
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[skip] {reason}")
        print(f"wrote {args.out} (status=substrate_absent)")
        sys.exit(0)

    substrate_sha256 = None
    if direction == 1:
        path = args.file or os.path.join("results", "vizdoom_knowledge.jsonl")
        if not os.path.exists(path):
            report_absent(path, f"direction 1 substrate not found: {path}")
        try:
            import vizdoom  # noqa: F401
        except Exception as ex:
            report_absent(
                path,
                f"direction 1 needs vizdoom on THIS host (the verifier "
                f"recomputes frames through the engine): {ex}")
        substrate = os.path.basename(path)
        substrate_sha256 = file_sha256(path)
        checks = verify_direction1(path, n_samples, args.seed, skip=args.skip)
    elif direction == 2:
        path = args.file or os.path.join("results", "keyed_file.json")
        if not os.path.exists(path):
            report_absent(
                path,
                f"direction 2 substrate not found: {path} — the keyed file "
                f"(P55) is produced on the x86 runner; run on the host that "
                f"holds it.")
        substrate = os.path.basename(path)
        substrate_sha256 = file_sha256(path)
        try:
            checks = verify_direction2(path, n_samples, args.seed)
        except FileNotFoundError as ex:
            report_absent(path, str(ex))
    else:
        path = args.file
        if not path:
            report_absent(
                "(none given)",
                "direction 3 requires --file <LiveStore directory>")
        manifest_path = os.path.join(path, "manifest.json")
        if not os.path.isdir(path) or not os.path.exists(manifest_path):
            report_absent(
                path,
                f"direction 3 substrate not found: {path} is not a "
                f"LiveStore directory (no manifest.json) — run on the host "
                f"that holds the graph, or build one first (MVP-4/P72).")
        substrate = os.path.basename(os.path.normpath(path))
        # A LiveStore is a directory, not a single file: the manifest is
        # its content-addressed table of contents, so hash THAT rather than
        # a single substrate file (there isn't one -- segments are
        # per-record hashed already, which is what the checks below verify
        # individually via store.verify()).
        substrate_sha256 = file_sha256(manifest_path)
        checks = verify_direction3(
            path, n_samples, args.seed, with_corpus_replay=args.with_corpus_replay
        )

    scoring = score(checks, n_target)
    direction_label = {
        1: "pixel-world (vizdoom)",
        2: "text keyed-file (C4)",
        3: "LIVE-CAUSAL graph (src/livecausal)",
    }[direction]
    payload = {
        "generated_by": "src/stranger_verify_run.py",
        "prediction": "P60 — stranger verification",
        "direction": direction,
        "direction_label": direction_label,
        "substrate_file": substrate,
        "substrate_sha256": substrate_sha256,
        "smoke": bool(args.smoke),
        "sample_seed": args.seed,
        "checks": checks,
        "scoring": scoring,
    }
    if direction == 3:
        n_base = sum(1 for c in checks if c["class"] == "base")
        n_inferred = sum(1 for c in checks if c["class"] == "inferred")
        payload["dir3_class_counts"] = {"base": n_base, "inferred": n_inferred}
        payload["with_corpus_replay"] = bool(args.with_corpus_replay)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print("=" * 74)
    print(f"P60 stranger verification — direction {direction} "
          f"({payload['direction_label']}), substrate {substrate}")
    print(f"  (a) verified : {scoring['p60a_fraction']}  "
          f"pass={scoring['p60a_verified_all']}")
    print(f"  (b) consensus: {scoring['p60b_fraction']}  "
          f"pass={scoring['p60b_consensus_all']}")
    print(f"wrote {args.out}")
    sys.exit(0)


if __name__ == "__main__":
    main()
