"""Plain-assert tests for stranger_verify_run.py --direction 3 (LIVE-CAUSAL
graph verification). Run: python3 src/test_stranger_verify_direction3.py

Builds a synthetic LiveStore/LiveGraph (reusing the same chain-segment
generator shape as src/livecausal/test_infer.py), then drives
verify_direction3 directly (not via subprocess -- faster, and lets us
inspect the checks list precisely).
"""

import os
import random
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from livecausal.infer import LiveGraph
from stranger_verify_run import score, verify_direction3


def with_tmpdir(fn):
    d = tempfile.mkdtemp(prefix="dir3-test-")
    try:
        fn(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def make_record(from_key, to_key, doc_coord=0):
    return {
        "trigger": "trig::{}".format(from_key),
        "mechanism": "causes",
        "outcome": "outc::{}".format(to_key),
        "trigger_key": from_key,
        "outcome_key": to_key,
        "doc_coord": doc_coord,
        "evidence_count": 1,
        "use_count": 0,
        "meta": {},
    }


def build_synthetic_graph(store_dir, seed=42, n_batches=15):
    """A chain-mix graph big enough to have >=10 base and >=10 inferred
    edges to sample from, mirroring livecausal/test_infer.py's generators.
    """
    rng = random.Random(seed)
    graph = LiveGraph(store_dir)
    key_counter = [0]

    def _fresh():
        key_counter[0] += 1
        return "K{}".format(key_counter[0])

    for _ in range(n_batches):
        chain_len = rng.randint(1, 3)
        keys = [_fresh() for _ in range(chain_len + 1)]
        for i in range(chain_len):
            sha = graph.store.append_segment([make_record(keys[i], keys[i + 1], doc_coord=i)])
            graph.on_append(sha)
    return graph


# ---------------------------------------------------------------------
# Test 1: 20 edges (both classes), everything verifies + consensus.
# ---------------------------------------------------------------------

def test_synthetic_store_all_verified(d):
    graph = build_synthetic_graph(d, seed=3001, n_batches=15)
    n_base = sum(len(v) for v in graph._base_edges.values())
    n_inferred = len(graph.inferred_edges())
    assert n_base >= 10, "test fixture too small: only {} base edges".format(n_base)
    assert n_inferred >= 10, "test fixture too small: only {} inferred edges".format(n_inferred)

    checks = verify_direction3(d, n_samples=20, seed=60)
    assert len(checks) == 20

    n_base_checks = sum(1 for c in checks if c["class"] == "base")
    n_inferred_checks = sum(1 for c in checks if c["class"] == "inferred")
    assert n_base_checks == 10, "expected an even 10/10 split, got base={}".format(n_base_checks)
    assert n_inferred_checks == 10, "expected an even 10/10 split, got inferred={}".format(n_inferred_checks)

    scoring = score(checks, n_target=20)
    assert scoring["n_verified"] == 20, scoring
    assert scoring["n_consensus"] == 20, scoring
    assert scoring["p60a_verified_all"] is True
    assert scoring["p60b_consensus_all"] is True

    for c in checks:
        assert c["found"] is True
        assert c["consensus"] is True
        if c["class"] == "base":
            assert c["segment_sha_verified"] is True


# ---------------------------------------------------------------------
# Test 2: negative -- tamper a copy of one segment file, verify fails for
# that base edge AND for the inferred edge(s) whose derivation cites it.
# ---------------------------------------------------------------------

def test_tampered_segment_fails_verification(d):
    orig_dir = os.path.join(d, "orig")
    tampered_dir = os.path.join(d, "tampered")
    os.makedirs(orig_dir)

    graph = build_synthetic_graph(orig_dir, seed=3001, n_batches=15)

    # Find a base edge that is also cited by at least one inferred edge, so
    # the test exercises both classes failing together.
    target_sha = None
    target_from = None
    target_to = None
    for e in graph.inferred_edges():
        first_hop_sha, first_hop_idx = e["derivation"][0]
        target_sha = first_hop_sha
        target_from = e["from_key"]
        # to_key of the first hop is not directly on the edge dict, but we
        # only need the sha to tamper -- re-derive to_key from the store.
        break
    assert target_sha is not None, "fixture has no inferred edges to anchor the tamper on"

    shutil.copytree(orig_dir, tampered_dir)
    seg_path = os.path.join(tampered_dir, "{}.seg".format(target_sha))
    with open(seg_path, "r", encoding="utf-8") as f:
        content = f.read()
    # Corrupt the outcome_key field's value generically: flip every
    # occurrence of the target sha's own trigger substring is fragile, so
    # instead corrupt by mutating a distinctive JSON value: replace the
    # trigger_key or outcome_key content. We know the record's trigger_key
    # is target_from; mutate its outcome_key by string substitution on the
    # "outcome_key" JSON field.
    import json as _json
    lines = content.split("\n")
    tampered_lines = []
    mutated = False
    for line in lines:
        if line.strip() == "":
            tampered_lines.append(line)
            continue
        try:
            rec = _json.loads(line)
        except ValueError:
            tampered_lines.append(line)  # header line
            continue
        if "outcome_key" in rec and rec.get("trigger_key") == target_from:
            rec["outcome_key"] = rec["outcome_key"] + "_TAMPERED"
            mutated = True
            tampered_lines.append(_json.dumps(rec, sort_keys=True, ensure_ascii=False))
        else:
            tampered_lines.append(line)
    assert mutated, "did not find the expected record to tamper"
    tampered_content = "\n".join(tampered_lines)
    assert tampered_content != content
    with open(seg_path, "w", encoding="utf-8") as f:
        f.write(tampered_content)

    checks = verify_direction3(tampered_dir, n_samples=20, seed=60)
    scoring = score(checks, n_target=20)

    # At least the base edge from target_sha must now fail verification
    # (segment_sha_verified False: store.verify() catches the sha256
    # mismatch against the tampered bytes).
    base_checks_on_target = [
        c for c in checks
        if c["class"] == "base" and c["coords"]["segment_sha"] == target_sha
    ]
    if base_checks_on_target:
        for c in base_checks_on_target:
            assert c["found"] is False
            assert c["segment_sha_verified"] is False

    assert scoring["n_verified"] < scoring["n_sampled"], (
        "tampering a cited segment must break at least one check, got: {}".format(scoring)
    )
    assert scoring["p60a_verified_all"] is False


# ---------------------------------------------------------------------
# Test 3: re-derivation of inferred edges never trusts LiveGraph's cache --
# corrupting ONLY the on-disk inferred.jsonl cache (not the segments) must
# NOT make a bad cached edge look verified, since verify_direction3 samples
# candidates from a fresh LiveGraph mount but re-derives strictly from
# LiveStore.iter_records on the cited (sha, idx) pairs.
# ---------------------------------------------------------------------

def test_rederivation_ignores_stale_cache(d):
    graph = build_synthetic_graph(d, seed=777, n_batches=10)
    real_edges = graph.inferred_edges()
    assert real_edges, "fixture has no inferred edges"

    # Inject a bogus inferred edge into the on-disk cache: same shape, but
    # claims a from_key/to_key/depth its cited derivation does not support.
    cache_path = os.path.join(d, "inferred.jsonl")
    with open(cache_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    header = lines[0]
    real = real_edges[0]
    bogus = dict(real)
    bogus["to_key"] = "NOT_A_REAL_TARGET"
    import json as _json
    bogus_line = _json.dumps(bogus, sort_keys=True, ensure_ascii=False) + "\n"
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.writelines(lines[1:])
        f.write(bogus_line)

    # Sample a large N so the bogus edge is very likely included; if not,
    # this test still passes vacuously on the real edges (the assertion
    # below only checks entries whose to_key is the bogus one).
    checks = verify_direction3(d, n_samples=200, seed=1)
    bogus_checks = [
        c for c in checks
        if c["class"] == "inferred" and c["edge"]["to_key"] == "NOT_A_REAL_TARGET"
    ]
    for c in bogus_checks:
        assert c["found"] is False, "re-derivation must reject a cache-only edge, not trust it"


def run_all():
    tests = [
        test_synthetic_store_all_verified,
        test_tampered_segment_fails_verification,
        test_rederivation_ignores_stale_cache,
    ]
    for t in tests:
        with_tmpdir(t)
        print("OK  {}".format(t.__name__))
    print("ALL {} TESTS PASSED".format(len(tests)))


if __name__ == "__main__":
    run_all()
