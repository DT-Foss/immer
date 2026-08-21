#!/usr/bin/env python3 -u
"""
Drive the REAL Kamradt v2 NIAH runner with O1 as the provider.
Real Paul Graham prose haystack, real single-needle task, real ExactMatch scoring, sweep over
growing context lengths. O1 streams the haystack into its fact index and answers from it.

Run:  PYTHONPATH=/Users/bhkmie/Documents/Forschung/O1/src python3 run_o1_niah.py
"""
import os, sys, json, asyncio, time

# O1 provider
sys.path.insert(0, "/Users/bhkmie/Documents/Forschung/O1/src")
from niah_o1_provider import O1Provider

# v2 components (real)
from needlehaystack.core.runner import Runner
from needlehaystack.core.sweep import Sweep
from needlehaystack.haystacks.files import FilesHaystack
from needlehaystack.stores.jsonl import JsonlResultStore
from needlehaystack.tasks.single_needle import SingleNeedleTask
from needlehaystack.tasks.multi_needle import MultiNeedleTask
from needlehaystack.tasks.uuid_chain import UuidChainTask

_TASKS = {"single": SingleNeedleTask, "multi": MultiNeedleTask, "uuid_chain": UuidChainTask}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="single", choices=list(_TASKS))
    ap.add_argument("--lengths", default="1000,2000,4000,8000,16000,32000,64000,128000")
    ap.add_argument("--depths", default="0,25,50,75,100")
    ap.add_argument("--out", default="results/o1_single_needle.jsonl")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    depths = [float(x) for x in args.depths.split(",")]

    task = _TASKS[args.task]()                       # the v2 task, unmodified
    provider = O1Provider()
    haystack = FilesHaystack(path="PaulGrahamEssays")   # real prose
    sweep = Sweep(lengths=lengths, depths=depths, seeds=[None])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    store = JsonlResultStore(args.out)

    n_cells = len(lengths) * len(depths)
    print("=" * 78)
    print(f"REAL Kamradt v2 NIAH [{args.task}] — O1 (state stream + fact index) on Paul Graham prose")
    print(f"  needle   : {getattr(task, 'expected_answer', getattr(task, 'expected_fragments', '?'))!r}")
    print(f"  question : {getattr(task, 'question_template', '(dynamic)')!r}")
    print(f"  lengths  : {lengths}")
    print(f"  depths   : {depths}   →  {n_cells} cells")
    print("=" * 78)

    done = {"n": 0}
    def on_result(r):
        done["n"] += 1
        sc = getattr(r, "score", None)
        val = sc.value if sc is not None else (r.error and "ERR")
        print(f"  [{done['n']:>3}/{n_cells}] len={getattr(r,'context_length','?'):>7} "
              f"depth={getattr(r,'depth_percent','?'):>5}  score={val}", flush=True)

    runner = Runner(task=task, provider=provider, haystack=haystack, sweep=sweep,
                    store=store, run_name="o1-single-needle",
                    concurrency=args.concurrency, retries=1, resume=False,
                    on_result=on_result)

    t0 = time.time()
    results = asyncio.run(runner.run())
    dt = time.time() - t0

    scored = [r for r in results if getattr(r, "score", None) is not None]
    if scored:
        acc = sum(r.score.value for r in scored) / len(scored)
        perfect = sum(1 for r in scored if r.score.value >= 0.999)
        print("\n" + "=" * 78)
        print(f"RESULT: mean recall {acc*100:.1f}%  ·  {perfect}/{len(scored)} cells at 100%  "
              f"·  {dt:.0f}s")
        # per-length breakdown
        bylen = {}
        for r in scored:
            bylen.setdefault(r.context_length, []).append(r.score.value)
        print("  per length:")
        for L in sorted(bylen):
            vs = bylen[L]
            print(f"    {L:>8,}: {sum(vs)/len(vs)*100:5.1f}%  ({sum(1 for v in vs if v>=0.999)}/{len(vs)} perfect)")
        print("=" * 78)
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
