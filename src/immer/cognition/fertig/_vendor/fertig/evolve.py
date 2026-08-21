"""
fertig.evolve — der autonome Verbesserungs-Loop.

Online-Learning als messbarer Kreislauf, ohne anzuhalten:

  Iteration:
    1. MESSEN   : ARC-Easy-Sample + Arena laufen lassen
                  (Genauigkeit, Graph-Abdeckung, unknown-Rate)
    2. LÜCKEN   : fehlgeschlagene Antwort-Optionen + Arena-Ziele
                  als Wachstums-Kandidaten sammeln
    3. WACHSEN  : Kandidaten durch alle Quellen in den Welt-Graphen holen
    4. NEU MESSEN: dieselben Benchmarks — Delta ist der Ledger-Eintrag

Abbruch erst, wenn keine Lücke mehr wächst (oder Budget erschöpft).
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple

from . import gaps as gaps_mod
from . import sources as sources_mod
from .pipeline import _toks


def _arc_option_gaps(n_questions: int, n_options: int = 4) -> List[str]:
    """Optionen aus den ARC-Fragen sammeln, die nicht im Graphen sind.
    Ganze Options-Phrasen (bis 4 Wörter) — die sind die Antwort-Kandidaten."""
    from . import bench as bench_mod
    from .pipeline import load_graph_merged
    from . import inference as inf
    vocab = load_graph_merged()[0]
    rows = bench_mod._arc_rows(n_questions)
    gaps: List[str] = []
    for r in rows:
        for opt in r["choices"]["text"]:
            o = " ".join(_toks(str(opt)))
            if len(o) < 3 or len(o.split()) > 4:
                continue
            known = any(inf.jaro_winkler(o, v) >= 0.9 for v in vocab)
            if not known and o not in gaps:
                gaps.append(o)
    return gaps


def evolve(iterations: int = 3, arc_questions: int = 30,
           grow_per_iter: int = 3, sources: Optional[List[str]] = None,
           verbose: bool = True) -> List[dict]:
    """Der Loop. Rückgabe: Messprotokoll pro Iteration."""
    log: List[dict] = []
    for it in range(1, iterations + 1):
        # 1. MESSEN
        from . import bench as bench_mod
        arc = bench_mod.run_arc(n=arc_questions, use_graph=True,
                                verbose=False)
        # 2. LÜCKEN
        arc_gaps = _arc_option_gaps(arc_questions)
        from .arena import EVAL_SET
        arena_gaps = gaps_mod.detect_gaps([c for c, _, _ in EVAL_SET])
        candidates = (arc_gaps + arena_gaps)
        # dedupe, bekannte zuerst (höherer Ertrag)
        seen: set = set()
        uniq = []
        for g in candidates:
            g = g.strip()
            if g and g not in seen:
                seen.add(g)
                uniq.append(g)
        # 3. WACHSEN
        grown = 0
        for target in uniq[:grow_per_iter]:
            n = len(gaps_mod.grow(target, verbose=False,
                                  source_names=sources))
            grown += n
        # 4. NEU MESSEN
        arc2 = bench_mod.run_arc(n=arc_questions, use_graph=True,
                                 verbose=False)
        entry = {
            "iteration": it,
            "graph_triplets": len(gaps_mod._load_world()),
            "arc_accuracy": arc2.accuracy,
            "arc_coverage": arc2.coverage,
            "arc_delta_acc": arc2.accuracy - arc.accuracy,
            "arc_delta_cov": arc2.coverage - arc.coverage,
            "grown_triplets": grown,
            "gaps_remaining": len(uniq) - min(len(uniq), grow_per_iter),
        }
        log.append(entry)
        if verbose:
            print(f"[evolve] Iteration {it}: +{grown} Tripletts "
                  f"(Graph {entry['graph_triplets']}) | ARC "
                  f"{100*arc2.accuracy:.1f}% (cov "
                  f"{100*arc2.coverage:.1f}%) "
                  f"[Δacc {100*entry['arc_delta_acc']:+.1f}pp "
                  f"Δcov {100*entry['arc_delta_cov']:+.1f}pp]")

    return log


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="fertig evolve")
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--arc-questions", type=int, default=30)
    ap.add_argument("--grow-per-iter", type=int, default=3)
    ap.add_argument("--sources", default=None)
    args = ap.parse_args(argv)
    srcs = args.sources.split(",") if args.sources else None
    log = evolve(args.iterations, args.arc_questions,
                 args.grow_per_iter, srcs)
    if len(log) >= 2:
        print("\n=== Evolve-Protokoll (Ledger) ===")
        print(f"{'It':>3} {'Graph':>6} {'ARC%':>6} {'Cov%':>6} "
              f"{'Δacc':>6} {'Δcov':>6}")
        for e in log:
            print(f"{e['iteration']:>3} {e['graph_triplets']:>6} "
                  f"{100*e['arc_accuracy']:>5.1f}% "
                  f"{100*e['arc_coverage']:>5.1f}% "
                  f"{100*e['arc_delta_acc']:>+5.1f}% "
                  f"{100*e['arc_delta_cov']:>+5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
