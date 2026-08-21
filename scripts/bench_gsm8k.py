"""bench_gsm8k.py — FERTIG gegen die Großen, gemessen im eigenen Repo.

Läuft den vendored FERTIG-Solver über den GSM8K-Test-Split und berichtet
correct / abstain / WRONG. Der Claim der Architektur: wrong = 0 (Abstinenz
statt Raten). Ausgabe: results/bench_gsm8k.json (immer.benchmark/v1-Geist).

    PYTHONPATH=src python3 scripts/bench_gsm8k.py [--limit 100]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "immer" / "cognition" / "fertig" / "_vendor"))

import pandas as pd  # noqa: E402

from fertig import solver  # noqa: E402

EVAL_PATH = ROOT / "evals" / "gsm8k_test.parquet"
RESULTS_DIR = ROOT / "results"


def gold_number(answer_field: str) -> float | None:
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_field)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def solver_number(output: str | None) -> float | None:
    if output is None:
        return None
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", str(output))
    if not numbers:
        return None
    return float(numbers[-1].replace(",", ""))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 = alle 1319")
    args = parser.parse_args()

    df = pd.read_parquet(EVAL_PATH)
    if args.limit:
        df = df.head(args.limit)

    correct = abstain = wrong = errors = 0
    t0 = time.time()
    for _, row in df.iterrows():
        gold = gold_number(str(row["answer"]))
        if gold is None:
            errors += 1
            continue
        try:
            out = solver.solve(str(row["question"]))
        except Exception:
            wrong += 1  # eine Exception ist ein falsches Verhalten, kein Abstinenz
            continue
        got = solver_number(out)
        if out is None or got is None:
            abstain += 1
        elif abs(got - gold) < 1e-6:
            correct += 1
        else:
            wrong += 1
    seconds = time.time() - t0
    attempted = correct + wrong
    accuracy_on_attempted = correct / attempted if attempted else 0.0

    report = {
        "schema": "immer.benchmark/v1",
        "benchmark": "GSM8K-test",
        "n": int(len(df)),
        "correct": correct,
        "abstained": abstain,
        "wrong": wrong,
        "wrong_must_be_zero": wrong == 0,
        "accuracy_on_attempted": round(accuracy_on_attempted, 4),
        "seconds": round(seconds, 1),
        "engine": "FERTIG vendored (bindings->semantic->math->miner), no neural net",
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "bench_gsm8k.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"geschrieben: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
