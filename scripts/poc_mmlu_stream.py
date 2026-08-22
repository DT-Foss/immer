"""poc_mmlu_stream.py — MMLU gegen gestreamte Qwen3.8-27B-Gewichte.

Der erste ehrliche Retrieval-Pfad auf dem Weg zur Demo-Tafel:

    Frage + Choices --(32B-Tokenizer, auch gestreamt)--> Token-IDs
    Token-IDs -------> Zeilen aus embed_tokens.weight NUR FÜR DIESE IDs
    Cosinus(Frage-Vektor, Choice-Vektor) -> Antwort

Gemessen wird BEIDES: accuracy UND bytes_pro_frage gegen 61 GB Modellgröße.
Das ist die Alpha-Stufe der Router-Wissenschaft (KIMI §8) — die Verbesserung
von hier bis zu FFN-Seiten-Routing ist die Forschungskurve, nicht der Anspruch.

    PYTHONPATH=src python3 scripts/poc_mmlu_stream.py [--subject high_school_geography] [--limit 30]
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from immer.knowledge import Streamer

ROOT = Path(__file__).resolve().parent.parent
REPO = "Qwen/Qwen3.8-27B"
CHOICE_KEYS = ["A", "B", "C", "D"]


def _fetch_rows(
    source: Streamer, tensor_name: str, row_indices: list[int]
) -> np.ndarray:
    """Read only requested rows through the installed tensor-source API."""
    wanted = [int(index) for index in row_indices]
    if not wanted:
        width = int(source.find(tensor_name)["shape"][1])
        return np.empty((0, width), dtype=np.float32)
    unique = sorted(set(wanted))
    runs: list[tuple[int, int]] = []
    start = previous = unique[0]
    for index in unique[1:]:
        if index != previous + 1:
            runs.append((start, previous - start + 1))
            start = index
        previous = index
    runs.append((start, previous - start + 1))

    def read(run: tuple[int, int]) -> tuple[int, np.ndarray]:
        start_row, count = run
        return start_row, source.rows(tensor_name, start_row, count)

    with ThreadPoolExecutor(max_workers=min(12, len(runs))) as pool:
        chunks = dict(pool.map(read, runs))
    by_index = {
        start_row + offset: row
        for start_row, matrix in chunks.items()
        for offset, row in enumerate(matrix)
    }
    return np.stack([by_index[index] for index in wanted])


def fetch_tokenizer_bytes(s: Streamer) -> bytes:
    """Der Tokenizer des 32B selbst kommt gestreamt aus dem Repo."""
    try:
        return s.reader.fetch_file("tokenizer.json")
    except Exception:
        return s.reader.fetch_file("tokenizer.model")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="high_school_geography")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--budget-mb", type=float, default=300.0)
    ap.add_argument("--lexik", type=float, default=0.3)
    args = ap.parse_args()

    local = ROOT / "evals" / f"mmlu_{args.subject}_test.parquet"
    if local.is_file():
        import pandas as pd  # type: ignore

        frame = pd.read_parquet(local)
        rows = list(frame.to_dict("records"))
    else:
        import datasets  # type: ignore

        ds = datasets.load_dataset("cais/mmlu", args.subject, split="test")
        rows = list(ds)

    t0 = time.time()
    s = Streamer(REPO, budget_mb=args.budget_mb)
    tok_raw = fetch_tokenizer_bytes(s)
    print(f"tokenizer.json gestreamt: {len(tok_raw) / 1048576:.1f} MB ({time.time() - t0:.0f}s)")

    import tempfile

    from transformers import PreTrainedTokenizerFast

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        handle.write(tok_raw)
        tok_path = handle.name
    tok = PreTrainedTokenizerFast(tokenizer_file=tok_path)

    inv = s.inventory()
    all_ts = s.tensors()
    names = [t["name"] for t in all_ts]
    emb_name = ("model.embed_tokens.weight" if "model.embed_tokens.weight" in names
                else next(n for n in names if n.endswith("embed_tokens.weight")))
    emb = next(t for t in all_ts if t["name"] == emb_name)
    print(f"Inventar: {len(inv['tensors'])} Tensoren, Modell "
          f"{inv['model_payload_bytes'] / 1024 ** 3:.1f} GB | Ziel-Tensor {emb_name} "
          f"{emb['bytes'] / 1048576:.0f} MB\n")

    def vec(text: str) -> tuple[np.ndarray, int]:
        """IDF-gewichtete Mittelung der embed_tokens-Zeilen — Stoppwörter und
        Satzzeichen fliegen raus, seltene Tokens tragen mehr (IR-Standard).
        Jede benötigte Zeile wird einzeln gestreamt (merge <32 KB)."""
        ids = tok.encode(text.lower(), add_special_tokens=False)
        toks = tok.convert_ids_to_tokens(ids)
        n_cols = int(emb["shape"][1])
        kept: list[tuple[int, float]] = []
        for i, t in zip(ids, toks):
            w = _clean_token(t)
            if w is None:
                continue
            weight = idf.get(w, 4.0)
            kept.append((i, weight))
        uniq = sorted({i for i, _ in kept})
        missing = [i for i in uniq if i not in row_cache]
        if missing:
            matm = _fetch_rows(s, emb_name, missing)
            for j, i in enumerate(missing):
                row_cache[i] = matm[j]
        acc = np.zeros(n_cols, dtype=np.float32)
        wsum = 0.0
        for i, weight in kept:
            acc += weight * row_cache[i]
            wsum += weight
        v = acc / (wsum + 1e-9)
        nrm = np.linalg.norm(v)
        return v / (nrm + 1e-9), len(missing) * n_cols * 2

    row_cache: dict[int, np.ndarray] = {}

    # --- IDF aus der Fragen-Menge selbst (lokal, keine externe Info) ---
    import math as _math
    import re as _re

    def _clean_token(t: str | None) -> str | None:
        if t is None:
            return None
        w = t.replace("Ġ", "").replace("▁", "").lower()
        w = _re.sub(r"[^a-zäöüß]", "", w)
        if len(w) < 3:
            return None
        if w in _STOPS:
            return None
        return w

    _STOPS = {
        "the", "and", "for", "with", "that", "this", "from", "was", "were",
        "are", "have", "has", "had", "not", "but", "its", "his", "her",
        "which", "what", "when", "where", "who", "how", "why", "into", "onto",
        "der", "die", "das", "und", "ist", "ein", "eine", "sich",
    }
    docs = []
    for item in rows[: args.limit]:
        words = set(_clean_token(t) for t in tok.convert_ids_to_tokens(
            tok.encode((item["question"] + " " + " ".join(item["choices"][:64])).lower(),
                       add_special_tokens=False)))
        docs.append({w for w in words if w})
    n_docs = max(len(docs), 1)
    df_counts: dict[str, int] = {}
    for dset in docs:
        for w in dset:
            df_counts[w] = df_counts.get(w, 0) + 1
    idf = {w: _math.log(n_docs / c) + 1.0 for w, c in df_counts.items()}

    def lexical_overlap(q: str, ch: str) -> float:
        qs = {_clean_token(t) for t in tok.convert_ids_to_tokens(tok.encode(q.lower(), add_special_tokens=False))}
        cs = {_clean_token(t) for t in tok.convert_ids_to_tokens(tok.encode(ch.lower(), add_special_tokens=False))}
        qs.discard(None)
        cs.discard(None)
        if not qs or not cs:
            return 0.0
        return len(qs & cs) / _math.sqrt(len(qs) * len(cs))

    correct = total = 0
    q_bytes_total = 0
    t_run = time.time()
    for item in rows[: args.limit]:
        frage = item["question"]
        choices = item["choices"]
        gold = int(item["answer"])
        before = s.bytes_moved()
        try:
            qv, _ = vec(frage)
            scores = []
            for ch in choices:
                cv, _b = vec(ch[:64])
                dense = float(np.dot(qv, cv))
                lexik = lexical_overlap(frage, ch[:64])
                scores.append((1 - args.lexik) * dense + args.lexik * lexik)
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP ({type(exc).__name__}: {exc})")
            continue
        used = s.bytes_moved() - before
        q_bytes_total += used
        pred = int(np.argmax(scores))
        total += 1
        correct += int(pred == gold)

    acc = correct / total if total else 0.0
    report = {
        "schema": "immer.benchmark/v1",
        "benchmark": f"MMLU:{args.subject}",
        "gegen": REPO,
        "modell_groesse_gb": round(inv["model_payload_bytes"] / 1024 ** 3, 1),
        "n": total,
        "accuracy_alpha_router": round(acc, 4),
        "scorer": f"dense+{args.lexik}lexik",
        "bytes_pro_frage_kb": round(q_bytes_total / max(total, 1) / 1024, 2),
        "anteil_modell_pct": round(100 * q_bytes_total / inv["model_payload_bytes"], 6),
        "sekunden": round(time.time() - t_run, 1),
        "note": "Alpha: Token-Embedding-Aehnlichkeit. Forschungskurve: FFN-Seiten-Routing (KIMI S8).",
    }
    out = ROOT / "results" / "poc_mmlu_stream.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
