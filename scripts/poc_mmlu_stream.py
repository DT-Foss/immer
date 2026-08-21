"""poc_mmlu_stream.py — MMLU gegen gestreamte Qwen3-32B-Gewichte.

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
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "immer" / "knowledge"))
sys.path.insert(0, str(ROOT / "vendor" / "mitglm"))

import numpy as np  # noqa: E402

from streamer import Streamer  # noqa: E402

REPO = "Qwen/Qwen3-32B"
CHOICE_KEYS = ["A", "B", "C", "D"]


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
    emb = next(t for t in s.tensors() if t["name"] == "model.embed_tokens.weight")
    print(f"Inventar: {len(inv['tensors'])} Tensoren, Modell "
          f"{inv['model_payload_bytes'] / 1024 ** 3:.1f} GB | Ziel-Tensor "
          f"{emb['bytes'] / 1048576:.0f} MB\n")

    def vec(text: str) -> tuple[np.ndarray, int]:
        """Mittelwert der embed_tokens-Zeilen aller Tokens — jede benoetigte
        Zeile wird EINZELN gestreamt (fetch_uniform_rows: merge bei Luecke
        <32 KB, sonst eigener Request). Kein contiguous Overfetch."""
        ids = tok.encode(text, add_special_tokens=False)
        n_cols = int(emb["shape"][1])
        uniq = sorted(set(ids))
        missing = [i for i in uniq if i not in row_cache]
        if missing:
            import hf_organ_reader as hor
            import casi_tensor_map as ctm

            raw = hor.fetch_uniform_rows(
                s.reader, emb["shard"], emb["data_start"],
                emb["offset_in_shard"][0], n_cols, 2, missing,
            )
            mat = ctm.bf16_rows_to_f32(raw.view(np.uint16), (len(missing), n_cols))
            for j, i in enumerate(missing):
                row_cache[i] = mat[j]
        mat2 = np.stack([row_cache[i] for i in ids])
        v = mat2.mean(axis=0)
        n = np.linalg.norm(v)
        return v / (n + 1e-9), len(missing) * n_cols * 2

    row_cache: dict[int, np.ndarray] = {}

    correct = total = 0
    q_bytes_total = 0
    t_run = time.time()
    for item in rows[: args.limit]:
        frage = item["question"]
        choices = item["choices"]
        gold = int(item["answer"])
        before = s.budget.body
        try:
            qv, _ = vec(frage)
            scores = []
            for ch in choices:
                cv, _b = vec(ch[:64])
                scores.append(float(np.dot(qv, cv)))
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP ({type(exc).__name__}: {exc})")
            continue
        used = s.budget.body - before
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
