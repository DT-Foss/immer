#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E01: CASI-Wissenslandkarte aller Tensoren von Qwen3.5-0.8B (multimodal checkpoint).

Methode (BINDEND, T02b-Lektion aus casi-collapse/FINDINGS.md):
  1. Natuerliche 2D-Matrixform (out_dim x in_dim) des Gewichtstensors beibehalten
     (3D+ conv → (out_channels, -1); Zeilenzahl < 100 → transponieren).
  2. Global-quantile-Mapping der 2D-Struktur auf uint8 (Rang → Byte),
     mit ZUFALLS-Tie-Breaking (BF16 erzeugt massenhaft exakte Ties;
     deterministische Reihenfolge wuerde kuenstliche Monotonie injizieren).
  3. CASI = ||Z|| ueber die 21 Byte-Level-Strategien von live_casiv2
     (compute_fast_casi — keine temporalen Permutationsstrategien).
  4. Null: CASI auf strukturzerstoerter Version gleicher Shape
     (Element-Permutation der quantile-Bytes — haelt Byte-Histogramm und Form exakt,
      zerstoert nur raeumliche Anordnung). n_null=5 Permutationen, Ratio = raw/null.
  => Ratio ≈ 1: Tensor ist CASI-aequivalent zu Zufall ("Mischer").
     Ratio ≫ 1: Tensor traegt raeumliche Struktur ("Wissen/Spezialisierung").

Sanity-Gates (Asserts, ABORT bei Fail — ein Fail waere selbst ein Befund):
  G1: BF16-quantisierte Zufallsmatrix (2 Seeds) → Ratio in [0.7, 1.3] (±30%).
  G2: Toeplitz-Fixture (stark strukturiert) → Ratio > 5.
  G3 (informational, kein Gate): 10% doppelte Zeilen ( partieller Kollaps).

RAM-Disziplin: ein Tensor gleichzeitig (np.memmap + zeilenweise Subsamples fuer
Riesen), Peak-RSS via resource.ru_maxrss im Report. Kein torch noetig —
BF16 wird in numpy dekodiert (u16 << 16 → view float32).

Resume: Output-JSON wird nach JEDEM Tensor atomar geschrieben; vorhandene
Eintraege werden uebersprungen (--fresh setzt zurueck).

Nur-Lesen fuer alle Quell-Ressourcen; geschrieben wird ausschliesslich
nach mitGLM/results/.
"""
import argparse
import hashlib
import json
import os
import re
import resource
import struct
import sys
import time
import zlib
from datetime import datetime, timezone

import numpy as np

# ── Pfade (Defaults) ────────────────────────────────────────────────────────
ST_FILE = ("/Volumes/INTENSO/KI Projekt/Qwen3.5-HCGM-Moonshots/source/"
           "Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors")
ENGINE_DIR = ("/Users/bhkmie/Downloads/Forschung/Alte AI Projekte/"
              "AI Fingerprintkram/casi-collapse")
OUT_JSON = ("/Users/bhkmie/Downloads/Forschung/Alte AI Projekte/mitGLM/results/"
            "e01_casi_tensor_map_qwen35_08b.json")

SEED = 42
MIN_SCORE_ELEMS = 4096      # bias-/norm-Vektoren darunter: nur zaehlen
N_NULL = 5                  # Permutations-Nulls pro Tensor
MAX_ELEMS = 8_388_608       # Element-Deckel (8*2^20) — sonst deterministische Zeilensubsample
MIN_ROWS = 100              # <100 Zeilen → transponieren (casi_nn-Konvention)
T0 = time.time()

# ── CASI-Engine: live_casiv2 importieren, sonst minimale Fallback-Impl. ─────
sys.path.insert(0, ENGINE_DIR)
ENGINE_SOURCE = None
try:
    from live_casiv2 import (compute_fast_casi, compute_fast_profile,  # noqa: E402
                             CRYPTO_STRATEGY_NAMES, IMPL_STRATEGY_NAMES)
    import live_casiv2
    ENGINE_SOURCE = f"live_casiv2 v{live_casiv2.__version__} @ {live_casiv2.__file__}"
    STRATEGY_NAMES = CRYPTO_STRATEGY_NAMES + IMPL_STRATEGY_NAMES
except Exception as e:  # pragma: no cover — Fallback falls Import schiefgeht
    print(f"[WARN] live_casiv2 nicht importierbar ({e}) — nutze minimale Fallback-CASI.")
    STRATEGY_NAMES = ["byte_frequency", "entropy", "min_entropy", "byte_skewness",
                      "runs", "autocorr", "multi_lag_ac", "corr_matrix_chi2"]

    def _byte_frequency(k):
        n = k.size
        h = np.bincount(k.ravel(), minlength=256) / n
        p = h[h > 0]
        ent = float(-(p * np.log2(p)).sum())
        return (ent - np.log2(256.0)) / (1.0 / np.sqrt(2.0 * n * np.log(2) ** 2))

    def _entropy(k):
        return _byte_frequency(k) * 0.9999

    def _min_entropy(k):
        n = k.size
        h = np.bincount(k.ravel(), minlength=256).astype(np.float64)
        hi = -np.log2(max(1.0, h.max()) / n)
        exp = 8.0 - (8.0 - 7.8425) * np.sqrt(160000.0 / n)
        return (hi - exp) / (0.02026 * np.sqrt(160000.0 / n))

    def _skew(k):
        x = k.ravel().astype(np.float64) - 127.5
        s = float(np.mean(x ** 3) / max(np.mean(x ** 2) ** 1.5, 1e-12))
        return s / (0.05 * np.sqrt(160000.0 / k.size))

    def _runs(k):
        b = k.ravel()
        n = len(b)
        runs = int(np.count_nonzero(b[1:] != b[:-1])) + 1
        exp = 2 * (n - 1) * (1 - 1.0 / 256.0) + 1
        return (runs - exp) / np.sqrt(n)

    def _autocorr(k):
        x = k.ravel().astype(np.float64) - 127.5
        r1 = float(np.mean(x[:-1] * x[1:])) / max(float(np.var(x)), 1e-12)
        return r1 * np.sqrt(len(x))

    def _multi_lag(k):
        x = k.ravel().astype(np.float64) - 127.5
        n = len(x)
        v = max(float(np.var(x)), 1e-12)
        Q = sum((float(np.mean(x[:-j] * x[j:])) / v) ** 2 / (n - j) for j in (1, 2, 4, 8, 16, 32, 64))
        Q *= n * (n + 2)
        return (Q - 7.0) / np.sqrt(14.0)

    def _corr_matrix_chi2(k):
        if k.shape[0] < 500:
            return 0.0
        bits = np.unpackbits(k, axis=1)[:, :256].astype(np.float64)
        c = bits - bits.mean(axis=0, keepdims=True)
        s = bits.std(axis=0)
        s[s < 1e-10] = 1.0
        c /= s
        corr = (c.T @ c) / k.shape[0]
        np.fill_diagonal(corr, 0)
        z = corr * np.sqrt(k.shape[0])
        iu = np.triu_indices(z.shape[0], k=1)
        sum_z2 = float(np.sum(z[iu] ** 2))
        npair = len(iu[0])
        return (sum_z2 - npair) / np.sqrt(2.0 * npair)

    _FB = {"byte_frequency": _byte_frequency, "entropy": _entropy,
           "min_entropy": _min_entropy, "byte_skewness": _skew, "runs": _runs,
           "autocorr": _autocorr, "multi_lag_ac": _multi_lag,
           "corr_matrix_chi2": _corr_matrix_chi2}

    def compute_fast_profile(keys):
        return np.array([_FB[n](keys) for n in STRATEGY_NAMES], dtype=np.float64)

    def compute_fast_casi(keys):
        return float(np.sqrt(np.sum(compute_fast_profile(keys) ** 2)))
    ENGINE_SOURCE = f"fallback_minimal ({len(STRATEGY_NAMES)} Strategien)"


# ── Hilfsfunktionen ─────────────────────────────────────────────────────────
def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def sha256_file(path, chunk=16 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_st_header(path):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen).decode("utf-8"))
    return header, 8 + hlen  # data-start-offset


_DT_SIZE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I32": 4, "I64": 8, "U8": 1}


def load_tensor_f32(path, data_start, entry):
    """Tensor via np.memmap einlesen und nach float32 dekodieren (BF16: u16<<16)."""
    dt = entry["dtype"]
    shape = tuple(entry["shape"])
    off_b, off_e = entry["data_offsets"]
    nbytes = off_e - off_b
    assert nbytes == int(np.prod(shape)) * _DT_SIZE[dt], f"Offset/Shape-Mismatch {entry}"
    mm = np.memmap(path, dtype=np.uint8, mode="r", offset=data_start + off_b,
                   shape=(nbytes,))
    if dt == "BF16":
        u16 = mm.view(np.uint16)
        # nur fuer die Analyse benoetigte Zeilen direkt aus dem memmap ziehen
        rows = shape[0] if len(shape) >= 2 else None
        return u16, shape, "bf16"
    if dt == "F32":
        return mm.view(np.float32), shape, "f32"
    raise ValueError(f"dtype {dt} nicht unterstuetzt")


def bf16_rows_to_f32(u16, shape, row_idx=None):
    """BF16-uint16 (2D-View) nach float32; optional nur ausgewaehlte Zeilen."""
    if len(shape) == 2:
        m2 = u16.reshape(shape)
        if row_idx is not None:
            m2 = m2[np.sort(row_idx)]
        u32 = np.ascontiguousarray(m2, dtype=np.uint32) << 16
        return u32.view(np.float32)
    flat = u16 if row_idx is None else u16  # 1D wird nie gescored
    u32 = np.ascontiguousarray(flat, dtype=np.uint32) << 16
    return u32.view(np.float32)


def f32_rows_to_f32(f32mm, shape, row_idx=None):
    m = f32mm.reshape(shape)
    if row_idx is not None and len(shape) == 2:
        m = m[np.sort(row_idx)]
    return np.array(m, dtype=np.float32, copy=True)


def global_quantile_u8(mat2d, rng):
    """Global-quantile auf 2D-Float-Matrix → uint8, Zufalls-Tie-Breaking.

    Rang jedes Elements in [0..n-1] → Byte. Identische Werte (BF16-Ties)
    bekommen zufaellige Rangfolge innerhalb ihrer Tie-Gruppe — verhindert
    das T02-Artefakt (kuenstliche Index-Monotonie innerhalb von Ties).
    """
    flat = mat2d.ravel()
    n = flat.size
    perm = rng.permutation(n)                      # Tie-Break-Reihenfolge
    order = np.argsort(flat[perm], kind="stable")  # stabile Sortierung der gemischten Werte
    ranks = np.empty(n, dtype=np.float64)
    ranks[perm[order]] = np.arange(n, dtype=np.float64)
    q = (ranks * (255.999 / n)).astype(np.uint8)
    return q.reshape(mat2d.shape)


def casi_with_nulls(bytes2d, rng, n_null=N_NULL):
    casi_raw = compute_fast_casi(bytes2d)
    profile = compute_fast_profile(bytes2d)
    flat = bytes2d.ravel()
    nulls = []
    for _ in range(n_null):
        p = rng.permutation(flat.size)
        nb = flat[p].reshape(bytes2d.shape)
        nulls.append(compute_fast_casi(nb))
    nulls = np.asarray(nulls, dtype=np.float64)
    null_mean = float(nulls.mean())
    ratio = float(casi_raw / null_mean) if null_mean > 0 else float("inf")
    return casi_raw, profile, nulls, null_mean, ratio


# ── tensor_class aus dem Namen ──────────────────────────────────────────────
CLASS_RULES = [
    (r"(^|\.)embed_tokens\.weight$", "embed"),
    (r"(^|\.)lm_head\.weight$", "lm_head"),
    (r"linear_attn\.in_proj_(qkv|z)\.weight$", "gdn_in_proj"),
    (r"linear_attn\.in_proj_[ab]\.weight$", "gdn_ba"),
    (r"linear_attn\.out_proj\.weight$", "gdn_out"),
    (r"linear_attn\.conv1d\.weight$", "gdn_conv"),
    (r"^mtp\..*self_attn\.[qkv]_proj\.weight$", "mtp_attn_qkv"),
    (r"^mtp\..*self_attn\.o_proj\.weight$", "mtp_attn_o"),
    (r"^mtp\..*mlp\.gate_proj\.weight$", "mtp_mlp_gate"),
    (r"^mtp\..*mlp\.up_proj\.weight$", "mtp_mlp_up"),
    (r"^mtp\..*mlp\.down_proj\.weight$", "mtp_mlp_down"),
    (r"^mtp\.fc\.weight$", "mtp_fc"),
    (r"self_attn\.[qkv]_proj\.weight$", "attn_qkv"),
    (r"self_attn\.o_proj\.weight$", "attn_o"),
    (r"mlp\.gate_proj\.weight$", "mlp_gate"),
    (r"mlp\.up_proj\.weight$", "mlp_up"),
    (r"mlp\.down_proj\.weight$", "mlp_down"),
    (r"(^|\.)norm\.weight$", "final_norm"),
    (r"visual\.patch_embed\.proj\.weight$", "visual_patch_embed"),
    (r"visual\.pos_embed\.weight$", "visual_pos_embed"),
    (r"visual\.blocks\.\d+\.attn\.qkv\.weight$", "visual_attn_qkv"),
    (r"visual\.blocks\.\d+\.attn\.proj\.weight$", "visual_attn_o"),
    (r"visual\.blocks\.\d+\.mlp\.linear_fc1\.weight$", "visual_mlp_up"),
    (r"visual\.blocks\.\d+\.mlp\.linear_fc2\.weight$", "visual_mlp_down"),
    (r"visual\.merger\.linear_fc1\.weight$", "visual_merger_fc1"),
    (r"visual\.merger\.linear_fc2\.weight$", "visual_merger_fc2"),
]
CLASS_RULES = [(re.compile(p), c) for p, c in CLASS_RULES]


def classify(name):
    for rx, cls in CLASS_RULES:
        if rx.search(name):
            return cls
    return "other"


def layer_index(name):
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


# ── Sanity-Gates ────────────────────────────────────────────────────────────
def quantize_bf16(f32):
    """float32 → BF16 → float32 (Identitaet der gespeicherten Precision simulieren)."""
    u32 = np.ascontiguousarray(f32, dtype=np.float32).view(np.uint32)
    return ((u32 >> 16).astype(np.uint32) << 16).view(np.float32)


def pipeline_ratio(mat2d_f32, seed, n_null=N_NULL):
    rng = np.random.default_rng([SEED, seed])
    b = global_quantile_u8(mat2d_f32, rng)
    casi_raw, _, nulls, null_mean, ratio = casi_with_nulls(b, rng, n_null=n_null)
    return casi_raw, nulls, null_mean, ratio


def run_sanity_gates():
    gates = {"passed": True, "fixtures": {}}
    # G1: Zufall (BF16-quantisiert, wie die Modellgewichte) → ratio ≈ 1 ± 30%
    for seed in (1001, 1002):
        rng = np.random.default_rng(seed)
        W = quantize_bf16(rng.normal(0.0, 0.02, (1024, 1024)).astype(np.float32))
        raw, nulls, nm, ratio = pipeline_ratio(W, seed)
        ok = 0.7 <= ratio <= 1.3
        gates["fixtures"][f"random_bf16_seed{seed}"] = {
            "shape": [1024, 1024], "casi_raw": raw,
            "casi_null_values": [float(x) for x in nulls],
            "casi_null_mean": nm, "ratio": ratio, "ok": ok}
        print(f"  G1 random(seed={seed}): CASI={raw:8.1f} null={nm:8.1f} "
              f"ratio={ratio:6.3f}  [{'PASS' if ok else 'FAIL'}]")
        gates["passed"] &= ok
    # G2: Toeplitz (jede Zeile = Verschub derselben Basissequenz) → ratio ≫ 5
    rng = np.random.default_rng(2001)
    base = rng.normal(0.0, 0.02, 2047).astype(np.float32)
    idx = np.arange(1024)
    Wt = quantize_bf16(base[(idx[:, None] - idx[None, :]) + 1023])
    raw, nulls, nm, ratio = pipeline_ratio(Wt, 2001)
    ok = ratio > 5.0
    gates["fixtures"]["toeplitz_bf16"] = {
        "shape": [1024, 1024], "casi_raw": raw,
        "casi_null_values": [float(x) for x in nulls],
        "casi_null_mean": nm, "ratio": ratio, "ok": ok}
    print(f"  G2 toeplitz:            CASI={raw:8.1f} null={nm:8.1f} "
          f"ratio={ratio:8.2f}  [{'PASS' if ok else 'FAIL'}]")
    gates["passed"] &= ok
    # G3 (informational, kein Gate): 10% doppelte Neuronen (partieller Kollaps)
    rng = np.random.default_rng(3001)
    Wd = quantize_bf16(rng.normal(0.0, 0.02, (1024, 1024)).astype(np.float32))
    Wd[::10] = Wd[0]
    raw, nulls, nm, ratio = pipeline_ratio(Wd, 3001)
    gates["fixtures"]["dup10_bf16_informational"] = {
        "shape": [1024, 1024], "casi_raw": raw,
        "casi_null_values": [float(x) for x in nulls],
        "casi_null_mean": nm, "ratio": ratio, "ok": None}
    print(f"  G3 dup10 (info):        CASI={raw:8.1f} null={nm:8.1f} ratio={ratio:8.2f}")
    return gates


# ── Hauptpipeline pro Tensor ────────────────────────────────────────────────
def natural_2d(shape):
    """Natuerliche 2D-Form: 2D as-is, 3D+ → (out, -1)."""
    if len(shape) == 2:
        return tuple(shape), False
    if len(shape) >= 2:
        return (shape[0], int(np.prod(shape[1:]))), True
    return None, False


def score_tensor(name, entry, path, data_start):
    rng = np.random.default_rng([SEED, zlib.crc32(name.encode("utf-8"))])
    raw_u, shape, kind = load_tensor_f32(path, data_start, entry)
    mat2d, reshaped = natural_2d(shape)
    if mat2d is None:  # 1D — wird vorher gefiltert, nur zur Sicherheit
        return None

    n_rows, n_cols = mat2d
    transposed = False
    row_idx = None
    if n_rows < MIN_ROWS and n_cols >= MIN_ROWS:
        # Transponieren, damit genug "Keys" (Zeilen) fuer die Strategien da sind
        raw_u_t = raw_u.reshape(mat2d).T.copy()  # (n_cols, n_rows)
        mat2d = (n_cols, n_rows)
        n_rows, n_cols = mat2d
        transposed = True
        base = (raw_u_t, mat2d, kind)
    else:
        base = (raw_u, mat2d, kind)

    numel = n_rows * n_cols
    subsampled = numel > MAX_ELEMS
    if subsampled:
        keep = MAX_ELEMS // n_cols
        row_idx = np.sort(rng.choice(n_rows, size=keep, replace=False))
        n_rows_analyzed = keep
    else:
        n_rows_analyzed = n_rows

    if transposed:
        f32 = bf16_rows_to_f32(base[0], mat2d, None) if kind == "bf16" \
            else f32_rows_to_f32(base[0], mat2d, None)
        if subsampled:
            f32 = f32[row_idx]
    else:
        if kind == "bf16":
            f32 = bf16_rows_to_f32(base[0], mat2d, row_idx)
        else:
            f32 = f32_rows_to_f32(base[0], mat2d, row_idx)

    nan_count = int(np.count_nonzero(np.isnan(f32)))
    if nan_count:
        print(f"    [WARN] {nan_count} NaN(s) in {name} — werden wie +inf behandelt "
              f"(sortieren ans Ende).")
    n_unique = int(np.unique(f32).size)
    tie_fraction = float(1.0 - n_unique / f32.size)

    b2d = global_quantile_u8(f32, rng)
    del f32, raw_u
    casi_raw, profile, nulls, null_mean, ratio = casi_with_nulls(b2d, rng)
    del b2d

    return {
        "name": name,
        "tensor_class": classify(name),
        "layer": layer_index(name),
        "shape": list(shape),
        "casi_input_shape_full": [n_rows, n_cols],
        "dtype": entry["dtype"],
        "numel": int(np.prod(shape)),
        "numel_analyzed": int(n_rows_analyzed * n_cols),
        "casi_input_shape": [n_rows_analyzed, n_cols],
        "reshaped_from_nd": bool(reshaped),
        "transposed": transposed,
        "subsampled_rows": subsampled,
        "nan_count": nan_count,
        "n_unique": n_unique,
        "tie_fraction": round(tie_fraction, 6),
        "casi_raw": round(float(casi_raw), 3),
        "casi_null_mean": round(null_mean, 3),
        "casi_null_std": round(float(nulls.std()), 3),
        "casi_null_values": [round(float(x), 3) for x in nulls],
        "ratio": round(ratio, 4),
        "profile_raw": {n: round(float(z), 2) for n, z in zip(STRATEGY_NAMES, profile)},
    }


def atomic_write_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=False)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description="E01 CASI-Tensor-Map")
    ap.add_argument("--fresh", action="store_true", help="Vorhandenes Ergebnis ignorieren")
    ap.add_argument("--limit", type=int, default=0, help="Nur erste N Tensoren (Smoke-Test)")
    ap.add_argument("--gates-only", action="store_true", help="Nur Sanity-Gates ausfuehren")
    ap.add_argument("--filter", type=str, default="", help="Substring-Filter auf Tensornamen")
    args = ap.parse_args()

    print("=" * 78)
    print("E01: CASI-Wissenslandkarte — Qwen3.5-0.8B")
    print(f"  engine: {ENGINE_SOURCE}")
    print(f"  model : {ST_FILE}")
    print(f"  seed={SEED}  n_null={N_NULL}  max_elems={MAX_ELEMS}")
    print("=" * 78)

    # 1) Header lesen (ohne Modell-RAM)
    header, data_start = read_st_header(ST_FILE)
    header.pop("__metadata__", None)
    print(f"Header: {len(header)} Tensoren, Daten ab Offset {data_start}")

    # 2) Sanity-Gates — ABORT bei Fail
    print("\n--- Sanity-Gates ---")
    gates = run_sanity_gates()
    if args.gates_only:
        print("\n--gates-only: Stop nach Gates.")
        return
    if not gates["passed"]:
        print("\n*** SANITY-GATE FEHLGESCHLAGEN — ABORT. Das ist selbst ein Befund: ***")
        print(json.dumps(gates["fixtures"], indent=2))
        atomic_write_json({"experiment": "E01_casi_tensor_map", "aborted": True,
                           "sanity_gates": gates,
                           "runtime": {"finished": datetime.now(timezone.utc).isoformat()}},
                          OUT_JSON)
        sys.exit(2)

    # 3) Resume / Vorhandenes Ergebnis laden
    results = None
    if os.path.exists(OUT_JSON) and not args.fresh:
        try:
            with open(OUT_JSON) as f:
                results = json.load(f)
            print(f"\nResume: {len(results.get('tensors', []))} vorhandene Eintraege.")
        except Exception as e:
            print(f"[WARN] Vorhandenes JSON unlesbar ({e}) — Neustart.")
            results = None
    if results is None:
        results = {
            "experiment": "E01_casi_tensor_map",
            "model": {"path": ST_FILE, "size_bytes": os.path.getsize(ST_FILE)},
            "engine": {"source": ENGINE_SOURCE, "mode": "fast (21 Byte-Level-Strategien)",
                       "strategies": STRATEGY_NAMES},
            "method": {
                "serialization": "natuerliche 2D-Form (out x in) + global-quantile uint8 "
                                 "mit Zufalls-Tie-Breaking (T02b)",
                "null": f"Element-Permutation der quantile-Bytes, n={N_NULL}",
                "ratio": "casi_raw / mean(casi_null)",
                "seed": SEED, "min_score_elements": MIN_SCORE_ELEMS,
                "max_elements": MAX_ELEMS, "min_rows": MIN_ROWS,
                "note": "3D+/5D → (out, -1); Zeilen<100 → transponiert; "
                        "Elemente>max → deterministische Zeilensubsample (seeded)"},
            "tensors": [],
            "counts_only": {},
        }
        # SHA einmalig streamen und cachen
        print("Berechne SHA256 (streaming) ...")
        t0 = time.time()
        results["model"]["sha256"] = sha256_file(ST_FILE)
        print(f"  sha256={results['model']['sha256']}  ({time.time()-t0:.1f}s)")
    else:
        # ggf. Gates des resumed Laufs aktualisieren
        results["sanity_gates"] = gates
        if "sha256" not in results.get("model", {}):
            results["model"]["sha256"] = sha256_file(ST_FILE)

    results["sanity_gates"] = gates
    results.setdefault("runtime", {})
    results["runtime"]["started"] = results["runtime"].get(
        "started", datetime.now(timezone.utc).isoformat())

    done = {t["name"] for t in results["tensors"]}
    counted = results.get("counts_only", {})

    names = sorted(header.keys())
    if args.filter:
        names = [n for n in names if args.filter in n]
    if args.limit:
        names = names[:args.limit]

    n_scored_new = 0
    for i, name in enumerate(names):
        entry = header[name]
        shape = tuple(entry["shape"])
        numel = int(np.prod(shape))
        cls = classify(name)
        if name in done:
            continue
        if numel < MIN_SCORE_ELEMS or len(shape) < 2:
            counted[name] = {"tensor_class": cls, "shape": list(shape),
                             "dtype": entry["dtype"], "numel": numel,
                             "layer": layer_index(name)}
            print(f"[{i+1:3d}/{len(names)}] {name}  -> nur gezaehlt ({cls}, numel={numel})")
            results["counts_only"] = counted
            atomic_write_json(results, OUT_JSON)
            continue
        t0 = time.time()
        try:
            rec = score_tensor(name, entry, ST_FILE, data_start)
        except Exception as e:
            rec = {"name": name, "tensor_class": cls, "shape": list(shape),
                   "dtype": entry["dtype"], "error": f"{type(e).__name__}: {e}"}
            print(f"[{i+1:3d}/{len(names)}] {name}  ERROR: {rec['error']}")
        results["tensors"].append(rec)
        n_scored_new += 1
        if "error" not in rec:
            print(f"[{i+1:3d}/{len(names)}] {name}  cls={rec['tensor_class']:>16s} "
                  f"CASI={rec['casi_raw']:9.1f} null={rec['casi_null_mean']:9.1f} "
                  f"ratio={rec['ratio']:8.3f}  ({time.time()-t0:5.1f}s, "
                  f"rss={peak_rss_mb():.0f}MB peak)")
        results["runtime"]["last_tensor_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(results, OUT_JSON)

    results["counts_only"] = counted
    results["runtime"]["finished"] = datetime.now(timezone.utc).isoformat()
    results["runtime"]["seconds"] = round(time.time() - T0, 1)
    results["runtime"]["peak_rss_mb"] = round(peak_rss_mb(), 1)
    atomic_write_json(results, OUT_JSON)

    scored = results["tensors"]
    print("\n" + "=" * 78)
    print(f"FERTIG: {len(scored)} gescoret ({n_scored_new} neu), "
          f"{len(counted)} nur gezaehlt")
    print(f"Laufzeit {results['runtime']['seconds']}s, Peak-RSS "
          f"{results['runtime']['peak_rss_mb']} MB")
    print(f"Output: {OUT_JSON}")


if __name__ == "__main__":
    main()
