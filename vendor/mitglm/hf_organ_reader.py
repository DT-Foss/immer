#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hf_organ_reader — On-Demand-Streaming von Tensor-Informationen aus
HuggingFace-Safetensors ueber HTTP-Range-Requests, OHNE Modell-Download.

Ziel (PARRHESIA v0.4, Modul 1 PATHOLOGIST — Weight-Level Forensics):
Ein kleines Modell soll sich gezielt "Faehigkeits-Organe" aus Frontier-Weights
lesen koennen, ohne die Frontier-Weights zu laden. Safetensors = 8-Byte-
Headerlaenge + JSON-Header (alle Tensornamen/Offets/Shapes/dtypes) + rohe
Payload-Bloecke. HuggingFace resolve-URLs folgen einem 302 auf einen CDN
(xet-bridge), der Single-Range-Requests (206 Partial Content) unterstuetzt.

Damit:
  1. model.safetensors.index.json holen  -> Shard-Liste
  2. pro Shard NUR den Header-Range lesen -> vollstaendiges Tensor-Inventar
     ohne Payload
  3. gezielt einzelne Tensoren / Zeilen-Bloecke per Range-Request streamen
  4. auf den gestreamten Bytes Struktur-Signaturen rechnen (CASI-Ratio nach
     T02b/E01 + SVD-alpha-Vorschau)

Messgroesse des Demos: Gesamttransfer < 200 MB fuer einen kompletten
Organ-Scan eines Modells > 20 GB — die Selektivitaet IST der Beweis.

Subsystem-Wegpunkte (Rotation, dokumentiert in hf_organ_reader_loop.md):
  - Multipart-Range (bytes=a-b,c-d) wird vom HF-CDN mit 416 abgewiesen ->
    Rotation auf Single-Range-Requests pro Block.
  - Uniform ueber den Tensor verteilte Zeilen-Samples + Luecken-Merging
    Fuehren zu massivem Overfetch (Faktor ~6 gemessen, siehe Loop-Log I2)
    -> Rotation auf BLOCK-KONTIGUIERLICHES Sampling: N Bloecke zu je
    zusammenhaengenden Zeilen, ebenverteilt + seeded Jitter. Transfer ==
    Analysematrix, Overfetch ~0.
  - CASI-Ratio ist analysisgroessen-abhaengig (Drift 4.98x bei sqrt(n)-
    Vorhersage 2.45x, Exponent ~0.87 — Loop-Log I2) -> kein parametrischer
    Korrekturfaktor; stattdessen MATCHED-PROTOCOL-Vergleich: das kleine
    Referenzmodell wird mit identischer Block-Geometrie geprobt.
  - Redirects werden MANUELL verfolgt (resolve -> CDN-URL gecacht bis
    Signatur-Expiry), Range geht direkt auf die finale URL.

Determinismus: seed 42; Zeilen-Subsamples via
rng = np.random.default_rng([42, crc32(name)]) — identisch zu E01
(casi_tensor_map.py), sodass --e01-exact Probes die E01-Ratios BITGENAU
reproduzieren (Validierung des HTTP-Pfads gegen die Datei-Pipeline —
I1 im Loop-Log: 6/6 Tensoren exakt).

Schreibt ausschliesslich nach mitGLM/results/hf_organ_reader*.{json,md}.
Quell-Ressourcen (E01-JSON, Taxonomie, CASI-Engine) werden nur gelesen.
"""
import argparse
import fnmatch
import json
import math
import os
import re
import struct
import sys
import time
import urllib.parse
import zlib
from datetime import datetime, timezone

import numpy as np
import requests

# ── E01-Pipeline (casi_tensor_map) fuer Bit-Exaktheit importieren ────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "results"))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import casi_tensor_map as ctm  # noqa: E402  (nur Lesen; keine Seiteneffekte)

SEED = ctm.SEED                       # 42
N_NULL = ctm.N_NULL                   # 5
E01_MAX_ELEMS = ctm.MAX_ELEMS         # 8_388_608 (E01-Deckel, --e01-exact)
MIN_SCORE_ELEMS = ctm.MIN_SCORE_ELEMS  # 4096

HF_ENDPOINT = "https://huggingface.co"
DEFAULT_BUDGET_MB = 200.0
HTTP_MAX_ELEMS = 1_048_576            # HTTP-Modus: Element-Deckel pro Tensor
HTTP_N_BLOCKS = 8                     # Block-kontiguierliches Sampling
ROWS_FLOOR = 128                      # Engine-Operabilitaet: <100 Zeilen -> 0.0
PER_TENSOR_CEIL = 24 * 1024 * 1024    # harte Byte-Obergrenze pro Tensor
REQUEST_TIMEOUT = (15, 180)

E01_JSON = os.path.join(RESULTS_DIR, "e01_casi_tensor_map_qwen35_08b.json")

# Dynamik-/Mischer-Klassen (Taxonomie tensor_taxonomy_qwen35.json:
# mixer_candidate). gdn_conv/gdn_ba = hochspezialisiert (Reasoning-Kandidaten),
# embed = Weltwissen-Anker, ViT ≈ 1 (Random-Aequivalent-Anker).
DYNAMICS_CLASSES = {"gdn_conv", "gdn_ba", "gdn_in_proj", "gdn_out",
                    "attn_qkv", "attn_o"}


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def slug(repo):
    return repo.replace("/", "_")


# ════════════════════════════════════════════════════════════════════════════
# Transfer-Budget (hart)
# ════════════════════════════════════════════════════════════════════════════
class BudgetExceeded(Exception):
    pass


class Budget:
    """Zaehlt JEDE uebertragene Byte: Response-Bodies exakt, HTTP-Header-
    Overhead als Approximation (Statuszeile + Header-Laengen). Sichtbar nach
    jeder Operation; Ueberschreitung der harten Grenze -> Abbruch."""

    def __init__(self, limit_mb):
        self.limit = int(limit_mb * 1024 * 1024)
        self.body = 0
        self.overhead = 0
        self.requests = 0
        self.log = []          # (tag, body, overhead)

    def charge(self, body, overhead, tag):
        self.body += body
        self.overhead += overhead
        self.requests += 1
        self.log.append((tag, body, overhead))
        if self.total > self.limit:
            raise BudgetExceeded(
                f"Budget hart ueberschritten: {self.total}/{self.limit} Bytes "
                f"({self.total / 1048576:.1f}/{self.limit / 1048576:.0f} MB) "
                f"bei '{tag}'")

    @property
    def total(self):
        return self.body + self.overhead

    def line(self):
        pct = 100.0 * self.total / self.limit
        return (f"[budget {self.total / 1048576:7.2f}/{self.limit / 1048576:.0f} MB "
                f"({pct:5.1f}%)  body={self.body / 1048576:7.2f} MB  "
                f"hdr~{self.overhead / 1024:6.1f} KB  reqs={self.requests}]")

    def as_dict(self):
        return {"limit_bytes": self.limit, "bytes_total": self.total,
                "bytes_body": self.body, "bytes_overhead_approx": self.overhead,
                "http_requests": self.requests}


# ════════════════════════════════════════════════════════════════════════════
# HTTP-Range-Reader (manueller Redirect, CDN-URL-Cache, Retries)
# ════════════════════════════════════════════════════════════════════════════
class HFRangeReader:
    def __init__(self, repo, revision="main", budget=None, session=None):
        self.repo = repo
        self.rev = revision
        self.budget = budget or Budget(DEFAULT_BUDGET_MB)
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = (
            "hf-organ-reader/1.0 (range-streaming weight forensics)")
        self._cdn = {}          # filename -> (url, expires_epoch)
        self.file_info = {}     # filename -> {size, etag, cas_url_hash}

    # -- Redirect-Aufloesung --------------------------------------------------
    def resolve_url(self, filename):
        return f"{HF_ENDPOINT}/{self.repo}/resolve/{self.rev}/{filename}"

    def _cdn_url(self, filename):
        ent = self._cdn.get(filename)
        if ent and ent[1] > time.time():
            return ent[0]
        url = self.resolve_url(filename)
        for attempt in range(4):
            try:
                r = self.session.get(url, allow_redirects=False,
                                     headers={"Range": "bytes=0-0"},
                                     timeout=REQUEST_TIMEOUT)
            except requests.RequestException as e:
                if attempt == 3:
                    raise RuntimeError(f"Netzfehler bei resolve {filename}: {e}") from e
                time.sleep(2 ** attempt)
                continue
            overhead = self._overhead(r)
            if r.status_code in (301, 302, 303, 307, 308):
                loc = r.headers.get("Location", "")
                exp = self._location_expiry(loc)
                self.budget.charge(len(r.content), overhead, f"resolve:{filename}")
                self._cdn[filename] = (loc, exp)
                # CAS/xet-Identitaet aus der signierten URL (Merkle-artig):
                m = re.search(r"/([0-9a-f]{64})\?", loc)
                info = self.file_info.setdefault(filename, {})
                if m:
                    info["cas_url_hash"] = m.group(1)
                info.setdefault("cdn_host", urllib.parse.urlparse(loc).netloc)
                return loc
            if r.status_code == 200:
                # Datei wird inline ausgeliefert (kleine Dateien) — kein CDN.
                self.budget.charge(len(r.content), overhead,
                                   f"resolve-inline:{filename}")
                self._cdn[filename] = (url, time.time() + 3600)
                return url
            if r.status_code in (401, 403):
                raise RuntimeError(
                    f"{r.status_code} bei {filename} — Repo gated/gesperrt. "
                    f"Rotation noetig (alternatives offenes Repo).")
            if r.status_code == 404:
                raise RuntimeError(
                    f"404: {filename} existiert nicht in {self.repo}@{self.rev}")
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(
                f"Unerwarteter Status {r.status_code} bei resolve {filename}")
        raise RuntimeError(f"resolve {filename} nach 4 Versuchen gescheitert")

    @staticmethod
    def _location_expiry(loc):
        try:
            q = urllib.parse.parse_qs(urllib.parse.urlparse(loc).query)
            return int(q.get("Expires", [time.time() + 900])[0]) - 60
        except Exception:
            return time.time() + 900

    @staticmethod
    def _overhead(resp):
        ov = 64  # Statuszeile/Request-Line-Pauschale
        try:
            ov += len(str(resp.request.headers))
        except Exception:
            ov += 512
        try:
            ov += len(str(resp.headers))
        except Exception:
            ov += 512
        return ov

    # -- Range-GET ------------------------------------------------------------
    def get_range(self, filename, start, end):
        """Bytes [start, end] (inklusive) aus filename — genau ein Range-Request."""
        url = self._cdn_url(filename)
        last = None
        for attempt in range(4):
            try:
                r = self.session.get(url, allow_redirects=False,
                                     headers={"Range": f"bytes={start}-{end}"},
                                     timeout=REQUEST_TIMEOUT)
            except requests.RequestException as e:
                last = e
                time.sleep(2 ** attempt)
                continue
            overhead = self._overhead(r)
            if r.status_code == 206:
                cr = r.headers.get("Content-Range", "")
                m = re.match(r"bytes (\d+)-(\d+)/(\d+)", cr)
                if not m or int(m.group(1)) != start or int(m.group(2)) != end:
                    raise RuntimeError(
                        f"Content-Range-Mismatch bei {filename}: erwartet "
                        f"{start}-{end}, bekommen '{cr}'")
                info = self.file_info.setdefault(filename, {})
                info["size"] = int(m.group(3))
                if "etag" not in info and r.headers.get("etag"):
                    info["etag"] = r.headers.get("etag")
                self.budget.charge(len(r.content), overhead,
                                   f"range:{filename}:{start}")
                return r.content
            if r.status_code == 200:
                # Range ignoriert — nur akzeptieren, wenn der Body exakt dem
                # Wunschfenster entspricht (inline-Fall), sonst Budget-Bruch.
                if start == 0 and len(r.content) - 1 == end:
                    self.budget.charge(len(r.content), overhead,
                                       f"inline:{filename}")
                    return r.content
                raise RuntimeError(
                    f"Server ignorierte Range bei {filename} "
                    f"(200 statt 206, {len(r.content)} Bytes)")
            if r.status_code == 416:
                raise RuntimeError(
                    f"416 Range Not Satisfiable bei {filename} [{start}-{end}]")
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(min(30, 2 ** attempt))
                continue
            raise RuntimeError(f"Status {r.status_code} bei Range-GET {filename}")
        raise RuntimeError(f"Range-GET {filename}[{start}-{end}] nach 4 Versuchen "
                           f"gescheitert: {last}")

    def fetch_file(self, filename):
        """Kleine Datei (JSON-Metadaten) komplett laden (folgt Redirect)."""
        url = self.resolve_url(filename)
        r = self.session.get(url, allow_redirects=True, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        self.budget.charge(len(r.content), self._overhead(r), f"file:{filename}")
        return r.content

    # -- Safetensors-Header ---------------------------------------------------
    def fetch_st_header(self, filename):
        b8 = self.get_range(filename, 0, 7)
        (hlen,) = struct.unpack("<Q", b8)
        if not (0 < hlen <= 64 * 1024 * 1024):
            raise RuntimeError(f"unsinnige Header-Laenge {hlen} bei {filename}")
        hdr = self.get_range(filename, 8, 8 + hlen - 1)
        header = json.loads(hdr.decode("utf-8"))
        meta = header.pop("__metadata__", None)
        info = self.file_info.setdefault(filename, {})
        info["header_len"] = hlen
        if meta:
            info["st_metadata"] = {k: str(v)[:200]
                                   for k, v in list(meta.items())[:8]}
        return header, 8 + hlen, info


# ════════════════════════════════════════════════════════════════════════════
# Inventar (scan)
# ════════════════════════════════════════════════════════════════════════════
def scan_inventory(reader, budget):
    """Index + alle Shard-Header per Range holen -> Tensor-Inventar OHNE Payload."""
    inv = {"tool": "hf_organ_reader", "command": "scan", "repo": reader.repo,
           "revision": reader.rev, "scanned_at": now_utc(), "shards": [],
           "tensors": [], "index_bytes": 0}
    shard_files = None
    try:
        idx_raw = reader.fetch_file("model.safetensors.index.json")
        idx = json.loads(idx_raw)
        shard_files = sorted(set(idx["weight_map"].values()))
        inv["index_bytes"] = len(idx_raw)
        inv["index_total_size"] = idx.get("metadata", {}).get("total_size")
    except (requests.HTTPError, RuntimeError):
        shard_files = ["model.safetensors"]
    total_bytes = 0
    for sf in shard_files:
        header, data_start, info = reader.fetch_st_header(sf)
        for name, e in header.items():
            off_b, off_e = e["data_offsets"]
            shape = list(e["shape"])
            numel = int(np.prod(shape)) if shape else 1
            inv["tensors"].append({
                "name": name, "tensor_class": ctm.classify(name),
                "layer": ctm.layer_index(name), "shard": sf, "dtype": e["dtype"],
                "shape": shape, "numel": numel, "bytes": off_e - off_b,
                "offset_in_shard": [off_b, off_e], "data_start": data_start})
            total_bytes += off_e - off_b
        inv["shards"].append({
            "file": sf, "n_tensors": len(header),
            "header_len": info.get("header_len"), "data_start": data_start,
            "size": info.get("size"), "etag": info.get("etag"),
            "cas_url_hash": info.get("cas_url_hash"),
            "cdn_host": info.get("cdn_host"),
            "st_metadata": info.get("st_metadata")})
        print(f"  shard {sf}: {len(header):4d} Tensoren, "
              f"header={info.get('header_len')} B size={info.get('size')} "
              f"cas={str(info.get('cas_url_hash'))[:16]}…  {budget.line()}")
    if inv.get("index_total_size"):
        dev = abs(total_bytes - int(inv["index_total_size"]))
        assert dev == 0, (f"Payload-Summe {total_bytes} != index.total_size "
                          f"{inv['index_total_size']}")
    inv["model_payload_bytes"] = total_bytes
    if not inv.get("index_total_size"):
        inv["index_total_size"] = total_bytes
    hist = {}
    for t in inv["tensors"]:
        hist[t["tensor_class"]] = hist.get(t["tensor_class"], 0) + 1
    inv["class_histogram"] = dict(sorted(hist.items(), key=lambda kv: -kv[1]))
    inv["budget"] = budget.as_dict()
    return inv


def scan_path(repo):
    return os.path.join(RESULTS_DIR, f"hf_organ_reader_scan_{slug(repo)}.json")


# ════════════════════════════════════════════════════════════════════════════
# Zeilen-Streaming: Block-kontiguierliches Sampling (Transfer == Analyse)
# ════════════════════════════════════════════════════════════════════════════
def fetch_full(reader, shard, data_start, off_b, numel, itemsize):
    data = reader.get_range(shard, data_start + off_b,
                            data_start + off_b + numel * itemsize - 1)
    return np.frombuffer(data, np.uint8), 1


def fetch_blocks(reader, shard, data_start, off_b, n_rows, n_cols, itemsize,
                 cap, name, n_blocks=HTTP_N_BLOCKS, rows_floor=ROWS_FLOOR,
                 rows_exact=0):
    """BLOCK-KONTIGUIERLICHES Sampling (Rotation R2): n_blocks Bloecke aus
    jeweils block_rows zusammenhaengenden Zeilen, ebenverteilt ueber den
    Tensor mit seeded Jitter (eigener RNG-Strom, beeinflusst den Analyse-RNG
    nicht). Jeder Block = EIN Range-Request. Transfer == Analysematrix.

    Rotation R4: Die CASI-Engine ist unter 100 Zeilen ("Keys") inoperabel
    (alle Strategien guarden auf 0.0 — empirisch: 96 Zeilen -> CASI 0.000,
    100 Zeilen -> 17.8, deckungsgleich mit E01 MIN_ROWS=100). Breite
    Matrizen (z.B. mlp_down 17408 Spalten) wuerden unter dem Element-Deckel
    in diese Zone fallen -> Zeilen-Floor 128 erzwingen (Byte-Ceiling sei
    Dank). rows_exact > 0 erzwingt exakte Zeilenzahl (matched protocol)."""
    row_bytes = n_cols * itemsize
    n_blocks = max(1, min(n_blocks, n_rows))
    if rows_exact:
        block_rows = max(1, rows_exact // n_blocks)
    else:
        block_rows = max(1, cap // (n_blocks * n_cols))
        if block_rows * n_blocks < rows_floor:
            block_rows = max(1, (rows_floor + n_blocks - 1) // n_blocks)
    block_rows = min(block_rows, n_rows)
    if block_rows * row_bytes > PER_TENSOR_CEIL:
        raise RuntimeError(
            f"{name}: Zeilen-Floor sprengt Tensor-Ceiling "
            f"({block_rows} Rows x {row_bytes} B > {PER_TENSOR_CEIL})")
    brng = np.random.default_rng([SEED, zlib.crc32(name.encode("utf-8")), 13])
    span = n_rows - block_rows * n_blocks          # Startverschiebungs-Raum
    out = np.empty(block_rows * n_blocks * row_bytes, dtype=np.uint8)
    abs_base = data_start + off_b
    reqs = 0
    for i in range(n_blocks):
        ideal = round(i * (span + 1) / n_blocks) if n_blocks > 1 else 0
        jitter = int(brng.integers(0, max(1, span // max(1, n_blocks) + 1)))
        start_row = min(span, max(0, ideal + jitter - (span // (2 * n_blocks))))
        s = abs_base + start_row * row_bytes
        data = reader.get_range(shard, s, s + block_rows * row_bytes - 1)
        reqs += 1
        i0 = i * block_rows * row_bytes
        out[i0:i0 + block_rows * row_bytes] = np.frombuffer(data, np.uint8)
    return out, reqs, block_rows * n_blocks


def fetch_uniform_rows(reader, shard, data_start, off_b, n_cols, itemsize,
                       rows_sorted, merge_gap=32768):
    """E01-Subsample-Geometrie (uniforme Zeilen) — nur fuer --e01-exact.
    Merge-Luecke klein halten; Overfetch wird gemessen und ausgewiesen."""
    row_bytes = n_cols * itemsize
    out = np.empty(len(rows_sorted) * row_bytes, dtype=np.uint8)
    abs_base = data_start + off_b
    row_abs = {r: abs_base + r * row_bytes for r in rows_sorted}
    idx_of = {r: i for i, r in enumerate(rows_sorted)}
    blocks = []
    for r in rows_sorted:
        s = row_abs[r]
        e = s + row_bytes
        if blocks and s - blocks[-1][1] <= merge_gap:
            blocks[-1][1] = e
            blocks[-1][2].append(r)
        else:
            blocks.append([s, e, [r]])
    for s, e, rows in blocks:
        data = reader.get_range(shard, s, e - 1)
        for r in rows:
            src = row_abs[r] - s
            i0 = idx_of[r] * row_bytes
            out[i0:i0 + row_bytes] = np.frombuffer(data, np.uint8, row_bytes, src)
    return out


# ════════════════════════════════════════════════════════════════════════════
# SVD-alpha-Vorschau (randomisierte Spektral-Skizze)
# ════════════════════════════════════════════════════════════════════════════
def svd_alpha_preview(mat_f32, name, k=48):
    """Spektrale Zerfalls-Skizze: sv_i ~ i^-alpha (log-log-Fit ueber die top-k
    Approximation einer randomisierten SVD, 1 Power-Iteration, seed 42)."""
    m, n = mat_f32.shape
    k = int(min(k, min(m, n) - 1))
    if k < 4:
        return None
    rng = np.random.default_rng([SEED, zlib.crc32(name.encode("utf-8")), 777])
    A = np.ascontiguousarray(mat_f32, dtype=np.float64)
    omega = rng.standard_normal((n, k))
    Y = A @ (A.T @ (A @ omega))             # 2 Power-Iterationen
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ A                             # k x n Sketch
    sv = np.linalg.svd(B, compute_uv=False)
    ii = np.arange(1, len(sv) + 1, dtype=np.float64)
    good = sv > 0
    if good.sum() < 4:
        return None
    slope, _ = np.polyfit(np.log(ii[good]), np.log(sv[good]), 1)
    p = sv ** 2 / np.sum(sv ** 2)
    eff_rank = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
    return {"k": k, "alpha_decay": round(float(-slope), 4),
            "sigma1": round(float(sv[0]), 4),
            "sigma_k": round(float(sv[min(k, len(sv)) - 1]), 6),
            "spectral_entropy_eff_rank": round(eff_rank, 2)}


# ════════════════════════════════════════════════════════════════════════════
# Tensor-Scoring ueber HTTP (spiegelt E01 score_tensor exakt)
# ════════════════════════════════════════════════════════════════════════════
def score_streamed(name, inv_entry, reader, max_elems, e01_exact=False,
                   n_blocks=HTTP_N_BLOCKS, rows_floor=ROWS_FLOOR,
                   rows_exact=0):
    """CASI-Ratio eines Tensors, der per HTTP-Range gestreamt wird.
    rng-Verbrauch identisch zu E01 (choice -> quantile-Tiebreak -> nulls),
    deshalb reproduziert e01_exact=True die E01-Ratios bitgenau (fuer
    Tensoren <= 8.4M Elemente, I1-validiert)."""
    t_fetch0 = reader.budget.total
    rng = np.random.default_rng([SEED, zlib.crc32(name.encode("utf-8"))])
    shape = tuple(inv_entry["shape"])
    dt = inv_entry["dtype"]
    itemsize = ctm._DT_SIZE[dt]
    mat2d, reshaped = ctm.natural_2d(shape)
    if mat2d is None:
        return None
    n_rows, n_cols = mat2d
    transposed = False
    if n_rows < ctm.MIN_ROWS and n_cols >= ctm.MIN_ROWS:
        transposed = True
        mat2d = (n_cols, n_rows)
        n_rows, n_cols = mat2d
    numel = n_rows * n_cols
    cap = E01_MAX_ELEMS if e01_exact else max_elems
    subsampled = numel > cap
    row_idx = None
    if subsampled and e01_exact:
        # E01-Geometrie: uniforme Zeilen aus dem ANALYSE-rng (Reihenfolge
        # des rng-Verbrauchs muss exakt E01 entsprechen).
        keep = cap // n_cols
        row_idx = np.sort(rng.choice(n_rows, size=keep, replace=False))
    shard = inv_entry["shard"]
    data_start = inv_entry["data_start"]
    off_b = inv_entry["offset_in_shard"][0]
    fetched = blocks = 0
    analyzed_rows = n_rows
    sampling = "full"

    if not transposed and row_idx is None:
        # Voll-Fetch nur wenn unter dem Deckel UND (kein rows_exact gewuenscht
        # ODER der Tensor eh weniger Zeilen hat) — sonst Block-Geometrie.
        want_full = numel <= cap and not (rows_exact and n_rows > rows_exact)
        if want_full:
            raw, blocks = fetch_full(reader, shard, data_start, off_b,
                                     numel, itemsize)
            fetched = len(raw)
            sampling = "full"
        else:
            raw, blocks, analyzed_rows = fetch_blocks(
                reader, shard, data_start, off_b, n_rows, n_cols, itemsize,
                cap, name, n_blocks=n_blocks, rows_floor=rows_floor,
                rows_exact=rows_exact)
            fetched = len(raw)
            sampling = f"blocks(n={blocks},rows={analyzed_rows})"
    elif not transposed:
        raw = fetch_uniform_rows(reader, shard, data_start, off_b, n_cols,
                                 itemsize, row_idx)
        fetched, blocks = len(raw), len(row_idx)
        analyzed_rows = len(row_idx)
        sampling = "uniform-rows(e01)"

    if transposed:
        # E01-Konvention: Analyse auf der Transponierten (Zeilen<100).
        # Eine Transponierte kann per Range nicht gestreamt werden -> volle
        # Tensor-Bytes laden (Guard; nur kleine Tensoren, z.B. gdn_ba).
        full_bytes = numel * itemsize
        if full_bytes > 32 * 1024 * 1024:
            raise RuntimeError(
                f"transponierter Analysefall bei {name} braucht vollen Tensor "
                f"({full_bytes} B) — ueber 32MB-Guard, Sampling-Rotation noetig")
        raw, blocks = fetch_full(reader, shard, data_start, off_b,
                                 numel, itemsize)
        fetched = len(raw)
        sampling = "full+transpose"
        # Nach dem Swap gilt: (n_cols, n_rows) == ORIGINAL-2D-Form.
        # E01-Semantik: Rohbytes auf Originalform reshaped, dann .T -> mat2d.
        if dt == "BF16":
            u16 = raw.view(np.uint16)
            m_t = u16.reshape((n_cols, n_rows)).T.copy()
            f32 = ctm.bf16_rows_to_f32(m_t, mat2d, None)
        else:
            f32 = raw.view(np.float32).reshape((n_cols, n_rows)).T.copy()
        if subsampled:
            # (nur e01_exact; 0.8B/27B-gdn_ba sind klein genug fuer voll)
            f32 = f32[row_idx]
            analyzed_rows = len(row_idx)
    else:
        if dt == "BF16":
            f32 = ctm.bf16_rows_to_f32(raw.view(np.uint16),
                                       (analyzed_rows, n_cols), None)
        else:
            f32 = ctm.f32_rows_to_f32(raw.view(np.float32),
                                      (analyzed_rows, n_cols), None)

    bytes_moved = reader.budget.total - t_fetch0
    nan_count = int(np.count_nonzero(np.isnan(f32)))
    n_unique = int(np.unique(f32).size)
    svd = svd_alpha_preview(f32, name)
    b2d = ctm.global_quantile_u8(f32, rng)
    del f32
    casi_raw, profile, nulls, null_mean, ratio = ctm.casi_with_nulls(b2d, rng)
    del b2d

    return {
        "name": name,
        "tensor_class": ctm.classify(name),
        "layer": ctm.layer_index(name),
        "shard": shard,
        "shape": list(shape),
        "dtype": dt,
        "numel": int(np.prod(shape)),
        "numel_analyzed": int(analyzed_rows * n_cols),
        "casi_input_shape": [analyzed_rows, n_cols],
        "sampling": sampling,
        "reshaped_from_nd": bool(reshaped),
        "transposed": transposed,
        "e01_exact_mode": bool(e01_exact),
        "range_requests": blocks,
        "bytes_moved": int(bytes_moved),
        "overfetch_ratio": round(fetched / max(1, analyzed_rows * n_cols * itemsize), 4),
        "nan_count": nan_count,
        "n_unique": n_unique,
        "tie_fraction": round(1.0 - n_unique / (analyzed_rows * n_cols), 6),
        "casi_raw": round(float(casi_raw), 3),
        "casi_null_mean": round(null_mean, 3),
        "casi_null_std": round(float(nulls.std()), 3),
        "ratio": round(ratio, 4),
        "svd_preview": svd,
        "profile_raw": {n: round(float(z), 2)
                        for n, z in zip(ctm.STRATEGY_NAMES, profile)},
    }


# ════════════════════════════════════════════════════════════════════════════
# report: Organ-Plan + Kalibrier-Vergleich
# ════════════════════════════════════════════════════════════════════════════
# Stratifikation: pro Klasse N Layer seeded ziehen; kleine Tensoren
# (gdn_conv/gdn_ba) komplett laden (volle Fidelitaet), grosse auf HTTP_MAX_ELEMS.
PLAN_K = [
    ("gdn_conv", 6), ("gdn_ba", 6), ("gdn_in_proj", 5), ("gdn_out", 5),
    ("attn_qkv", 3), ("attn_o", 4),
    ("mlp_gate", 3), ("mlp_up", 3), ("mlp_down", 3),
    ("embed", 1), ("lm_head", 1),
    ("visual_attn_qkv", 2), ("visual_mlp_up", 2), ("visual_mlp_down", 2),
    ("mtp_fc", 1), ("mtp_mlp_gate", 1),
]


def build_plan(inv, max_elems, k_scale=1.0, only_classes=None):
    by_class = {}
    for t in inv["tensors"]:
        if t["numel"] < MIN_SCORE_ELEMS or len(t["shape"]) < 2:
            continue
        by_class.setdefault(t["tensor_class"], []).append(t)
    plan = []
    for cls, k in PLAN_K:
        if cls not in by_class:
            continue
        if only_classes and cls not in only_classes:
            continue
        cand = by_class[cls]
        k = max(1, int(round(k * k_scale)))
        rng = np.random.default_rng([SEED, zlib.crc32(cls.encode("utf-8"))])
        if len(cand) <= k:
            chosen = cand
        else:
            # Schichtung ueber Layer: Layer ziehen, dann alle Tensoren der Layer
            layers = sorted({c["layer"] for c in cand if c["layer"] is not None})
            if layers and len(layers) > k:
                sel = np.sort(rng.choice(len(layers), size=k, replace=False))
                sel_layers = set(np.array(layers)[sel].tolist())
                chosen = [c for c in cand if c["layer"] in sel_layers]
            else:
                idx = np.sort(rng.choice(len(cand), size=k, replace=False))
                chosen = [cand[i] for i in idx]
        plan.extend(chosen)
    for t in plan:
        if t["tensor_class"] in ("gdn_conv", "gdn_ba") and t["numel"] <= 2_000_000:
            t["_max_elems"] = t["numel"]
        else:
            t["_max_elems"] = max_elems
    return plan


def estimate_plan_bytes(plan, n_blocks=HTTP_N_BLOCKS):
    est = 0
    for t in plan:
        n_cols = t["shape"][-1] if len(t["shape"]) >= 2 else t["shape"][0]
        eff = min(t["numel"], t["_max_elems"])
        eff = max(eff, min(t["numel"], ROWS_FLOOR * n_cols))  # Zeilen-Floor
        est += eff * ctm._DT_SIZE.get(t["dtype"], 2)
        est += (n_blocks + 2) * 1400  # Request-Overhead-Pauschale
    return int(est * 1.02)


def load_calibration(path=E01_JSON):
    with open(path) as f:
        d = json.load(f)
    per_class = {}
    for t in d.get("tensors", []):
        if "ratio" in t:
            per_class.setdefault(t["tensor_class"], []).append(t["ratio"])
    cal = {}
    for cls, rs in per_class.items():
        cal[cls] = {"n": len(rs), "mean": round(sum(rs) / len(rs), 3),
                    "min": round(min(rs), 3), "max": round(max(rs), 3)}
    cal["_model"] = d.get("model", {})
    return cal


def aggregate_classes(records):
    agg = {}
    for r in records:
        if "ratio" not in r:
            continue
        agg.setdefault(r["tensor_class"], []).append(r)
    out = {}
    for cls, rs in agg.items():
        ratios = np.array([r["ratio"] for r in rs])
        alphas = [r["svd_preview"]["alpha_decay"] for r in rs
                  if r.get("svd_preview")]
        analyzed = int(np.median([r["numel_analyzed"] for r in rs]))
        rows_an = int(np.median([r["casi_input_shape"][0] for r in rs]))
        out[cls] = {"n": len(rs), "mean": round(float(ratios.mean()), 3),
                    "median": round(float(np.median(ratios)), 3),
                    "min": round(float(ratios.min()), 3),
                    "max": round(float(ratios.max()), 3),
                    "std": round(float(ratios.std()), 3),
                    "numel_analyzed_median": analyzed,
                    "rows_analyzed_median": rows_an,
                    "svd_alpha_mean": round(sum(alphas) / len(alphas), 3)
                    if alphas else None}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["mean"]))


def atomic_write_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def load_or_scan(reader, budget, refresh=False):
    path = scan_path(reader.repo)
    if os.path.exists(path) and not refresh:
        with open(path) as f:
            inv = json.load(f)
        print(f"Inventar-Cache: {path} ({len(inv['tensors'])} Tensoren, "
              f"Modell {inv['model_payload_bytes'] / 1e9:.2f} GB) — 0 neue Bytes.")
        return inv
    inv = scan_inventory(reader, budget)
    atomic_write_json(inv, path)
    print(f"Inventar gespeichert: {path}")
    return inv


# ── Demo-Markdown ────────────────────────────────────────────────────────────
def write_demo_md(path, demo):
    repo = demo["repo"]
    total_gb = demo["model_payload_bytes"] / 1e9
    b = demo["budget"]
    sel = 100.0 * b["bytes_total"] / demo["model_payload_bytes"]
    cls_rows = demo["class_aggregate"]
    cal = demo.get("calibration_e01", {})
    matched = (demo.get("matched_calibration") or {}).get("classes", {})
    emb = cls_rows.get("embed", {}).get("mean") or 1.0
    lines = []
    A = lines.append
    A(f"# HF Organ-Reader — On-Demand-Organ-Scan: `{repo}`")
    A("")
    A(f"*Generiert {demo['finished_at']} — `hf_organ_reader.py report` "
      f"(seed 42, deterministisch).*")
    A("")
    A("**Methodik-Rahmen:** PARRHESIA v0.4, Modul 1 (PATHOLOGIST) — Weight-Level "
      "Forensics: Spektral-Signatur-Analyse (SVD) und Weight-Anomalie-Detektion, "
      "hier realisiert als remote CASI-Struktursignatur (T02b: global-quantile "
      "uint8 mit Zufalls-Tie-Breaking, Ratio gegen permutations-zerstoerte Null) "
      "plus SVD-alpha-Zerfallsskizze — ausgefuehrt als HTTP-Range-Streaming direkt "
      "auf die Safetensors-Shards, ohne Modell-Download.")
    A("")
    A("## Beweis-Satz")
    A("")
    A(f"> Wir haben **{b['bytes_total'] / 1048576:.1f} MB** eines "
      f"**{total_gb:.1f}-GB-Modells** uebertragen ({sel:.4f} % des Gewichts, "
      f"{b['http_requests']} HTTP-Range-Requests) und koennen dennoch sagen, "
      f"welche Komponenten die strukturierten sind: komplettes Tensor-Inventar "
      f"aller {demo['n_shards']} Shards plus CASI-Organ-Signaturen von "
      f"{demo['n_scored']} Tensoren in {len(cls_rows)} Klassen.")
    A("")
    A("## Transfer-Bilanz")
    A("")
    A("| Posten | Bytes |")
    A("|---|---:|")
    A(f"| index.json | {demo['index_bytes']:,} |")
    A(f"| Shard-Header (JSON, kein Payload) | {demo['header_bytes']:,} |")
    A(f"| Tensor-Payload (Block-Zeilen-Slices) | "
      f"{b['bytes_body'] - demo['index_bytes'] - demo['header_bytes']:,} |")
    A(f"| HTTP-Header-Overhead (approx) | {b['bytes_overhead_approx']:,} |")
    A(f"| **Gesamt** | **{b['bytes_total']:,}** |")
    A("")
    A(f"Budget: {b['bytes_total'] / 1048576:.1f} / "
      f"{b['limit_bytes'] / 1048576:.0f} MB harte Grenze — eingehalten.")
    A("")
    A("## Organ-Scan: Klassen-Ranking (CASI-Ratio = Struktur gegenueber Zufall)")
    A("")
    A("| Klasse | n | Zeilen analysiert | ratio mean | ratio min–max | "
      "0.8B matched-protocol (Zeilen) | Organ-Score (vs. embed) | SVD-alpha |")
    A("|---|---:|---:|---:|---|---:|---:|---:|")
    for cls, st in cls_rows.items():
        mc = matched.get(cls, {})
        if mc:
            mm = f"{mc.get('mean', '—')} ({mc.get('rows_analyzed_median', '?')}r)"
        else:
            mm = "—"
        al = st.get("svd_alpha_mean")
        full_fid = "*" if st["rows_analyzed_median"] > 1000 else ""
        A(f"| {cls}{full_fid} | {st['n']} | {st['rows_analyzed_median']:,} | "
          f"{st['mean']:.2f} | {st['min']:.2f}–{st['max']:.2f} | {mm} | "
          f"{st['mean'] / emb:.2f} | {al if al is not None else '—'} |")
    A("")
    A("`*` = volle Fidelitaet (gesamter Tensor gestreamt, keine Stichprobe) — "
      "Analysegeometrie folgt der Tensorform und ist MODELLABHAENGIG "
      "(z.B. gdn_ba: 0.8B analysiert 1024x16 voll, 27B 5120x48 voll); "
      "Vergleich mit 0.8B-Vollwerten semantisch ordinal, nicht kardinal. "
      "Alle ohne `*` sind zeilengematcht (200 Zeilen) — dort ist der "
      "Cross-Modell-Vergleich kardinal.")
    A("")
    A("`Organ-Score` = Klassen-mitlere Ratio normalisiert auf die embed-Ratio "
      "desselben Modells und Protokolls (embed = Weltwissen-Anker). "
      "`0.8B matched-protocol` = Qwen3.5-0.8B mit identischer Block-Geometrie "
      "ueber denselben Reader geprobt (gleicher Element-Deckel) — "
      "CASI-Ratios sind analysegroessen-abhaengig (Loop-Log I2: Drift 4.98x "
      "bei sqrt(n)-Vorhersage 2.45x), deshalb matched statt E01-voll. "
      "E01-Kontext (volle Tensoren): embed 38.5, gdn_conv 85.2, gdn_ba 49.1, "
      "ViT-Attn ~1.4–1.6.")
    A("")
    A("## Top-Organ-Kandidaten (Dynamik-Klassen, Tensor-Ebene)")
    A("")
    A("| Rang | Tensor | Klasse | ratio | SVD-alpha | Sampling | Bytes bewegt |")
    A("|---:|---|---|---:|---:|---|---:|")
    for i, r in enumerate(demo["top_candidates"], 1):
        al = r["svd_preview"]["alpha_decay"] if r.get("svd_preview") else None
        A(f"| {i} | `{r['name']}` | {r['tensor_class']} | {r['ratio']:.2f} | "
          f"{al if al is not None else '—'} | {r.get('sampling', '—')} | "
          f"{r['bytes_moved']:,} |")
    A("")
    A("## Sampling-Entscheidungen")
    A("")
    A(f"- Zeilengematchtes Protokoll (R5): alle gesampelten Klassen mit exakt "
      f"200 analysierten Zeilen (Element-Deckel {demo['http_max_elems']:,} als "
      "Obergrenze, 8 kontiguierliche Bloecke, ebenverteilt + seeded Jitter, "
      "seed 42) — Transfer == Analysematrix, Overfetch ~0 (`overfetch_ratio` "
      "pro Tensor im JSON). 200 > 100 = Engine-Operabilitaetsgrenze (R4).")
    A("- Kleine hochspezialisierte Dynamik-Organe (gdn_conv, gdn_ba) wurden "
      "VOLLSTAENDIG gestreamt (markiert `*`) — Analyse auf der ganzen "
      "Tensorform wie in E01, modellabhaengige Groesse.")
    A("- Riesen-Tensoren (embed/lm_head > 1 GB) nur als Block-Stichprobe — "
      "die Sampling-Entscheidung steht im JSON pro Tensor.")
    A("- Transponierten-Fall (gdn_ba, Zeilenzahl < 100): voller Tensor (klein), "
      "Analyse auf der Transponierten wie E01.")
    A("- Shard-Identitaet ueber xet-CAS-Hash der signierten CDN-URL "
      "(Merkle-artige Content-Adresse) — ohne den Shard zu laden.")
    A("")
    A("## Validierung")
    A("")
    A(demo.get("validation_note", "—"))
    A("")
    A("## Naechste Ausbaustufen des Readers")
    A("")
    A("1. Zeilen-Slice-Granularitaet < Zeile (Byte-Sub-Blocks im Tensor-Inneren).")
    A("2. Merkle-artige Partial-Hashes: CASI/SHA pro Block unterwegs "
      "aggregieren, statt Bytes zu puffern.")
    A("3. Multipart-Range erneut verhandeln (heute 416) fuer 1 Request/Tensor.")
    A("4. Budget-allokierte Planskalierung: Organ-Rangliste zuerst, Rest-Budget "
      "in Verfeinerung.")
    A("")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ════════════════════════════════════════════════════════════════════════════
# W6 (A1) — Merkle-Streaming: Partial-Hashes + exakte CASI ohne Byte-Puffer
# ════════════════════════════════════════════════════════════════════════════
# NEU (additiv, beruehrt keine bestehende Semantik): Der Merkle-Pfad streamt
# dieselbe Block-Geometrie wie fetch_blocks(), haelt aber die Tensor-Bytes
# NICHT im RAM:
#   Phase 1 (Netzwerk): je Block -> SHA-256 (leaf), Chain/Tree-Aggregation,
#      exakte Wert-Histogramm-Akkumulation (BF16: 65536 Bins) + lokale
#      CASI-Partial-Statistik; die 2-Byte-Schluessel werden auf Platte
#      gespilled (np.memmap), die Rohbytes verworfen.
#   Phase 2 (ohne Netzwerk): exakte Rekonstruktion der global-quantile-uint8-
#      Codes aus Histogramm + RNG-Replay (rng.permutation(n) — identischer
#      rng-Strom wie ctm.global_quantile_u8), chunkweise Rang-Zuweisung.
#      Danach ctm.casi_with_nulls — bitgenau identisch zum gepufferten Pfad.
# Speichermodell (n = analysierte Elemente, itemsize 2 fuer BF16):
#   gepuffert : raw 2n + f32 4n + perm 8n + flat[perm] 4n + order 8n +
#               ranks(f64) 8n + codes n  ~ 35n Bytes Peak
#   streaming : Phase 1: Blockpuffer + Histogramm (512 KB) ~ konstant;
#               Phase 2: perm 8n (RNG-Replay) + codes n + Chunk-Scratch
#               ~ 9n Bytes Peak   (perm ist rng-Pflicht, kein Tensor-Payload)
import hashlib
import resource
import subprocess
import tempfile
import tracemalloc

MERKLE_CHUNK = 262_144                  # Phase-2-Chunkgroesse (Elemente)


def ru_peak_mb():
    """Prozess-Peak-RSS (macOS: Bytes, Linux: KB — hier darwin)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0


class _SegTM:
    """Segmentiertes tracemalloc fuer die ASSEMBLY-Messung (Primarmetrik):
    Engine-/Fenster-Aufrufe werden pausiert (tracemalloc auf den O(n)-
    Python-Loops der Engine kostet Faktor ~14); tracemalloc sieht nach einem
    Restart nur noch NEUE Allokationen, deshalb wird das Lebend-Volumen
    ueber Segmentgrenzen aufgetragen: Peak = max(live_vor_Segment +
    Segment-Peak). Naherung: Freigaben PAUSIERTER Allokationen bleiben
    unsichtbar (max. ~1 Block-Puffer, < 0.6 MB)."""

    def __init__(self):
        self._live = 0
        self._peak = 0
        self._on = False

    def begin(self):
        tracemalloc.start()
        self._on = True

    def _take(self):
        cur, pk = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self._on = False
        self._live += cur
        self._peak = max(self._peak, self._live + pk)
        if os.environ.get("HF_MK_SEG_DEBUG"):
            sys.stderr.write(
                f"[seg] cur={cur/1048576:.2f}MB segpeak={pk/1048576:.2f}MB "
                f"live_sum={self._live/1048576:.2f}MB "
                f"peak_cand={self._peak/1048576:.2f}MB\n")

    def pause(self):
        if self._on:
            self._take()

    def resume(self):
        if not self._on:
            tracemalloc.start()
            self._on = True

    def finish(self):
        if self._on:
            self._take()
        return self._peak


def _warm_engine():
    """Einmalige Vorwarmung VOR der Baseline-Messung: Quantile-Maschinerie,
    Engine-Strategien (inkl. corr_matrix >= 500 Zeilen) und Permutation,
    damit der ERSTE Pfad keine Cold-Start/lazy-Allokations-Seiten in seiner
    RSS-Messung traegt. 512x512: klein genug, um die Wasserlinie nicht so
    hochzulegen, dass sie die Pfade maskiert (1. Version mit 1M-Warmup
    maskierte k_proj komplett — siehe Loop-Log W6)."""
    r = np.random.default_rng(0)
    m = r.standard_normal((512, 512)).astype(np.float32)
    rng = np.random.default_rng(1)
    b = ctm.global_quantile_u8(m, rng)
    ctm.compute_fast_casi(b)
    ctm.compute_fast_casi(
        b.ravel()[rng.permutation(b.size)].reshape(b.shape))
    del m, b


class MerkleChain:
    """SHA-256-Aggregation ueber eine Blockfolge OHNE Bytes zu puffern.

    Exakte Vorschrift (Version merkle-chain-v1), Bloecke in Ankunfts- bzw.
    Montagereihenfolge i = 0..k-1:
      leaf_i    = SHA256(block_bytes_i)                         (32-B-Digest)
      chain_0   = SHA256(b"merkle-chain-v1")
      chain_i   = SHA256(chain_{i-1}.digest() || LE64(len_i) || leaf_i)
      chain     = hex(chain_k)                                  (Laenge: 64)
      order_sha = SHA256(block_bytes_0 || ... || block_bytes_{k-1})
                  — Bytestrom in Montagereihenfolge; referenzvergleichbar
                  mit dem gepufferten Pfad (SHA ueber den zusammengesetzten
                  Puffer).
      merkle_root = paarweiser Baum ueber [leaf_0..leaf_{k-1}]:
                    solange >1 Blatt: bei UNGERADER Anzahl wird das letzte
                    Element dupliziert; parent_j = SHA256(b"\\x01" ||
                    L[2j] || L[2j+1]). Root = hex(letztes Einzelblatt).
    """

    def __init__(self):
        self._chain = hashlib.sha256(b"merkle-chain-v1")
        self.leaves = []
        self._last_block_sha = ""
        self.total_bytes = 0

    def update(self, block_bytes):
        h = hashlib.sha256(block_bytes)
        self._last_block_sha = h.hexdigest()
        self.leaves.append(h.digest())
        self._chain = hashlib.sha256(
            self._chain.digest() + struct.pack("<Q", len(block_bytes))
            + h.digest())
        self.total_bytes += len(block_bytes)

    @property
    def block_sha(self):
        return self._last_block_sha

    @property
    def chain_sha(self):
        return self._chain.hexdigest()

    @property
    def merkle_root(self):
        lvl = list(self.leaves)
        while len(lvl) > 1:
            if len(lvl) % 2:
                lvl.append(lvl[-1])
            lvl = [hashlib.sha256(b"\x01" + lvl[2 * j] + lvl[2 * j + 1]).digest()
                   for j in range(len(lvl) // 2)]
        return lvl[0].hex() if lvl else hashlib.sha256(b"").hexdigest()


def iter_blocks_layout(n_rows, n_cols, itemsize, cap, name,
                       n_blocks=HTTP_N_BLOCKS, rows_floor=ROWS_FLOOR,
                       rows_exact=0):
    """Repliziert die Block-Geometrie aus fetch_blocks() als Generator:
    identischer rng-Strom ([SEED, crc32(name), 13]) und identische Formeln,
    liefert (start_row, block_rows) je Block in Montage-Reihenfolge.
    Die Identitaet zum gepufferten fetch_blocks wird per End-SHA-Vergleich
    im --selftest-merkle BEWIESEN (nicht nur behauptet)."""
    row_bytes = n_cols * itemsize
    n_blocks = max(1, min(n_blocks, n_rows))
    if rows_exact:
        block_rows = max(1, rows_exact // n_blocks)
    else:
        block_rows = max(1, cap // (n_blocks * n_cols))
        if block_rows * n_blocks < rows_floor:
            block_rows = max(1, (rows_floor + n_blocks - 1) // n_blocks)
    block_rows = min(block_rows, n_rows)
    if block_rows * row_bytes > PER_TENSOR_CEIL:
        raise RuntimeError(
            f"{name}: Zeilen-Floor sprengt Tensor-Ceiling "
            f"({block_rows} Rows x {row_bytes} B > {PER_TENSOR_CEIL})")
    brng = np.random.default_rng([SEED, zlib.crc32(name.encode("utf-8")), 13])
    span = n_rows - block_rows * n_blocks
    for i in range(n_blocks):
        ideal = round(i * (span + 1) / n_blocks) if n_blocks > 1 else 0
        jitter = int(brng.integers(0, max(1, span // max(1, n_blocks) + 1)))
        start_row = min(span, max(0, ideal + jitter - (span // (2 * n_blocks))))
        yield start_row, block_rows


def _mono_u16(u16arr):
    """IEEE-750 sign-magnitude Bits -> monotone unsigned Ordnung (Sort-Key).
    (-inf < ... < -0 -> 0x0000..0x7FFF, +0 < ... < +inf/NaN -> 0x8000..0xFFFF)
    EinschlRAENKUNG: float-compare behandelt -0.0 == +0.0 und NaN == NaN als
    gleich, der Key nicht — der Merkle-Pfad detektiert beide Faelle am
    Histogramm und flaggt sie (exactness_caveat); bei den Referenz-Tensoren
    ist beides 0 (vgl. nan_count in hf_organ_reader_demo.json)."""
    u = u16arr.astype(np.uint16, copy=True)
    neg = (u & np.uint16(0x8000)).astype(bool)
    out = np.where(neg, u ^ np.uint16(0xFFFF), u | np.uint16(0x8000))
    return out.astype(np.uint16)


def _keys_to_f32(keys):
    """Inverse des monotonen u16-Schluessels zurueck nach float32 (BF16)."""
    u = keys.astype(np.uint16, copy=True)
    pos = (u & np.uint16(0x8000)).astype(bool)
    u[pos] = u[pos] & np.uint16(0x7FFF)
    u[~pos] = u[~pos] ^ np.uint16(0xFFFF)
    return (u.astype(np.uint32) << np.uint32(16)).view(np.float32)


def merkle_stream_tensor(name, inv_entry, reader, mode="blocks", rows_exact=400,
                         n_chunks=8, chunk=MERKLE_CHUNK, run_stats=True):
    """Streaming-Pfad: SHA/Merkle je Block + exakte CASI am Ende, ohne die
    Rohbytes im RAM (Disk-Spill der 2-Byte-Sortier-Schluessel statt Puffer).
    rng-Konsum: genau EIN rng.permutation(n) vor ctm.casi_with_nulls —
    identisch zu ctm.global_quantile_u8 + casi_with_nulls im gepufferten
    Pfad (bewiesen im Selbsttest)."""
    shape = tuple(inv_entry["shape"])
    dt = inv_entry["dtype"]
    itemsize = ctm._DT_SIZE[dt]
    if dt not in ("BF16", "F32"):
        raise RuntimeError(f"merkle: dtype {dt} nicht unterstuetzt (BF16/F32)")
    mat2d, _ = ctm.natural_2d(shape)
    if mat2d is None:
        raise RuntimeError("1D-Tensor")
    n_rows, n_cols = mat2d
    if n_rows < ctm.MIN_ROWS and n_cols >= ctm.MIN_ROWS:
        raise RuntimeError(
            "transponierter Analysefall (Zeilen<100, z.B. gdn_ba): Zeilen "
            "liegen spaltenweise im File — nicht ohne Voll-Puffer streambar "
            "(dokumentierter Geltungsbereich des Merkle-Pfads)")
    shard = inv_entry["shard"]
    data_start = inv_entry["data_start"]
    off_b = inv_entry["offset_in_shard"][0]
    abs_base = data_start + off_b
    row_bytes = n_cols * itemsize
    keys16 = dt == "BF16"

    if mode == "full":
        q, r = divmod(n_rows, n_chunks)
        layout, s = [], 0
        for i in range(n_chunks):
            br = q + (1 if i < r else 0)
            if br:
                layout.append((s, br))
                s += br
        analyzed_rows = n_rows
    else:
        layout = list(iter_blocks_layout(n_rows, n_cols, itemsize,
                                         HTTP_MAX_ELEMS, name,
                                         n_blocks=n_chunks,
                                         rows_floor=ROWS_FLOOR,
                                         rows_exact=rows_exact))
        analyzed_rows = layout[0][1] * len(layout)
    n_elem = analyzed_rows * n_cols

    # ── Phase 1: Netzwerk-Streaming, Histogramm, Partial-Stats, Spill ──────
    chain = MerkleChain()
    order_h = hashlib.sha256()      # SHA ueber Bytestrom in Montageordnung
    hist = np.zeros(65536, dtype=np.int64) if keys16 else {}
    spill_path = tempfile.mktemp(prefix=".merkle_spill_",
                                 suffix=".npy", dir=RESULTS_DIR)
    keysmm = np.lib.format.open_memmap(
        spill_path, mode="w+", dtype=np.uint16 if keys16 else np.uint32,
        shape=(n_elem,))
    pos = 0
    rows_so_far = 0
    run = {"n": 0, "sum": 0.0, "sq": 0.0}
    block_records = []
    rss_cp = {"base": ru_peak_mb()}
    seg = _SegTM()
    seg.begin()
    for i, (sr, br) in enumerate(layout):
        s = abs_base + sr * row_bytes
        data = reader.get_range(shard, s, s + br * row_bytes - 1)
        chain.update(data)
        order_h.update(data)
        if keys16:
            u16 = np.frombuffer(data, np.uint16)
            keys = _mono_u16(u16)
            hist += np.bincount(keys, minlength=65536)
            f32 = (u16.astype(np.uint32) << np.uint32(16)).view(np.float32)
            f32 = f32.reshape(br, n_cols)
        else:
            u32 = np.frombuffer(data, np.uint32)
            neg = (u32 & np.uint32(0x80000000)).astype(bool)
            keys = np.where(neg, u32 ^ np.uint32(0xFFFFFFFF),
                            u32 | np.uint32(0x80000000)).astype(np.uint32)
            ukeys, uc = np.unique(keys, return_counts=True)
            for k, c in zip(ukeys.tolist(), uc.tolist()):
                hist[k] = hist.get(k, 0) + c
            f32 = np.frombuffer(data, np.float32).reshape(br, n_cols)
        keysmm[pos:pos + keys.size] = keys
        pos += keys.size
        rows_so_far += br
        run["n"] += int(f32.size)
        f64 = f32.astype(np.float64)
        run["sum"] += float(f64.sum())
        run["sq"] += float(np.square(f64).sum())
        n_prev = pos - keys.size
        u16 = u32 = None               # Views loesen (Referenzen fallen)
        del data, keys, f32, f64   # VOR der Pause freigeben: tracemalloc
        # sieht Freigaben aus FRUEHEREN Segmenten nach stop/start nicht
        # mehr (neue Tabelle) — sonst akkumuliert _SegTM Phantomsalden.
        # Laufende CASI-Partial-Statistik (EIGENE rng-Stroeme — beruehrt
        # den Analyse-rng nicht). Block allein nur ab Engine-Zeilen-Floor
        # 100 (R4); ZUSAETZLICH kumulative Fenster-Statistik — der
        # eigentliche Verlaufs-Indikator. Beide rekonstruieren ihre f32
        # bit-identisch aus dem Spill (gleiche Bytes, gleicherng-Reihen-
        # folge wie in der 1. Version).
        lrng = np.random.default_rng(
            [SEED, zlib.crc32(name.encode("utf-8")), 901, i])
        lcasi = lnull = None
        rcasi = rnull = None
        wr_rows = None
        seg.pause()                     # Engine nicht tracen (14x)
        if keys16 and br >= ctm.MIN_ROWS:
            lf = _keys_to_f32(np.array(keysmm[n_prev:pos]))
            lb = ctm.global_quantile_u8(lf.reshape(br, n_cols), lrng)
            lcasi = float(ctm.compute_fast_casi(lb))
            lnull = float(ctm.compute_fast_casi(
                lb.ravel()[lrng.permutation(lb.size)].reshape(lb.shape)))
            del lf, lb
        if run_stats and keys16 and rows_so_far >= ctm.MIN_ROWS:
            # Fenster: Engine-Floor 100 Zeilen, ~128k-Elemente-Deckel —
            # sonst dominiert der Engine-Aufruf auf der vollen Matrix
            # (~121 B/Element transient) den Pfad-Peak (Messartefakt der
            # 1. Version, siehe Loop-Log W6).
            wrows = max(ctm.MIN_ROWS, 131072 // n_cols)
            wrows = min(wrows, rows_so_far)
            wr_rows = wrows
            wr_idx = np.unique(
                np.linspace(0, rows_so_far - 1, wrows).astype(np.int64))
            keys_win = keysmm[:pos].reshape(rows_so_far, n_cols)[wr_idx]
            rf = _keys_to_f32(keys_win.ravel())
            rrng = np.random.default_rng(
                [SEED, zlib.crc32(name.encode("utf-8")), 902, i])
            rb = ctm.global_quantile_u8(rf.reshape(len(wr_idx), n_cols), rrng)
            rcasi = float(ctm.compute_fast_casi(rb))
            rnull = float(ctm.compute_fast_casi(
                rb.ravel()[rrng.permutation(rb.size)].reshape(rb.shape)))
            del rf, rb, keys_win
        seg.resume()
        cum_distinct = (int(np.count_nonzero(hist)) if keys16
                        else len(hist))
        block_records.append({
            "i": i, "start_row": int(sr), "rows": int(br),
            "bytes": int(br * row_bytes), "sha256": chain.block_sha,
            "block_local_casi_raw": round(lcasi, 3) if lcasi else None,
            "block_local_ratio": (round(lcasi / lnull, 4)
                                  if lcasi and lnull and lnull > 0 else None),
            "running_rows": rows_so_far,
            "running_window_rows": (int(wr_rows)
                                    if rcasi is not None and wr_rows
                                    else None),
            "running_casi_raw": round(rcasi, 3) if rcasi else None,
            "running_casi_null": round(rnull, 3) if rnull else None,
            "running_ratio": (round(rcasi / rnull, 4)
                              if rcasi and rnull and rnull > 0 else None),
            "cum_n": run["n"],
            "cum_distinct_values": cum_distinct,
            "running_mean": round(run["sum"] / max(1, run["n"]), 8),
            "running_var": round(
                run["sq"] / max(1, run["n"])
                - (run["sum"] / max(1, run["n"])) ** 2, 8)})
    keysmm.flush()
    del keysmm
    spill = np.load(spill_path, mmap_mode="r")
    assert pos == n_elem, (pos, n_elem)

    # ── Phase 2: exakte Code-Rekonstruktion (Histogramm + RNG-Replay) ─────
    if keys16:
        assert int(hist.sum()) == n_elem
        base = np.cumsum(hist) - hist
        distinct = int(np.count_nonzero(hist))
    else:
        uk = np.array(sorted(hist), dtype=np.uint32)
        uc = np.array([hist[k] for k in uk.tolist()], dtype=np.int64)
        base_v = np.cumsum(uc) - uc
        distinct = len(uk)
    rng = np.random.default_rng([SEED, zlib.crc32(name.encode("utf-8"))])
    perm = rng.permutation(n_elem)     # rng-Replay == global_quantile_u8
    codes = np.empty(n_elem, dtype=np.uint8)
    cnt = np.zeros(65536, dtype=np.int64) if keys16 else np.zeros(distinct,
                                                                  dtype=np.int64)
    # Adaptiver Chunk: Temporaries (~30x8 Byte je Chunk-Element) konstant
    # halten — bei kleinem n schrumpft der Chunk mit, damit der Assembly-
    # Peak auch unter 1M Elementen unter dem gepufferten Pfad bleibt.
    chunk = min(chunk, max(16384, n_elem // 8))
    for c0 in range(0, n_elem, chunk):
        pc = perm[c0:c0 + chunk]
        vals = spill[pc]
        order = np.argsort(vals, kind="stable")
        vs = vals[order]
        newg = np.empty(vs.size, dtype=bool)
        newg[0] = True
        np.not_equal(vs[1:], vs[:-1], out=newg[1:])
        starts = np.flatnonzero(newg)
        gid = np.cumsum(newg) - 1
        sg = starts[gid]
        del gid
        within = np.arange(vs.size, dtype=np.int64)
        within -= sg
        del sg
        if keys16:
            grank = base[vs]
            grank += cnt[vs]
        else:
            ix = np.searchsorted(uk, vs)
            grank = base_v[ix]
            grank += cnt[ix]
            del ix
        grank += within
        del within, vs
        bsorted = (grank * (255.999 / n_elem)).astype(np.uint8)
        del grank
        tmp8 = np.empty(bsorted.size, dtype=np.uint8)
        tmp8[order] = bsorted
        codes[pc] = tmp8
        if keys16:
            cnt += np.bincount(vals, minlength=65536)
        else:
            cnt += np.bincount(np.searchsorted(uk, vals), minlength=distinct)
        del vals, order, bsorted, tmp8
    del perm, cnt
    caveat = {}
    if keys16:
        caveat = {"pos_nan": int(hist[0xFF81:].sum()),
                  "neg_nan": int(hist[:0x007F].sum()),
                  "neg_zero": int(hist[0x7FFF]), "pos_zero": int(hist[0x8000])}
    b2d = codes.reshape(analyzed_rows, n_cols)
    assembly_peak = seg.finish()          # vor der gemeinsamen Null-Phase
    rss_cp["pre_engine"] = ru_peak_mb()
    casi_raw, profile, nulls, null_mean, ratio = ctm.casi_with_nulls(b2d, rng)
    rss_cp["final"] = ru_peak_mb()
    rec = {
        "name": name, "mode": mode, "rows_exact": rows_exact,
        "n_chunks": len(layout), "dtype": dt,
        "casi_input_shape": [analyzed_rows, n_cols],
        "n_elements": int(n_elem),
        "blocks": block_records,
        "order_sha256": order_h.hexdigest(),
        "chain_sha256": chain.chain_sha,
        "merkle_root": chain.merkle_root,
        "block_bytes_total": chain.total_bytes,
        "distinct_values": distinct,
        "exactness_caveat_keys": caveat,
        "casi_raw": round(float(casi_raw), 3),
        "casi_null_mean": round(null_mean, 3),
        "casi_null_std": round(float(nulls.std()), 3),
        "ratio": round(ratio, 4),
        "assembly_peak_bytes": int(assembly_peak),
        "run_stats": bool(run_stats),
        "rss_checkpoints_mb": {k: round(v, 1) for k, v in rss_cp.items()},
        "profile_raw": {n: round(float(z), 2)
                        for n, z in zip(ctm.STRATEGY_NAMES, profile)},
    }
    del spill, codes, b2d
    try:
        os.remove(spill_path)
    except OSError:
        pass
    return rec


def buffered_reference(name, inv_entry, reader, mode="blocks", rows_exact=400):
    """Der BESTEHENDE gepufferte Pfad als Referenz: nutzt fetch_full()/
    fetch_blocks() (Originalfunktionen!) + ctm-Pipeline exakt wie
    score_streamed — die Bytes bleiben bis zum Ende im RAM gepuffert."""
    shape = tuple(inv_entry["shape"])
    dt = inv_entry["dtype"]
    itemsize = ctm._DT_SIZE[dt]
    mat2d, _ = ctm.natural_2d(shape)
    n_rows, n_cols = mat2d
    if n_rows < ctm.MIN_ROWS and n_cols >= ctm.MIN_ROWS:
        raise RuntimeError("transponierter Fall — nicht vergleichbar")
    shard = inv_entry["shard"]
    data_start = inv_entry["data_start"]
    off_b = inv_entry["offset_in_shard"][0]
    numel = int(np.prod(shape))
    if mode == "full":
        raw, reqs = fetch_full(reader, shard, data_start, off_b, numel,
                               itemsize)
        analyzed_rows = n_rows
        sampling = "full"
    else:
        raw, reqs, analyzed_rows = fetch_blocks(
            reader, shard, data_start, off_b, n_rows, n_cols, itemsize,
            HTTP_MAX_ELEMS, name, n_blocks=HTTP_N_BLOCKS,
            rows_floor=ROWS_FLOOR, rows_exact=rows_exact)
        sampling = f"blocks(n={reqs},rows={analyzed_rows})"
    order_sha = hashlib.sha256(raw).hexdigest()
    rss_cp = {"base": ru_peak_mb()}
    seg = _SegTM()
    seg.begin()
    if dt == "BF16":
        f32 = ctm.bf16_rows_to_f32(raw.view(np.uint16),
                                   (analyzed_rows, n_cols), None)
    else:
        f32 = ctm.f32_rows_to_f32(raw.view(np.float32),
                                  (analyzed_rows, n_cols), None)
    n_unique = int(np.unique(f32).size)
    rng = np.random.default_rng([SEED, zlib.crc32(name.encode("utf-8"))])
    b2d = ctm.global_quantile_u8(f32, rng)
    del f32
    assembly_peak = seg.finish()          # Quantile-Assembly ENTHALTEN —
    rss_cp["pre_engine"] = ru_peak_mb()   # genau das ersetzt der Merkle-Pfad
    casi_raw, profile, nulls, null_mean, ratio = ctm.casi_with_nulls(b2d, rng)
    rss_cp["final"] = ru_peak_mb()
    del b2d, raw
    return {"name": name, "sampling": sampling,
            "casi_input_shape": [analyzed_rows, n_cols],
            "n_elements": int(analyzed_rows * n_cols),
            "range_requests": reqs, "order_sha256": order_sha,
            "n_unique": n_unique,
            "tie_fraction": round(1.0 - n_unique / (analyzed_rows * n_cols), 6),
            "casi_raw": round(float(casi_raw), 3),
            "casi_null_mean": round(null_mean, 3),
            "casi_null_std": round(float(nulls.std()), 3),
            "ratio": round(ratio, 4),
            "assembly_peak_bytes": int(assembly_peak),
            "rss_checkpoints_mb": {k: round(v, 1) for k, v in rss_cp.items()},
            "profile_raw": {n: round(float(z), 2)
                            for n, z in zip(ctm.STRATEGY_NAMES, profile)}}


# Referenz-Tensoren des Selbsttests: (a) kleiner Voll-Tensor (conv1d, 3D-
# Reshape, im Demo-Report ratio 119.353), (b) Gross-Tensor embed 400 Zeilen
# (RSS-Demonstration), (c) k_proj L19 mit exakt der Demo-Geometrie (200
# Zeilen — Cross-Check gegen das vorhandene Demo-Record).
MERKLE_SELFTEST_REFS = [
    ("model.language_model.layers.34.linear_attn.conv1d.weight", "full", 0),
    ("model.language_model.embed_tokens.weight", "blocks", 400),
    ("model.language_model.layers.19.self_attn.k_proj.weight", "blocks", 200),
]


def _run_merkle_child(repo, revision, name, mode, rows_exact, blocks, kind,
                      budget_mb, run_stats=True, timeout=1500):
    """Ein Pfad in einem FRISCHEN Subprozess (isoliertes ru_maxrss-Hochwasser
    — im gemeinsamen Prozess wuerde der erste Pfad das Hochwasser des
    zweiten maskieren; die gemeinsame Null-Phase der Engine dominiert)."""
    cmd = [sys.executable, os.path.abspath(__file__), "_merkle_child", repo,
           "--revision", revision, "--name", name, "--mode", mode,
           "--rows-exact", str(rows_exact), "--blocks", str(blocks),
           "--kind", kind, "--budget-mb", str(budget_mb),
           "--running-stats" if run_stats else "--no-running-stats"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"child-timeout nach {timeout}s"}
    for line in p.stdout.splitlines():
        if line.startswith("MERKLE_CHILD_JSON "):
            return json.loads(line[len("MERKLE_CHILD_JSON "):])
    return {"error": f"child rc={p.returncode}: {p.stderr[-400:]}"}


def cmd_merkle(args):
    budget = Budget(args.budget_mb)
    reader = HFRangeReader(args.repo, args.revision, budget)
    inv = load_or_scan(reader, budget, refresh=args.refresh)
    by_name = {t["name"]: t for t in inv["tensors"]}
    if args.globs:
        names = sorted(by_name)
        sel = []
        for g in args.globs:
            sel += [n for n in names if fnmatch.fnmatch(n, g)]
        refs = [(n, "blocks", args.rows_exact)
                for n in sorted(set(sel))[:args.limit or 8]]
    else:
        refs = MERKLE_SELFTEST_REFS
    out = {"tool": "hf_organ_reader", "command": "merkle",
           "repo": args.repo, "revision": args.revision,
           "selftest": bool(args.selftest_merkle or not args.globs),
           "started_at": now_utc(),
           "measurement_design": (
               "PRIMAER: Assembly-Peak je Pfad via segmentiertem tracemalloc "
               "(alles ausser den byte-identischen Engine-Aufrufen; Engine-"
               "Python-Loops wuerden unter tracemalloc Faktor ~14 bremsen, "
               "deshalb pro Segment pausiert und Lebend-Volumen "
               "weitergetragen). SEKUNDAER: ru_maxrss-Gesamtpeak je Pfad in "
               "einem frischen Subprozess (Checkpoints base/pre_engine/final "
               "nach 512x512-Warmup; monotones Hochwasser). Die Engine "
               "allokiert ~121 B/Element transient und dominiert die "
               "Gesamtpeaks beider Pfade gemeinsam."),
           "aggregation_rule": {
               "leaf": "SHA256(block_bytes_i)",
               "chain": "chain_0=SHA256(b'merkle-chain-v1'); "
                        "chain_i=SHA256(chain_{i-1}.digest()||LE64(len_i)"
                        "||leaf_i)",
               "order_sha": "SHA256(block_bytes_0||..||block_bytes_{k-1}) "
                            "(Montagereihenfolge == gepufferter Puffer)",
               "merkle_root": "Baum ueber leaf-Digests; ungerade Ebene: "
                              "letztes Element dupliziert; parent=SHA256("
                              "b'\\x01'||L[2j]||L[2j+1])"},
           "tensors": []}
    ledger = {"bytes_body": 0, "bytes_overhead_approx": 0, "http_requests": 0}
    n_pass = 0
    for (name, mode, rex) in refs:
        if name not in by_name:
            out["tensors"].append({"name": name, "error": "nicht im Inventar"})
            continue
        print(f"\n=== {name} (mode={mode}, rows_exact={rex}) ===", flush=True)
        recs = {}
        failed = False
        for kind in ("streaming", "buffered"):
            r = _run_merkle_child(args.repo, args.revision, name, mode, rex,
                                  args.blocks, kind, args.budget_mb,
                                  run_stats=args.running_stats)
            if "error" in r:
                out["tensors"].append({"name": name, "error": r["error"]})
                print(f"  {kind} ERROR: {r['error'][:200]}", flush=True)
                failed = True
                break
            recs[kind] = r
            b = r["budget"]
            ledger["bytes_body"] += b["bytes_body"]
            ledger["bytes_overhead_approx"] += b["bytes_overhead_approx"]
            ledger["http_requests"] += b["http_requests"]
        if failed:
            continue
        mk, bu = recs["streaming"]["tensor"], recs["buffered"]["tensor"]
        eq = {"order_sha256": mk["order_sha256"] == bu["order_sha256"],
              "ratio": mk["ratio"] == bu["ratio"],
              "casi_raw": mk["casi_raw"] == bu["casi_raw"],
              "casi_null_mean": mk["casi_null_mean"] == bu["casi_null_mean"],
              "n_unique_vs_distinct": mk["distinct_values"] == bu["n_unique"]}
        eq["all"] = all(eq.values())
        n_pass += int(eq["all"])
        sc, bc = mk["rss_checkpoints_mb"], bu["rss_checkpoints_mb"]
        s_path = round(sc.get("pre_engine", 0) - sc.get("base", 0), 1)
        b_path = round(bc.get("pre_engine", 0) - bc.get("base", 0), 1)
        s_asm = mk["assembly_peak_bytes"] / 1048576
        b_asm = bu["assembly_peak_bytes"] / 1048576
        rec = {"name": name, "tensor_class": ctm.classify(name),
               "mode": mode, "rows_exact": rex,
               "n_elements": bu["n_elements"],
               "casi_input_shape": bu["casi_input_shape"],
               "buffered": {k: bu[k] for k in
                            ("sampling", "range_requests", "order_sha256",
                             "n_unique", "tie_fraction", "casi_raw",
                             "casi_null_mean", "ratio", "rss_checkpoints_mb",
                             "assembly_peak_bytes")},
               "buffered_seconds": recs["buffered"]["seconds"],
               "streaming": {k: mk[k] for k in
                             ("n_chunks", "order_sha256", "chain_sha256",
                              "merkle_root", "block_bytes_total",
                              "distinct_values", "exactness_caveat_keys",
                              "casi_raw", "casi_null_mean", "ratio",
                              "rss_checkpoints_mb", "assembly_peak_bytes",
                              "run_stats")},
               "streaming_seconds": recs["streaming"]["seconds"],
               "streaming_blocks": mk["blocks"],
               "assembly_peak": {
                   "streaming_mb": round(s_asm, 2),
                   "buffered_mb": round(b_asm, 2),
                   "saving_mb": round(b_asm - s_asm, 2),
                   "factor_buffered_over_streaming": round(
                       b_asm / max(1e-9, s_asm), 2)},
               "peak_rss": {
                   "streaming_path_mb": s_path,
                   "buffered_path_mb": b_path,
                   "path_saving_mb": round(b_path - s_path, 1),
                   "streaming_total_mb": round(
                       sc.get("final", 0) - sc.get("base", 0), 1),
                   "buffered_total_mb": round(
                       bc.get("final", 0) - bc.get("base", 0), 1),
                   "engine_common_mb_streaming": round(
                       sc.get("final", 0) - sc.get("pre_engine", 0), 1),
                   "engine_common_mb_buffered": round(
                       bc.get("final", 0) - bc.get("pre_engine", 0), 1)},
               "equal": eq}
        out["tensors"].append(rec)
        print(f"  streaming: sha={mk['order_sha256'][:16]}… ratio="
              f"{mk['ratio']} assembly={s_asm:.1f}MB rss(gesamt) "
              f"+{rec['peak_rss']['streaming_total_mb']}MB "
              f"({recs['streaming']['seconds']}s) "
              f"root={mk['merkle_root'][:10]}…", flush=True)
        print(f"  buffered : sha={bu['order_sha256'][:16]}… ratio="
              f"{bu['ratio']} assembly={b_asm:.1f}MB rss(gesamt) "
              f"+{rec['peak_rss']['buffered_total_mb']}MB "
              f"({recs['buffered']['seconds']}s)", flush=True)
        print(f"  ASSEMBLY-Ersparnis: {rec['assembly_peak']['saving_mb']}MB "
              f"(Faktor {rec['assembly_peak']['factor_buffered_over_streaming']})"
              f" | GLEICH: {eq}", flush=True)
        out["network_ledger"] = dict(ledger)
        atomic_write_json(
            out, os.path.join(RESULTS_DIR, "hf_reader_merkle.json"))
    out["network_ledger"] = dict(ledger)
    out["n_pass"] = n_pass
    out["n_total"] = len(refs)
    out["finished_at"] = now_utc()
    atomic_write_json(out, os.path.join(RESULTS_DIR, "hf_reader_merkle.json"))
    write_merkle_md(os.path.join(RESULTS_DIR, "hf_reader_merkle.md"), out)
    print(f"\nFERTIG merkle: {n_pass}/{len(refs)} identisch; Netz-Ledger: "
          f"{(ledger['bytes_body'] + ledger['bytes_overhead_approx']) / 1048576:.2f} MB, "
          f"{ledger['http_requests']} Requests", flush=True)
    if out.get("selftest") and n_pass != len(refs):
        sys.exit(2)


def _merkle_child_cmd(args):
    """Intern: ein Merkle-Pfad (streaming|buffered) in isolation. Protokoll:
    eine Zeile 'MERKLE_CHILD_JSON <json>' auf stdout."""
    budget = Budget(args.budget_mb)
    reader = HFRangeReader(args.repo, args.revision, budget)
    inv = load_or_scan(reader, budget)
    by_name = {t["name"]: t for t in inv["tensors"]}
    ent = by_name[args.name]
    _warm_engine()
    t0 = time.time()
    if args.kind == "streaming":
        tensor = merkle_stream_tensor(args.name, ent, reader, args.mode,
                                      args.rows_exact, args.blocks,
                                      run_stats=args.running_stats)
    else:
        tensor = buffered_reference(args.name, ent, reader, args.mode,
                                    args.rows_exact)
    payload = {"kind": args.kind, "tensor": tensor,
               "seconds": round(time.time() - t0, 1),
               "budget": budget.as_dict(),
               "ru_peak_mb": round(ru_peak_mb(), 1)}
    sys.stdout.write("MERKLE_CHILD_JSON " + json.dumps(payload) + "\n")
    sys.stdout.flush()


def write_merkle_md(path, data):
    lines = []
    A = lines.append
    A("# HF Reader — Merkle-Streaming (W6/A1): Partial-Hashes ohne Byte-Puffer")
    A("")
    A(f"*Generiert {data['finished_at']} — `hf_organ_reader.py merkle "
      f"--selftest-merkle {data['repo']}` (seed 42).*")
    A("")
    A("## Beweis-Satz")
    A("")
    A(f"> Der Streaming-Pfad aggregiert SHA-256 je Block + Merkle-Chain/Root "
      f"und rekonstruiert die global-quantile-CASI **ohne die Tensor-Bytes "
      f"im RAM** (Disk-Spill der 2-Byte-Sortierschluessel). Validiert an "
      f"{data['n_pass']}/{data['n_total']} Referenz-Tensoren: End-SHA und "
      f"CASI-Ratio **bitgenau identisch** zum gepufferten Pfad.")
    A("")
    A("## Aggregationsvorschrift (exakt, Version merkle-chain-v1)")
    A("")
    A("```")
    A("leaf_i     = SHA256(block_bytes_i)")
    A("chain_0    = SHA256(b\"merkle-chain-v1\")")
    A("chain_i    = SHA256(chain_{i-1}.digest() || LE64(len_i) || leaf_i)")
    A("order_sha  = SHA256(block_bytes_0 || ... || block_bytes_{k-1})")
    A("             (Montagereihenfolge — vergleichbar mit SHA ueber den")
    A("              gepufferten Gesamt-Puffer)")
    A("merkle_root= Baum ueber [leaf_0..leaf_{k-1}]: ungerade Ebene ->")
    A("             letztes Element dupliziert; parent = SHA256(b\"\\x01\" ||")
    A("             L[2j] || L[2j+1]); Root = letztes Einzelblatt (hex)")
    A("```")
    A("")
    A("## Speichermodell und Peak-Messung")
    A("")
    A("**Primärmetrik — Assembly-Peak** (segmentiertes tracemalloc): gemessen "
      "wird ALLES ausser den gemeinsamen Engine-Aufrufen (`casi_with_nulls` "
      "laeuft in beiden Pfaden byte-identisch; tracemalloc auf den O(n)-"
      "Python-Loops der Engine kostet Faktor ~14 und wird deshalb pro "
      "Segment pausiert — lebende Allokationen werden über Segmentgrenzen "
      "weitergetragen). Der Assembly-Pfad ist genau der Teil, den der "
      "Merkle-Redesign ersetzt: gepuffert = Fetch-Buffer + f32 + "
      "global_quantile_u8-Workspace; streaming = Blockpuffer + Histogramm + "
      "Spill + RNG-Replay + Codes.")
    A("")
    A("| Tensor | Elemente | Assembly streaming | Assembly gepuffert | "
      "Ersparnis | Faktor |")
    A("|---|---:|---:|---:|---:|---:|")
    for t in data["tensors"]:
        if "error" in t:
            continue
        p = t["assembly_peak"]
        A(f"| `{t['name'].split('.')[-2]}.{t['name'].split('.')[-1]}` "
          f"| {t['n_elements']:,} | {p['streaming_mb']} MB | "
          f"{p['buffered_mb']} MB | {p['saving_mb']} MB | "
          f"{p['factor_buffered_over_streaming']}x |")
    A("")
    A("Analytisch (n Elemente, BF16): gepuffert ≈ 35n B (raw 2n + f32 4n + "
      "perm 8n + flat[perm] 4n + order 8n + ranks 8n + codes n), "
      "streaming ≈ 9n B (perm-Replay 8n + codes n; Phase 1: Blockpuffer + "
      "Histogramm 512 KB konstant, Spill file-backed). Die Tensor-Bytes "
      "(2n) und die f32-Kopie (4n) entfallen komplett.")
    A("")
    A("**Sekundärmetrik — ru_maxrss-Gesamtpeak** (je Pfad ein frischer "
      "Subprozess, Checkpoints base/pre_engine/final nach 512x512-Warmup; "
      "monotones Hochwasser, deshalb nur gesamt ausweisbar):")
    A("")
    A("| Tensor | gesamt streaming | gesamt gepuffert | Engine-Zuwachs "
      "stream/gepuffert |")
    A("|---|---:|---:|---:|")
    for t in data["tensors"]:
        if "error" in t:
            continue
        p = t["peak_rss"]
        A(f"| `{t['name'].split('.')[-2]}.{t['name'].split('.')[-1]}` "
          f"| +{p['streaming_total_mb']} MB | +{p['buffered_total_mb']} MB | "
          f"+{p['engine_common_mb_streaming']}/"
          f"+{p['engine_common_mb_buffered']} MB |")
    A("")
    A("Die CASI-Engine (live_casiv2) allokiert transient ~121 B/Element — "
      "bei 1-2M analysierten Elementen dominieren ~120-235 MB die "
      "Gesamtpeaks BEIDER Pfade (identische Aufrufe). Der Streaming-Pfad "
      "traegt zusaetzlich die optionale Fenster-Statistik (Engine auf "
      "<=512k Elementen, ~62 MB transient bei 5120 Spalten); mit "
      "`--no-running-stats` entfaellt sie. Der Assembly-Vergleich "
      "(Primärmetrik) bleibt davon unberuehrt.")
    A("")
    A("## Validierung (Selbsttest)")
    A("")
    A("| Tensor | Geometrie | End-SHA gleich | ratio gleich | casi_raw | "
      "null_mean | ratio (beide Pfade) |")
    A("|---|---|:-:|:-:|:-:|:-:|---:|")
    for t in data["tensors"]:
        if "error" in t:
            A(f"| `{t['name']}` | — | ERROR: {t['error'][:40]} | | | | |")
            continue
        eq = t["equal"]
        A(f"| `{t['name']}` | {t['mode']}"
          f"{', rows_exact=' + str(t['rows_exact']) if t['rows_exact'] else ''} "
          f"| {'JA' if eq['order_sha256'] else 'NEIN'} "
          f"| {'JA' if eq['ratio'] else 'NEIN'} "
          f"| {'JA' if eq['casi_raw'] else 'NEIN'} "
          f"| {'JA' if eq['casi_null_mean'] else 'NEIN'} "
          f"| {t['buffered']['ratio']} |")
    A("")
    A("Vergleichbarkeit der Geometrie: `iter_blocks_layout()` repliziert den "
      "rng-Strom `[42, crc32(name), 13]` und die Formeln aus `fetch_blocks()` "
      "— die Identitaet wird nicht behauptet, sondern durch den End-SHA-"
      "Gleichheit BEWIESEN (ungleiche Blockwahl wuerde den SHA brechen).")
    A("")
    A("Geltungsgrenzen: transponierte Analyse (Zeilen<100, gdn_ba) ist nicht "
      "streambar (Zeilen liegen spaltenweise im File) — dort bleibt es beim "
      "vollen Puffer (kleine Tensoren). F16 wird nicht unterstuetzt (BF16/F32 "
      "ja). Der Schluessel-Raum (monotone u16) ist exakt in der float-"
      "Ordnung, AUSSER -0.0/+0.0 und NaN-Payloads — beide werden am "
      "Histogramm detektiert und als `exactness_caveat_keys` ausgewiesen "
      "(bei den Referenz-Tensoren: alle 0).")
    A("")
    A("## Laufende Partial-Statistiken (Phase 1, exemplarisch embed 400r)")
    A("")
    emb = [t for t in data["tensors"] if "streaming_blocks" in t
           and t.get("rows_exact") == 400]
    if emb:
        A("| Block i | Zeilen | Bytes | SHA-16 | block-local ratio | "
          "kumul. ratio (Bloecke 0..i) | cum distinct | running mean |")
        A("|---:|---:|---:|---|---:|---:|---:|---:|")
        for brec in emb[0]["streaming_blocks"][:4]:
            A(f"| {brec['i']} | {brec['rows']} | {brec['bytes']:,} | "
              f"`{brec['sha256'][:16]}…` | "
              f"{brec.get('block_local_ratio', '—')} | "
              f"{brec.get('running_ratio', '—')} | "
              f"{brec['cum_distinct_values']:,} | {brec['running_mean']} |")
        A("")
        A("Beide Partial-Masse nutzen LOCALE Quantile (eigene rng-Stroeme, "
          "[42, crc32(name), 901/902, i]) — nach Befund W3/I2 ist CASI "
          "analysegrössenabhängig: Partial-Ratios sind Verlaufsanzeige, "
          "nicht dem globalen Ratio kardinal vergleichbar. Block-lokal "
          "unter 100 Zeilen liefert die Engine 0 (R4-Floor) — dort steht "
          "nur die kumulative Statistik.")
    A("")
    A("## Transfer-Bilanz (Netz-Ledger ueber alle Subprozesse)")
    A("")
    led = data.get("network_ledger", {})
    A(f"- Body {(led.get('bytes_body', 0)) / 1048576:.2f} MB + Header-Overhead "
      f"~{(led.get('bytes_overhead_approx', 0)) / 1024:.0f} KB, "
      f"{led.get('http_requests', 0)} Requests (beide Pfade je Tensor "
      "gefetcht — Vergleichspflicht; jedes Child lost den CDN-Redirect "
      "selbst).")
    A("- Inventar aus Cache (0 neue Bytes).")
    A("")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ════════════════════════════════════════════════════════════════════════════
# W7 (A2) — Budget-allozierter Planer (Trockenplan + optionale Ausfuehrung)
# ════════════════════════════════════════════════════════════════════════════
def load_ranking_source(path):
    """Rangquelle: hf_organ_reader_demo.json — top_candidates (Tensor-Rang)
    + class_aggregate (Klassen-Mittel als Rang fuer den Rest) + bereits
    gemessene Records (kosten 0 neue Bytes)."""
    with open(path) as f:
        d = json.load(f)
    top = [t["name"] for t in d.get("top_candidates", []) if "name" in t]
    class_mean = {c: s["mean"] for c, s in d.get("class_aggregate", {}).items()}
    scored = {t["name"]: t for t in d.get("tensors", [])
              if isinstance(t, dict) and "ratio" in t}
    return {"repo": d.get("repo"), "top_candidates": top,
            "class_mean": class_mean, "scored": scored,
            "n_top": len(top), "n_scored": len(scored)}


def build_rank_list(inv, ranking, rows_base=200, n_blocks=HTTP_N_BLOCKS):
    """Vollstaendige, deterministische Rangliste aller scorebaren Tensoren:
    Rang 1..n_top = top_candidates (Demo, ratio-desc), danach Klassen nach
    class_aggregate-Mittel (desc), innerhalb Klasse layer asc, name asc."""
    ents = []
    for t in inv["tensors"]:
        if t["numel"] < MIN_SCORE_ELEMS or len(t["shape"]) < 2:
            continue
        m2, _ = ctm.natural_2d(tuple(t["shape"]))
        if m2 is None:
            continue
        n_rows, n_cols = m2
        transposed = n_rows < ctm.MIN_ROWS and n_cols >= ctm.MIN_ROWS
        itemsize = ctm._DT_SIZE.get(t["dtype"], 2)
        full_bytes = t["numel"] * itemsize
        row_b = n_cols * itemsize
        rungs = []
        if transposed:
            if full_bytes <= 32 * 1024 * 1024:
                rungs.append({"rows": n_rows, "full": True,
                              "bytes": full_bytes + 3 * 1400})
        else:
            for rr in (rows_base, 800, 3200, 12800, 51200, n_rows):
                rb = rr * row_b
                if rr > n_rows or rb > PER_TENSOR_CEIL:
                    continue
                if rungs and rungs[-1]["rows"] == rr:
                    continue
                rungs.append({"rows": min(rr, n_rows),
                              "full": rr >= n_rows and full_bytes <= PER_TENSOR_CEIL,
                              "bytes": (full_bytes if rr >= n_rows
                                        else rb) + (n_blocks + 2) * 1400})
            if not rungs:
                continue
        rungs.sort(key=lambda g: g["rows"])
        # Deduplizieren (n_rows-Rung == full)
        ded = []
        for g in rungs:
            if ded and ded[-1]["rows"] == g["rows"]:
                ded[-1] = g
            else:
                ded.append(g)
        rec = ranking["scored"].get(t["name"])
        cached_rows = None
        if rec:
            smp = rec.get("sampling", "")
            if smp.startswith("blocks"):
                cached_rows = rec["casi_input_shape"][0]
            elif "full" in smp:
                cached_rows = n_rows
            elif smp.startswith("uniform"):
                cached_rows = rec["casi_input_shape"][0]
        ents.append({"name": t["name"], "tensor_class": t["tensor_class"],
                     "layer": t["layer"], "shape": t["shape"],
                     "n_rows": n_rows, "n_cols": n_cols,
                     "transposed": transposed, "full_bytes": full_bytes,
                     "rungs": ded, "cached_rows": cached_rows})
    top = ranking["top_candidates"]
    cm = ranking["class_mean"]
    def sort_key(e):
        if e["name"] in top:
            return (0, top.index(e["name"]))
        return (1, -cm.get(e["tensor_class"], -1.0),
                e["tensor_class"], e["layer"] if e["layer"] is not None else -1,
                e["name"])
    ents.sort(key=sort_key)
    for i, e in enumerate(ents, 1):
        e["rank"] = i
    return ents


def plan_for_budget(rank_list, budget_bytes, header_cost, rows_base=200):
    """Phasen: (0) Header-Inventar, (A) Top-Organe in VOLLER Fidelitaet bis
    Budget — STRICT-STOP beim ersten Voll-Tensor, der nicht mehr passt
    (deterministische Literal-Umsetzung von 'voll bis Budget'), (B) Basis-
    Rung (rows_base Zeilen) fuer weitere Range in Rangordnung, (C) Rest-
    Budget in Verfeinerung (naechsthoehere Rung, wiederholend).

    Budget-Semantik = TRANSFER: bereits gemessene Demo-Tensoren (Cache)
    kosten 0 NEUE Bytes und zaehlen trotzdem zur Abdeckung."""
    avail = budget_bytes - header_cost

    def cost(e, rung):
        if e["cached_rows"] is not None and e["cached_rows"] >= rung["rows"]:
            return 0
        return rung["bytes"]

    assign = {}
    phases = {"A": [], "B": [], "C": []}
    for e in rank_list:                                   # Phase A
        rung = e["rungs"][-1]
        if not rung.get("full"):
            continue
        c = cost(e, rung)
        if c <= avail:
            assign[e["name"]] = dict(rung, phase="A", bytes_gross=rung["bytes"],
                                     bytes_new=c)
            avail -= c
            phases["A"].append(e["name"])
        else:
            break                                         # STRICT-STOP
    for e in rank_list:                                   # Phase B
        if e["name"] in assign:
            continue
        base = [g for g in e["rungs"] if g["rows"] >= rows_base] or e["rungs"]
        rung = base[0]
        c = cost(e, rung)
        if c <= avail:
            assign[e["name"]] = dict(rung, phase="B",
                                     bytes_gross=rung["bytes"], bytes_new=c)
            avail -= c
            phases["B"].append(e["name"])
    changed = True
    while changed:                                        # Phase C
        changed = False
        for e in rank_list:
            cur = assign.get(e["name"])
            if cur is None or cur.get("full"):
                continue
            higher = [g for g in e["rungs"] if g["rows"] > cur["rows"]]
            if not higher:
                continue
            nxt = higher[0]
            delta = cost(e, nxt) - cost(e, cur)
            if delta <= avail:
                avail -= delta
                assign[e["name"]] = dict(nxt, phase="C",
                                         bytes_gross=nxt["bytes"],
                                         bytes_new=cost(e, nxt))
                if e["name"] not in phases["C"]:
                    phases["C"].append(e["name"])
                changed = True
    # kumulierte NEUE Bytes ueber Raenge (Abdeckungs-/Rang-Kurve)
    curve, cum, cum_gross = [], 0, 0
    for e in rank_list:
        g = assign.get(e["name"])
        if g:
            cum += g["bytes_new"]
            cum_gross += g["bytes_gross"]
        curve.append({"rank": e["rank"], "name": e["name"],
                      "tensor_class": e["tensor_class"],
                      "assigned": bool(g),
                      "rows": g["rows"] if g else None,
                      "full": bool(g and g.get("full")),
                      "bytes": g["bytes_gross"] if g else 0,
                      "bytes_new": g["bytes_new"] if g else 0,
                      "cached_free": bool(g and g["bytes_new"] == 0),
                      "phase": g["phase"] if g else None,
                      "cum_bytes_new": cum,
                      "cum_bytes_gross": cum_gross})
    n_cov = sum(1 for c in curve if c["assigned"])
    n_full_top = sum(1 for c in curve[:15] if c["full"])
    n_free = sum(1 for c in curve if c["cached_free"])
    return {"assign": assign, "phases": phases, "curve": curve,
            "header_cost_bytes": header_cost,
            "planned_payload_bytes": cum_gross,
            "planned_new_transfer_bytes": cum,
            "planned_total_bytes": header_cost + cum,
            "unspent_bytes": avail,
            "coverage": {"n_ranked": len(rank_list),
                         "n_covered_base_or_better": n_cov,
                         "coverage_share": round(n_cov / len(rank_list), 4),
                         "top15_at_full_fidelity": n_full_top,
                         "n_cache_covered_free": n_free}}


def validate_estimator_on_demo(ranking, inv, rows_base=200,
                               n_blocks=HTTP_N_BLOCKS):
    """Kostenmodell-Validierung OFFLINE gegen die 65 vorhandenen Demo-
    Records: geplante Kosten (Rung-Formel) vs. gemessene bytes_moved."""
    by_name = {t["name"]: t for t in inv["tensors"]}
    rows = []
    for name, rec in ranking["scored"].items():
        ent = by_name.get(name)
        if ent is None:
            continue
        itemsize = ctm._DT_SIZE.get(ent["dtype"], 2)
        m2, _ = ctm.natural_2d(tuple(ent["shape"]))
        if m2 is None:
            continue
        n_rows, n_cols = m2
        smp = rec.get("sampling", "")
        if smp.startswith("blocks"):
            rows_an = rec["casi_input_shape"][0]
            est = int((rows_an * n_cols * itemsize) * 1.02
                      + (n_blocks + 2) * 1400)
        elif "full" in smp:
            est = int(ent["bytes"] * 1.02 + 3 * 1400)
        else:
            continue
        meas = rec.get("bytes_moved")
        if meas:
            rows.append({"name": name, "estimated": est, "measured": meas,
                         "ratio_est_over_meas": round(est / meas, 4)})
    if rows:
        rs = sorted(r["ratio_est_over_meas"] for r in rows)
        stat = {"n": len(rows), "min": rs[0], "median": rs[len(rs) // 2],
                "max": rs[-1]}
    else:
        stat = {"n": 0}
    return {"records": rows, "stats": stat}


def cmd_plan(args):
    budget = Budget(args.budget_mb)
    reader = HFRangeReader(args.repo, args.revision, budget)
    inv = load_or_scan(reader, budget, refresh=args.refresh)
    ranking = load_ranking_source(args.ranking_json)
    if ranking["repo"] != args.repo:
        print(f"[WARN] Rangquelle {args.ranking_json} stammt von "
              f"{ranking['repo']}, Planer laeuft auf {args.repo}.")
    header_cost = int(inv.get("index_bytes", 0)
                      + sum(s.get("header_len") or 0 for s in inv["shards"])
                      + (2 * len(inv["shards"]) + 2) * 1400)
    rank_list = build_rank_list(inv, ranking, rows_base=args.rows_base,
                                n_blocks=args.blocks)
    budget_mb_list = [float(x) for x in str(args.plan_mb).split(",")]
    out = {"tool": "hf_organ_reader", "command": "plan",
           "repo": args.repo, "revision": reader.rev,
           "ranking_source": {"path": args.ranking_json,
                              "n_top_candidates": ranking["n_top"],
                              "n_scored_records": ranking["n_scored"]},
           "rows_base": args.rows_base, "n_blocks": args.blocks,
           "header_cost_bytes": header_cost, "plans": []}
    out["estimator_validation_offline"] = validate_estimator_on_demo(
        ranking, inv, rows_base=args.rows_base, n_blocks=args.blocks)
    print(f"Rangliste: {len(rank_list)} scorebare Tensoren; Header-Kosten "
          f"{header_cost / 1048576:.2f} MB; Rangquelle "
          f"{ranking['n_top']} top_candidates + "
          f"{ranking['n_scored']} vorhandene Records.")
    for bmb in budget_mb_list:
        plan = plan_for_budget(rank_list, int(bmb * 1024 * 1024),
                               header_cost, rows_base=args.rows_base)
        plan["budget_mb"] = bmb
        out["plans"].append(plan)
        print(f"\n=== PLAN B={bmb:.0f} MB (trocken) ===")
        print(f"  Phase 0 Header: {header_cost / 1048576:.2f} MB | "
              f"Phase A voll: {len(plan['phases']['A'])} Tensoren | "
              f"Phase B Basis: {len(plan['phases']['B'])} | "
              f"Phase C Verfeinerung: {len(plan['phases']['C'])} | "
              f"ungeplant: {plan['unspent_bytes'] / 1048576:.2f} MB")
        print(f"  Abdeckung: {plan['coverage']['n_covered_base_or_better']}"
              f"/{plan['coverage']['n_ranked']} Ränge "
              f"({100 * plan['coverage']['coverage_share']:.1f} %, davon "
              f"{plan['coverage']['n_cache_covered_free']} aus Cache gratis), "
              f"Top-15 voll: {plan['coverage']['top15_at_full_fidelity']}/15")
        for c in plan["curve"][:12]:
            print(f"    Rang {c['rank']:3d} {c['tensor_class']:<14s} "
                  f"rows={str(c['rows']):>6s} full={int(c['full'])} "
                  f"neu={c['bytes_new'] / 1048576:6.2f} MB  cum "
                  f"{c['cum_bytes_new'] / 1048576:7.2f} MB  "
                  f"phase={c['phase']}{' [Cache]' if c['cached_free'] else ''}")
    out["budget"] = budget.as_dict()
    out["planned_at"] = now_utc()
    atomic_write_json(out, os.path.join(RESULTS_DIR,
                                        "hf_reader_budget_planner.json"))

    if args.execute and out["plans"]:
        plan = out["plans"][0]     # nur der ERSTE Plan wird ausgefuehrt
        print(f"\n=== AUSFUEHRUNG Plan B={plan['budget_mb']:.0f} MB ===")
        by_name = {t["name"]: t for t in inv["tensors"]}
        rl = {e["name"]: e for e in rank_list}
        executed = []
        spent0 = budget.total
        for c in plan["curve"]:
            if not c["assigned"]:
                continue
            e = rl[c["name"]]
            if e["cached_rows"] is not None and e["cached_rows"] >= c["rows"]:
                executed.append({"name": c["name"], "skipped_cached": True,
                                 "cached_rows": e["cached_rows"]})
                continue
            ent = by_name[c["name"]]
            try:
                rec = score_streamed(
                    c["name"], ent, reader,
                    max_elems=ent["numel"] if c["full"] else HTTP_MAX_ELEMS,
                    n_blocks=args.blocks,
                    rows_exact=0 if c["full"] else c["rows"])
            except Exception as ex:
                rec = {"name": c["name"], "error": f"{type(ex).__name__}: {ex}"}
                print(f"  {c['name']} ERROR {rec['error']}")
            rec["planned_bytes"] = c["bytes_new"]
            rec["planned_rows"] = c["rows"]
            rec["measured_bytes_moved"] = rec.get("bytes_moved")
            executed.append(rec)
            if "error" not in rec:
                print(f"  [{c['rank']:3d}] {c['name'][-60:]} "
                      f"ratio={rec['ratio']:9.3f} rows={rec['casi_input_shape'][0]} "
                      f"moved={rec['bytes_moved'] / 1048576:.2f}MB "
                      f"(plan {c['bytes'] / 1048576:.2f}MB)")
            out["executed_plan_budget_mb"] = plan["budget_mb"]
            out["executed"] = executed
            out["budget"] = budget.as_dict()
            atomic_write_json(out, os.path.join(RESULTS_DIR,
                                                "hf_reader_budget_planner.json"))
        out["executed"] = executed
        out["executed_spent_bytes"] = budget.total - spent0
    out["budget"] = budget.as_dict()
    out["finished_at"] = now_utc()
    atomic_write_json(out, os.path.join(RESULTS_DIR,
                                        "hf_reader_budget_planner.json"))
    write_planner_md(os.path.join(RESULTS_DIR, "hf_reader_budget_planner.md"),
                     out)
    print(f"\nFERTIG plan: {budget.line()}")


def write_planner_md(path, data):
    lines = []
    A = lines.append
    A("# HF Reader — Budget-Planer (W7/A2): Rangliste vor Bytes")
    A("")
    A(f"*Generiert {data['finished_at']} — `hf_organ_reader.py plan "
      f"{data['repo']}` (Rangquelle: {data['ranking_source']['path']}).*")
    A("")
    A("## Algorithmus (exakt)")
    A("")
    A(f"0. **Header-Inventar** komplett ({data['header_cost_bytes'] / 1048576:.2f} MB: "
      "index.json + alle Shard-Header + Request-Pauschale) — aus Cache, 0 Bytes.")
    A("1. **Rangliste**: Rang 1-15 = top_candidates des 27B-Demo-Reports "
      "(ratio-desc), danach alle scorebaren Tensoren nach Klassen-Mittel "
      "(class_aggregate, desc), in der Klasse layer-aufsteigend.")
    A(f"2. **Phase A** — Top-Organe in VOLLER Fidelitaet bis Budget "
      f"(STRICT-STOP beim ersten nicht passenden Voll-Tensor; ganze Tensoren, "
      f"durch {PER_TENSOR_CEIL // 1048576}-MB-Tensor-Ceiling bzw. 32-MB-"
      "Transposed-Guard begrenzt).")
    A(f"3. **Phase B** — weitere Raenge auf Basis-Rung "
      f"({data['rows_base']} Zeilen, matched protocol R5), in Rangordnung.")
    A("4. **Phase C** — Rest-Budget in VERFEINERUNG: naechsthoehere Rung "
      "(mehr Zeilen: 800/3200/12800/51200/voll) in Rangordnung, solange es "
      "passt. TRANSFER-Budget-Semantik: bereits gemessene Demo-Tensoren "
      "kosten 0 neue Bytes und zaehlen zur Abdeckung.")
    A("")
    A("**Ranking-Befund:** Die Rangliste nach Klassen-Mittel (class_aggregate)"
      " konzentriert das Budget auf die Top-Klasse (gdn_ba: hoechster Mittel-"
      "wert) — Phasigkeit statt Diversitaet. Ein Diversitaets-Constraint pro "
      "Klasse waere der naechste Ausbauschritt (dokumentiert, nicht "
      "implementiert).")
    A("")
    ev = data.get("estimator_validation_offline", {}).get("stats", {})
    if ev.get("n"):
        A(f"**Kostenmodell-Validierung (offline)** gegen die "
          f"{ev['n']} vorhandenen Demo-Records: geplant/gemessen "
          f"(bytes_moved) median {ev['median']:.3f}, min {ev['min']:.3f}, "
          f"max {ev['max']:.3f} — Plan <= Messung +2 % (1.02-Faktor + "
          f"Request-Pauschale).")
    A("")
    A("**Hinweis Phase C:** In beiden Beispielplaenen bleibt C leer — die "
      "Basis-Deckung (Phase B, 200 Zeilen auf breiten Matrizen: 0.45-6.9 MB "
      "pro Rang) verbraucht das Rest-Budget vollständig, bevor Verfeinerung "
      "greift. C aktiviert erst bei Budgets, die die Basis-Deckung der "
      "naechsten Ränge uebersteigen (Algorithmus unterstuetzt sie, die "
      "Beispiel-Budgets erreichen sie nicht).")
    A("")
    for plan in data["plans"]:
        cov = plan["coverage"]
        A(f"## Beispielplan B = {plan['budget_mb']:.0f} MB")
        A("")
        A(f"- Header: {plan['header_cost_bytes'] / 1048576:.2f} MB | "
          f"Payload gesamt (brutto): "
          f"{plan['planned_payload_bytes'] / 1048576:.2f} MB | "
          f"NEUE Transfers: "
          f"{plan['planned_new_transfer_bytes'] / 1048576:.2f} MB | "
          f"ungeplant übrig: {plan['unspent_bytes'] / 1048576:.2f} MB")
        A(f"- Phase A (voll): {len(plan['phases']['A'])} Tensoren, "
          f"Phase B (Basis): {len(plan['phases']['B'])}, "
          f"Phase C (Verfeinerung): {len(plan['phases']['C'])}")
        A(f"- Abdeckung: {cov['n_covered_base_or_better']}/{cov['n_ranked']} "
          f"Rängen = {100 * cov['coverage_share']:.1f} % "
          f"(davon {cov['n_cache_covered_free']} gratis aus Demo-Cache); "
          f"Top-15-Organ-Ränge voll: {cov['top15_at_full_fidelity']}/15")
        A("")
        A("| Rang | Tensor | Klasse | Rung (Zeilen/voll) | Bytes neu | "
          "kumuliert neu |")
        A("|---:|---|---|---|---:|---:|")
        shown = 0
        for c in plan["curve"]:
            if not c["assigned"]:
                continue
            rung_str = "VOLL" if c["full"] else str(c["rows"]) + " Zeilen"
            tag = " [Cache]" if c["cached_free"] else ""
            A(f"| {c['rank']} | `{c['name']}` | {c['tensor_class']} | "
              f"{rung_str} ({c['phase']}){tag} | "
              f"{c['bytes_new'] / 1048576:.2f} MB | "
              f"{c['cum_bytes_new'] / 1048576:.2f} MB |")
            shown += 1
            if shown >= 20:
                break
        A("")
        A("### Abdeckungs-/Rang-Kurve (kumulierte NEUE Bytes vs. kumulierte Ränge)")
        A("")
        A("```")
        cum_final = plan["planned_new_transfer_bytes"]   # Payload ohne Header
        for m in range(11):
            thr = cum_final * m / 10
            k = 0
            for c in plan["curve"]:
                if c["cum_bytes_new"] >= thr:
                    k = c["rank"]
                    break
            bar = "#" * int(60 * m / 10)
            A(f"{m * 10:3d}% {bar:<60s} bei Rang {k}")
        A("```")
        A("")
    if "executed" in data:
        ex = [e for e in data["executed"] if not e.get("skipped_cached")]
        sk = len(data["executed"]) - len(ex)
        meas = [e for e in ex if "error" not in e]
        A(f"## Ausführung (B = {data.get('executed_plan_budget_mb'):.0f} MB)")
        A("")
        A(f"- {len(ex)} Tensoren neu geprobt, {sk} aus Demo-Cache uebernommen "
          f"(0 Bytes).")
        if meas:
            devs = [e["measured_bytes_moved"] / e["planned_bytes"] for e in meas
                    if e.get("planned_bytes")]
            A(f"- gemessen/geplant Bytes: median "
              f"{sorted(devs)[len(devs) // 2]:.3f}, min {min(devs):.3f}, "
              f"max {max(devs):.3f}.")
        A(f"- Netz dieser Ausfuehrung: "
          f"{data.get('executed_spent_bytes', 0) / 1048576:.2f} MB "
          f"(Budget-Klasse haelt die harte Grenze).")
        A("")
    b = data["budget"]
    A("## Transfer-Bilanz (Planer-Lauf)")
    A("")
    A(f"- {b['bytes_total'] / 1048576:.2f} MB / "
      f"{b['limit_bytes'] / 1048576:.0f} MB, {b['http_requests']} Requests "
      "(Trockenplan 0; Ausfuehrung nur auf Anforderung).")
    A("")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ════════════════════════════════════════════════════════════════════════════
# W8 (A3) — Multipart-Range-Verhandlung (begrenzte empirische Sonde)
# ════════════════════════════════════════════════════════════════════════════
MULTIPART_MAX_REQUESTS = 15        # hartes Limit der Sonde
MULTIPART_READ_CAP = 4096          # Lesestopp (falls Server Range ignoriert)


def _probe_range(session, url, headers, method="GET", tag=""):
    r = session.request(method, url, headers=headers or {}, stream=True,
                        allow_redirects=False, timeout=REQUEST_TIMEOUT)
    body = b""
    try:
        for chunkb in r.iter_content(chunk_size=1024):
            body += chunkb
            if len(body) >= MULTIPART_READ_CAP:
                break
    except Exception:
        pass
    finally:
        r.close()
    return {"tag": tag, "method": method,
            "request_headers": dict(headers or {}),
            "status": r.status_code,
            "etag": r.headers.get("ETag", "") or r.headers.get("Etag", ""),
            "content_type": r.headers.get("Content-Type", "")[:60],
            "content_range": r.headers.get("Content-Range", "")[:80],
            "accept_ranges": r.headers.get("Accept-Ranges", ""),
            "content_length_header": r.headers.get("Content-Length", ""),
            "body_bytes_read": len(body),
            "body_prefix_hex": body[:16].hex(),
            "looks_multipart": "multipart" in r.headers.get("Content-Type",
                                                            "").lower()}


def cmd_multipart(args):
    budget = Budget(args.budget_mb)
    reader = HFRangeReader(args.repo, args.revision, budget)
    inv = load_or_scan(reader, budget, refresh=args.refresh)
    fname = args.file or inv["shards"][0]["file"]
    url = reader._cdn_url(fname)          # 1 resolve-Request
    print(f"Sonde auf {fname} -> {urllib.parse.urlparse(url).netloc}")
    tests = [
        ("control-single", {"Range": "bytes=0-7"}),
        ("baseline-2parts", {"Range": "bytes=0-7,100-107"}),
        ("adjacent-2parts", {"Range": "bytes=0-7,8-15"}),
        ("3parts", {"Range": "bytes=0-7,100-107,200-207"}),
        ("accept-byteranges", {"Range": "bytes=0-7,100-107",
                               "Accept": "multipart/byteranges"}),
        ("if-range-etag", {"Range": "bytes=0-7,100-107", "If-Range": "@ETAG@"}),
        ("descending", {"Range": "bytes=100-107,0-7"}),
        ("overlapping", {"Range": "bytes=0-15,8-23"}),
        ("big-gap", {"Range": "bytes=0-7,65536-65543"}),
        ("semicolon-malformed", {"Range": "bytes=0-7;100-107"}),
    ]
    if args.only:
        want = set(args.only.split(","))
        tests = [t for t in tests if t[0] in want]
    out_path = os.path.join(RESULTS_DIR, "hf_reader_multipart_probe.json")
    out = {"tool": "hf_organ_reader", "command": "multipart",
           "repo": args.repo, "file": fname,
           "cdn_host": urllib.parse.urlparse(url).netloc,
           "note": ("Lesestopp nach 4 KiB pro Antwort (Schutz vor 200-"
                    "Vollauslieferung); gezaehlt werden gelesene Bytes. "
                    "If-Range-ETag wird aus der Control-Response "
                    "(control-single) eingesetzt."),
           "max_requests_cap": MULTIPART_MAX_REQUESTS,
           "results": []}
    if os.path.exists(out_path):   #_merge earlier probe results (dedupe by tag)
        try:
            with open(out_path) as f:
                prev = json.load(f)
            out["results"] = list(prev.get("results", []))
            out["requests_used_total_estimate"] = prev.get(
                "requests_used_total_estimate", 0)
        except Exception:
            pass
    used = 1
    etag = ""
    results = out["results"]
    for tag, hdrs in tests:
        if "@ETAG@" in (hdrs.get("If-Range") or "") and etag:
            hdrs = dict(hdrs, **{"If-Range": etag})
        res = _probe_range(reader.session, url, hdrs, tag=tag)
        if not etag and res.get("etag"):
            etag = res["etag"]
        budget.charge(res["body_bytes_read"], 1200, f"multipart:{tag}")
        used += 1
        results[:] = [r for r in results if r["tag"] != tag]
        results.append(res)
        print(f"  [{tag:<22s}] {res['status']} ct={res['content_type'][:40]} "
              f"cr={res['content_range'][:30]} body={res['body_bytes_read']}B "
              f"multipart={res['looks_multipart']}")
    if not args.only or "head-cdn" in (args.only or ""):
        res = _probe_range(reader.session, url, None, method="HEAD",
                           tag="head-cdn")
        budget.charge(0, 800, "multipart:head")
        used += 1
        results[:] = [r for r in results if r["tag"] != "head-cdn"]
        results.append(res)
        print(f"  [head-cdn               ] {res['status']} "
              f"accept-ranges={res['accept_ranges']} "
              f"clen={res['content_length_header']}")
    if not args.only or "origin-multi" in (args.only or ""):
        origin = reader.resolve_url(fname)
        res = _probe_range(reader.session, origin,
                           {"Range": "bytes=0-7,100-107"}, tag="origin-multi")
        budget.charge(res["body_bytes_read"], 1200, "multipart:origin-multi")
        used += 1
        results[:] = [r for r in results if r["tag"] != "origin-multi"]
        results.append(res)
        print(f"  [origin-multi           ] {res['status']} "
              f"ct={res['content_type'][:40]} loc={bool(res['etag'])}")
    out["etag_from_control"] = etag
    out["requests_used"] = used
    out["requests_used_total_estimate"] = (
        out.get("requests_used_total_estimate", 0) + used)
    out["budget_this_run"] = budget.as_dict()
    out["finished_at"] = now_utc()
    atomic_write_json(out, out_path)
    print(f"\nFERTIG multipart: {used} Requests in diesem Lauf "
          f"(kumuliert ueber Laeufe ~"
          f"{out['requests_used_total_estimate']}), {budget.line()}")


# ════════════════════════════════════════════════════════════════════════════
# W9 (A4) — Zeilen-Slice < Zeile: Byte-Sub-Blocks als Analyseeinheit
# ════════════════════════════════════════════════════════════════════════════
def cmd_sliceprobe(args):
    budget = Budget(args.budget_mb)
    reader = HFRangeReader(args.repo, args.revision, budget)
    inv = load_or_scan(reader, budget, refresh=args.refresh)
    by_name = {t["name"]: t for t in inv["tensors"]}
    names = sorted(by_name)
    sel = []
    for g in args.globs:
        sel += [n for n in names if fnmatch.fnmatch(n, g)]
    sel = sorted(set(sel))[:args.limit]
    out = {"tool": "hf_organ_reader", "command": "sliceprobe",
           "repo": args.repo, "slices": args.slices,
           "rows_exact": args.rows_exact, "tensors": [],
           "started_at": now_utc()}
    for name in sel:
        ent = by_name[name]
        mat2d, _ = ctm.natural_2d(tuple(ent["shape"]))
        if mat2d is None:
            continue
        n_rows, n_cols = mat2d
        if n_rows < ctm.MIN_ROWS:
            continue
        itemsize = ctm._DT_SIZE[ent["dtype"]]
        if ent["dtype"] != "BF16":
            continue
        row_bytes = n_cols * itemsize
        if args.slices < 2 or row_bytes % args.slices:
            print(f"  {name}: row_bytes {row_bytes} nicht durch "
                  f"{args.slices} teilbar — uebersprungen")
            continue
        raw, reqs, rows_an = fetch_blocks(
            reader, ent["shard"], ent["data_start"],
            ent["offset_in_shard"][0], n_rows, n_cols, itemsize,
            HTTP_MAX_ELEMS, name, n_blocks=args.blocks,
            rows_floor=ROWS_FLOOR, rows_exact=args.rows_exact)
        half = rows_an // 2
        unit = row_bytes // args.slices
        halves = [raw[:half * row_bytes], raw[half * row_bytes:2 * half * row_bytes]]
        rec = {"name": name, "tensor_class": ctm.classify(name),
               "rows_exact": args.rows_exact, "rows_analyzed": rows_an,
               "rows_per_half": half, "slice_unit_bytes": unit,
               "range_requests": reqs, "bytes_moved": len(raw), "halves": []}
        for hi, hraw in enumerate(halves):
            f32 = ctm.bf16_rows_to_f32(hraw.view(np.uint16),
                                       (half, n_cols), None)
            rrng = np.random.default_rng(
                [SEED, zlib.crc32(name.encode("utf-8")), 31, hi])
            rb = ctm.global_quantile_u8(f32, rrng)
            rcasi, _, _, rnull, rratio = ctm.casi_with_nulls(rb, rrng)
            # Slice-Modus: jede Zeile in S Byte-Slices -> Einheiten-Matrix
            m = hraw.reshape(half * args.slices, unit).astype(np.float32)
            srng = np.random.default_rng(
                [SEED, zlib.crc32(name.encode("utf-8")), 32, hi])
            sb = ctm.global_quantile_u8(m, srng)
            scasi, _, _, snull, sratio = ctm.casi_with_nulls(sb, srng)
            rec["halves"].append({
                "half": hi, "rows_mode_ratio": round(rratio, 4),
                "rows_mode_casi": round(rcasi, 3),
                "rows_mode_null": round(rnull, 3),
                "slice_mode_ratio": round(sratio, 4),
                "slice_mode_casi": round(scasi, 3),
                "slice_mode_null": round(snull, 3)})
            del f32, rb, m, sb
        rr = [h["rows_mode_ratio"] for h in rec["halves"]]
        sr = [h["slice_mode_ratio"] for h in rec["halves"]]
        rec["stability_rows_rel_dev"] = round(
            abs(rr[0] - rr[1]) / max(1e-9, np.mean(rr)), 4)
        rec["stability_slices_rel_dev"] = round(
            abs(sr[0] - sr[1]) / max(1e-9, np.mean(sr)), 4)
        out["tensors"].append(rec)
        print(f"  {name}\n    rows : {rr} rel_dev="
              f"{rec['stability_rows_rel_dev']}\n    slice: {sr} rel_dev="
              f"{rec['stability_slices_rel_dev']}  ({budget.line()})")
        out["budget"] = budget.as_dict()
        atomic_write_json(out, os.path.join(RESULTS_DIR,
                                            "hf_reader_slice_probe.json"))
    out["budget"] = budget.as_dict()
    out["finished_at"] = now_utc()
    atomic_write_json(out, os.path.join(RESULTS_DIR,
                                        "hf_reader_slice_probe.json"))
    md = ["# HF Reader — Zeilen-Slice < Zeile (W9/A4): Machbarkeitsnachweis", "",
          f"*Generiert {out['finished_at']} — subcommand `sliceprobe`.*", "",
          "Konzept: Eine Zeile (row_bytes B) wird in S Byte-Slices zerlegt; "
          "Analyseeinheit ist das Slice statt der Zeile. Die Einheiten-"
          "Matrix (n_units x unit_bytes) wird wie die Zeilen-Matrix "
          "global-quantile-codiert und CASI-gescoret. Stabilitaetsmass: "
          "relative Abweichung des Ratios zwischen den beiden Block-Haelften "
          "(Bloecke 0-3 vs 4-7) je Modus — verglichen wird die STABILITAET, "
          "nicht der absolute Wert (Ratios sind analysegeometrie-abhaengig, "
          "Befund W3/W5).", ""]
    for rec in out["tensors"]:
        md.append(f"## `{rec['name']}` ({rec['tensor_class']})")
        md.append("")
        md.append(f"- Geometrie: {rec['rows_analyzed']} Zeilen in 8 Bloecken, "
                  f"Hälfte = {rec['rows_per_half']} Zeilen; Slice-Einheit "
                  f"{rec['slice_unit_bytes']} B "
                  f"(1/{out['slices']} Zeile, {rec['slice_unit_bytes'] // 2} "
                  f"BF16-Elemente quer zur Zeile).")
        md.append("")
        md.append("| Hälfte | rows ratio | slice ratio |")
        md.append("|---:|---:|---:|")
        for h in rec["halves"]:
            md.append(f"| {h['half']} | {h['rows_mode_ratio']} | "
                      f"{h['slice_mode_ratio']} |")
        md.append("")
        md.append(f"**Stabilität:** rows rel_dev = "
                  f"{rec['stability_rows_rel_dev']}, slices rel_dev = "
                  f"{rec['stability_slices_rel_dev']} → "
                  + ("Slice-Statistik ist STABILER als Zeilen-Statistik."
                     if rec["stability_slices_rel_dev"]
                     < rec["stability_rows_rel_dev"] else
                     "Zeilen-Statistik ist stabiler (Slice-Statistik rauscht "
                     "mehr) — Slice-Granularitaet bringt hier keinen "
                     "Stabilitaetsgewinn."))
        md.append("")
    with open(os.path.join(RESULTS_DIR, "hf_reader_slice_probe.md"),
              "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nFERTIG sliceprobe: {len(out['tensors'])} Tensoren, "
          f"{budget.line()}")


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════
def cmd_scan(args):
    budget = Budget(args.budget_mb)
    reader = HFRangeReader(args.repo, args.revision, budget)
    inv = scan_inventory(reader, budget)
    atomic_write_json(inv, scan_path(args.repo))
    print(f"\nFERTIG scan: {len(inv['tensors'])} Tensoren, "
          f"{len(inv['shards'])} Shards, "
          f"Modell {inv['model_payload_bytes'] / 1e9:.2f} GB")
    print(budget.line())
    print("Output:", scan_path(args.repo))


def cmd_probe(args):
    budget = Budget(args.budget_mb)
    reader = HFRangeReader(args.repo, args.revision, budget)
    inv = load_or_scan(reader, budget, refresh=args.refresh)
    by_name = {t["name"]: t for t in inv["tensors"]}
    names = sorted(by_name)
    sel = []
    for g in args.globs:
        sel += [n for n in names if fnmatch.fnmatch(n, g)]
    sel = sorted(set(sel))
    if args.limit:
        sel = sel[:args.limit]
    if not sel:
        print("Keine Tensoren passen auf die Globs.")
        return
    suffix = ""
    if len(sel) > 12:
        suffix = "_" + zlib.crc32("|".join(args.globs).encode()).hex(8)
    out_path = os.path.join(
        RESULTS_DIR,
        f"hf_organ_reader_probe_{slug(args.repo)}"
        + ("_e01exact" if args.e01_exact else "") + suffix + ".json")
    out = {"tool": "hf_organ_reader", "command": "probe", "repo": args.repo,
           "globs": args.globs, "started_at": now_utc(),
           "max_elems": E01_MAX_ELEMS if args.e01_exact else args.max_elems,
           "n_blocks": args.blocks, "tensors": [], "budget": None}
    for i, n in enumerate(sel):
        t = by_name[n]
        if t["numel"] < MIN_SCORE_ELEMS or len(t["shape"]) < 2:
            print(f"[{i + 1}/{len(sel)}] {n} — nur gezaehlt (numel={t['numel']})")
            continue
        t0 = time.time()
        try:
            rec = score_streamed(
                n, t, reader,
                max_elems=E01_MAX_ELEMS if args.e01_exact else args.max_elems,
                e01_exact=args.e01_exact, n_blocks=args.blocks)
        except Exception as e:
            rec = {"name": n, "tensor_class": t["tensor_class"],
                   "error": f"{type(e).__name__}: {e}"}
        out["tensors"].append(rec)
        if "error" in rec:
            print(f"[{i + 1}/{len(sel)}] {n} ERROR {rec['error']}")
        else:
            print(f"[{i + 1}/{len(sel)}] {n}\n    cls={rec['tensor_class']} "
                  f"ratio={rec['ratio']:.4f} samp={rec['sampling']} "
                  f"moved={rec['bytes_moved'] / 1048576:.2f}MB "
                  f"overfetch={rec['overfetch_ratio']} ({time.time() - t0:.1f}s)")
        print("   ", budget.line())
        out["budget"] = budget.as_dict()
        atomic_write_json(out, out_path)
    out["budget"] = budget.as_dict()
    out["finished_at"] = now_utc()
    atomic_write_json(out, out_path)
    print(f"\nFERTIG probe: {len(out['tensors'])} Tensoren -> {out_path}")
    print(budget.line())


def cmd_report(args):
    budget = Budget(args.budget_mb)
    reader = HFRangeReader(args.repo, args.revision, budget)
    inv = load_or_scan(reader, budget, refresh=args.refresh)
    header_bytes = sum(s.get("header_len") or 0 for s in inv["shards"])

    print("\n--- Sanity-Gates (E01-Disziplin, lokale Fixtures) ---")
    gates = ctm.run_sanity_gates()
    if not gates["passed"]:
        print("*** GATE-FAIL — ABORT (ist selbst ein Befund)")
        sys.exit(2)

    plan = build_plan(inv, args.max_elems, k_scale=args.k_scale,
                      only_classes=(set(args.classes.split(","))
                                    if args.classes else None))
    est = estimate_plan_bytes(plan, n_blocks=args.blocks)
    print(f"\nPlan: {len(plan)} Tensoren, geschaetzt {est / 1048576:.1f} MB "
          f"(Grenze {args.budget_mb:.0f} MB, Zielpfad <=75%)")
    while est > 0.75 * budget.limit and args.max_elems > 262_144:
        args.max_elems //= 2
        plan = build_plan(inv, args.max_elems, k_scale=args.k_scale,
                          only_classes=(set(args.classes.split(","))
                                        if args.classes else None))
        est = estimate_plan_bytes(plan, n_blocks=args.blocks)
        print(f"  Rotation: max_elems -> {args.max_elems}, "
              f"neu {est / 1048576:.1f} MB")

    if args.canonical:
        out_json = os.path.join(RESULTS_DIR, "hf_organ_reader_demo.json")
        out_md = os.path.join(RESULTS_DIR, "hf_organ_reader_demo.md")
    else:
        out_json = os.path.join(
            RESULTS_DIR, f"hf_organ_reader_demo_{slug(args.repo)}.json")
        out_md = os.path.join(
            RESULTS_DIR, f"hf_organ_reader_demo_{slug(args.repo)}.md")
    demo = None
    prev_budget = None
    if os.path.exists(out_json) and not args.fresh:
        try:
            with open(out_json) as f:
                demo = json.load(f)
            if demo.get("repo") != args.repo or \
                    demo.get("http_max_elems") != args.max_elems:
                print("Vorhandenes Demo-JSON passt nicht (Repo/Deckel) — neu.")
                demo = None
        except Exception:
            demo = None
        if demo is not None:
            prev_budget = demo.get("budget")
    if demo is None:
        demo = {"tool": "hf_organ_reader", "command": "report",
                "repo": args.repo, "revision": reader.rev,
                "started_at": now_utc(),
                "model_payload_bytes": inv["model_payload_bytes"],
                "n_shards": len(inv["shards"]),
                "inventory_class_histogram": inv["class_histogram"],
                "http_max_elems": args.max_elems,
                "index_bytes": inv.get("index_bytes", 0),
                "header_bytes": header_bytes,
                "shard_identities": [
                    {k: s.get(k) for k in ("file", "size", "etag", "cas_url_hash")}
                    for s in inv["shards"]],
                "tensors": []}
    done = {t["name"] for t in demo["tensors"]}
    cal = load_calibration(args.calibration)

    if args.redo_classes:
        redo = set(args.redo_classes.split(","))
        before = len(demo["tensors"])
        demo["tensors"] = [t for t in demo["tensors"]
                           if t.get("tensor_class") not in redo]
        print(f"Redo-Klassen {sorted(redo)}: {before} -> "
              f"{len(demo['tensors'])} Eintraege (alte werden verworfen, "
              f"neue Geometrie: rows_floor={ROWS_FLOOR}, rows_exact={args.rows_exact})")
        done = {t["name"] for t in demo["tensors"]}

    for i, t in enumerate(plan):
        if t["name"] in done:
            continue
        t0 = time.time()
        try:
            rec = score_streamed(t["name"], t, reader,
                                 max_elems=t["_max_elems"],
                                 n_blocks=args.blocks,
                                 rows_exact=args.rows_exact)
        except Exception as e:
            rec = {"name": t["name"], "tensor_class": t["tensor_class"],
                   "error": f"{type(e).__name__}: {e}"}
        demo["tensors"].append(rec)
        if "error" in rec:
            print(f"[{i + 1}/{len(plan)}] {t['name']} ERROR {rec['error']}")
        else:
            al = rec["svd_preview"]["alpha_decay"] if rec["svd_preview"] else "—"
            print(f"[{i + 1}/{len(plan)}] {t['name']}\n    "
                  f"cls={rec['tensor_class']:>14s} ratio={rec['ratio']:9.3f} "
                  f"alpha={al} samp={rec['sampling']} "
                  f"moved={rec['bytes_moved'] / 1048576:.2f}MB "
                  f"({time.time() - t0:.1f}s)")
        print("   ", budget.line())
        atomic_write_json(demo, out_json)

    scored = [t for t in demo["tensors"] if "ratio" in t]
    demo["n_scored"] = len(scored)
    demo["class_aggregate"] = aggregate_classes(scored)
    demo["calibration_e01"] = cal
    if args.matched and os.path.exists(args.matched):
        with open(args.matched) as f:
            mref = json.load(f)
        demo["matched_calibration"] = {
            "source": args.matched,
            "repo": mref.get("repo"),
            "http_max_elems": mref.get("http_max_elems"),
            "classes": mref.get("class_aggregate", {})}
    top = [t for t in scored if t["tensor_class"] in DYNAMICS_CLASSES]
    demo["top_candidates"] = sorted(top, key=lambda r: -r["ratio"])[:15]
    this_run = budget.as_dict()
    if prev_budget and not args.fresh:
        # Kumulative Transfer-Bilanz ueber alle Laeufe desselben Demo-JSONs
        demo["budget"] = {
            "limit_bytes": this_run["limit_bytes"],
            "bytes_total": this_run["bytes_total"] + prev_budget.get("bytes_total", 0),
            "bytes_body": this_run["bytes_body"] + prev_budget.get("bytes_body", 0),
            "bytes_overhead_approx": this_run["bytes_overhead_approx"]
            + prev_budget.get("bytes_overhead_approx", 0),
            "http_requests": this_run["http_requests"]
            + prev_budget.get("http_requests", 0),
            "cumulative_over_runs": True}
    else:
        demo["budget"] = this_run
    demo["budget_this_run"] = this_run
    demo["finished_at"] = now_utc()
    demo["validation_note"] = args.validation_note
    atomic_write_json(demo, out_json)
    write_demo_md(out_md, demo)
    print(f"\nFERTIG report: {len(scored)} Tensoren gescoret, {budget.line()}")
    print(f"Output: {out_json}\n        {out_md}")


def main():
    ap = argparse.ArgumentParser(
        description="HF Organ-Reader: Safetensors-On-Demand-Streaming "
                    "ueber HTTP-Range")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--budget-mb", type=float, default=DEFAULT_BUDGET_MB,
                        help="hartes Transfer-Budget in MB (Default 200)")
    common.add_argument("--revision", default="main")

    p_scan = sub.add_parser("scan", parents=[common],
                            help="Tensor-Inventar ueber Header-Ranges, ohne Payload")
    p_scan.add_argument("repo")
    p_scan.set_defaults(func=cmd_scan)

    p_probe = sub.add_parser("probe", parents=[common],
                             help="Tensoren/Zeilen-Bloecke streamen, CASI+SVD-alpha")
    p_probe.add_argument("repo")
    p_probe.add_argument("globs", nargs="+", help="fnmatch-Globs auf Tensornamen")
    p_probe.add_argument("--max-elems", type=int, default=HTTP_MAX_ELEMS,
                         help="Element-Deckel pro Tensor (HTTP-Modus)")
    p_probe.add_argument("--blocks", type=int, default=HTTP_N_BLOCKS)
    p_probe.add_argument("--e01-exact", action="store_true",
                         help="E01-Deckel (8.4M) — reproduziert E01-Ratios bitgenau")
    p_probe.add_argument("--limit", type=int, default=0)
    p_probe.add_argument("--refresh", action="store_true")
    p_probe.set_defaults(func=cmd_probe)

    p_rep = sub.add_parser("report", parents=[common],
                           help="Organ-Scan + Kalibrier-Vergleich -> Demo-Artefakte")
    p_rep.add_argument("repo")
    p_rep.add_argument("--max-elems", type=int, default=HTTP_MAX_ELEMS)
    p_rep.add_argument("--blocks", type=int, default=HTTP_N_BLOCKS)
    p_rep.add_argument("--k-scale", type=float, default=1.0)
    p_rep.add_argument("--classes", default="",
                       help="Kommagetrennte Klassen-Whitelist (Default: PLAN_K)")
    p_rep.add_argument("--redo-classes", default="",
                       help="Klassen aus vorhandenem Demo-JSON verwerfen und "
                            "mit aktueller Geometrie neu proben (Patch-Modus)")
    p_rep.add_argument("--rows-exact", type=int, default=0,
                       help="exakte Zeilenzahl erzwingen (matched protocol)")
    p_rep.add_argument("--calibration", default=E01_JSON)
    p_rep.add_argument("--matched", default="",
                       help="JSON eines matched-protocol Referenz-Reports")
    p_rep.add_argument("--canonical", action="store_true",
                       help="Schreibt hf_organ_reader_demo.{json,md} (Demo-Artefakt)")
    p_rep.add_argument("--validation-note", default="—")
    p_rep.add_argument("--fresh", action="store_true")
    p_rep.add_argument("--refresh", action="store_true")
    p_rep.set_defaults(func=cmd_report)

    p_mk = sub.add_parser(
        "merkle", parents=[common],
        help="W6/A1: Merkle-Streaming — SHA/Chain/Root je Block + exakte "
             "CASI ohne Byte-Puffer (Disk-Spill); --selftest-merkle "
             "validiert gegen den gepufferten Pfad")
    p_mk.add_argument("repo")
    p_mk.add_argument("globs", nargs="*",
                      help="optional: fnmatch-Globs (Default: Selbsttest-Set)")
    p_mk.add_argument("--selftest-merkle", action="store_true",
                      help="3 Referenz-Tensoren, Abbruch mit Exit 2 bei "
                           "Abweichung")
    p_mk.add_argument("--blocks", type=int, default=HTTP_N_BLOCKS)
    p_mk.add_argument("--rows-exact", type=int, default=400)
    p_mk.add_argument("--limit", type=int, default=0)
    p_mk.add_argument("--running-stats", dest="running_stats",
                      action="store_true", default=True,
                      help="laufende Fenster-CASI-Statistik (Default an; "
                           "--no-running-stats fuer reinen Benchmark-Pfad)")
    p_mk.add_argument("--no-running-stats", dest="running_stats",
                      action="store_false")
    p_mk.add_argument("--refresh", action="store_true")
    p_mk.set_defaults(func=cmd_merkle)

    p_mc = sub.add_parser(
        "_merkle_child", parents=[common],
        help=argparse.SUPPRESS)   # intern: ein Pfad isoliert (RSS-Messung)
    p_mc.add_argument("repo")
    p_mc.add_argument("--name", required=True)
    p_mc.add_argument("--mode", default="blocks")
    p_mc.add_argument("--rows-exact", type=int, default=0)
    p_mc.add_argument("--blocks", type=int, default=HTTP_N_BLOCKS)
    p_mc.add_argument("--kind", choices=("streaming", "buffered"),
                      required=True)
    p_mc.add_argument("--running-stats", dest="running_stats",
                      action="store_true", default=True)
    p_mc.add_argument("--no-running-stats", dest="running_stats",
                      action="store_false")
    p_mc.set_defaults(func=_merkle_child_cmd)

    p_plan = sub.add_parser(
        "plan", parents=[common],
        help="W7/A2: Budget-allozierter Planer — Trockenplan (Header, "
             "Top-Organe voll, Rest-Verfeinerung) + optionale Ausfuehrung")
    p_plan.add_argument("repo")
    p_plan.add_argument("--plan-mb", default="50,116",
                        help="Komma-Liste der Plan-Budgets in MB")
    p_plan.add_argument("--rows-base", type=int, default=200)
    p_plan.add_argument("--blocks", type=int, default=HTTP_N_BLOCKS)
    p_plan.add_argument("--ranking-json",
                        default=os.path.join(RESULTS_DIR,
                                             "hf_organ_reader_demo.json"))
    p_plan.add_argument("--execute", action="store_true",
                        help="fuehrt NUR den ERSTEN Plan aus (Netzwerk!)")
    p_plan.add_argument("--refresh", action="store_true")
    p_plan.set_defaults(func=cmd_plan)

    p_mp = sub.add_parser(
        "multipart", parents=[common],
        help="W8/A3: empirische Multipart-Range-Sonde (hart auf 15 Requests "
             "begrenzt, Lesestopp 4 KiB)")
    p_mp.add_argument("repo")
    p_mp.add_argument("--file", default="")
    p_mp.add_argument("--only", default="",
                      help="Kommagetrennte Tag-Filter (control-single,"
                           "if-range-etag,head-cdn,origin-multi,...)")
    p_mp.add_argument("--refresh", action="store_true")
    p_mp.set_defaults(func=cmd_multipart)

    p_sl = sub.add_parser(
        "sliceprobe", parents=[common],
        help="W9/A4: Zeilen-Slice < Zeile — CASI auf Byte-Sub-Blocks, "
             "Stabilitaet Halften-Vergleich")
    p_sl.add_argument("repo")
    p_sl.add_argument("globs", nargs="+")
    p_sl.add_argument("--rows-exact", type=int, default=256)
    p_sl.add_argument("--slices", type=int, default=4)
    p_sl.add_argument("--blocks", type=int, default=HTTP_N_BLOCKS)
    p_sl.add_argument("--limit", type=int, default=2)
    p_sl.add_argument("--refresh", action="store_true")
    p_sl.set_defaults(func=cmd_sliceprobe)

    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
