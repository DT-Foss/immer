"""
fertig.grounding — die Grounding-Schicht: Symbole an Nicht-Wort-Anker binden.

Der Dictionary-Regress endet, wenn ein Symbol an etwas hängt, das KEIN
Wort ist:

  1. Perzeptuelle Anker (CLIP): Das Symbol wird an Bild-Raum gebunden —
     Text-Embedding des Wortes vs. Embedding seines Wikipedia-Bildes.
     Ähnlichkeit > Schwelle = perzeptuell gebunden.
  2. Quantitative Anker: Zahlen mit Einheiten aus Web-Text
     ("runs at 110 km/h", "weighs 5 tons") — Messungen, keine Wörter.
  3. Coverage-Metrik: Anteil der Symbole mit mindestens einem
     Nicht-Wort-Anker. Das ist die Fortschrittszahl des Moonshots.

Gebundene Fakten wandern in den Welt-Graphen (z. B. (cheetah, runs_at,
"110 km/h")) und machen quantitative Fragen beantwortbar.
"""

from __future__ import annotations

import io
import re
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import sources

# ---------------------------------------------------------------------------
# CLIP (perzeptueller Anker) — optional, dependency-frei degradierbar
# ---------------------------------------------------------------------------

_CLIP = None
_CLIP_PREPROCESS = None
_CLIP_TOKENIZER = None


def _ensure_clip():
    """CLIP lazy-laden (open_clip, ViT-B-32). None wenn nicht verfügbar."""
    global _CLIP, _CLIP_PREPROCESS, _CLIP_TOKENIZER
    if _CLIP is not None:
        return True
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k")
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        model.eval()
        _CLIP, _CLIP_PREPROCESS, _CLIP_TOKENIZER = model, preprocess, tokenizer
        return True
    except Exception:
        return False


def _clip_text_embedding(text: str) -> Optional[np.ndarray]:
    if not _ensure_clip():
        return None
    import torch
    with torch.no_grad():
        toks = _CLIP_TOKENIZER([text])
        emb = _CLIP.encode_text(toks)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].numpy()


def _clip_image_embedding(image_bytes: bytes) -> Optional[np.ndarray]:
    if not _ensure_clip():
        return None
    import torch
    from PIL import Image
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = _CLIP_PREPROCESS(img).unsqueeze(0)
        with torch.no_grad():
            emb = _CLIP.encode_image(x)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb[0].numpy()
    except Exception:
        return None


def _fetch_image(url: str) -> Optional[bytes]:
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "fertig/1.0 (research)"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    except Exception:
        return None


def perceptual_anchor(word: str, threshold: float = 0.22) -> Optional[dict]:
    """Symbol an Bild-Raum binden: Wikipedia-Bild des Wortes + CLIP.

    Liefert {similarity, image_url} wenn gebunden, sonst None."""
    if not _ensure_clip():
        return None
    title, desc, extract = sources.scrape.wikipedia_summary(word)
    if not extract:
        return None
    # Bild-URL aus dem REST-Summary (thumbnail)
    import json
    import urllib.parse
    try:
        d = json.loads(sources._http(
            "https://en.wikipedia.org/api/rest_v1/page/summary/" +
            urllib.parse.quote(title)))
        img_url = d.get("thumbnail", {}).get("source")
    except Exception:
        img_url = None
    if not img_url:
        return None
    img_bytes = _fetch_image(img_url)
    if not img_bytes:
        return None
    te = _clip_text_embedding(word)
    ie = _clip_image_embedding(img_bytes)
    if te is None or ie is None:
        return None
    sim = float(np.dot(te, ie))
    if sim < threshold:
        return None
    return {"similarity": sim, "image_url": img_url}


# ---------------------------------------------------------------------------
# Quantitative Anker (Zahlen + Einheiten — Messungen, keine Wörter)
# ---------------------------------------------------------------------------

_UNIT_RE = re.compile(
    r"\b(\d+(?:[.,]\d+)?)\s*(km/h|kmh|kph|mph|m/s|kg|g|ton|tons|t|cm|m|km|"
    r"mm|liters|litres|l|ml|years|year|months|days|hours|minutes|seconds|"
    r"degrees?|°c|°f|celsius|fahrenheit|meters?|metres?|kilometers?|"
    r"kilometres?|percent|%)\b", flags=re.I)

# Relations-Verben, vor denen quantitative Fakten stehen ("runs at",
# "weighs", "measures", "reaches", "grows to", "can reach")
_QUANT_VERBS = {
    "runs at", "run at", "can run at", "running at", "speeds up to",
    "at speeds of", "up to", "reaches", "reach", "weighs", "weigh",
    "weighing", "measures", "measure", "measuring", "grows to", "grow to",
    "can reach", "can weigh", "can grow", "grows up to", "can measure",
    "has a top speed of", "top speed of", "can travel at", "travels at",
    "lives for", "live for", "can live for", "lasts", "last", "holds",
    "contains up to", "costs", "cost", "produces up to", "takes", "takes up",
}


def _parse_number(s: str) -> Optional[float]:
    """Zahl parsen — Tausendertrenner vs. Dezimalzeichen unterscheiden.

    '6,350' -> 6350.0 (engl. Tausendertrenner)
    '3.14'  -> 3.14   (Dezimalpunkt)
    '12,5'  -> 12.5   (europäisches Dezimalkomma)
    '1,234,567' -> 1234567.0
    """
    s = s.strip()
    if "," in s:
        parts = s.split(",")
        # Alle Komma-Gruppen (außer der ersten) sind genau 3 Ziffern
        # lang -> Tausendertrenner; sonst Dezimalkomma.
        if all(re.fullmatch(r"\d{3}(?:\.\d+)?", p) for p in parts[1:]):
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    return float(s)


def extract_quantitative(text: str) -> List[Tuple[str, str, float]]:
    """Quantitative Fakten: (Verb-Kontext, 'Zahl Einheit', Konfidenz)."""
    out = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        low = sent.lower()
        for verb in _QUANT_VERBS:
            idx = low.find(verb)
            if idx < 0:
                continue
            window = sent[idx + len(verb):idx + len(verb) + 60]
            m = _UNIT_RE.search(window)
            if m:
                try:
                    val = _parse_number(m.group(1))
                except ValueError:
                    continue
                out.append((verb, f"{m.group(1)} {m.group(2).lower()}",
                            val))
            break
    # Dedupe
    seen = set()
    uniq = []
    for v, u, val in out:
        key = (v, u)
        if key not in seen:
            seen.add(key)
            uniq.append((v, u, val))
    return uniq


# Infobox-Felder, die quantitative Anker sind (strukturiert, hochkonfident)
_QUANT_INFOBAR = {"mass", "weight", "speed", "top_speed", "length",
                  "height", "wingspan", "lifespan", "life_span",
                  "diameter", "depth", "width", "area", "volume",
                  "capacity", "range", "maximum_speed", "average_speed",
                  "max_speed", "max_length", "max_height", "max_weight",
                  "gestation", "incubation", "age", "size"}


def quantitative_anchors(word: str, sources_list: Optional[List[str]] = None,
                         ) -> List[Tuple[str, str, float]]:
    """Quantitative Anker: Wikipedia-Infobox (strukturiert) zuerst, dann
    Wikipedia-Text, dann Web (nur als Ergänzung)."""
    anchors = []
    # 1. Infobox: strukturierte quantitative Felder (höchste Konfidenz)
    try:
        infobox = sources.scrape.wikipedia_infobox(word)
    except Exception:
        infobox = {}
    for key, val in infobox.items():
        if key in _QUANT_INFOBAR:
            m = _UNIT_RE.search(val)
            if m and _parse_number(m.group(1)) is not None:
                anchors.append((key, f"{m.group(1)} {m.group(2).lower()}",
                                0.70))
    # 2. Wikipedia-Text (Summary zuerst, dann Volltext-Fallback)
    title, desc, extract = sources.scrape.wikipedia_summary(word)
    if extract:
        anchors += [(v, u, 0.50) for v, u, _ in
                    extract_quantitative(extract[:4000])]
    if not anchors:
        # Volltext: den ganzen Artikel crawlen (Trafilatura)
        import urllib.parse
        full = sources.extract_text(
            "https://en.wikipedia.org/wiki/" +
            urllib.parse.quote(title.replace(" ", "_")))
        if len(full) > 500:
            anchors += [(v, u, 0.45) for v, u, _ in
                        extract_quantitative(full[:12000])]
    # 3. Web nur, wenn noch keine Anker gefunden wurden
    if not anchors:
        for url in sources._ddg_urls(f"{word} speed weight size")[:2]:
            text = sources.extract_text(url)
            if len(text) > 200:
                anchors += [(v, u, 0.35) for v, u, _ in
                            extract_quantitative(text)]
    seen = set()
    uniq = []
    for v, u, conf in anchors:
        key = (v, u)
        if key not in seen:
            seen.add(key)
            uniq.append((v, u, conf))
    return uniq[:6]


# ---------------------------------------------------------------------------
# Grounding-Coverage — die Moonshot-Metrik
# ---------------------------------------------------------------------------

# Erdungs-Ebenen (ehrliche Terminologie — nichts davon ist "volle Erdung"
# im Harnad-Sinn; alles ist menschlich vermittelt, nur unterschiedlich nah
# an primärer Transduktion):
#   L0: nur Wort-Wort-Kanten (reiner Regress)
#   L1: quantitative Anker — Zahlen aus Webtext. Zahlen sind nicht-
#       lexikalische Symbole mit fester Denotation, ABER die Quelle ist
#       menschlicher Text (vermittelt).
#   L2: perzeptuelle Bindung — CLIP/Commons-Bilder. Die Pixel sind primär,
#       die Wort↔Bild-Zuordnung ist menschenkuratiert (vermittelt).
#   L3: unüberwachte Kategorien aus Pixel-Struktur (fertig.vision) — die
#       Kategorie entsteht ohne Labels; Wörter werden NACH der Kategorie-
#       Bildung zugeordnet. Die nächste Annäherung an Harnads
#       sensorische Transduktion, die ein textbasiertes System erreichen
#       kann.
GROUNDING_LEVELS = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}


def grounding_coverage(graph_triplets: List[Tuple[str, str, str, float]],
                       anchored: Dict[str, int]) -> Dict[str, float]:
    """Anteil der Graph-Symbole pro Erdungs-Ebene.

    anchored: Symbol -> höchste erreichte Ebene (0-3). Die Metrik ist
    ehrlich: "verankert" heißt nicht "geerdet" — volle Erdung verlangt
    sensorische Transduktion, die kein textbasiertes System hat."""
    symbols = set()
    for a, b, c, _ in graph_triplets:
        symbols.add(a)
        symbols.add(c)
    grounded = sum(1 for s in symbols if anchored.get(s))
    return {
        "symbols": len(symbols),
        "grounded": grounded,
        "coverage": grounded / max(len(symbols), 1),
    }


def ground_symbol(word: str, store: bool = False,
                  verbose: bool = True) -> dict:
    """Anker-Evidenz für ein Symbol sammeln: perzeptuell (CLIP-
    Quer-Modalität) + quantitativ (Text-Zeugnis). `store=True` nur mit
    explizitem Ziel — Default dry-run (Research-API-Prinzip).

    Ehrliche Terminologie: das ist EVIDENZ-SAMMLUNG, keine vollständige
    Harnad-Erdung. Die Datenquellen sind menschlich vermittelt
    (cross_modal_proxy / social_testimony).
    """
    from . import gaps as gaps_mod
    result = {"word": word}
    # 1. Perzeptuell
    pa = perceptual_anchor(word)
    result["perceptual"] = pa
    # 2. Quantitativ
    qa = quantitative_anchors(word)
    result["quantitative"] = qa
    # 3. Fakten in den Graphen
    if store and (pa or qa):
        trips = gaps_mod._load_world()
        best = {t[:3]: t[3] for t in trips}
        if pa:
            key = (word, "has_image", "perceptual")
            best[key] = max(best.get(key, 0.0), pa["similarity"])
        for verb, unit, val in qa:
            key = (word, verb.replace(" ", "_"), unit)
            best[key] = max(best.get(key, 0.0), 0.55)
        gaps_mod._save_world([(a, b, c, conf)
                              for (a, b, c), conf in best.items()])
        result["graph_triplets"] = len(best)
    if verbose:
        print(f"[ground] {word}:")
        if pa:
            print(f"  perzeptuell : gebunden (CLIP-Sim {pa['similarity']:.3f})")
        else:
            print(f"  perzeptuell : nicht gebunden")
        print(f"  quantitativ : {len(qa)} Anker")
        for v, u, val in qa[:4]:
            print(f"    {v} {u} ({val})")
    return result
