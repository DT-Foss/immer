"""
fertig.scrape — Weltwissen-Beschaffung: Wikipedia -> .causal-Tripletts.

Der Gap-Loop (F4: externer Index, on demand):
  Lücke erkannt -> Query formuliert (Ziel-Entität) -> Wikipedia geholt ->
  deterministische Extraktion (Infobox + is-a-Muster) -> Tripletts mit
  ehrlicher Konfidenz -> in den wachsenden Graphen gespeichert.

Extraktions-Konfidenz (gemessen, nicht behauptet):
  Infobox key=value : 0.7   (strukturiert, hoch)
  "X is a Y"        : 0.5   (Muster, mittel)
  Description       : 0.4   (frei, niedrig)

Kein LLM, kein Training — reines deterministisches Parsen. Mehrere Quellen,
die dasselbe Triplett liefern, heben die Konfidenz (max).
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

_REST = "https://en.wikipedia.org/api/rest_v1/page/summary/{t}"
_ACTION = ("https://en.wikipedia.org/w/api.php?action=query&prop=revisions"
           "&rvprop=content&rvslots=main&titles={t}&format=json&formatversion=2")

_INFOBAR_SKIP = {"name", "image", "image_size", "image_caption", "caption",
                 "logo", "logo_size", "alt", "website", "logo_caption",
                 "image_alt", "flag", "flag_size", "seal", "seal_size",
                 "map", "map_size", "map_caption", "coordinates", "footnotes",
                 "footnote", "module", "embedded", "header", "header1",
                 "header2", "header3", "width", "collapsible", "label1",
                 "data1"}


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent":
                                               "fertig/1.0 (research)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def wikipedia_summary(title: str) -> Tuple[str, Optional[str], str]:
    """(title, description, extract) — None bei Fehler/Fehlen."""
    try:
        d = _fetch_json(_REST.format(t=urllib.parse.quote(title)))
    except Exception:
        return title, None, ""
    return d.get("title", title), d.get("description"), d.get("extract", "")


def wikipedia_infobox(title: str) -> Dict[str, str]:
    """Infobox key=value-Paare aus dem Wikitext (deterministisch)."""
    try:
        d = _fetch_json(_ACTION.format(t=urllib.parse.quote(title)))
    except Exception:
        return {}
    pages = d.get("query", {}).get("pages", [])
    if not pages:
        return {}
    content = (pages[0].get("revisions", [{}])[0]
               .get("slots", {}).get("main", {}).get("content", ""))
    out: Dict[str, str] = {}
    m = re.search(r"\{\{\s*(?:Infobox|Speciesbox|Taxobox|Automatic_taxobox|"
                  r"Subspeciesbox|Infobox_animal|Infobox_food|Infobox_drug)",
                  content)
    if not m:
        return out
    block = content[m.start():]
    end = re.search(r"\n\}\}", block)
    if end:
        block = block[:end.start()]
    for km in re.finditer(r"\|\s*([a-z][a-z0-9_ ]*?)\s*=\s*([^\n|]+)",
                          block, flags=re.I):
        key = km.group(1).strip().lower()
        val = km.group(2).strip()
        val = re.sub(r"\[\[([^\]|]*?)(?:\|[^\]]*)?\]\]", r"\1", val)
        val = re.sub(r"<ref[^>]*>.*?</ref>", "", val, flags=re.S)
        val = re.sub(r"\{\{[^}]*\}\}", "", val).strip()
        val = re.sub(r"<[^>]+>", "", val).strip()
        if key not in _INFOBAR_SKIP and len(val) > 1 and len(val) < 200:
            out[key] = val
    return out


def extract_triplets(title: str, description: Optional[str],
                     extract: str, infobox: Dict[str, str]
                     ) -> List[Tuple[str, str, str, float]]:
    """Deterministische Triplett-Extraktion mit Konfidenz-Tiers."""
    trips: List[Tuple[str, str, str, float]] = []
    t = title.strip().lower()

    # Tier 1: Infobox (strukturiert, hoch)
    for key, val in infobox.items():
        trips.append((t, key, val.lower(), 0.7))

    # Tier 2: "X is a/an/the Y" aus den ersten Sätzen
    for sent in re.split(r"(?<=[.!?])\s+", extract)[:4]:
        s = sent.strip()
        m = re.match(
            r"^" + re.escape(t) + r"\s+is\s+(?:a|an|the)\s+"
            r"([a-z][a-z \-]*)", s, flags=re.I)
        if m:
            y = re.sub(r"[,.]$", "", m.group(1)).strip()
            if y and len(y) > 2:
                trips.append((t, "is_a", y, 0.5))

    # Tier 3: Description (frei, niedrig)
    if description:
        trips.append((t, "described_as", description.lower(), 0.4))

    # Dedupe: gleiches Triplett -> max Konfidenz
    best: Dict[Tuple[str, str, str], float] = {}
    for a, b, c, conf in trips:
        key = (a, b, c)
        best[key] = max(best.get(key, 0.0), conf)
    return [(a, b, c, conf) for (a, b, c), conf in best.items()]
