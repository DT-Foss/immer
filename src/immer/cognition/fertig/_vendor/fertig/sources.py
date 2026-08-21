"""
fertig.sources — das Quellen-Registry: das Internet als Online-Learning.

Eine Schnittstelle, viele Quellen. Jede Quelle liefert
(trigger, mechanism, outcome, confidence)-Tripletts; der gemeinsame Kern
ist der kausale Satz-Extraktor (deterministisch, mit Negations-Schutz).

Quellen (v1.1, stdlib-only):
  wikipedia         : Infobox + is-a-Muster + Description      (0.70/0.50/0.40)
  wiktionary        : Definitions-Seiten (JSON)                (0.45)
  duckduckgo        : Suchergebnis-Snippets -> Kausal-Saetze    (0.30)
  arxiv             : Titel + Abstracts -> Kausal-Saetze        (0.35)
  pubmed            : Abstracts (Textformat) -> Kausal-Saetze   (0.40)
  semantic_scholar  : Titel + Abstracts (Graph-API)            (0.35)
  openalex          : Abstracts (inverted index)               (0.35)

Jede Quelle ist getrennt in fetch (Netz) und parse (Text) — parse ist
offline testbar. Aggregation: max-Konfidenz + Evidenz-Boost pro
bestaetigender Quelle. Neue Quellen = parse-Funktion + Konfidenz-Tier.
"""

from __future__ import annotations

import html
import html.parser
import json
import re
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

from . import scrape

# ---------------------------------------------------------------------------
# Kausaler Satz-Extraktor (der gemeinsame Kern)
# ---------------------------------------------------------------------------

_CAUSAL = [
    ("causes", "causes"), ("cause", "causes"),
    ("leads to", "leads_to"), ("lead to", "leads_to"),
    ("results in", "results_in"), ("result in", "results_in"),
    ("reduces", "reduces"), ("reduce", "reduces"),
    ("increases", "increases"), ("increase", "increases"),
    ("prevents", "prevents"), ("prevent", "prevents"),
    ("promotes", "promotes"), ("promote", "promotes"),
    ("associated with", "associated_with"),
    ("contributes to", "contributes_to"),
    ("induces", "induces"), ("induce", "induces"),
    ("triggers", "triggers"), ("raises", "raises"), ("lowers", "lowers"),
    ("improves", "improves"), ("improve", "improves"),
    ("worsens", "worsens"), ("inhibits", "inhibits"),
    ("enhances", "enhances"), ("impairs", "impairs"),
    ("linked to", "linked_to"), ("linked with", "linked_with"),
    # funktionale/definitorische Relationen (für Wissens-QA)
    ("coordinates", "coordinates"), ("controls", "controls"),
    ("regulates", "regulates"), ("produces", "produces"),
    ("transports", "transports"), ("carries", "carries"),
    ("responsible for", "responsible_for"),
    ("used to", "used_to"), ("used for", "used_for"),
    ("made of", "made_of"), ("composed of", "composed_of"),
    ("consists of", "consists_of"), ("contains", "contains"),
    ("absorbs", "absorbs"), ("releases", "releases"),
    ("converts", "converts"), ("provides", "provides"),
    ("supports", "supports"), ("protects", "protects"),
    ("filters", "filters"), ("removes", "removes"),
]
_CAUSAL_RE = re.compile(
    r"\b(" + "|".join(re.escape(v) for v, _ in _CAUSAL) + r")\b",
    flags=re.I)

# Fuehrende Funktionswoerter (werden von NPs abgestreift; interne bleiben)
_STOP_LEAD = {"the", "a", "an", "this", "that", "these", "those", "such",
              "some", "any", "all", "both", "each", "every", "no", "my",
              "your", "his", "her", "its", "our", "their", "in", "on", "at",
              "to", "for", "with", "by", "from", "of", "as", "than", "it",
              "its", "which", "who", "whom", "whose", "there", "here",
              "however", "while", "when", "also", "but", "and", "or", "not"}


def _negated(sentence: str) -> bool:
    return bool(re.search(r"\b(not|no|never|without|rarely|seldom)\b",
                          sentence, flags=re.I))


def _np(phrase: str) -> str:
    """Nominalphrase: fuehrende Funktionswoerter abstreifen, interne
    Struktur erhalten ('cavities in children' bleibt erhalten)."""
    toks = phrase.lower().split()
    while toks and toks[0] in _STOP_LEAD:
        toks = toks[1:]
    return " ".join(toks)


def _ok_np(np_: str) -> bool:
    """1-8 Wörter, min 2 Zeichen — Funktionswörter sind durch _np
    bereits abgestreift, daher sind Einzelwort-NPs legitim."""
    words = np_.split()
    return 1 <= len(words) <= 8 and len(np_) >= 2


def extract_causal(text: str, base_conf: float = 0.35,
                   max_triplets: int = 25) -> List[Tuple[str, str, str, float]]:
    """Kausal-Saetze -> Tripletts. Negation schuetzt vor falschen Kanten."""
    out: List[Tuple[str, str, str, float]] = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if _negated(sent):
            continue
        m = _CAUSAL_RE.search(sent)
        if not m:
            continue
        verb = m.group(1).lower()
        canon = dict((v.lower(), c) for v, c in _CAUSAL).get(verb, verb)
        from . import primitives as _primitives
        canon = _primitives.normalize_mechanism(canon)
        before = sent[:m.start()].strip()
        after = sent[m.end():].strip()
        # Kontext-Grenzen: letzter Nebensatz vor dem Verb, erster nach
        before = re.split(r"[,;]|(?:^|\s)(?:in|with|among|for|through) ",
                          before)[-1]
        after = re.split(r"[,;.]", after)[0]
        subj, obj = _np(before), _np(after)
        if _ok_np(subj) and _ok_np(obj):
            out.append((subj, canon, obj, base_conf))
    # Dedupe (max-Konfidenz)
    best: Dict[Tuple[str, str, str], float] = {}
    for a, b, c, conf in out:
        key = (a, b, c)
        best[key] = max(best.get(key, 0.0), conf)
    return [(a, b, c, conf) for (a, b, c), conf in best.items()][:max_triplets]




# ---------------------------------------------------------------------------
# Generelle Relations-Extraktion: delegiert an das Primitiv-Schema
# ---------------------------------------------------------------------------

def extract_relations(text: str, base_conf: float = 0.40,
                      max_triplets: int = 40
                      ) -> List[Tuple[str, str, str, float]]:
    """Schema-getrieben: fertig.relations -> fertig.primitives.RELATIONS."""
    from .relations import extract_relations as _schema_extract
    return _schema_extract(text, base_conf=base_conf,
                           max_triplets=max_triplets)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _http(url: str) -> str:
    req = urllib.request.Request(url, headers={
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0 Safari/537.36"),
        "Accept": "text/html,application/json,*/*"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Quellen: fetch (Netz) + parse (Text, offline testbar)
# ---------------------------------------------------------------------------

def fetch_wikipedia(target: str) -> List[Tuple[str, str, str, float]]:
    title, desc, extract = scrape.wikipedia_summary(target)
    if not extract and not desc:
        return []
    infobox = scrape.wikipedia_infobox(title)
    return scrape.extract_triplets(title, desc, extract, infobox)


def parse_duckduckgo(page: str, base_conf: float = 0.30):
    snips = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', page,
                       flags=re.S)
    text = " ".join(html.unescape(re.sub(r"<[^>]+>", "", s)) for s in snips)
    return extract_relations(text, base_conf=base_conf)


def fetch_duckduckgo(target: str) -> List[Tuple[str, str, str, float]]:
    url = ("https://html.duckduckgo.com/html/?q=" +
           urllib.parse.quote(f"{target} causes effect"))
    try:
        return parse_duckduckgo(_http(url))
    except Exception:
        return []


def parse_arxiv(xml: str, base_conf: float = 0.35):
    trips: List[Tuple[str, str, str, float]] = []
    for e in re.findall(r"<entry>(.*?)</entry>", xml, flags=re.S):
        tm = re.search(r"<title>(.*?)</title>", e, flags=re.S)
        title = html.unescape(re.sub(r"<[^>]+>", "", tm.group(1))) if tm else ""
        sm = re.search(r"<summary>(.*?)</summary>", e, flags=re.S)
        abstract = (html.unescape(re.sub(r"<[^>]+>", "", sm.group(1)))
                    if sm else "")
        trips += extract_relations(title + ". " + abstract,
                                base_conf=base_conf)
    return trips


def fetch_arxiv(target: str) -> List[Tuple[str, str, str, float]]:
    url = ("https://export.arxiv.org/api/query?search_query=all:" +
           urllib.parse.quote(target) + "&max_results=3")
    try:
        return parse_arxiv(_http(url))
    except Exception:
        return []


def parse_pubmed(txt: str, base_conf: float = 0.40):
    return extract_causal(txt, base_conf=base_conf)


def fetch_pubmed(target: str) -> List[Tuple[str, str, str, float]]:
    try:
        j = json.loads(_http(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            "?db=pubmed&retmode=json&retmax=3&term=" +
            urllib.parse.quote(target)))
        ids = j["esearchresult"].get("idlist", [])
        if not ids:
            return []
        txt = _http(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            "?db=pubmed&rettype=abstract&retmode=text&id=" + ",".join(ids))
    except Exception:
        return []
    return parse_pubmed(txt)


def parse_semantic_scholar(j: dict, base_conf: float = 0.35):
    trips = []
    for p in j.get("data", []):
        title = p.get("title", "")
        abstract = p.get("abstract") or ""
        trips += extract_relations(title + ". " + abstract,
                                base_conf=base_conf)
    return trips


def fetch_semantic_scholar(target: str) -> List[Tuple[str, str, str, float]]:
    url = ("https://api.semanticscholar.org/graph/v1/paper/search"
           "?query=" + urllib.parse.quote(target) +
           "&fields=title,abstract&limit=3")
    try:
        return parse_semantic_scholar(json.loads(_http(url)))
    except Exception:
        return []


def _inverted_to_text(inv: dict) -> str:
    """OpenAlex abstract_inverted_index -> Text."""
    if not inv:
        return ""
    pos = [(p, w) for w, ps in inv.items() for p in ps]
    pos.sort()
    return " ".join(w for _, w in pos)


def parse_openalex(j: dict, base_conf: float = 0.35):
    trips = []
    for w in j.get("results", []):
        title = w.get("title") or ""
        abstract = _inverted_to_text(w.get("abstract_inverted_index") or {})
        trips += extract_relations(title + ". " + abstract,
                                base_conf=base_conf)
    return trips


def fetch_openalex(target: str) -> List[Tuple[str, str, str, float]]:
    url = ("https://api.openalex.org/works?search=" +
           urllib.parse.quote(target) + "&per-page=3")
    try:
        return parse_openalex(json.loads(_http(url)))
    except Exception:
        return []


def parse_wiktionary(j: dict, title: str, base_conf: float = 0.45):
    """Wiktionary-Definitionen -> is_a/defined_as-Tripletts."""
    trips: List[Tuple[str, str, str, float]] = []
    entries = j.get("en", [])
    if not entries:
        entries = [j]
    t = title.strip().lower()
    for e in entries:
        for part in e.get("partOfSpeech", []):
            for d in part.get("definitions", []):
                defn = re.sub(r"<[^>]+>", "", d.get("definition", "")).strip()
                if not defn:
                    continue
                m = re.match(r"(?:any of |a |an )?(?:type of |kind of |form of |"
                             r"substance )?(.+?)(?:[,;.]| that| which)", defn)
                if m:
                    y = _np(m.group(1))
                    if len(y) > 2:
                        trips.append((t, "is_a", y, base_conf))
                trips.append((t, "defined_as", defn.lower(), base_conf * 0.8))
    return trips


def fetch_wiktionary(target: str) -> List[Tuple[str, str, str, float]]:
    url = ("https://en.wiktionary.org/api/rest_v1/page/definition/" +
           urllib.parse.quote(target))
    try:
        return parse_wiktionary(json.loads(_http(url)), target)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Generische Web-Schicht: jede beliebige URL -> Text -> Tripletts
# ---------------------------------------------------------------------------

_HAS_TRAFILATURA = False
try:
    import trafilatura  # noqa: F401
    _HAS_TRAFILATURA = True
except ImportError:
    pass


class _TextExtractor(html.parser.HTMLParser):
    """Stdlib-Fallback: sichtbarer Text aus HTML (ohne Skripte/Styles)."""

    def __init__(self):
        super().__init__()
        self.parts: List[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "svg", "head"):
            self._skip += 1
        if tag in ("p", "div", "br", "li", "h1", "h2", "h3", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg", "head"):
            self._skip = max(0, self._skip - 1)

    def handle_data(self, data):
        if self._skip == 0:
            self.parts.append(data)

    def text(self) -> str:
        t = " ".join("".join(self.parts).split())
        return t


def _extract_text_stdlib(html_text: str) -> str:
    p = _TextExtractor()
    try:
        p.feed(html_text)
    except Exception:
        return ""
    return p.text()


def extract_text(url: str) -> str:
    """URL -> sichtbarer Text. Trafilatura (Boilerplate-Removal) mit
    stdlib-Fallback — das System bleibt dependency-frei lauffähig."""
    try:
        if _HAS_TRAFILATURA:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                return trafilatura.extract(downloaded) or ""
        html_text = _http(url)
        return _extract_text_stdlib(html_text)
    except Exception:
        return ""


def _parse_ddg_urls(page: str, n: int = 3) -> List[str]:
    hrefs = re.findall(r'class="result__a"[^>]*href="([^"]+)"', page)
    out = []
    for h in hrefs:
        h = html.unescape(h)
        m = re.search(r"uddg=([^&]+)", h)
        if m:
            h = urllib.parse.unquote(m.group(1))
        if h.startswith("http") and "duckduckgo.com" not in h:
            out.append(h)
        if len(out) >= n:
            break
    return out


def _ddg_urls(target: str, n: int = 3) -> List[str]:
    """DDG-Suche -> Ergebnis-URLs (generische Ziel-Findung)."""
    url = ("https://html.duckduckgo.com/html/?q=" +
           urllib.parse.quote(target))
    try:
        page = _http(url)
    except Exception:
        return []
    return _parse_ddg_urls(page, n=n)


def parse_web(text: str, base_conf: float = 0.28):
    """Web-Text (generisch) -> Kausal-Tripletts. Niedrige Konfidenz:
    beliebige Seiten sind lauter als kuratierte Quellen."""
    return extract_causal(text[:6000], base_conf=base_conf)


def fetch_web(target: str) -> List[Tuple[str, str, str, float]]:
    """Generische Web-Quelle: Suche -> Seiten -> Text -> Tripletts.
    Das ist der Ersatz für tausend Einzel-Adapter: eine Schicht für
    jede beliebige Website."""
    trips: List[Tuple[str, str, str, float]] = []
    for url in _ddg_urls(target):
        text = extract_text(url)
        if len(text) > 200:
            trips += parse_web(text)
    return trips


def fetch_url_direct(url: str) -> List[Tuple[str, str, str, float]]:
    """Direkter URL-Modus für `fertig crawl <url>`."""
    text = extract_text(url)
    if len(text) < 100:
        return []
    return parse_web(text)


SOURCES = {
    "wikipedia": fetch_wikipedia,
    "wiktionary": fetch_wiktionary,
    "duckduckgo": fetch_duckduckgo,
    "web": fetch_web,
    "arxiv": fetch_arxiv,
    "pubmed": fetch_pubmed,
    "semantic_scholar": fetch_semantic_scholar,
    "openalex": fetch_openalex,
}

# Konfidenz-Boost pro zusaetzlicher Evidenz-Quelle:
# conf_final = min(CONF_CAP, max_conf + EVIDENCE_BOOST * (n_quellen - 1))
EVIDENCE_BOOST = 0.06
CONF_CAP = 0.95


def fetch_all(target: str, sources: Optional[List[str]] = None,
              verbose: bool = False) -> List[Tuple[str, str, str, float]]:
    """Alle Quellen -> gemergte Tripletts mit Evidenz-Boost."""
    names = sources or list(SOURCES)
    best: Dict[Tuple[str, str, str], Tuple[float, int]] = {}
    for name in names:
        fn = SOURCES.get(name)
        if fn is None:
            continue
        try:
            trips = fn(target)
        except Exception as e:
            if verbose:
                print(f"  [source {name}] Fehler: {e}")
            continue
        if verbose:
            print(f"  [source {name}] {len(trips)} Tripletts")
        for a, b, c, conf in trips:
            key = (a, b, c)
            cur = best.get(key)
            if cur is None:
                best[key] = (conf, 1)
            else:
                best[key] = (max(cur[0], conf), cur[1] + 1)
    out = []
    for (a, b, c), (conf, n) in best.items():
        boosted = min(CONF_CAP, conf + EVIDENCE_BOOST * (n - 1))
        out.append((a, b, c, boosted))
    return out
