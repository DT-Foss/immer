"""
fertig.vision — deterministische Bilderkennung (kein neuronales Netz).

Harnad (2005): "To cognize is to categorize" — Wahrnehmung ist die
Fähigkeit, über Instanzen invariante Kategorien zu bilden. Diese Schicht
macht genau das mit deterministischen Signaturen statt Modellen:

  Signatur(Bild) = HSV-Histogramme ⊕ Textur ⊕ Graustufen-Gitter
  Kategorie(wort) = Prototyp = Mittelwert der Signaturen ihrer Bilder
  Erkennen(Bild) = argmin_Distanz(Signatur, Prototypen)

Die kategorielle Wahrnehmung (Harnads Kern-Eigenschaft) ist messbar:
  within-category-Distanz < between-category-Distanz -> harnad_ratio < 1

Bilder kommen aus Wikimedia Commons (Datei-Suche). Kein Training, keine
Gewichte: jede Signatur ist deterministisch, jedes Erkennen ist eine
Distanz-Berechnung.
"""

from __future__ import annotations

import io
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Signaturen (deterministisch, numpy-only)
# ---------------------------------------------------------------------------

_HIST_BINS = (8, 4, 4)      # HSV: H 8, S 4, V 4
_GRID = 8                   # 8x8 Graustufen-Gitter


def _load_image(image_bytes: bytes):
    from PIL import Image
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _signature_parts(img):
    import colorsys
    arr = np.asarray(img, dtype=np.float32) / 255.0
    h, w, _ = arr.shape
    hsv = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            r, g, b = arr[i, j]
            hsv[i, j] = colorsys.rgb_to_hsv(r, g, b)
    h_hist = np.histogram(hsv[..., 0].ravel(), bins=_HIST_BINS[0],
                          range=(0, 1))[0]
    s_hist = np.histogram(hsv[..., 1].ravel(), bins=_HIST_BINS[1],
                          range=(0, 1))[0]
    v_hist = np.histogram(hsv[..., 2].ravel(), bins=_HIST_BINS[2],
                          range=(0, 1))[0]
    # Textur: Gradienten-Magnitude (vereinfachtes LBP-Ersatzsignal)
    gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    tex = np.histogram(mag, bins=16, range=(0, 1))[0]
    # Form-Gitter
    small = np.asarray(img.convert("L").resize((_GRID, _GRID)),
                       dtype=np.float32) / 255.0
    return h_hist, s_hist, v_hist, tex, small.ravel()


def signature(image_bytes: bytes) -> np.ndarray:
    """Deterministische Signatur: Histogramme + Textur + Form-Gitter."""
    img = _load_image(image_bytes)
    parts = _signature_parts(img)
    sig = np.concatenate([p.astype(np.float64) for p in parts])
    total = sig.sum()
    if total > 0:
        sig = sig / total
    return sig


def signature_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L1-Distanz der normalisierten Signaturen (deterministisch)."""
    return float(np.abs(a - b).sum())


# ---------------------------------------------------------------------------
# Kategorien: Prototypen aus Bild-Mengen (Harnad: Kategorisierung)
# ---------------------------------------------------------------------------

def prototype(signatures: List[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(signatures), axis=0)


def within_distance(signatures: List[np.ndarray]) -> float:
    """Mittlere paarweise Distanz innerhalb der Kategorie."""
    n = len(signatures)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total += signature_distance(signatures[i], signatures[j])
    return total / (n * (n - 1) / 2)


def between_distance(proto_a: np.ndarray, proto_b: np.ndarray) -> float:
    return signature_distance(proto_a, proto_b)


# ---------------------------------------------------------------------------
# Bildbeschaffung (Wikimedia Commons)
# ---------------------------------------------------------------------------

def _http(url: str) -> bytes:
    import time
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent":
                                               "fertig/1.0 (research)"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except Exception as e:
            if "429" in str(e) or "503" in str(e):
                time.sleep(2.0 * (attempt + 1))  # Rate-Limit respektieren
                continue
            raise
    raise RuntimeError("Rate-Limit trotz Retries")


_CACHE_DIR = None


def _cache_dir() -> "Path":
    global _CACHE_DIR
    if _CACHE_DIR is None:
        from pathlib import Path
        _CACHE_DIR = (Path(__file__).resolve().parent.parent /
                      "data" / "images")
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _cached_image(name: str, url: str) -> Optional[bytes]:
    """Bild-Cache: einmal geladen, nie wieder (Rate-Limit + Geschwindigkeit)."""
    import hashlib
    h = hashlib.sha256(url.encode()).hexdigest()[:16]
    path = _cache_dir() / f"{name}_{h}.img"
    if path.exists():
        return path.read_bytes()
    try:
        data = _http(url)
    except Exception:
        return None
    if len(data) > 1000:
        path.write_bytes(data)
        return data
    return None


def commons_images(word: str, n: int = 5) -> List[bytes]:
    """n Bilder für ein Wort aus dem Commons-Datei-Namespace (gecacht)."""
    import json
    import time
    import urllib.parse
    url = ("https://commons.wikimedia.org/w/api.php?action=query"
           "&list=search&srsearch=" + urllib.parse.quote(word) +
           "&srnamespace=6&srlimit=" + str(n) + "&format=json")
    try:
        d = json.loads(_http(url).decode("utf-8"))
    except Exception:
        return []
    out = []
    for s in d.get("query", {}).get("search", []):
        fpath = ("https://commons.wikimedia.org/wiki/Special:FilePath/" +
                 urllib.parse.quote(s["title"].replace("File:", "")))
        data = _cached_image(word.replace(" ", "_"), fpath)
        if data:
            out.append(data)
        time.sleep(0.5)  # Rate-Limit-Höflichkeit
    return out


def cluster_unsupervised(images: List[bytes], k: int = 4, seed: int = 0
                         ) -> Tuple[List[List[int]], List[np.ndarray]]:
    """Harnad-Ebene: Kategorien OHNE Wörter. Signaturen -> k-means.
    Die Cluster entstehen aus der Pixel-Struktur selbst — kein mensch-
    liches Label, kein trainiertes Netz, nur Distanzen im Signatur-Raum.
    Rückgabe: (Cluster-Mitglieds-Indizes, Cluster-Zentren)."""
    sigs = []
    for img in images:
        try:
            sigs.append(signature(img))
        except Exception:
            continue
    if len(sigs) < k:
        return [], []
    X = np.stack(sigs)
    rng = np.random.RandomState(seed)
    # k-means++-ähnliche Initialisierung (deterministisch via Seed)
    centers = [X[rng.randint(len(X))]]
    for _ in range(k - 1):
        d = np.array([min(np.abs(x - c).sum() for c in centers)
                      for x in X])
        d = d / d.sum()
        centers.append(X[rng.choice(len(X), p=d)])
    centers = np.stack(centers)
    assign = np.zeros(len(X), dtype=int)
    for _ in range(12):
        dists = np.array([[np.abs(x - c).sum() for c in centers]
                          for x in X])
        new_assign = np.argmin(dists, axis=1)
        if np.array_equal(new_assign, assign):
            break
        assign = new_assign
        for j in range(k):
            members = X[assign == j]
            if len(members):
                centers[j] = members.mean(axis=0)
    clusters = [[] for _ in range(k)]
    for i, c in enumerate(assign):
        clusters[int(c)].append(i)
    return clusters, centers


def cluster_purity(clusters: List[List[int]], true_labels: List[str]
                   ) -> float:
    """Wie gut decken sich die unüberwachten Cluster mit den wahren
    Klassen? (Nur zur Validierung — das Clustering selbst sah nie Labels.)"""
    total = 0
    for cl in clusters:
        if not cl:
            continue
        from collections import Counter
        majority = Counter(true_labels[i] for i in cl).most_common(1)[0][1]
        total += majority
    return total / max(sum(len(c) for c in clusters), 1)


def _recover_signatures(word: str, n_images: int) -> List[np.ndarray]:
    sigs = []
    for img in commons_images(word, n_images):
        try:
            sigs.append(signature(img))
        except Exception:
            continue
    return sigs


class CategoryBank:
    """Wort -> Prototyp-Signatur. Erkennen = nächster Prototyp."""

    def __init__(self):
        self.prototypes: Dict[str, np.ndarray] = {}
        self.consistency: Dict[str, float] = {}   # 1/within — Kategorien-Güte

    def add(self, word: str, signatures: List[np.ndarray]) -> None:
        if not signatures:
            return
        self.prototypes[word] = prototype(signatures)
        w = within_distance(signatures)
        self.consistency[word] = 1.0 / (w + 1e-9)

    def add_from_word(self, word: str, n_images: int = 4) -> bool:
        """Bilder für das Wort holen und als Kategorie registrieren."""
        sigs = _recover_signatures(word, n_images)
        if len(sigs) < 2:
            return False
        self.add(word, sigs)
        return True

    def recognize(self, image_bytes: bytes) -> Tuple[Optional[str], float]:
        """Bild -> ähnlichste Kategorie (Bilderkennung ohne Netz)."""
        sig = signature(image_bytes)
        best, best_d = None, float("inf")
        for word, proto in self.prototypes.items():
            d = signature_distance(sig, proto)
            if d < best_d:
                best, best_d = word, d
        return best, best_d

    def harnad_ratio(self) -> Optional[float]:
        """Kategorielle Wahrnehmung: within < between -> Ratio < 1.
        Je kleiner, desto stärker komprimiert die Kategorie ihre
        Mitglieder gegenüber der Trennung von Fremden."""
        words = list(self.prototypes)
        if len(words) < 2:
            return None
        betweens = []
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                betweens.append(between_distance(self.prototypes[words[i]],
                                                 self.prototypes[words[j]]))
        within = [1.0 / self.consistency[w] for w in words]
        mean_between = float(np.mean(betweens))
        mean_within = float(np.mean(within))
        if mean_within <= 0:
            return None
        return mean_within / mean_between


def build_bank(words: List[str], n_images: int = 4) -> CategoryBank:
    """Kategorien-Bank aus Wörtern bauen (die Erkennungs-Welt)."""
    bank = CategoryBank()
    for word in words:
        ok = bank.add_from_word(word, n_images)
        if not ok:
            print(f"[vision] {word}: keine Bilder — Kategorie übersprungen")
    return bank
