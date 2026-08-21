"""
fertig.video — Video-Verständnis und -Generierung auf der FERTIG-Architektur.

Kein Transformer, keine Modelle — dieselben Bausteine wie überall:

  Video = Frame-Sequenz von Signaturen (fertig.vision)
  Verstehen  = gemessene Übergänge + Noether-Primitive:
               Bewegungs-Profil (Frame-Distanzen), Periodizität,
               Parität, Szenenwechsel (Distanz-Sprünge)
  Kategorien = VideoBank: Prototyp aus Mean-Frame + Bewegungs-Profil
               (Harnad-L3: Cluster ohne Labels über Sequenz-Signaturen)
  Generieren = gemessene Frame-Übergänge -> Kontraktions-Sampler ->
               neue Sequenz (wie run_symbolic, nur im Bild-Raum)

Eingabe-Format v1: GIF (PIL nativ, kein ffmpeg). Ein GIF IST ein Video —
Frames über die Zeit.
"""

from __future__ import annotations

import io
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import sampler
from .vision import signature, signature_distance
from . import primitives


# ---------------------------------------------------------------------------
# Frames + Sequenz-Signaturen
# ---------------------------------------------------------------------------

def extract_frames(gif_bytes: bytes, max_frames: int = 64
                   ) -> List[np.ndarray]:
    """GIF -> Roh-Frames (h, w, 3) — deterministisch. Die Roh-Frames
    sind nötig für das Pixel-Differenz-Signal (Translation)."""
    from PIL import Image
    img = Image.open(io.BytesIO(gif_bytes))
    frames = []
    try:
        while True:
            frames.append(np.asarray(img.convert("RGB"),
                                     dtype=np.float32) / 255.0)
            if len(frames) >= max_frames:
                break
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    return frames


def frame_signatures(raw_frames: List[np.ndarray]) -> List[np.ndarray]:
    """Roh-Frames -> Signatur-Sequenz (für VideoBank/Generierung)."""
    return [signature(_png_bytes(_to_pil(f))) for f in raw_frames]


def _to_pil(frame: np.ndarray):
    from PIL import Image
    return Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8))


def _png_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def movement_profile(frames: List[np.ndarray]) -> np.ndarray:
    """Bewegungs-Profil: Signatur-Distanz aufeinanderfolgender Frames."""
    if len(frames) < 2:
        return np.array([0.0])
    return np.array([signature_distance(frames[i], frames[i + 1])
                     for i in range(len(frames) - 1)])


# Rausch-Kalibrierung der Bewegungserkennung (GEMESSEN, nicht geraten):
# zwei identische Frames mit ±4/255-Pixelrauschen ergeben
# Signatur-Distanz ~0.0177 und Pixel-Differenz-Mean ~0.0105.
SIGNATURE_NOISE = 0.0177
PIXEL_NOISE = 0.0105
SIG_MOTION_THRESHOLD = 3.0 * SIGNATURE_NOISE
PIX_MOTION_THRESHOLD = 3.0 * PIXEL_NOISE


# ---------------------------------------------------------------------------
# Verständnis: Noether-Primitive auf der Sequenz
# ---------------------------------------------------------------------------

def understand(raw_frames: List[np.ndarray]) -> Dict:
    """Video-Verständnis: gemessene Struktur der Sequenz.

    Zwei Bewegungs-Primitive (rausch-kalibriert, gemessen):
    - signatur: Signatur-Distanz (fängt Radius-Änderung, 'puls')
    - pixel   : mittlere Pixel-Differenz (fängt Translation, 'wander' —
                Histogramme sind translations-invariant!)
    bewegt = (signatur > 3x Rausch) ODER (pixel > 3x Rausch).
    Dazu Noether-Primitive: Periodizität, Parität, Szenenwechsel.
    """
    if len(raw_frames) < 3:
        return {"frames": len(raw_frames), "bewegt": False,
                "periodisch": None, "szenenwechsel": 0, "paritaet": None,
                "signatur": 0.0, "pixel": 0.0}
    sigs = frame_signatures(raw_frames)
    prof = movement_profile(sigs)
    sig_motion = float(prof.mean())
    pix = np.mean([np.abs(raw_frames[i + 1] - raw_frames[i]).mean()
                   for i in range(len(raw_frames) - 1)])
    period = primitives.detect_periodicity(prof.tolist(), tol=0.15)
    med = float(np.median(prof))
    cuts = int((prof > max(3.0 * med, 0.05)).sum())
    parity = primitives.detect_parity(prof.tolist(), tol=0.15)
    return {
        "frames": len(raw_frames),
        "bewegt": (sig_motion > SIG_MOTION_THRESHOLD or
                   pix > PIX_MOTION_THRESHOLD),
        "signatur": sig_motion,
        "pixel": pix,
        "periodisch": period,
        "szenenwechsel": cuts,
        "paritaet": parity,
    }


# ---------------------------------------------------------------------------
# VideoBank: Kategorien über Sequenz-Signaturen (Harnad-L3)
# ---------------------------------------------------------------------------

def sequence_signature(frames: List[np.ndarray]) -> np.ndarray:
    """Sequenz-Signatur: Mean-Frame + Bewegungs-Statistik."""
    if not frames:
        return np.zeros(10)
    mean_frame = np.mean(np.stack(frames), axis=0)
    prof = movement_profile(frames)
    stats = np.array([prof.mean(), prof.std(), prof.max(),
                      float(np.median(prof))])
    return np.concatenate([mean_frame, stats])


class VideoBank:
    def __init__(self, threshold: float = 0.25):
        self.prototypes: Dict[str, np.ndarray] = {}
        self.threshold = threshold   # über der Schwelle: ehrlich 'unbekannt'
        self._tree = None            # cKDTree-Index (lazy rebuild)
        self._names: List[str] = []

    def _rebuild_index(self) -> None:
        """KD-Baum über die Prototypen — O(log n) Erkennung statt O(n).
        L1-Distanz (p=1) entspricht der Signatur-Distanz."""
        if not self.prototypes:
            self._tree, self._names = None, []
            return
        X = np.stack(list(self.prototypes.values()))
        try:
            from scipy.spatial import cKDTree
            self._tree = cKDTree(X)
        except ImportError:  # pragma: no cover
            self._tree = None
        self._names = list(self.prototypes.keys())

    def save(self, path) -> None:
        import json
        from pathlib import Path
        Path(path).write_text(json.dumps(
            {k: v.tolist() for k, v in self.prototypes.items()}))
        self._rebuild_index()

    def load(self, path) -> "VideoBank":
        import json
        from pathlib import Path
        p = Path(path)
        if p.exists():
            data = json.loads(p.read_text())
            for k, arr in data.items():
                self.prototypes[k] = np.array(arr)
        self._rebuild_index()
        return self

    def add(self, name: str, frames: List[np.ndarray]) -> None:
        self.prototypes[name] = sequence_signature(frames)
        self._rebuild_index()

    def add_from_learner(self, name: str, learner) -> None:
        """Kategorie aus einem StreamLearner übernehmen: der gelernte
        Prototyp + Bewegungs-Statistik wird eine VideoBank-Kategorie."""
        self.prototypes[name] = learner.sequence_signature()

    def recognize_signature(self, sig: np.ndarray) -> Tuple[Optional[str],
                                                            float]:
        """Sequenz-Signatur gegen die Bank erkennen — O(log n) über den
        KD-Baum. Über der Schwelle: ehrlich 'unbekannt' statt raten."""
        if self._tree is None:
            self._rebuild_index()
        if self._tree is None:
            return None, float("inf")
        d, idx = self._tree.query(sig, p=1)
        d = float(d)
        if d > self.threshold:
            return None, d
        return self._names[int(idx)], d

    def recognize(self, frames: List[np.ndarray]) -> Tuple[Optional[str],
                                                           float]:
        sig = sequence_signature(frames)
        best, best_d = None, float("inf")
        for name, proto in self.prototypes.items():
            d = signature_distance(sig, proto)
            if d < best_d:
                best, best_d = name, d
        return best, best_d


# ---------------------------------------------------------------------------
# Generierung: gemessene Übergänge + Kontraktions-Sampler
# ---------------------------------------------------------------------------

def frame_code(frames: List[np.ndarray], n_bins: int = 16) -> List[int]:
    """Frames -> diskrete Codes (Quantisierung im Signatur-Raum).

    Die Codes sind die 'Wörter' des Video-Korpus — Übergänge zwischen
    ihnen sind die gemessene Video-Grammatik."""
    if not frames:
        return []
    base = frames[0]
    return [_code_for(f, base, n_bins) for f in frames]


def _code_for(f, base, n_bins):
    d = signature_distance(f, base)
    # Distanz-basierte Codes: 0 = identisch, wächst mit Abweichung
    return min(int(d * 40), n_bins - 1)


def learn_transitions(codes: List[int]) -> Dict[int, Dict[int, int]]:
    """Gemessene Frame-Übergänge (die Video-Grammatik)."""
    trans: Dict[int, Dict[int, int]] = {}
    for a, b in zip(codes, codes[1:]):
        row = trans.setdefault(a, {})
        row[b] = row.get(b, 0) + 1
    return trans


def generate_frames(seed_code: int, transitions: Dict[int, Dict[int, int]],
                    n: int = 20, tau: float = 0.3,
                    n_bins: int = 16, recency: int = 4) -> List[int]:
    """Neue Sequenz durch Kontraktions-Sampling über den gemessenen
    Übergängen — Video-Generierung als Korpus-Fortsetzung. Rezenz-Penalty
    verhindert Attraktor-Schleifen (derselbe Guard wie im Text-Modus)."""
    out = [seed_code]
    for _ in range(n - 1):
        nbrs = transitions.get(out[-1])
        if not nbrs:
            best = max(transitions.items(),
                       key=lambda kv: sum(kv[1].values()))[0]
            out.append(best)
            continue
        logits = np.full(n_bins, -30.0)
        for b, c in nbrs.items():
            logits[b] = np.log(c)
        # Anti-Wiederholung: kürzlich genutzte Codes bestrafen
        for k, code in enumerate(out[-recency:]):
            logits[code] -= 2.0 * (k + 1) / recency
        nxt = sampler.contraction_sample(logits, tau=tau, top_k=8)
        out.append(int(nxt))
    return out
