"""
fertig.stream — permanentes Lernen aus Video-Streams (O(1)-State).

Die o1-state-Philosophie auf Video: Der Stream ist ein Iterator, kein
Objekt. Der StreamLearner verdichtet jeden Frame in gleitende
Statistiken — konstantes Memory bei unendlichem Input:

  Prototyp      : EMA der Frame-Signaturen (was ist typisch?)
  Bewegung      : EMA der Frame-Distanzen (wie stark ändert es sich?)
  Übergänge     : begrenzte Code-Grammatik (Top-K pro Code, verdrängt)
  Periodizität  : Noether-Detektor über dem gleitenden Bewegungs-Profil
  Szenenwechsel : Distanz-Sprünge über der Rausch-Schwelle

Quellen: ffmpeg (Datei/URL) + yt-dlp (YouTube/Live-Streams). Die
schnelle numpy-Signatur vermeidet den PIL/PNG-Umweg pro Frame.
"""

from __future__ import annotations

import subprocess
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import primitives
from .vision import signature_distance
from . import video as video_mod
from .video import SIGNATURE_NOISE

# ---------------------------------------------------------------------------
# Schnelle numpy-Signatur (Stream-Pfad, kein PIL)
# ---------------------------------------------------------------------------

def fast_signature(frame: np.ndarray, bins: int = 8) -> np.ndarray:
    """Signatur direkt aus dem numpy-Frame (h, w, 3) in [0,1].
    HSV-Histogramme + Gradient-Statistik — deterministisch, kein PNG."""
    f = frame.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mx, mn = np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b)
    d = mx - mn
    h = np.zeros_like(mx)
    ok = d > 1e-6
    h[ok] = np.where(mx[ok] == r[ok],
                     ((g[ok] - b[ok]) / d[ok]) % 6,
                     np.where(mx[ok] == g[ok],
                              (b[ok] - r[ok]) / d[ok] + 2,
                              (r[ok] - g[ok]) / d[ok] + 4))
    h = (h / 6.0).clip(0, 1)
    s = np.where(d > 1e-6, d / np.maximum(mx, 1e-6), 0)
    v = mx
    hist = np.concatenate([
        np.histogram(h, bins=bins, range=(0, 1))[0],
        np.histogram(s, bins=bins // 2, range=(0, 1))[0],
        np.histogram(v, bins=bins // 2, range=(0, 1))[0],
    ]).astype(np.float64)
    gy, gx = np.gradient(np.mean(f, axis=2))
    mag = np.sqrt(gx ** 2 + gy ** 2)
    tex = np.histogram(mag, bins=bins, range=(0, 1))[0].astype(np.float64)
    sig = np.concatenate([hist, tex])
    total = sig.sum()
    return sig / total if total > 0 else sig


# ---------------------------------------------------------------------------
# StreamLearner: O(1)-Verdichtung
# ---------------------------------------------------------------------------

class StreamLearner:
    """Lernt permanent aus einem Frame-Stream mit konstantem Memory."""

    def __init__(self, n_bins: int = 16, ema: float = 0.05,
                 code_budget: int = 8):
        self.n_bins = n_bins
        self.ema = ema
        self.code_budget = code_budget
        self.frames_seen = 0
        self.prototype: Optional[np.ndarray] = None   # EMA der Signaturen
        self.motion_ema = 0.0                          # EMA der Sig-Distanz
        self.pixel_ema = 0.0                           # EMA der Pixel-Diff
        self.prev_sig: Optional[np.ndarray] = None
        self.prev_frame: Optional[np.ndarray] = None
        self.transitions: Dict[int, Dict[int, int]] = {}
        self.cut_count = 0
        self.motion_history: List[float] = []          # begrenztes Fenster
        self.window = 32

    def update(self, frame: np.ndarray) -> Dict:
        """Ein Frame verdichten. Rückgabe: aktueller Verständnis-Zustand."""
        sig = fast_signature(frame, bins=self.n_bins // 2)
        if self.prototype is None:
            self.prototype = sig.copy()
        else:
            self.prototype = (1 - self.ema) * self.prototype + self.ema * sig
        if self.prev_sig is not None:
            d = signature_distance(sig, self.prev_sig)
            self.motion_ema = (1 - self.ema) * self.motion_ema + self.ema * d
            # Pixel-Differenz: fängt Translation (Histogramme sind
            # translations-invariant — die Lektion aus video.py)
            pd = float(np.abs(frame - self.prev_frame).mean())
            self.pixel_ema = (1 - self.ema) * self.pixel_ema + self.ema * pd
            motion = max(d, pd)
            self.motion_history.append(motion)
            if len(self.motion_history) > self.window:
                self.motion_history.pop(0)
            if motion > 3.0 * SIGNATURE_NOISE and len(self.motion_history) > 1:
                med = float(np.median(self.motion_history[:-1]))
                if motion > max(3.0 * med, 3.0 * SIGNATURE_NOISE):
                    self.cut_count += 1
            code = min(int(motion * 40), self.n_bins - 1)
            row = self.transitions.setdefault(code, {})
            row[self.prev_code] = row.get(self.prev_code, 0) + 1
            if len(row) > self.code_budget:
                drop = min(row, key=row.get)
                del row[drop]
        else:
            code = 0
        self.prev_sig = sig
        self.prev_frame = frame
        self.prev_code = code
        self.frames_seen += 1
        return self.state()

    def state(self) -> Dict:
        """Aktueller Verständnis-Zustand (konstante Größe)."""
        period = primitives.detect_periodicity(
            self.motion_history, tol=0.15) if len(self.motion_history) > 8 \
            else None
        return {
            "frames": self.frames_seen,
            "bewegung": self.motion_ema,
            "pixel": self.pixel_ema,
            "periodisch": period,
            "szenenwechsel": self.cut_count,
            "grammatik_kanten": sum(len(r) for r in self.transitions.values()),
            "memory_konstant": True,
        }

    def generate(self, n: int = 12) -> List[int]:
        """Kurze Fortsetzung aus der gelernten Code-Grammatik."""
        if not self.transitions:
            return []
        out = [max(self.transitions, key=lambda k: sum(
            self.transitions[k].values()))]
        for _ in range(n - 1):
            nbrs = self.transitions.get(out[-1])
            if not nbrs:
                break
            out.append(max(nbrs, key=nbrs.get))
        return out

    def sequence_signature(self) -> np.ndarray:
        """Gelernte Sequenz-Signatur (Prototyp + Bewegungs-Statistik) —
        die Brücke zur VideoBank-Kategorie."""
        proto = self.prototype if self.prototype is not None \
            else np.zeros(16)
        stats = np.array([self.motion_ema, self.pixel_ema,
                          float(np.median(self.motion_history))
                          if self.motion_history else 0.0])
        return np.concatenate([proto, stats])

    def to_graph_facts(self, name: str, store: bool = True) -> List[tuple]:
        """Gelernte Stream-Struktur -> Welt-Graph-Fakten.
        (name, hat_bewegung, wert), (name, ist_periodisch, P),
        (name, szenenwechsel, n) — der Stream wird Wissen."""
        from .gaps import _load_world, _save_world
        st = self.state()
        facts = []
        if st["bewegung"] > 3.0 * SIGNATURE_NOISE or st["pixel"] > 0.01:
            facts.append((name, "hat_bewegung",
                          f"{max(st['bewegung'], st['pixel']):.4f}", 0.5))
        if st["periodisch"]:
            facts.append((name, "ist_periodisch",
                          str(st["periodisch"]), 0.6))
        if st["szenenwechsel"]:
            facts.append((name, "hat_szenenwechsel",
                          str(st["szenenwechsel"]), 0.5))
        if facts and store:
            trips = _load_world()
            best = {t[:3]: t[3] for t in trips}
            for a, b, c, conf in facts:
                best[(a, b, c)] = max(best.get((a, b, c), 0.0), conf)
            _save_world([(a, b, c, conf)
                         for (a, b, c), conf in best.items()])
        return facts


# ---------------------------------------------------------------------------
# Quellen: ffmpeg + yt-dlp
# ---------------------------------------------------------------------------

def ytdlp_url(url: str) -> str:
    """YouTube/Live-Stream -> direkte Medien-URL (yt-dlp)."""
    import yt_dlp
    opts = {"format": "best", "quiet": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return info["url"]


def ffmpeg_frames(source: str, fps: int = 2, scale: int = 64,
                  max_frames: Optional[int] = None):
    """Frame-Generator: ffmpeg dekodiert, liefert numpy-Frames (h, w, 3).
    source = Dateipfad oder (Stream-)URL. Der Generator IST der Iterator —
    nichts wird gespeichert."""
    cmd = ["ffmpeg", "-loglevel", "error", "-i", source,
           "-vf", f"fps={fps},scale={scale}:{scale}",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    frame_size = scale * scale * 3
    try:
        while max_frames is None or max_frames > 0:
            data = proc.stdout.read(frame_size)
            if len(data) < frame_size:
                break
            yield np.frombuffer(data, dtype=np.uint8).reshape(
                scale, scale, 3).astype(np.float32) / 255.0
            if max_frames is not None:
                max_frames -= 1
    finally:
        proc.kill()


def learn_from(source: str, seconds: int = 15, fps: int = 2,
               learner: Optional["StreamLearner"] = None,
               verbose: bool = True) -> StreamLearner:
    """Permanent lernen: ffmpeg-Frames -> StreamLearner (O(1))."""
    learner = learner or StreamLearner()
    n_frames = seconds * fps
    for frame in ffmpeg_frames(source, fps=fps, max_frames=n_frames):
        st = learner.update(frame)
        if verbose and learner.frames_seen % fps == 0:
            print(f"  frames={st['frames']} bewegung={st['bewegung']:.4f} "
                  f"periodisch={st['periodisch']} cuts={st['szenenwechsel']}")
    return learner
