"""
fertig.interp — Interpolations-Lernen mit Stützrädern (Curriculum).

Fahrradfahren: Lücke k=1 (benachbarte Frames) ist fast gegeben; die Lücke
wächst, bis das Modell echte Dynamik beherrscht; k=inf wäre freies
Träumen. Die beherrschte Lücke IST die Fortschritts-Metrik.

Kein Warp, kein optischer Flow — Interpolation im CODE-Raum:
  Frames -> Codes (Quantisierung im Signatur-Raum, video.py)
  Grammatik = gemessene Übergänge (was folgt auf was?)
  Code-Prototyp = Mittel der gesehenen Frames pro Code (O(bins) Memory)
  Interpolieren(A, B, k) = Pfad durch die gemessene Grammatik von
  code(A) zu code(B), Frames aus den Code-Prototypen — Artefakt-frei
  per Konstruktion (kein Warping möglich, nur Code-Unschärfe).

Self-paced Curriculum (F1-Prinzip): die Surprise (Interpolations-Fehler
gegen die echte Mitte) steuert die Lücke selbst — Fehler klein -> Lücke
wächst, Fehler groß -> Lücke schrumpft. Der Organism stellt seine
eigenen Stützräder ein.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from . import video as video_mod
from .vision import signature_distance


class CodeFrameBank:
    """Code -> Frame-Prototyp (Mittel der gesehenen Frames pro Code).
    O(n_bins) Memory — konstant, unabhängig von der Stream-Länge."""

    def __init__(self, n_bins: int = 16):
        self.n_bins = n_bins
        self.prototypes: Dict[int, np.ndarray] = {}
        self.counts: Dict[int, int] = {}

    def update(self, code: int, frame: np.ndarray) -> None:
        if code not in self.prototypes:
            self.prototypes[code] = frame.astype(np.float64).copy()
            self.counts[code] = 1
        else:
            n = self.counts[code]
            self.prototypes[code] = (
                (self.prototypes[code] * n + frame) / (n + 1))
            self.counts[code] = n + 1

    def frame_for(self, code: int) -> Optional[np.ndarray]:
        p = self.prototypes.get(code)
        return p if p is None else np.clip(p, 0, 1)


class InterpLearner:
    """Lernt Übergänge + Code-Prototypen; interpoliert über die
    gemessene Grammatik; misst die beherrschte Lücke."""

    def __init__(self, n_bins: int = 16, quality_threshold: float = 0.08):
        self.n_bins = n_bins
        self.quality_threshold = quality_threshold
        self.transitions: Dict[int, Dict[int, int]] = {}
        self.bank = CodeFrameBank(n_bins)
        self.prev_code: Optional[int] = None
        self.prev_frame: Optional[np.ndarray] = None
        self.frames_seen = 0

    # -- Lernen ----------------------------------------------------------

    def update(self, frame: np.ndarray) -> int:
        """Ein Frame: Code bestimmen, Übergang + Prototyp lernen.
        Der Code trägt ein RICHTUNGS-BIT (wächst/schrumpft die
        Differenz?) — ohne es wäre der Code phasenblind und der
        Prototyp mittelte Anstieg und Abfall zu einer Unschärfe."""
        if self.prev_frame is None:
            code = 0
            self.prev_d = 0.0
        else:
            d = float(np.abs(frame - self.prev_frame).mean())
            wachsend = d > self.prev_d
            bucket = min(int(d * 40), self.n_bins // 2 - 1)
            code = bucket * 2 + (1 if wachsend else 0)
            row = self.transitions.setdefault(self.prev_code, {})
            row[code] = row.get(code, 0) + 1
            self.prev_d = d
        self.bank.update(code, frame)
        self.prev_code = code
        self.prev_frame = frame.copy()
        self.frames_seen += 1
        return code

    # -- Interpolation ----------------------------------------------------

    def _path(self, start: int, goal: int, max_steps: int = 8
              ) -> List[int]:
        """Pfad von start zu goal durch die gemessene Grammatik.
        start == goal: der gemessene ZYKLUS (die Dynamik zwischen zwei
        gleichen Ankern — genau das muss Interpolation wissen)."""
        if start == goal:
            cyc = [start]
            cur = start
            for _ in range(max_steps):
                nbrs = self.transitions.get(cur)
                if not nbrs:
                    break
                nxt = max(nbrs, key=nbrs.get)
                if nxt == start:
                    break
                cyc.append(nxt)
                cur = nxt
            return cyc
        from collections import deque
        q = deque([(start, [start])])
        seen = {start}
        while q:
            c, path = q.popleft()
            if len(path) > max_steps:
                break
            for nxt in sorted(self.transitions.get(c, {}),
                              key=lambda k: -self.transitions[c][k]):
                if nxt == goal:
                    return path + [nxt]
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, path + [nxt]))
        path = [start]
        cur = start
        for _ in range(max_steps):
            nbrs = self.transitions.get(cur)
            if not nbrs:
                break
            nxt = min(nbrs, key=lambda k: (abs(k - goal), -nbrs[k]))
            path.append(nxt)
            cur = nxt
            if cur == goal:
                break
        return path

    def interpolate(self, start_frame: np.ndarray, end_frame: np.ndarray,
                    gap: int) -> Optional[List[np.ndarray]]:
        """Zwischenframes für eine Lücke von `gap` erzeugen.
        Anker: Code des Starts, Code des Ziels (aus der Frame-Differenz);
        Pfad durch die gemessene Grammatik, Frames aus Code-Prototypen."""
        if gap <= 1:
            return [start_frame]
        d0 = float(np.abs(start_frame - end_frame).mean())
        c_start = self.prev_code or 0
        bucket = min(int(d0 * 40), self.n_bins // 2 - 1)
        c_goal = bucket * 2 + (1 if d0 > (self.prev_d or 0.0) else 0)
        path = self._path(c_start, c_goal, max_steps=gap * 2)
        if len(path) < 2:
            return None
        out = []
        for step in range(1, gap):
            idx = min(int(step * (len(path) - 1) / gap), len(path) - 1)
            fr = self.bank.frame_for(path[idx])
            if fr is None:
                fr = start_frame
            out.append(fr)
        return out

    # -- Qualität + Stützräder --------------------------------------------

    def quality(self, frames: List[np.ndarray], gap: int) -> float:
        """Interpolations-Qualität auf einem echten Segment: mittlere
        Distanz zwischen generierten und echten Zwischenframes."""
        if len(frames) < gap + 1 or gap <= 1:
            return 0.0
        gen = self.interpolate(frames[0], frames[gap], gap)
        if not gen:
            return 1.0
        real = frames[1:gap]
        return float(np.mean([
            np.abs(g - r).mean() for g, r in zip(gen, real)]))

    def mastered_gap(self, frames: List[np.ndarray], max_gap: int = 8
                     ) -> int:
        """Die GRÖSSTE Lücke, deren Interpolation unter der
        Qualitäts-Schwelle bleibt — die Stützräder-Metrik.
        (Die Qualitätskurve ist nicht monoton — Code-Quantisierung —
        daher Maximum statt Abbruch bei der ersten Überschreitung.)"""
        best = 0
        for gap in range(2, max_gap + 1):
            if self.quality(frames, gap) < self.quality_threshold:
                best = gap
        return best

    def self_paced_learn(self, frames: List[np.ndarray], max_gap: int = 8,
                         grow: int = 1, shrink: int = 1,
                         verbose: bool = False) -> List[Tuple[int, int]]:
        """F1-Prinzip: die eigene Surprise stellt die Stützräder ein.
        Rückgabe: [(gap, frames_seen), ...] — die Lern-Kurve."""
        gap = 2
        curve = []
        for i in range(len(frames)):
            self.update(frames[i])
            if i < gap:  # noch kein Segment verfügbar
                continue
            seg = frames[max(0, i - gap - 2):i + 1]
            if len(seg) < gap + 1:
                continue
            err = self.quality(seg, gap)
            if err < self.quality_threshold:
                gap = min(gap + grow, max_gap)
            else:
                gap = max(gap - shrink, 2)
            curve.append((gap, self.frames_seen))
            if verbose:
                print(f"  frames={self.frames_seen} gap={gap} "
                      f"err={err:.4f}")
        return curve
