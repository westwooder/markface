"""IoU + centroid tracker with coasting, built for "never drop a frame".

Not a re-identification tracker: the job is narrower. We need every frame that
plausibly contains a head to carry a mosaic, so a lost track keeps coasting on
its last known velocity for `max_age` frames instead of disappearing at once.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..detect.boxes import Box, iou
from ..detect.occlusion import repair_from_history


@dataclass
class Track:
    tid: int
    box: Box
    age: int = 0              # frames since last real detection
    hits: int = 1
    vx: float = 0.0
    vy: float = 0.0
    history: list[Box] = field(default_factory=list)
    # Largest box this identity has ever shown, used to resist occlusion shrink.
    peak_w: float = 0.0
    peak_h: float = 0.0
    repair: bool = True

    def predict(self) -> Box:
        """Where the box should be next frame, assuming constant velocity."""
        b = self.box
        # Damp the velocity as a track goes stale; drifting boxes look worse
        # than slightly lagging ones.
        damp = max(0.0, 1.0 - self.age * 0.12)
        dx, dy = self.vx * damp, self.vy * damp
        return Box(b.x1 + dx, b.y1 + dy, b.x2 + dx, b.y2 + dy,
                   b.score * 0.95, "track")

    def update(self, box: Box) -> None:
        prev = self.box
        if box.source in ("face", "yunet"):
            # Track the peak only from real face detections; pose heads and
            # filled boxes are coarser and would inflate the reference.
            self.peak_w = max(self.peak_w, box.w)
            self.peak_h = max(self.peak_h, box.h)
        if self.repair and self.peak_h > 0:
            box = repair_from_history(box, self.peak_w, self.peak_h)
        # Exponential velocity estimate; raw frame-to-frame deltas are too jumpy.
        self.vx = 0.6 * self.vx + 0.4 * (box.cx - prev.cx)
        self.vy = 0.6 * self.vy + 0.4 * (box.cy - prev.cy)
        self.box = box
        self.age = 0
        self.hits += 1
        self.history.append(box)
        if len(self.history) > 64:
            self.history.pop(0)

    def coast(self) -> None:
        self.box = self.predict()
        self.age += 1
        self.history.append(self.box)
        if len(self.history) > 64:
            self.history.pop(0)


class IouTracker:
    """Greedy IoU/centroid association with coasting for missed detections."""

    def __init__(self, iou_thr: float = 0.25, max_age: int = 30,
                 dist_scale: float = 1.2, repair_occlusion: bool = True):
        self.iou_thr = float(iou_thr)
        self.max_age = int(max_age)
        self.dist_scale = float(dist_scale)
        self.repair_occlusion = bool(repair_occlusion)
        self.tracks: list[Track] = []
        self._next_id = 1

    def _cost(self, track: Track, det: Box) -> float:
        """Association score in [0, 1]; higher is a better match."""
        pred = track.predict()
        score = iou(pred, det)
        if score > 0:
            return score
        # Boxes can leapfrog each other entirely between frames on fast motion,
        # so fall back to normalised centre distance.
        ref = max(pred.w, pred.h, det.w, det.h, 1.0) * self.dist_scale
        d = float(np.hypot(pred.cx - det.cx, pred.cy - det.cy))
        if d > ref:
            return 0.0
        size_ratio = min(pred.area, det.area) / (max(pred.area, det.area) + 1e-9)
        if size_ratio < 0.25:
            return 0.0
        return 0.15 * (1.0 - d / ref) * size_ratio

    def update(self, dets: list[Box]) -> list[Box]:
        """Feed one frame of detections; return boxes to mask this frame."""
        pairs: list[tuple[float, int, int]] = []
        for ti, tr in enumerate(self.tracks):
            for di, det in enumerate(dets):
                c = self._cost(tr, det)
                if c > 0:
                    pairs.append((c, ti, di))
        pairs.sort(reverse=True)

        used_t: set[int] = set()
        used_d: set[int] = set()
        for c, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            if c < self.iou_thr and c < 0.15:
                continue
            self.tracks[ti].update(dets[di])
            used_t.add(ti)
            used_d.add(di)

        for ti, tr in enumerate(self.tracks):
            if ti not in used_t:
                tr.coast()

        for di, det in enumerate(dets):
            if di not in used_d:
                self.tracks.append(
                    Track(self._next_id, det, repair=self.repair_occlusion))
                self._next_id += 1

        self.tracks = [t for t in self.tracks if t.age <= self._age_limit(t)]
        self._suppress_duplicates()

        # Emit every live track, coasting ones included - that is the whole point.
        return [t.box for t in self.tracks]

    def _age_limit(self, tr: Track) -> int:
        """How long this track may coast without a detection.

        A track seen only once or twice is probably a detector blip; letting it
        coast the full max_age leaves a stationary phantom box sitting in the
        frame long after the real subject has moved on. Confidence in a track
        grows with its hit count, so scale its licence to coast accordingly.
        """
        if tr.hits >= 8:
            return self.max_age
        if tr.hits <= 2:
            return min(self.max_age, 3)
        return min(self.max_age, 3 + (tr.hits - 2) * 4)

    def _suppress_duplicates(self) -> None:
        """Drop a coasting track that a better-established one already covers.

        Intermittent detectors (the pose model especially) re-appear a few pixels
        off and spawn a parallel track. Keep the one with more hits.
        """
        if len(self.tracks) < 2:
            return
        ranked = sorted(self.tracks, key=lambda t: (t.hits, -t.age), reverse=True)
        keep: list[Track] = []
        for tr in ranked:
            dup = False
            for k in keep:
                if iou(tr.box, k.box) > 0.5:
                    dup = True
                    break
                # Centre of a stale track swallowed by a live one.
                if tr.age > 0 and k.box.x1 <= tr.box.cx <= k.box.x2                         and k.box.y1 <= tr.box.cy <= k.box.y2:
                    dup = True
                    break
            if not dup:
                keep.append(tr)
        if len(keep) != len(self.tracks):
            kept = {id(t) for t in keep}
            self.tracks = [t for t in self.tracks if id(t) in kept]

    def reset(self) -> None:
        self.tracks.clear()
