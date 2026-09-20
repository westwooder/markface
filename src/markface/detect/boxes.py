"""Box primitives and geometry helpers shared by every detector."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Box:
    """Axis-aligned box in pixel coordinates, plus provenance."""
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    source: str = "face"     # face | yunet | pose | track | fill

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) * 0.5

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) * 0.5

    @property
    def area(self) -> float:
        return self.w * self.h

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def clipped(self, w: int, h: int) -> "Box":
        return Box(
            max(0.0, min(self.x1, w - 1.0)),
            max(0.0, min(self.y1, h - 1.0)),
            max(0.0, min(self.x2, w - 1.0)),
            max(0.0, min(self.y2, h - 1.0)),
            self.score,
            self.source,
        )

    def scaled(self, sx: float, sy: float, dy: float = 0.0) -> "Box":
        """Grow/shrink about the centre; dy shifts down by a fraction of height."""
        cx, cy = self.cx, self.cy + self.h * dy
        hw, hh = self.w * sx * 0.5, self.h * sy * 0.5
        return Box(cx - hw, cy - hh, cx + hw, cy + hh, self.score, self.source)


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    return inter / (a.area + b.area - inter + 1e-9)


def contain_ratio(a: Box, b: Box) -> float:
    """Fraction of `a` that lies inside `b` - catches nested duplicates that IoU misses."""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / (a.area + 1e-9)


def nms(boxes: list[Box], thr: float = 0.45) -> list[Box]:
    """Greedy NMS, highest score first."""
    if not boxes:
        return []
    order = sorted(boxes, key=lambda b: b.score, reverse=True)
    keep: list[Box] = []
    for cand in order:
        if all(iou(cand, k) <= thr and contain_ratio(cand, k) <= 0.80 for k in keep):
            keep.append(cand)
    return keep


def same_face(a: Box, b: Box) -> bool:
    """True if two boxes are two detectors' opinions of one face.

    Different architectures frame the same face differently - YOLO tends to run
    tall and narrow, YuNet short and wide - so their IoU can sit near 0.4 while
    both clearly describe one head. Plain NMS keeps both, and each one becomes
    its own mosaic patch, so a single face ends up wearing two ellipses.

    Judging by centre proximity relative to box size is far more robust here
    than overlap area: two readings of one face share a centre, whatever their
    aspect ratios.
    """
    if a.area <= 0 or b.area <= 0:
        return False
    # Comparable size? A face and a background object rarely match closely.
    ratio = min(a.area, b.area) / max(a.area, b.area)
    if ratio < 0.25:
        # Unless the smaller sits almost entirely inside the larger.
        return contain_ratio(a, b) > 0.75 or contain_ratio(b, a) > 0.75
    ref = (min(a.w, b.w) + min(a.h, b.h)) * 0.5
    if ref <= 0:
        return False
    dist = float(np.hypot(a.cx - b.cx, a.cy - b.cy))
    # Near-identical box sizes are strong evidence of one face seen twice, so
    # allow the centres to sit further apart before splitting them.
    limit = 0.55 + (0.25 if ratio > 0.7 else 0.0)
    if dist < ref * limit:
        return True
    return iou(a, b) > 0.30


def merge_faces(boxes: list[Box], prefer: tuple[str, ...] = ("face", "yunet")) -> list[Box]:
    """Collapse boxes that describe one face into a single box.

    The survivor is the union of the cluster, so the patch covers everything any
    detector flagged - keeping the "rather over-cover than miss" bias while
    emitting one patch per face.
    """
    if len(boxes) < 2:
        return list(boxes)
    order = sorted(boxes, key=lambda b: (b.score, b.area), reverse=True)
    clusters: list[list[Box]] = []
    for b in order:
        for cl in clusters:
            if any(same_face(b, c) for c in cl):
                cl.append(b)
                break
        else:
            clusters.append([b])

    out: list[Box] = []
    for cl in clusters:
        if len(cl) == 1:
            out.append(cl[0])
            continue
        x1 = min(c.x1 for c in cl)
        y1 = min(c.y1 for c in cl)
        x2 = max(c.x2 for c in cl)
        y2 = max(c.y2 for c in cl)
        best = max(cl, key=lambda c: c.score)
        # Name the merged box after a real face detector when one contributed,
        # so downstream occlusion logic still treats it as a face.
        src = next((p for p in prefer if any(c.source == p for c in cl)), best.source)
        out.append(Box(x1, y1, x2, y2, best.score, src))
    return out


def merge_union(groups: list[list[Box]], thr: float = 0.55) -> list[Box]:
    """Union detections from several detectors, deduplicating overlaps.

    Deliberately biased toward keeping boxes: a region flagged by any detector
    survives, because a missed face is worse than an extra mosaic patch. What it
    must not do is emit two patches for one face - hence merge_faces.
    """
    flat = [b for g in groups for b in g]
    return merge_faces(nms(flat, thr))


def batch_nms_numpy(xyxy: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
    """Vectorised NMS over raw arrays; returns kept indices."""
    if xyxy.size == 0:
        return []
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        ovr = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[ovr <= thr]
    return keep
