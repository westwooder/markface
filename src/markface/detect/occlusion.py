"""Compensation for detectors that shrink their box when a face is occluded.

Observed behaviour: with eyes and mouth covered, a YOLO face box collapses to
roughly the visible strip (often just the forehead) - measured at ~1/3 of the
unoccluded height on test footage. Masking that box alone leaves the covered
features exposed, which is exactly the case the user cares about.

Two corrections, both erring toward a larger patch:
  1. Aspect repair. A frontal face box is normally ~1.25-1.45x taller than wide.
     A much flatter box means the detector only saw a slice, so restore the
     expected height, growing downward since the forehead is what stays visible.
  2. Track-history repair. Within one track we remember the largest box seen,
     and never mask much less than that. A face does not shrink because a
     blindfold went on.
"""
from __future__ import annotations

from .boxes import Box

# Height/width of a typical unoccluded frontal face box.
NOMINAL_ASPECT = 1.32
# Below this, treat the box as a partial observation.
FLAT_ASPECT = 1.05


def repair_aspect(b: Box, strength: float = 1.0) -> Box:
    """Restore a plausible face height when the box looks like a visible slice."""
    if b.w <= 1 or b.h <= 1:
        return b
    aspect = b.h / b.w
    if aspect >= FLAT_ASPECT:
        return b
    target_h = b.w * NOMINAL_ASPECT
    # Partial credit: a slightly flat box gets a slight correction.
    grow = (target_h - b.h) * min(1.0, max(0.0, strength))
    if grow <= 0:
        return b
    # Visible part is usually the upper face, so extend mostly downward.
    return Box(b.x1, b.y1 - grow * 0.18, b.x2, b.y2 + grow * 0.82,
               b.score, b.source)


def repair_from_history(b: Box, peak_w: float, peak_h: float,
                        floor: float = 0.72) -> Box:
    """Keep a box from collapsing far below the largest size this track has had."""
    if peak_w <= 1 or peak_h <= 1:
        return b
    min_w, min_h = peak_w * floor, peak_h * floor
    if b.w >= min_w and b.h >= min_h:
        return b
    new_w = max(b.w, min_w)
    new_h = max(b.h, min_h)
    cx = b.cx
    # Anchor on the top edge: the forehead is the part that stayed visible, so
    # the missing area is below it.
    top = b.y1 if b.h < min_h else b.cy - new_h * 0.5
    return Box(cx - new_w * 0.5, top, cx + new_w * 0.5, top + new_h,
               b.score, b.source)
