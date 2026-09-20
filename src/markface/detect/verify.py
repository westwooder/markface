"""Cheap rejection of detections that cannot be faces.

The ensemble is deliberately trigger-happy, which is right for occlusion but
does let through the occasional patch of bare skin - a hand, a shoulder, a
forearm - reported at low confidence by one detector. Those show up as extra
mosaic ellipses on the body.

The discriminator used here is internal structure. A face, even blindfolded or
gagged, carries edges: the covering itself has a rim, and the head has a
silhouette against the background. A flat area of skin has almost none. So a
low-confidence, low-detail box is dropped, while anything a detector is
confident about is left alone regardless.
"""
from __future__ import annotations

import cv2
import numpy as np

from .boxes import Box

# Below this Laplacian variance the patch has essentially no internal structure.
FLAT_DETAIL = 26.0
# Detections at or above this score are trusted without the structure test.
TRUST_SCORE = 0.62


def _detail(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_plausible_face(frame: np.ndarray, box: Box,
                      flat_detail: float = FLAT_DETAIL,
                      trust_score: float = TRUST_SCORE) -> bool:
    """False only for boxes that are both weakly scored and visually featureless."""
    if box.score >= trust_score:
        return True
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
    x2, y2 = min(w, int(box.x2)), min(h, int(box.y2))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return True          # too small to judge; keep it
    patch = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    if _detail(gray) >= flat_detail:
        return True
    # One more chance: a strong vertical edge pair (head silhouette) also counts.
    edges = cv2.Canny(gray, 50, 150)
    return float(edges.mean()) > 6.0


def filter_implausible(frame: np.ndarray, boxes: list[Box],
                       enabled: bool = True) -> list[Box]:
    """Drop featureless low-confidence boxes; keep everything else."""
    if not enabled or not boxes:
        return boxes
    return [b for b in boxes if is_plausible_face(frame, b)]
