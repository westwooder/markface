"""Mosaic rendering.

Goals from the brief: keep the skull/hair outline readable, pixelate the
features, use smallish blocks, and avoid the pure-blur look that reads as
uncanny. So: an elliptical patch that is narrower vertically than the detected
box, filled with block pixelation, only lightly blurred, with a feathered edge
so it sits on the face instead of being stamped on top of it.
"""
from __future__ import annotations

import cv2
import numpy as np

from .detect.boxes import Box
from .settings import Settings


def _block_size(w: int, h: int, st: Settings) -> int:
    """Block edge in pixels.

    `strength` multiplies the block size: that is the only thing that actually
    controls how much detail survives. Re-running the downsample does not - once
    a region is flat, averaging it again returns the same values.
    """
    base = max(w, h) * st.block_ratio * max(0.25, st.strength)
    return int(np.clip(round(base), st.block_min, st.block_max))


def _pixelate(patch: np.ndarray, block: int, flatten: bool = True) -> np.ndarray:
    """Block-average the patch into flat square cells.

    `flatten` first blurs at roughly the block scale. Without it, each cell's
    average is pulled around by whatever high-contrast edge happens to fall
    inside it (an eyelash, the rim of a blindfold), leaving a faint but legible
    trace of the features in the block pattern. Pre-blurring removes that trace
    while keeping the cells crisp, because the blur happens before the
    downsample, not after.
    """
    h, w = patch.shape[:2]
    if h < 2 or w < 2:
        return patch
    src = patch
    if flatten and block >= 3:
        k = max(3, (block | 1))
        src = cv2.GaussianBlur(patch, (k, k), 0)
    bw = max(1, w // max(1, block))
    bh = max(1, h // max(1, block))
    # INTER_AREA down then NEAREST up gives clean, flat blocks.
    small = cv2.resize(src, (bw, bh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _mask_for(shape: str, w: int, h: int, feather: int) -> np.ndarray:
    """Single-channel float mask in [0, 1] with a soft edge."""
    m = np.zeros((h, w), np.uint8)
    if shape == "rect":
        cv2.rectangle(m, (0, 0), (w - 1, h - 1), 255, -1)
    else:
        cv2.ellipse(m, (w // 2, h // 2), (max(1, w // 2), max(1, h // 2)),
                    0, 0, 360, 255, -1)
    if feather > 0:
        k = int(feather) | 1
        m = cv2.GaussianBlur(m, (k, k), 0)
    return (m.astype(np.float32) / 255.0)[..., None]


def apply_mosaic(frame: np.ndarray, boxes: list[Box], st: Settings,
                 draw_boxes: bool = False) -> np.ndarray:
    """Composite a mosaic patch for each box. Returns the same array, modified.

    `draw_boxes` overlays debug rectangles. It is an explicit argument rather
    than a settings lookup so the final render can never pick it up - those
    rectangles are a tuning aid for the preview pane only.
    """
    if not boxes:
        return frame
    H, W = frame.shape[:2]
    for raw in boxes:
        b = raw.scaled(st.scale_x, st.scale_y, st.offset_y).clipped(W, H)
        x1, y1 = int(np.floor(b.x1)), int(np.floor(b.y1))
        x2, y2 = int(np.ceil(b.x2)), int(np.ceil(b.y2))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        w, h = x2 - x1, y2 - y1
        if w < 3 or h < 3:
            continue

        roi = frame[y1:y2, x1:x2]
        block = _block_size(w, h, st)
        mosaic = _pixelate(roi, block, st.flatten)

        if st.darken > 0:
            # A touch darker reads as a deliberate solid patch rather than a
            # blurry face, and kills any remaining low-contrast detail.
            mosaic = cv2.convertScaleAbs(mosaic, alpha=1.0 - st.darken, beta=0)

        if st.blur_mix > 0:
            # Just enough blur to soften block edges; full blur looks eerie.
            k = max(3, (block // 2) | 1)
            blurred = cv2.GaussianBlur(mosaic, (k, k), 0)
            mosaic = cv2.addWeighted(mosaic, 1.0 - st.blur_mix, blurred, st.blur_mix, 0)

        mask = _mask_for(st.shape, w, h, st.feather)
        frame[y1:y2, x1:x2] = (roi * (1.0 - mask) + mosaic * mask).astype(np.uint8)

        if draw_boxes:
            color = {
                "face": (0, 255, 0), "yunet": (0, 200, 255),
                "pose": (255, 128, 0), "track": (255, 0, 255),
                "fill": (128, 128, 255),
            }.get(raw.source, (200, 200, 200))
            cv2.rectangle(frame, (int(raw.x1), int(raw.y1)),
                          (int(raw.x2), int(raw.y2)), color, 1)
    return frame


def preview_tile(frame: np.ndarray, boxes: list[Box], st: Settings,
                 max_side: int = 720) -> np.ndarray:
    """Downscaled before/after pair for the UI preview."""
    h, w = frame.shape[:2]
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        small = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
        sb = [Box(b.x1 * scale, b.y1 * scale, b.x2 * scale, b.y2 * scale,
                  b.score, b.source) for b in boxes]
    else:
        small, sb = frame.copy(), boxes
    return apply_mosaic(small.copy(), sb, st, draw_boxes=st.preview_boxes)
