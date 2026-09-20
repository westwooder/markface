"""Temporal smoothing and dropout bridging over a whole-video detection pass.

Running this offline (rather than streaming) lets us look *backwards* as well as
forwards, so a face missed for a few frames in the middle of a shot gets filled
by interpolating between the detections on either side.
"""
from __future__ import annotations

import numpy as np

from ..detect.boxes import Box, iou


def _lerp(a: Box, b: Box, t: float, source: str = "fill") -> Box:
    return Box(
        a.x1 + (b.x1 - a.x1) * t,
        a.y1 + (b.y1 - a.y1) * t,
        a.x2 + (b.x2 - a.x2) * t,
        a.y2 + (b.y2 - a.y2) * t,
        min(a.score, b.score),
        source,
    )


def build_chains(per_frame: list[list[Box]], iou_thr: float = 0.2,
                 dist_scale: float = 1.5) -> list[dict[int, Box]]:
    """Group boxes across frames into per-identity chains {frame_index: box}."""
    chains: list[dict[int, Box]] = []
    last_seen: list[tuple[int, Box]] = []   # (frame, box) per chain

    for fi, boxes in enumerate(per_frame):
        taken: set[int] = set()
        for bi, b in enumerate(boxes):
            best, best_score = -1, 0.0
            for ci, (lf, lb) in enumerate(last_seen):
                if ci in taken or fi - lf > 90:
                    continue
                s = iou(lb, b)
                if s <= 0:
                    ref = max(lb.w, lb.h, b.w, b.h, 1.0) * dist_scale
                    d = float(np.hypot(lb.cx - b.cx, lb.cy - b.cy))
                    gap = max(1, fi - lf)
                    if d < ref * gap:
                        s = 0.05 * (1.0 - min(1.0, d / (ref * gap)))
                if s > best_score:
                    best, best_score = ci, s
            if best >= 0 and best_score >= min(iou_thr, 0.2):
                chains[best][fi] = b
                last_seen[best] = (fi, b)
                taken.add(best)
            else:
                chains.append({fi: b})
                last_seen.append((fi, b))
                taken.add(len(chains) - 1)
    return chains


def fill_gaps(chains: list[dict[int, Box]], max_gap: int) -> list[dict[int, Box]]:
    """Interpolate across short dropouts inside each chain."""
    if max_gap <= 0:
        return chains
    out: list[dict[int, Box]] = []
    for ch in chains:
        keys = sorted(ch)
        filled = dict(ch)
        for a, b in zip(keys, keys[1:]):
            gap = b - a
            if 1 < gap <= max_gap + 1:
                for k in range(a + 1, b):
                    filled[k] = _lerp(ch[a], ch[b], (k - a) / gap)
        out.append(filled)
    return out


def smooth_chains(chains: list[dict[int, Box]], window: int) -> list[dict[int, Box]]:
    """Moving-average the box corners along each chain to kill jitter."""
    if window <= 1:
        return chains
    half = window // 2
    out: list[dict[int, Box]] = []
    for ch in chains:
        keys = sorted(ch)
        arr = np.array([[ch[k].x1, ch[k].y1, ch[k].x2, ch[k].y2] for k in keys],
                       dtype=np.float64)
        n = len(keys)
        sm = np.empty_like(arr)
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            sm[i] = arr[lo:hi].mean(axis=0)
        out.append({
            k: Box(*sm[i].tolist(), ch[k].score, ch[k].source)
            for i, k in enumerate(keys)
        })
    return out


def pad_chain_ends(chains: list[dict[int, Box]], total: int, pad: int) -> list[dict[int, Box]]:
    """Extend each chain a few frames past its first/last detection.

    Detectors usually pick a face up a frame or two late and drop it a frame or
    two early; padding hides those edges.
    """
    if pad <= 0:
        return chains
    out: list[dict[int, Box]] = []
    for ch in chains:
        keys = sorted(ch)
        if not keys:
            out.append(ch)
            continue
        filled = dict(ch)
        first, last = keys[0], keys[-1]
        for k in range(max(0, first - pad), first):
            filled[k] = Box(*ch[first].as_tuple(), ch[first].score * 0.9, "fill")
        for k in range(last + 1, min(total, last + pad + 1)):
            filled[k] = Box(*ch[last].as_tuple(), ch[last].score * 0.9, "fill")
        out.append(filled)
    return out


def chains_to_frames(chains: list[dict[int, Box]], total: int) -> list[list[Box]]:
    out: list[list[Box]] = [[] for _ in range(total)]
    for ch in chains:
        for fi, b in ch.items():
            if 0 <= fi < total:
                out[fi].append(b)
    return out
