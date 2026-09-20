"""Build a synthetic test clip from a still image: moving + occluded faces.

Not a substitute for real footage, but it exercises tracking, dropout fill and
the occlusion path deterministically.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="still image containing a face")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    img = cv2.imread(args.src)
    if img is None:
        print("cannot read", args.src)
        return 1
    face = cv2.resize(img, (200, 200))
    H, W = 480, 854
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    for i in range(args.frames):
        frame = np.full((H, W, 3), 40, np.uint8)
        t = i / max(1, args.frames - 1)
        x = int(30 + t * (W - 260))
        y = int(120 + 60 * np.sin(t * 6.28))
        patch = face.copy()
        # Middle third: blindfold. Last third: gag as well.
        if 0.33 <= t < 0.66:
            cv2.rectangle(patch, (10, 60), (190, 100), (30, 30, 30), -1)
        elif t >= 0.66:
            cv2.rectangle(patch, (10, 60), (190, 100), (30, 30, 30), -1)
            cv2.rectangle(patch, (40, 130), (160, 170), (200, 200, 200), -1)
        frame[y:y + 200, x:x + 200] = patch
        vw.write(frame)
    vw.release()
    print("wrote", args.out, Path(args.out).stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
