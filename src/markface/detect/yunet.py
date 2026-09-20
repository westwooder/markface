"""OpenCV YuNet face detector.

Kept as a second opinion: it is a different architecture from YOLO, so it
tends to fail on different inputs. Cheap enough to always run.
"""
from __future__ import annotations

import cv2
import numpy as np

from .boxes import Box


class YunetDetector:
    def __init__(self, model_path: str, conf: float = 0.5, nms_thr: float = 0.3,
                 top_k: int = 500, max_side: int = 1280):
        self.conf = float(conf)
        self.max_side = int(max_side)
        self._det = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320), self.conf, float(nms_thr), int(top_k)
        )
        self._size = (320, 320)

    def detect(self, img: np.ndarray, conf: float | None = None) -> list[Box]:
        h, w = img.shape[:2]
        # YuNet is sensitive to input size; cap it for speed on 4K frames.
        scale = 1.0
        if max(w, h) > self.max_side:
            scale = self.max_side / max(w, h)
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        ih, iw = img.shape[:2]
        if self._size != (iw, ih):
            self._det.setInputSize((iw, ih))
            self._size = (iw, ih)
        if conf is not None and abs(conf - self.conf) > 1e-6:
            self.conf = float(conf)
            self._det.setScoreThreshold(self.conf)

        try:
            _, faces = self._det.detect(img)
        except cv2.error:
            return []
        if faces is None or len(faces) == 0:
            return []

        out: list[Box] = []
        inv = 1.0 / scale
        for f in faces:
            x, y, fw, fh, score = f[0], f[1], f[2], f[3], f[-1]
            out.append(
                Box(x * inv, y * inv, (x + fw) * inv, (y + fh) * inv,
                    float(score), "yunet").clipped(w, h)
            )
        return out
