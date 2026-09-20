"""YOLO-family face detector (single class) over onnxruntime."""
from __future__ import annotations

import numpy as np

from .boxes import Box, batch_nms_numpy
from .runtime import make_session, run_locked, session_device


class YoloFaceDetector:
    """Runs a yolov8/v11-style face model exported with output [1, 5, N]."""

    def __init__(self, model_path: str, device: str = "auto", threads: int = 0,
                 input_size: int = 640, conf: float = 0.25, nms_thr: float = 0.45):
        self.sess = make_session(model_path, device, threads)
        self.input_name = self.sess.get_inputs()[0].name
        self.input_size = int(input_size)
        self.conf = float(conf)
        self.nms_thr = float(nms_thr)
        self.device_label = session_device(self.sess)
        # Static-shape exports must be fed exactly what they declare.
        shape = self.sess.get_inputs()[0].shape
        if isinstance(shape[2], int) and isinstance(shape[3], int):
            self.input_size = int(shape[2])

    def _letterbox(self, img: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        h, w = img.shape[:2]
        s = self.input_size
        r = min(s / w, s / h)
        nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
        # Centre the padding so off-centre faces are not biased toward an edge.
        import cv2
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((s, s, 3), 114, np.uint8)
        dx, dy = (s - nw) // 2, (s - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized
        return canvas, r, dx, dy

    def detect(self, img: np.ndarray, conf: float | None = None) -> list[Box]:
        thr = self.conf if conf is None else float(conf)
        canvas, r, dx, dy = self._letterbox(img)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = run_locked(self.sess, {self.input_name: blob})[0]

        pred = out[0]                      # (5, N)
        if pred.shape[0] < pred.shape[1]:  # (5, N) -> (N, 5)
            pred = pred.T
        scores = pred[:, 4]
        keep = scores >= thr
        if not np.any(keep):
            return []
        cxcywh = pred[keep, :4]
        scores = scores[keep]

        xyxy = np.empty_like(cxcywh)
        xyxy[:, 0] = cxcywh[:, 0] - cxcywh[:, 2] * 0.5
        xyxy[:, 1] = cxcywh[:, 1] - cxcywh[:, 3] * 0.5
        xyxy[:, 2] = cxcywh[:, 0] + cxcywh[:, 2] * 0.5
        xyxy[:, 3] = cxcywh[:, 1] + cxcywh[:, 3] * 0.5
        # Undo letterbox: remove padding, then rescale.
        xyxy[:, [0, 2]] -= dx
        xyxy[:, [1, 3]] -= dy
        xyxy /= r

        idx = batch_nms_numpy(xyxy, scores, self.nms_thr)
        h, w = img.shape[:2]
        return [
            Box(*xyxy[i].tolist(), float(scores[i]), "face").clipped(w, h)
            for i in idx
        ]
