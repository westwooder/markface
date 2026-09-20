"""Head-region inference from body pose.

Why this exists: a blindfolded, gagged, masked or back-turned head defeats
face detectors, but the body is still plainly visible. COCO pose gives 17
keypoints of which five (nose, both eyes, both ears) sit on the head and two
(the shoulders) bound it from below. From those we can synthesise a head box
even when no facial texture is visible at all.
"""
from __future__ import annotations

import cv2
import numpy as np

from .boxes import Box
from .runtime import make_session, run_locked, session_device

# COCO keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
HEAD_KPTS = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)


class PoseHeadDetector:
    """Wraps a YOLO26-pose ONNX export: output [1, 300, 57], NMS already applied.

    Layout per row: x1 y1 x2 y2 conf cls, then 17 * (x, y, conf).
    Box and keypoint coordinates come out normalised to the letterboxed square.
    """

    def __init__(self, model_path: str, device: str = "auto", threads: int = 0,
                 input_size: int = 640, conf: float = 0.35, kpt_conf: float = 0.30):
        self.sess = make_session(model_path, device, threads)
        self.input_name = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape
        self.input_size = int(shape[2]) if isinstance(shape[2], int) else int(input_size)
        self.conf = float(conf)
        self.kpt_conf = float(kpt_conf)
        self.device_label = session_device(self.sess)

    def _letterbox(self, img: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        h, w = img.shape[:2]
        s = self.input_size
        r = min(s / w, s / h)
        nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
        canvas = np.full((s, s, 3), 114, np.uint8)
        dx, dy = (s - nw) // 2, (s - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(img, (nw, nh),
                                                    interpolation=cv2.INTER_LINEAR)
        return canvas, r, dx, dy

    def detect(self, img: np.ndarray, conf: float | None = None) -> list[Box]:
        """Return synthesised head boxes, one per detected person."""
        thr = self.conf if conf is None else float(conf)
        h, w = img.shape[:2]
        canvas, r, dx, dy = self._letterbox(img)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = run_locked(self.sess, {self.input_name: blob})[0][0]  # (300, 57)

        s = float(self.input_size)
        heads: list[Box] = []
        self.last_bodies: list[Box] = []
        self.last_visible: list[int] = []
        for row in out:
            score = float(row[4])
            if score < thr:
                continue
            # De-normalise, undo padding, undo scale.
            px = lambda v: (float(v) * s - dx) / r   # noqa: E731
            py = lambda v: (float(v) * s - dy) / r   # noqa: E731
            body = Box(px(row[0]), py(row[1]), px(row[2]), py(row[3]), score, "pose")
            kpts = row[6:].reshape(-1, 3).astype(np.float64)
            head = self._head_from_keypoints(kpts, body, px, py, score)
            if head is not None and head.w > 2 and head.h > 2:
                visible = int(sum(1 for i in HEAD_KPTS
                                  if kpts[i, 2] >= self.kpt_conf))
                heads.append(head.clipped(w, h))
                self.last_bodies.append(body.clipped(w, h))
                self.last_visible.append(visible)
        return heads

    def _head_from_keypoints(self, kpts: np.ndarray, body: Box, px, py,
                             score: float) -> Box | None:
        """Synthesise a head box from whatever head/shoulder keypoints are visible.

        Three tiers, most reliable first:
          1. Two or more head keypoints -> span them and pad by their spread.
          2. One head keypoint + shoulders -> size the head from shoulder width.
          3. Shoulders only -> place the head above the shoulder line.
        Falling all the way through, we use the top slice of the body box, which
        is crude but still better than leaving the head unmasked.
        """
        head_pts = [(px(kpts[i, 0]), py(kpts[i, 1]), kpts[i, 2]) for i in HEAD_KPTS]
        good = [(x, y) for x, y, c in head_pts if c >= self.kpt_conf]

        ls, rs = kpts[L_SHOULDER], kpts[R_SHOULDER]
        sh_ok = ls[2] >= self.kpt_conf and rs[2] >= self.kpt_conf
        if sh_ok:
            lsx, lsy = px(ls[0]), py(ls[1])
            rsx, rsy = px(rs[0]), py(rs[1])
            sh_w = abs(lsx - rsx)
            sh_cx, sh_cy = (lsx + rsx) * 0.5, (lsy + rsy) * 0.5
        else:
            sh_w = sh_cx = sh_cy = 0.0

        if len(good) >= 2:
            xs = [p[0] for p in good]
            ys = [p[1] for p in good]
            spread = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
            # Facial keypoints cover roughly the middle of the head: pad generously.
            pad_x = spread * 0.95
            pad_y = spread * 1.15
            cx = (min(xs) + max(xs)) * 0.5
            cy = (min(ys) + max(ys)) * 0.5
            half_w = (max(xs) - min(xs)) * 0.5 + pad_x
            half_h = (max(ys) - min(ys)) * 0.5 + pad_y
            if sh_ok:
                # A head is ~0.55-0.75x shoulder width; use it as a floor.
                half_w = max(half_w, sh_w * 0.34)
                half_h = max(half_h, sh_w * 0.42)
            return Box(cx - half_w, cy - half_h, cx + half_w, cy + half_h, score, "pose")

        if len(good) == 1 and sh_ok and sh_w > 1:
            cx, cy = good[0]
            half_w = sh_w * 0.38
            half_h = sh_w * 0.46
            return Box(cx - half_w, cy - half_h, cx + half_w, cy + half_h, score, "pose")

        if sh_ok and sh_w > 1:
            # No visible facial landmarks at all (back of head, full mask).
            half_w = sh_w * 0.40
            half_h = sh_w * 0.48
            cy = sh_cy - half_h * 0.85   # sit the head above the shoulders
            return Box(sh_cx - half_w, cy - half_h, sh_cx + half_w, cy + half_h,
                       score, "pose")

        if body.h > 8 and body.w > 4:
            # Last resort: the head occupies roughly the top 18% of a standing body.
            half_w = min(body.w * 0.5, body.h * 0.11)
            cx = body.cx
            top = body.y1
            return Box(cx - half_w, top, cx + half_w, top + body.h * 0.18,
                       score * 0.8, "pose")
        return None
