"""The detection ensemble: several detectors unioned, biased toward over-masking."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..settings import PRESETS, Settings
from .boxes import Box, contain_ratio, iou, merge_union
from .occlusion import repair_aspect
from .pose_head import PoseHeadDetector
from .runtime import session_device
from .verify import filter_implausible
from .yolo_face import YoloFaceDetector
from .yunet import YunetDetector

log = logging.getLogger(__name__)


class FaceEnsemble:
    """Runs the configured detectors on a frame and returns one merged box list.

    Design bias: recall over precision. A face the user has to censor by hand is
    a failure; one extra mosaic patch on a background object is cosmetic.
    """

    def __init__(self, models_dir: Path, settings: Settings, manifest: dict):
        preset = PRESETS[settings.preset]
        self.settings = settings
        self.preset = preset
        self.models_dir = Path(models_dir)
        self.manifest = manifest
        self.repair_occlusion = bool(getattr(settings, "repair_occlusion", True))
        self.reject_flat = bool(getattr(settings, "reject_flat", True))

        face_key = preset["model"]
        face_file = manifest["models"][face_key]["file"]
        self.face = YoloFaceDetector(
            str(self.models_dir / face_file),
            device=settings.device,
            threads=settings.threads,
            input_size=preset["input_size"],
            conf=settings.conf_face,
        )
        self.device_label = self.face.device_label

        self.yunet: YunetDetector | None = None
        if settings.use_yunet and preset["use_yunet"]:
            p = self.models_dir / manifest["models"]["yunet"]["file"]
            if p.exists():
                try:
                    # Higher bar than the YOLO pass: YuNet gets noisy below ~0.5.
                    self.yunet = YunetDetector(str(p), conf=max(0.45, settings.conf_face + 0.2))
                except Exception as exc:  # noqa: BLE001
                    log.warning("YuNet unavailable: %s", exc)

        self.pose: PoseHeadDetector | None = None
        if settings.use_pose and preset["use_pose"]:
            p = self.models_dir / manifest["models"]["pose"]["file"]
            if p.exists():
                try:
                    self.pose = PoseHeadDetector(
                        str(p), device=settings.device, threads=settings.threads,
                        conf=settings.conf_pose,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("pose model unavailable: %s", exc)

    def describe(self) -> str:
        parts = [f"人脸模型 {self.preset['model']}"]
        if self.yunet:
            parts.append("YuNet 二次确认")
        if self.pose:
            parts.append("姿态推断头部（抗遮挡）")
        return f"{self.device_label} · " + " · ".join(parts)

    @staticmethod
    def _is_novel(head: Box, known: list[Box]) -> bool:
        """True if a pose-derived head covers ground no face detector claimed.

        A pose head is a coarse estimate, so it can sit noticeably off the real
        face while still describing the same person. Several tests, because any
        one of them alone lets duplicates through:
          - meaningful overlap either way round
          - one box's centre inside the other
          - centres closer than the face box is wide (same head, sloppy estimate)
        """
        for f in known:
            if iou(f, head) > 0.28:
                return False
            if contain_ratio(f, head) > 0.55 or contain_ratio(head, f) > 0.55:
                return False
            if (f.x1 <= head.cx <= f.x2 and f.y1 <= head.cy <= f.y2) or                (head.x1 <= f.cx <= head.x2 and head.y1 <= f.cy <= head.y2):
                return False
            # Horizontally aligned and vertically adjacent means same person:
            # the pose estimate simply sits high or low on the real face.
            x_gap = max(0.0, max(f.x1, head.x1) - min(f.x2, head.x2))
            if x_gap < max(f.w, head.w) * 0.5:
                y_gap = max(0.0, max(f.y1, head.y1) - min(f.y2, head.y2))
                if y_gap < max(f.h, head.h) * 1.6:
                    return False
            ref = max(f.w, f.h, 1.0)
            if float(np.hypot(f.cx - head.cx, f.cy - head.cy)) < ref * 1.15:
                return False
        return True

    def detect(self, frame: np.ndarray) -> list[Box]:
        groups: list[list[Box]] = []
        faces = self.face.detect(frame)
        groups.append(faces)

        if self.yunet is not None:
            groups.append(self.yunet.detect(frame))

        if self.pose is not None:
            heads = self.pose.detect(frame)
            known = faces + (groups[1] if len(groups) > 1 else [])
            bodies = getattr(self.pose, "last_bodies", [])
            visible = getattr(self.pose, "last_visible", [])
            novel = []
            for i, h in enumerate(heads):
                if not self._is_novel(h, known):
                    continue
                # A head with no visible facial landmarks is inferred purely from
                # the torso. That is the case we want for a fully covered face,
                # but it also fires on false-positive bodies, so require the body
                # box to be big enough to actually be one.
                if i < len(visible) and visible[i] == 0:
                    body = bodies[i] if i < len(bodies) else None
                    if body is None or body.h < h.h * 2.2:
                        continue
                novel.append(h)
            groups.append(novel)

        if self.repair_occlusion:
            # Repair before merging, not after: a box that collapsed under
            # occlusion is the wrong size to compare against other detectors'
            # boxes, so merging first would split one face into two patches.
            groups = [
                [repair_aspect(b) if b.source in ("face", "yunet") else b
                 for b in g]
                for g in groups
            ]
        merged = merge_union(groups, thr=0.55)
        # Last gate: discard weakly-scored boxes with no facial structure, which
        # are usually bare skin elsewhere on the body.
        return filter_implausible(frame, merged, self.reject_flat)
