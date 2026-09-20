"""User-facing settings: defaults, persistence, validation."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from .paths import config_file


# Bump when a default changes in a way that should override what users already
# have saved. A stored file from an older schema is discarded rather than merged,
# because silently keeping an old default is how "I changed the default and
# nothing happened" bugs happen.
SCHEMA_VERSION = 3


@dataclass
class Settings:
    schema: int = SCHEMA_VERSION

    # --- compute ---
    device: str = "auto"          # auto | gpu | cpu
    threads: int = 0              # 0 = let onnxruntime decide

    # --- detection ---
    preset: str = "balanced"      # fast | balanced | thorough
    conf_face: float = 0.25       # low on purpose: over-detect rather than miss
    conf_pose: float = 0.35
    use_yunet: bool = False       # second face detector, different failure modes
    use_pose: bool = False        # head-from-body when the face is fully hidden
    repair_occlusion: bool = False  # grow boxes that collapsed under occlusion
    reject_flat: bool = True      # drop weak, featureless boxes (bare skin etc.)
    detect_every: int = 1         # detect every Nth frame; gaps filled by tracking

    # --- tracking / interpolation ---
    track_enabled: bool = False
    track_max_age: int = 10       # keep coasting a lost track this many frames
    track_iou: float = 0.25
    smooth_window: int = 5        # temporal box smoothing (odd number)
    forward_fill: int = 12        # bridge short detection dropouts

    # --- mosaic geometry ---
    shape: str = "ellipse"        # ellipse | rect
    scale_x: float = 1.00         # relative to detected face box
    scale_y: float = 1.02         # >1 so the eyes stay covered near the sides,
                                  # where an ellipse loses most of its height
    offset_y: float = -0.05       # shifts the patch down (+) or up (-)
    feather: int = 6              # soft edge in px, avoids the harsh cut-out look

    # --- mosaic appearance ---
    block_ratio: float = 0.085    # block size as a fraction of patch width
    block_min: int = 6
    block_max: int = 40
    blur_mix: float = 0.12        # slight blur over blocks; 0 = pure pixelation
    strength: float = 1.2         # multiplies block size; the real "thickness" knob
    flatten: bool = True          # pre-blur so block averages ignore stray edges
    darken: float = 0.10          # slightly darken the patch so it reads as solid

    # --- output ---
    keep_audio: bool = True
    quality_mode: str = "match"   # match | crf | lossless
    quality: int = 18             # ffmpeg CRF, used when quality_mode == "crf"
    preview_boxes: bool = False   # preview overlay only, never in the output

    def clamp(self) -> "Settings":
        self.schema = SCHEMA_VERSION
        self.conf_face = min(max(self.conf_face, 0.02), 0.95)
        self.conf_pose = min(max(self.conf_pose, 0.05), 0.95)
        self.detect_every = min(max(self.detect_every, 1), 10)
        self.scale_x = min(max(self.scale_x, 0.3), 2.5)
        self.scale_y = min(max(self.scale_y, 0.3), 2.5)
        self.offset_y = min(max(self.offset_y, -0.6), 0.6)
        self.block_ratio = min(max(self.block_ratio, 0.01), 0.3)
        self.blur_mix = min(max(self.blur_mix, 0.0), 1.0)
        self.strength = min(max(self.strength, 0.3), 4.0)
        self.darken = min(max(self.darken, 0.0), 0.5)
        self.block_min = min(max(self.block_min, 2), 40)
        self.block_max = min(max(self.block_max, self.block_min), 120)
        self.feather = min(max(self.feather, 0), 60)
        self.smooth_window = max(1, self.smooth_window | 1)
        self.forward_fill = min(max(self.forward_fill, 0), 120)
        self.track_max_age = min(max(self.track_max_age, 1), 300)
        self.quality = min(max(self.quality, 0), 51)
        if self.device not in ("auto", "gpu", "cpu"):
            self.device = "auto"
        if self.preset not in ("fast", "balanced", "thorough"):
            self.preset = "balanced"
        if self.shape not in ("ellipse", "rect"):
            self.shape = "ellipse"
        if self.quality_mode not in ("match", "crf", "lossless"):
            self.quality_mode = "match"
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known}).clamp()

    def save(self) -> None:
        try:
            config_file().write_text(
                json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            pass  # settings are a convenience, never fatal

    @classmethod
    def load(cls) -> "Settings":
        p = config_file()
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if int(data.get("schema", 0)) == SCHEMA_VERSION:
                    return cls.from_dict(data)
                # Older schema: keep the user's file as a backup and start from
                # the current defaults.
                try:
                    p.with_suffix(".json.bak").write_text(
                        json.dumps(data, indent=2, ensure_ascii=False),
                        encoding="utf-8")
                except OSError:
                    pass
            except (OSError, ValueError, TypeError):
                pass
        return cls()


# Preset overrides applied on top of the user's settings.
PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "model": "face_s", "conf_face": 0.35, "use_yunet": False,
        "use_pose": False, "detect_every": 2, "input_size": 640,
    },
    "balanced": {
        "model": "face_m", "conf_face": 0.25, "use_yunet": False,
        "use_pose": False, "detect_every": 1, "input_size": 640,
    },
    "thorough": {
        "model": "face_m", "conf_face": 0.15, "use_yunet": True,
        "use_pose": True, "repair_occlusion": True,
        "detect_every": 1, "input_size": 960,
    },
}
