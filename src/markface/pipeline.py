"""Two-pass video processing.

Pass 1 detects and tracks across the whole clip, collecting per-frame boxes.
Pass 2 re-reads the video and renders, using the *completed* box timeline - so
a face missed mid-shot can be filled by interpolating from both sides, which a
single streaming pass cannot do.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2

from .detect.boxes import Box
from .detect.ensemble import FaceEnsemble
from .mosaic import apply_mosaic
from .paths import models_dir
from .settings import PRESETS, Settings
from .track.smooth import (build_chains, chains_to_frames, fill_gaps,
                           pad_chain_ends, smooth_chains)
from .track.tracker import IouTracker
from .video import FfmpegWriter, FrameReader, VideoInfo, probe

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, float, str], None]   # stage, 0..1, message


class Cancelled(Exception):
    """Raised internally when the user stops a run."""


@dataclass
class Result:
    ok: bool
    out_path: str = ""
    frames: int = 0
    boxes_total: int = 0
    frames_with_box: int = 0
    seconds: float = 0.0
    message: str = ""


class Pipeline:
    def __init__(self, settings: Settings, progress: ProgressFn | None = None):
        self.st = settings.clamp()
        self.progress = progress or (lambda *_: None)
        self._cancel = threading.Event()
        self.ensemble: FaceEnsemble | None = None

    def cancel(self) -> None:
        self._cancel.set()

    def _check(self) -> None:
        if self._cancel.is_set():
            raise Cancelled()

    def _load_manifest(self) -> dict:
        p = models_dir() / "manifest.json"
        return json.loads(p.read_text(encoding="utf-8"))

    def ensure_models(self) -> str | None:
        """Return an error message if a required model file is missing."""
        man = self._load_manifest()
        preset = PRESETS[self.st.preset]
        need = [man["models"][preset["model"]]["file"]]
        missing = [f for f in need if not (models_dir() / f).exists()]
        if missing:
            return ("缺少模型文件：" + "、".join(missing)
                    + f"\n请把它们放到 {models_dir()}")
        return None

    def build_ensemble(self) -> FaceEnsemble:
        if self.ensemble is None:
            from .detect.cache import get_ensemble
            self.ensemble = get_ensemble(models_dir(), self.st, self._load_manifest())
        return self.ensemble

    # ---------- pass 1: detect + track ----------

    def detect_pass(self, info: VideoInfo) -> list[list[Box]]:
        """Collect per-frame boxes for the whole clip."""
        ens = self.build_ensemble()
        tracker = IouTracker(iou_thr=self.st.track_iou, max_age=self.st.track_max_age)
        per_frame: list[list[Box]] = []
        raw_frame: list[list[Box]] = []
        every = max(1, self.st.detect_every)

        t0 = time.time()
        with FrameReader(info.path) as reader:
            for idx, frame in enumerate(reader):
                self._check()
                if idx % every == 0:
                    dets = ens.detect(frame)
                else:
                    # Skipped frame: let the tracker coast, do not clear boxes.
                    dets = []
                raw_frame.append(dets)
                boxes = tracker.update(dets) if self.st.track_enabled else dets
                per_frame.append(list(boxes))

                if idx % 5 == 0 or idx + 1 == info.frames:
                    done = (idx + 1) / max(1, info.frames)
                    el = time.time() - t0
                    eta = el / max(1e-6, done) - el
                    self.progress("detect", min(0.999, done),
                                  f"检测 {idx + 1}/{info.frames} 帧 · "
                                  f"当前 {len(boxes)} 个目标 · 剩余约 {int(eta)}s")

        # Offline refinement: chain the raw detections, bridge dropouts, smooth.
        total = len(per_frame)
        if self.st.track_enabled:
            chains = build_chains(raw_frame, iou_thr=self.st.track_iou)
            chains = fill_gaps(chains, self.st.forward_fill)
            chains = pad_chain_ends(chains, total, min(6, self.st.forward_fill))
            chains = smooth_chains(chains, self.st.smooth_window)
            refined = chains_to_frames(chains, total)
            # Union the refined timeline with the streaming result; whichever
            # source found a head, the head gets covered.
            from .detect.boxes import merge_union
            per_frame = [merge_union([a, b], thr=0.6)
                         for a, b in zip(per_frame, refined)]
        self.progress("detect", 1.0, f"检测完成，共 {total} 帧")
        return per_frame

    # ---------- pass 2: render ----------

    def render_pass(self, info: VideoInfo, per_frame: list[list[Box]],
                    out_path: str) -> Result:
        audio = info.path if (self.st.keep_audio and info.has_audio) else None
        # Re-encode with the source's own codec family and bitrate so the
        # untouched parts of the picture look the same as they went in.
        vcodec = "libx265" if info.vcodec in ("hevc", "h265") else "libx264"
        writer = FfmpegWriter(
            out_path, info.width, info.height, info.fps,
            crf=self.st.quality, audio_from=audio,
            quality_mode=self.st.quality_mode,
            src_bitrate_kbps=info.vbitrate_kbps,
            pix_fmt=info.pix_fmt or "yuv420p",
            vcodec=vcodec,
        )
        t0 = time.time()
        n = 0
        boxes_total = 0
        frames_with_box = 0
        try:
            with FrameReader(info.path) as reader:
                for idx, frame in enumerate(reader):
                    self._check()
                    boxes = per_frame[idx] if idx < len(per_frame) else []
                    if boxes:
                        apply_mosaic(frame, boxes, self.st)
                        boxes_total += len(boxes)
                        frames_with_box += 1
                    writer.write(frame)
                    n += 1
                    if idx % 5 == 0 or idx + 1 == info.frames:
                        done = (idx + 1) / max(1, info.frames)
                        el = time.time() - t0
                        eta = el / max(1e-6, done) - el
                        self.progress("render", min(0.999, done),
                                      f"输出 {idx + 1}/{info.frames} 帧 · "
                                      f"剩余约 {int(eta)}s")
        except Cancelled:
            writer.abort()
            raise
        err = writer.close()
        if err:
            return Result(ok=False, message=f"编码失败：{err}")
        self.progress("render", 1.0, "输出完成")
        return Result(ok=True, out_path=out_path, frames=n,
                      boxes_total=boxes_total, frames_with_box=frames_with_box)

    # ---------- orchestration ----------

    def run(self, src: str, dst: str) -> Result:
        t0 = time.time()
        self.progress("probe", 0.0, "读取视频信息…")
        info = probe(src)
        if info is None:
            return Result(ok=False, message="无法识别该视频格式")
        if info.frames <= 0:
            return Result(ok=False, message="视频没有可用帧")

        log.info("input %s %dx%d %.3ffps %d frames audio=%s",
                 src, info.width, info.height, info.fps, info.frames, info.has_audio)

        out = Path(dst)
        out.parent.mkdir(parents=True, exist_ok=True)

        try:
            per_frame = self.detect_pass(info)
            res = self.render_pass(info, per_frame, str(out))
        except Cancelled:
            return Result(ok=False, message="已取消")

        res.seconds = time.time() - t0
        if res.ok:
            cover = res.frames_with_box / max(1, res.frames) * 100
            res.message = (f"完成：{res.frames} 帧，{res.frames_with_box} 帧打码"
                           f"（{cover:.0f}%），耗时 {res.seconds:.1f}s")
            log.info(res.message)
        return res

    def probe_only(self, src: str) -> VideoInfo | None:
        return probe(src)
