"""Video probing, decoding and encoding.

Decoding goes through OpenCV (fast, no extra process). Encoding goes through
ffmpeg so we can copy the original audio track and control quality properly -
OpenCV's writer cannot do either.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2

log = logging.getLogger(__name__)

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frames: int
    has_audio: bool = False
    # Source encoding, so the output can match it instead of guessing.
    vcodec: str = ""          # e.g. h264, hevc
    vbitrate_kbps: int = 0    # video stream bitrate as reported by ffmpeg
    pix_fmt: str = ""         # e.g. yuv420p

    @property
    def duration(self) -> float:
        return self.frames / self.fps if self.fps > 0 else 0.0

    def label(self) -> str:
        m, s = divmod(int(self.duration), 60)
        return (f"{self.width}×{self.height} · {self.fps:.2f}fps · "
                f"{self.frames} 帧 · {m:02d}:{s:02d}"
                + ("（含音频）" if self.has_audio else "（无音频）"))


def ffmpeg_exe() -> str:
    """Locate the bundled ffmpeg, falling back to PATH."""
    from .paths import resource_root

    for cand in (resource_root() / "ffmpeg" / "ffmpeg.exe",
                 resource_root() / "ffmpeg.exe"):
        if cand.exists():
            return str(cand)
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return "ffmpeg"


def _run_quiet(cmd: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          creationflags=_CREATE_NO_WINDOW)


_CODEC_RE = re.compile(r"Video:\s*([a-z0-9_]+)")
_BITRATE_RE = re.compile(r"Video:.*?,\s*(\d+)\s*kb/s")
_PIXFMT_RE = re.compile(r"Video:[^,]*,\s*([a-z0-9]+(?:\([^)]*\))?)")
_TOTAL_BITRATE_RE = re.compile(r"bitrate:\s*(\d+)\s*kb/s")


def probe_streams(path: str) -> dict[str, object]:
    """Read codec / bitrate / pixel format from ffmpeg's own stream report.

    imageio-ffmpeg ships ffmpeg but not ffprobe, so parse `ffmpeg -i` stderr.
    Every field is optional - callers fall back to sensible defaults.
    """
    out: dict[str, object] = {"has_audio": False, "vcodec": "",
                              "vbitrate_kbps": 0, "pix_fmt": ""}
    try:
        r = _run_quiet([ffmpeg_exe(), "-hide_banner", "-i", path])
    except Exception:  # noqa: BLE001
        return out
    text = r.stderr or ""
    out["has_audio"] = "Audio:" in text

    video_lines = [ln for ln in text.splitlines() if "Video:" in ln]
    if video_lines:
        line = video_lines[0]
        m = _CODEC_RE.search(line)
        if m:
            out["vcodec"] = m.group(1)
        m = _BITRATE_RE.search(line)
        if m:
            out["vbitrate_kbps"] = int(m.group(1))
        else:
            # Some containers only report a container-level bitrate.
            m = _TOTAL_BITRATE_RE.search(text)
            if m:
                out["vbitrate_kbps"] = int(m.group(1))
        for token in line.split("Video:", 1)[1].split(","):
            t = token.strip().split("(")[0].strip()
            if t.startswith("yuv") or t in ("gbrp", "nv12", "p010le", "rgb24"):
                out["pix_fmt"] = t
                break
    return out


def probe_audio(path: str) -> bool:
    """True if the file carries at least one audio stream."""
    exe = ffmpeg_exe()
    ffprobe = str(Path(exe).with_name("ffprobe.exe"))
    if Path(ffprobe).exists():
        try:
            r = _run_quiet([ffprobe, "-v", "error", "-select_streams", "a",
                            "-show_entries", "stream=index", "-of", "json", path])
            return bool(json.loads(r.stdout or "{}").get("streams"))
        except Exception:  # noqa: BLE001
            pass
    # ffmpeg itself reports stream info on stderr; good enough as a fallback.
    try:
        r = _run_quiet([exe, "-hide_banner", "-i", path])
        return "Audio:" in (r.stderr or "")
    except Exception:  # noqa: BLE001
        return False


def probe(path: str) -> VideoInfo | None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if w <= 0 or h <= 0:
        return None
    if n <= 0:
        n = _count_frames(path)
    meta = probe_streams(path)
    return VideoInfo(
        path, w, h, fps, n,
        has_audio=bool(meta["has_audio"]),
        vcodec=str(meta["vcodec"]),
        vbitrate_kbps=int(meta["vbitrate_kbps"] or 0),
        pix_fmt=str(meta["pix_fmt"]),
    )


def _count_frames(path: str) -> int:
    """Walk the file when the container has no reliable frame count."""
    cap = cv2.VideoCapture(path)
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


class FrameReader:
    """Sequential frame iterator with a context manager."""

    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise OSError(f"无法打开视频：{path}")
        self._seekable: bool | None = None

    def __iter__(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame

    def _check_seekable(self) -> bool:
        """Some containers report a frame position of -1 after a seek and hand
        back the same picture for every index. Detect that once, up front."""
        if self._seekable is not None:
            return self._seekable
        ok = False
        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
            pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            ok = pos >= 0
        except cv2.error:
            ok = False
        self._seekable = ok
        if not ok:
            log.info("%s: frame seeking unreliable, decoding sequentially",
                     os.path.basename(self.path))
        return ok

    def read_at(self, index: int):
        """Return one frame by index, decoding from the start if seeking fails."""
        index = max(0, int(index))
        if self._check_seekable():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            if self.cap.get(cv2.CAP_PROP_POS_FRAMES) >= 0:
                ok, frame = self.cap.read()
                if ok:
                    return frame
            self._seekable = False

        # Sequential fallback: reopen and walk forward. Slower, but correct, and
        # a preview only needs one frame.
        self.cap.release()
        self.cap = cv2.VideoCapture(self.path)
        frame = None
        for _ in range(index + 1):
            ok, f = self.cap.read()
            if not ok:
                break
            frame = f
        return frame

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> "FrameReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class FfmpegWriter:
    """Pipes BGR frames into ffmpeg, optionally muxing the source audio.

    Quality policy: by default the output tracks the *source* bitrate rather
    than a fixed CRF, so a video does not visibly change just because it went
    through here. Re-encoding is unavoidable (we are repainting pixels), so the
    aim is to keep the loss below the noise floor of the original.
    """

    def __init__(self, out_path: str, width: int, height: int, fps: float,
                 crf: int = 20, audio_from: str | None = None,
                 quality_mode: str = "match", src_bitrate_kbps: int = 0,
                 pix_fmt: str = "yuv420p", vcodec: str = "libx264"):
        self.out_path = out_path
        exe = ffmpeg_exe()
        cmd = [
            exe, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps:.6f}",
            "-i", "pipe:0",
        ]
        if audio_from:
            # Copy the original audio untouched; no reason to re-encode it.
            cmd += ["-i", audio_from, "-map", "0:v:0", "-map", "1:a:0?",
                    "-c:a", "copy", "-shortest"]
        cmd += ["-c:v", vcodec, "-preset", "slow" if quality_mode == "match" else "medium"]

        if quality_mode == "match" and src_bitrate_kbps > 0:
            # Give the encoder headroom over the source: mosaic blocks are cheap
            # to encode, but the untouched background must not degrade.
            target = int(src_bitrate_kbps * 1.15)
            cmd += ["-b:v", f"{target}k",
                    "-maxrate", f"{int(target * 1.5)}k",
                    "-bufsize", f"{int(target * 3)}k"]
        elif quality_mode == "lossless":
            cmd += ["-crf", "0"]
        else:
            cmd += ["-crf", str(int(crf))]

        cmd += ["-pix_fmt", pix_fmt or "yuv420p",
                "-movflags", "+faststart", out_path]
        self.cmd = cmd
        log.info("ffmpeg: %s", " ".join(cmd[6:]))
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, creationflags=_CREATE_NO_WINDOW,
        )

    def write(self, frame) -> None:
        try:
            self.proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError) as exc:
            err = self._drain_err()
            raise OSError(f"ffmpeg 写入失败：{err or exc}") from exc

    def _drain_err(self) -> str:
        try:
            return (self.proc.stderr.read() or b"").decode("utf-8", "replace")[-800:]
        except Exception:  # noqa: BLE001
            return ""

    def close(self) -> str | None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except OSError:
            pass
        err = self._drain_err()
        code = self.proc.wait()
        if code != 0:
            return err or f"ffmpeg 退出码 {code}"
        return None

    def abort(self) -> None:
        try:
            self.proc.kill()
        except Exception:  # noqa: BLE001
            pass
