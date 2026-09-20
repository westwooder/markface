"""Headless runner, for scripted use and for verifying a build without the GUI.

    python tools/cli.py input.mp4 output.mp4 --preset thorough --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from markface import logsetup                      # noqa: E402
from markface.pipeline import Pipeline             # noqa: E402
from markface.settings import Settings             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="视频人脸打码（命令行）")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--preset", choices=["fast", "balanced", "thorough"])
    ap.add_argument("--device", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--scale-x", type=float)
    ap.add_argument("--scale-y", type=float)
    ap.add_argument("--block-ratio", type=float)
    ap.add_argument("--strength", type=float, help="mosaic thickness, 0.5-3.0")
    ap.add_argument("--darken", type=float)
    ap.add_argument("--blur-mix", type=float)
    ap.add_argument("--quality-mode", choices=["match", "crf", "lossless"],
                    dest="quality_mode")
    ap.add_argument("--shape", choices=["ellipse", "rect"])
    ap.add_argument("--conf", type=float, dest="conf_face")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--no-reject-flat", action="store_true",
                    help="keep weak featureless detections")
    ap.add_argument("--boxes", action="store_true", help="draw detection boxes")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logsetup.setup()
    st = Settings.load()
    for name in ("preset", "device", "scale_x", "scale_y", "block_ratio",
                 "blur_mix", "shape", "conf_face", "strength", "darken",
                 "quality_mode"):
        v = getattr(args, name, None)
        if v is not None:
            setattr(st, name, v)
    if args.no_audio:
        st.keep_audio = False
    if args.no_reject_flat:
        st.reject_flat = False
    if args.boxes:
        st.preview_boxes = True
    st.clamp()

    last = [""]

    def progress(stage: str, frac: float, msg: str) -> None:
        if args.quiet:
            return
        key = f"{stage}{int(frac * 20)}"
        if key != last[0]:
            last[0] = key
            print(f"[{stage}] {frac * 100:5.1f}%  {msg}")

    pipe = Pipeline(st, progress=progress)
    err = pipe.ensure_models()
    if err:
        print(err, file=sys.stderr)
        return 2
    res = pipe.run(args.src, args.dst)
    print(res.message)
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
