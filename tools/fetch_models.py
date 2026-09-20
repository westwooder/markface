"""Download the ONNX weights listed in models/manifest.json.

Run before building:  python tools/fetch_models.py
Mirrors: pass --mirror hf-mirror.com if huggingface.co is unreachable.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024.0
    return f"{n:.1f}GB"


def download(url: str, dest: Path, mirror: str | None) -> None:
    if mirror:
        url = url.replace("huggingface.co", mirror)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "markface/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        got = 0
        with open(tmp, "wb") as fh:
            while chunk := resp.read(1 << 16):
                fh.write(chunk)
                got += len(chunk)
                if total:
                    pct = got * 100 // total
                    print(f"\r  {dest.name}: {pct}% ({human(got)}/{human(total)})",
                          end="", flush=True)
    print(f"\r  {dest.name}: done ({human(got)})            ")
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", help="host to substitute for huggingface.co")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()

    manifest = json.loads((MODELS / "manifest.json").read_text(encoding="utf-8"))
    failures = []
    for key, spec in manifest["models"].items():
        dest = MODELS / spec["file"]
        if dest.exists() and not args.force:
            print(f"[skip] {key}: {spec['file']} already present ({human(dest.stat().st_size)})")
            continue
        print(f"[get ] {key}: {spec['desc']}")
        try:
            download(spec["url"], dest, args.mirror)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  FAILED: {exc}")
            failures.append(key)

    if failures:
        print(f"\n{len(failures)} model(s) failed: {', '.join(failures)}")
        print("Retry, or use --mirror hf-mirror.com from mainland China.")
        return 1
    print("\nAll models present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
