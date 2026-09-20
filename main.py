"""Launcher: `python main.py` from source, and the PyInstaller entry script."""
from __future__ import annotations

import os
import sys
from pathlib import Path

if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# Qt on Windows: prefer the software-independent scaling path, and never let a
# stale plugin path from another install leak in.
os.environ.pop("QT_PLUGIN_PATH", None)
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

from markface.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
