"""Filesystem layout, resolved for both source runs and the frozen bundle."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def app_root() -> Path:
    """Directory that holds the executable (frozen) or the project root (source)."""
    if is_frozen():
        return Path(sys.executable).parent
    return Path(__file__).resolve().parents[2]


def resource_root() -> Path:
    """Directory that holds bundled read-only resources."""
    if is_frozen():
        # PyInstaller onedir puts data under _internal; onefile uses _MEIPASS.
        return Path(getattr(sys, "_MEIPASS", app_root()))
    return app_root()


def models_dir() -> Path:
    """Where .onnx weights live. Bundled copy wins; user dir is the fallback."""
    bundled = resource_root() / "models"
    if (bundled / "manifest.json").exists():
        return bundled
    return app_root() / "models"


def config_dir() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home())
    d = Path(base) / "markface"
    d.mkdir(parents=True, exist_ok=True)
    return d


def config_file() -> Path:
    return config_dir() / "settings.json"


def log_file() -> Path:
    return config_dir() / "markface.log"
