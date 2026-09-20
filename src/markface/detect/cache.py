"""Process-wide cache of built detector ensembles.

Rebuilding onnxruntime sessions costs a second or more and, on DirectML, also
allocates GPU memory. The preview pane re-runs on every slider nudge, and only
a handful of settings actually change which sessions are needed - so key the
cache on exactly those and reuse otherwise.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from ..settings import PRESETS, Settings
from .ensemble import FaceEnsemble

log = logging.getLogger(__name__)

_lock = threading.RLock()
_key: tuple | None = None
_ensemble: FaceEnsemble | None = None


def _cache_key(st: Settings) -> tuple:
    """Only settings that change which sessions exist belong in the key."""
    preset = PRESETS[st.preset]
    return (
        st.device,
        st.threads,
        preset["model"],
        preset["input_size"],
        bool(st.use_yunet and preset["use_yunet"]),
        bool(st.use_pose and preset["use_pose"]),
    )


def get_ensemble(models_dir: Path, st: Settings, manifest: dict) -> FaceEnsemble:
    """Return a cached ensemble, rebuilding only when the session set changes.

    Thresholds and mosaic geometry are applied per call by the caller, so a
    cached ensemble stays valid when the user only drags a slider.
    """
    global _key, _ensemble
    want = _cache_key(st)
    with _lock:
        if _ensemble is not None and _key == want:
            # Cheap knobs can be updated in place.
            _ensemble.settings = st
            _ensemble.repair_occlusion = st.repair_occlusion
            _ensemble.face.conf = st.conf_face
            if _ensemble.pose is not None:
                _ensemble.pose.conf = st.conf_pose
            return _ensemble
        log.info("building detector ensemble %s", want)
        _ensemble = FaceEnsemble(models_dir, st, manifest)
        _key = want
        return _ensemble


def clear() -> None:
    """Drop the cached sessions (called on shutdown)."""
    global _key, _ensemble
    with _lock:
        _ensemble = None
        _key = None
