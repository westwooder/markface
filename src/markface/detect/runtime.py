"""onnxruntime session construction and device selection."""
from __future__ import annotations

import logging
import threading
from typing import Any

import onnxruntime as ort

log = logging.getLogger(__name__)

# Building or running a DirectML session concurrently from several threads
# crashes the provider outright (access violation inside
# onnxruntime_pybind11_state.pyd, no Python traceback). onnxruntime's DML
# execution provider is not thread-safe across sessions on one device, so all
# GPU work in this process is funnelled through one lock.
GPU_LOCK = threading.RLock()

# Preference order per device choice. DirectML covers NVIDIA/AMD/Intel on Win10+.
_GPU_EPS = ("DmlExecutionProvider", "CUDAExecutionProvider")


def available_devices() -> list[tuple[str, str]]:
    """Return [(value, label)] for the UI, best option first."""
    provs = set(ort.get_available_providers())
    out = [("auto", "自动选择（推荐）")]
    for ep in _GPU_EPS:
        if ep in provs:
            name = "DirectML" if ep.startswith("Dml") else "CUDA"
            out.append(("gpu", f"GPU 加速（{name}）"))
            break
    out.append(("cpu", "CPU（兼容性最好）"))
    return out


def has_gpu() -> bool:
    provs = set(ort.get_available_providers())
    return any(ep in provs for ep in _GPU_EPS)


def resolve_providers(device: str) -> list[Any]:
    """Map a device choice onto an onnxruntime provider list."""
    provs = set(ort.get_available_providers())
    if device in ("auto", "gpu"):
        for ep in _GPU_EPS:
            if ep in provs:
                if ep == "DmlExecutionProvider":
                    return [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
                return [ep, "CPUExecutionProvider"]
        if device == "gpu":
            log.warning("GPU requested but no GPU provider available; falling back to CPU")
    return ["CPUExecutionProvider"]


def make_session(path: str, device: str, threads: int = 0) -> ort.InferenceSession:
    """Build a session, degrading to CPU if the GPU provider fails to init."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    if threads > 0:
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = max(1, threads // 2)
    providers = resolve_providers(device)
    gpu = providers[0] != "CPUExecutionProvider"
    lock = GPU_LOCK if gpu else _NULL_LOCK
    try:
        with lock:
            sess = ort.InferenceSession(path, sess_options=opts, providers=providers)
    except Exception as exc:  # noqa: BLE001 - GPU init is genuinely flaky
        if not gpu:
            raise
        log.warning("provider init failed (%s); retrying on CPU", exc)
        sess = ort.InferenceSession(path, sess_options=opts,
                                    providers=["CPUExecutionProvider"])
    return sess


class _NullLock:
    """No-op stand-in so CPU sessions skip the GPU lock entirely."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NULL_LOCK = _NullLock()


def run_locked(sess: ort.InferenceSession, feeds: dict) -> list:
    """Run inference, serialising the call when it executes on the GPU."""
    if sess.get_providers()[0] == "CPUExecutionProvider":
        return sess.run(None, feeds)
    with GPU_LOCK:
        return sess.run(None, feeds)


def session_device(sess: ort.InferenceSession) -> str:
    """Human-readable description of what the session actually runs on."""
    active = sess.get_providers()
    if active and active[0] == "DmlExecutionProvider":
        return "GPU / DirectML"
    if active and active[0] == "CUDAExecutionProvider":
        return "GPU / CUDA"
    return "CPU"
