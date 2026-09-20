"""Background workers.

Thread lifetime is handled carefully here. A QThread that Python garbage
collects while it is still running takes the process down with it, and when the
thread owns onnxruntime sessions the crash lands inside the native provider
(access violation in onnxruntime_pybind11_state.pyd) with no Python traceback.
So every thread is kept in a module-level registry until Qt confirms it has
finished, and a new job never starts on top of a live one.
"""
from __future__ import annotations

import logging

from PySide6.QtCore import QObject, QThread, Signal, Slot

from ..pipeline import Pipeline, Result
from ..settings import Settings

log = logging.getLogger(__name__)

# Keeps (thread, worker) pairs alive for exactly as long as Qt needs them.
_LIVE: set[tuple[QThread, QObject]] = set()


class ProcessWorker(QObject):
    """Runs the full pipeline on a worker thread."""

    progress = Signal(str, float, str)     # stage, fraction, message
    finished = Signal(object)              # Result

    def __init__(self, src: str, dst: str, settings: Settings):
        super().__init__()
        self.src = src
        self.dst = dst
        self.settings = settings
        self.pipeline: Pipeline | None = None

    @Slot()
    def run(self) -> None:
        res: Result
        try:
            self.pipeline = Pipeline(
                self.settings,
                progress=lambda s, f, m: self.progress.emit(s, f, m),
            )
            err = self.pipeline.ensure_models()
            if err:
                res = Result(ok=False, message=err)
            else:
                res = self.pipeline.run(self.src, self.dst)
        except Exception as exc:  # noqa: BLE001 - surface anything to the UI
            log.exception("pipeline failed")
            res = Result(ok=False, message=f"处理失败：{exc}")
        finally:
            self.pipeline = None
        self.finished.emit(res)

    def cancel(self) -> None:
        if self.pipeline:
            self.pipeline.cancel()


class PreviewWorker(QObject):
    """Renders a single-frame preview with the current settings."""

    ready = Signal(object, int, str)       # BGR image, box count, description
    failed = Signal(str)

    def __init__(self, src: str, frame_index: int, settings: Settings):
        super().__init__()
        self.src = src
        self.frame_index = frame_index
        self.settings = settings

    @Slot()
    def run(self) -> None:
        pipe = None
        try:
            from ..mosaic import apply_mosaic
            from ..video import FrameReader

            pipe = Pipeline(self.settings)
            err = pipe.ensure_models()
            if err:
                self.failed.emit(err)
                return
            ens = pipe.build_ensemble()
            with FrameReader(self.src) as reader:
                frame = reader.read_at(self.frame_index)
            if frame is None:
                self.failed.emit("无法读取该帧")
                return
            boxes = ens.detect(frame)
            desc = ens.describe()
            out = apply_mosaic(frame.copy(), boxes, self.settings,
                               draw_boxes=self.settings.preview_boxes)
            self.ready.emit(out, len(boxes), desc)
        except Exception as exc:  # noqa: BLE001
            log.exception("preview failed")
            self.failed.emit(str(exc))
        finally:
            # The ensemble is shared through the cache; just drop our reference.
            if pipe is not None:
                pipe.ensemble = None
                del pipe


def start_worker(worker: QObject, done_signal: str | None = None) -> QThread:
    """Run `worker.run()` on a dedicated thread, managing the thread's lifetime.

    The (thread, worker) pair is held in a registry until QThread.finished
    fires, so neither object can be collected while native code is still using
    it. Callers may keep the returned handle, but do not have to.
    """
    thread = QThread()
    worker.moveToThread(thread)
    entry = (thread, worker)
    _LIVE.add(entry)

    thread.started.connect(worker.run)

    # Ask the thread's event loop to stop once the worker reports completion.
    for name in ("finished", "ready", "failed"):
        sig = getattr(worker, name, None)
        if sig is not None and (done_signal is None or name == done_signal):
            sig.connect(thread.quit)

    def _release() -> None:
        _LIVE.discard(entry)

    thread.finished.connect(_release)
    thread.start()
    return thread


def wait_for_idle(timeout_ms: int = 5000) -> None:
    """Block until every worker thread has finished. Used on shutdown."""
    for thread, _worker in list(_LIVE):
        if thread.isRunning():
            thread.quit()
            thread.wait(timeout_ms)


# Backwards-compatible alias.
run_in_thread = start_worker
