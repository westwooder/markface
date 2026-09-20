"""Small reusable widgets."""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QSizePolicy, QSlider,
                               QVBoxLayout, QWidget)


def bgr_to_pixmap(frame: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img.copy())


class ImageView(QLabel):
    """Aspect-preserving image display with a placeholder message."""

    def __init__(self, placeholder: str = ""):
        super().__init__()
        self._pix: QPixmap | None = None
        self._placeholder = placeholder
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(360, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            "QLabel { background: #1b1d22; color: #8a8f98; border: 1px solid #2c2f36;"
            " border-radius: 6px; }"
        )
        self.setText(placeholder)

    def set_frame(self, frame: np.ndarray) -> None:
        self._pix = bgr_to_pixmap(frame)
        self._rescale()

    def clear_frame(self) -> None:
        self._pix = None
        self.setText(self._placeholder)

    def _rescale(self) -> None:
        if self._pix is None:
            return
        self.setPixmap(self._pix.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._rescale()


class LabeledSlider(QWidget):
    """Slider over a float range with a live value readout."""

    valueChanged = Signal(float)

    def __init__(self, label: str, lo: float, hi: float, value: float,
                 step: float = 0.01, suffix: str = "", decimals: int = 2):
        super().__init__()
        self.lo, self.hi, self.step = lo, hi, step
        self.decimals = decimals
        self.suffix = suffix
        self._n = max(1, int(round((hi - lo) / step)))

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self._n)
        self.slider.setValue(self._to_pos(value))
        self.name = QLabel(label)
        self.name.setMinimumWidth(96)
        self.readout = QLabel(self._fmt(value))
        self.readout.setMinimumWidth(58)
        self.readout.setAlignment(Qt.AlignmentFlag.AlignRight
                                  | Qt.AlignmentFlag.AlignVCenter)
        self.readout.setStyleSheet("color:#9aa0a8;")

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.name)
        row.addWidget(self.slider, 1)
        row.addWidget(self.readout)
        self.slider.valueChanged.connect(self._on_change)

    def _to_pos(self, v: float) -> int:
        return int(round((v - self.lo) / (self.hi - self.lo) * self._n))

    def _from_pos(self, p: int) -> float:
        return self.lo + (self.hi - self.lo) * p / self._n

    def _fmt(self, v: float) -> str:
        return f"{v:.{self.decimals}f}{self.suffix}"

    def _on_change(self, pos: int) -> None:
        v = self._from_pos(pos)
        self.readout.setText(self._fmt(v))
        self.valueChanged.emit(v)

    def value(self) -> float:
        return self._from_pos(self.slider.value())

    def set_value(self, v: float) -> None:
        self.slider.setValue(self._to_pos(v))
