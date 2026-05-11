"""Reusable, self-contained Qt widgets used by the MedVision workspace."""

import numpy as np

from PyQt6.QtWidgets import QScrollArea, QLabel, QWidget
# Phase 2: pyqtSignal is needed so the Fourier spectrum canvas can report clicked pixels.
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QColor, QLinearGradient, QBrush
)

from ..DIP.utils import ensure_gray


# ─────────────────────────────────────────────────────────────────────────────
#  IMAGE CANVAS
# ─────────────────────────────────────────────────────────────────────────────

class ImageCanvas(QScrollArea):
    """
    Scrollable canvas that renders a numpy uint8 image array.
    Automatically scales the image to fill the available viewport while
    preserving the aspect ratio.
    """

    # Phase 2: emitted when the user clicks the Fourier spectrum image.
    image_clicked = pyqtSignal(int, int)  # row, col in original image-array coordinates

    def __init__(self, parent=None):
        super().__init__(parent)
        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet(
            "background: #080a0d; color: #3a4d64; font-size: 13px;"
        )
        self._label.setText("Open or drop an image")
        self.setWidget(self._label)
        self.setWidgetResizable(True)
        self.setStyleSheet("background: #080a0d; border: none;")
        self._array: np.ndarray | None = None
        self._zoom_percent = 100
        self._mode: str = "none"  # 'none', 'pan', 'annotate'
        self._annotations: list[tuple[float, float]] = []  # relative coords (x_ratio, y_ratio)
        self._pan_active = False
        self._pan_start = None
        self._h_start = 0
        self._v_start = 0

        # Phase 2: when True, mouse clicks are converted to image row/col coordinates.
        self._click_reporting = False

        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self._label.setMouseTracking(True)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_array(self, arr: np.ndarray) -> None:
        """Display a new numpy image array (grayscale or RGB, uint8)."""
        self._array = arr
        self._refresh()

    def get_array(self) -> np.ndarray | None:
        """Return the currently displayed array, or None."""
        return self._array

    def clear(self) -> None:
        self._array = None
        self._label.setPixmap(QPixmap())
        self._label.setText("Open or drop an image")

    def set_display_zoom(self, percent: int) -> None:
        self._zoom_percent = max(5, min(400, int(percent)))
        self._refresh()

    def fit_to_window(self) -> None:
        self.set_display_zoom(100)

    # Phase 2: enable/disable direct click selection on the Fourier spectrum.
    def set_click_reporting(self, enabled: bool) -> None:
        """Enable/disable emitting image_clicked(row, col) on left click."""
        self._click_reporting = bool(enabled)
        if enabled:
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    # Phase 2: remove notch markers from the Fourier spectrum canvas.
    def clear_markers(self) -> None:
        """Clear annotation markers painted on the canvas."""
        self._annotations.clear()
        self._refresh()

    # Phase 2: add a visible marker at an original image-array pixel position.
    def add_marker_at_pixel(self, row: int, col: int) -> None:
        """Add marker using original image pixel coordinates."""
        if self._array is None:
            return

        h, w = self._array.shape[:2]
        if h <= 0 or w <= 0:
            return

        rx = col / max(1, w - 1)
        ry = row / max(1, h - 1)
        self._annotations.append((rx, ry))
        self._refresh()

    # Phase 2: map mouse click on scaled Qt pixmap back to true array row/col.
    def _event_to_array_pixel(self, event) -> tuple[int, int] | None:
        """Map mouse click on scaled pixmap back to original array row/col."""
        if self._array is None:
            return None

        pix = self._label.pixmap()
        if pix is None or pix.isNull():
            return None

        viewport_pos = event.position().toPoint()
        label_pos = self._label.mapFrom(self.viewport(), viewport_pos)

        pix_w, pix_h = pix.width(), pix.height()
        offset_x = max(0, (self._label.width() - pix_w) // 2)
        offset_y = max(0, (self._label.height() - pix_h) // 2)

        x = label_pos.x() - offset_x
        y = label_pos.y() - offset_y

        if not (0 <= x < pix_w and 0 <= y < pix_h):
            return None

        arr_h, arr_w = self._array.shape[:2]
        col = int(round(x * (arr_w - 1) / max(1, pix_w - 1)))
        row = int(round(y * (arr_h - 1) / max(1, pix_h - 1)))

        return row, col

    # ── Internal ──────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        if self._array is None:
            self._label.setPixmap(QPixmap())
            self._label.setText("Open or drop an image")
            return

        arr = self._array
        if arr.ndim == 2:
            h, w = arr.shape
            qimg = QImage(
                arr.astype(np.uint8).tobytes(), w, h, w,
                QImage.Format.Format_Grayscale8
            )
        else:
            h, w = arr.shape[:2]
            arr3 = arr[:, :, :3].astype(np.uint8)
            qimg = QImage(arr3.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)

        pixmap  = QPixmap.fromImage(qimg)
        target_w = max(1, int(self.viewport().width() * self._zoom_percent / 100.0))
        target_h = max(1, int(self.viewport().height() * self._zoom_percent / 100.0))
        scaled  = pixmap.scaled(
            target_w,
            target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        # Paint annotations onto a copy of the scaled pixmap
        display = QPixmap(scaled)
        if self._annotations:
            painter = QPainter(display)
            painter.setPen(QColor("#ff4d6d"))
            painter.setBrush(QColor("#ff4d6d"))
            for rx, ry in self._annotations:
                x = int(rx * display.width())
                y = int(ry * display.height())
                painter.drawEllipse(x - 4, y - 4, 8, 8)
            painter.end()

        self._label.setText("")
        self._label.setPixmap(display)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    # ----------------- Interaction modes: pan & annotate -----------------
    def set_interaction_mode(self, mode: str) -> None:
        self._mode = mode or "none"
        if self._mode == "pan":
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        elif self._mode == "annotate":
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)

    def mousePressEvent(self, event):
        # Phase 2: Fourier spectrum click selection for notch filtering.
        if self._click_reporting and event.button() == Qt.MouseButton.LeftButton:
            pixel = self._event_to_array_pixel(event)
            if pixel is not None:
                row, col = pixel
                self.image_clicked.emit(row, col)
                return

        if self._mode == "pan" and event.button() == Qt.MouseButton.LeftButton:
            self._pan_active = True
            self._pan_start = event.position().toPoint()
            self._h_start = self.horizontalScrollBar().value()
            self._v_start = self.verticalScrollBar().value()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if self._mode == "annotate" and event.button() == Qt.MouseButton.LeftButton:
            # map viewport coords to label coordinates
            vp_pos = event.position().toPoint()
            label_pos = self._label.mapFrom(self.viewport(), vp_pos)
            pix = self._label.pixmap()
            if pix is None:
                return
            pw, ph = pix.width(), pix.height()
            # compute offset due to centering inside label
            ox = max(0, (self._label.width() - pw) // 2)
            oy = max(0, (self._label.height() - ph) // 2)
            x = label_pos.x() - ox
            y = label_pos.y() - oy
            if 0 <= x < pw and 0 <= y < ph:
                rx = x / pw
                ry = y / ph
                self._annotations.append((rx, ry))
                self._refresh()
            return

        return super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._mode == "pan" and self._pan_active:
            cur = event.position().toPoint()
            dx = cur.x() - self._pan_start.x()
            dy = cur.y() - self._pan_start.y()
            self.horizontalScrollBar().setValue(self._h_start - dx)
            self.verticalScrollBar().setValue(self._v_start - dy)
            return
        return super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._mode == "pan" and event.button() == Qt.MouseButton.LeftButton:
            self._pan_active = False
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            return
        return super().mouseReleaseEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
#  HISTOGRAM WIDGET
# ─────────────────────────────────────────────────────────────────────────────

class HistogramWidget(QWidget):
    """
    Custom-painted 256-bin luminance histogram with a blue gradient fill.
    Call set_array() to update the display.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hist: np.ndarray | None = None
        self.setMinimumHeight(80)
        self.setMaximumHeight(100)

    def set_array(self, arr: np.ndarray) -> None:
        """Compute and redraw the histogram for the given image array."""
        gray = ensure_gray(arr)
        hist = np.zeros(256, dtype=np.int64)
        for val in gray.flatten():
            hist[val] += 1
        self._hist = hist
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0f1117"))

        if self._hist is None:
            return

        w, h = self.width(), self.height()
        mx = self._hist.max()
        if mx == 0:
            return

        bar_w = w / 256.0
        grad  = QLinearGradient(0, h, 0, 0)
        grad.setColorAt(0.0, QColor("#00c8ff"))
        grad.setColorAt(1.0, QColor("#0057ff"))
        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.PenStyle.NoPen)

        for i in range(256):
            bh = int((self._hist[i] / mx) * (h - 4))
            x  = int(i * bar_w)
            painter.drawRect(x, h - bh, max(1, int(bar_w)), bh)