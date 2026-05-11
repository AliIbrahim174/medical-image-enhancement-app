"""Reusable, self-contained Qt widgets used by the MedVision workspace."""

import numpy as np

from PyQt6.QtWidgets import QScrollArea, QLabel, QWidget
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import (
    QImage,
    QPixmap,
    QPainter,
    QPen,
    QColor,
    QLinearGradient,
    QBrush,
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

    # Phase 2 / Member 2: emitted when the user clicks the Fourier spectrum image.
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
        self._mode: str = "none"  # 'none', 'pan', 'marker', 'eraser', 'ruler'
        
        # Annotation strokes: each stroke is a dict with 'points', 'color', 'width', 'type'
        self._strokes: list[dict] = []
        self._current_stroke: list[tuple[float, float]] = []  # In-progress stroke (relative coords)
        
        # Drawing settings
        self._pen_color = "#ff4d6d"  # Default red
        self._pen_width = 2
        self._eraser_width = 15

        self._pan_active = False
        self._pan_start = None
        self._h_start = 0
        self._v_start = 0

        # Phase 2 / Member 2: when True, mouse clicks become image row/col coordinates.
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
        self._strokes.clear()
        self._current_stroke.clear()
        self._label.setPixmap(QPixmap())
        self._label.setText("Open or drop an image")

    def set_display_zoom(self, percent: int) -> None:
        self._zoom_percent = max(5, min(400, int(percent)))
        self._refresh()

    def fit_to_window(self) -> None:
        self.set_display_zoom(100)

    # Phase 2 / Member 2: enable/disable direct click selection on the Fourier spectrum.
    def set_click_reporting(self, enabled: bool) -> None:
        """Enable/disable emitting image_clicked(row, col) on left click."""
        self._click_reporting = bool(enabled)
        if enabled:
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        elif self._mode == "none":
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)

    # Phase 2 / Member 2: remove notch markers from the Fourier spectrum canvas.
    def clear_markers(self) -> None:
        """Clear all annotation strokes."""
        self._strokes.clear()
        self._current_stroke.clear()
        self._refresh()

    # Phase 2 / Member 2: add a visible marker at an original image-array pixel position.
    def add_marker_at_pixel(self, row: int, col: int) -> None:
        """Add marker (small circle) using original image pixel coordinates."""
        if self._array is None:
            return

        h, w = self._array.shape[:2]
        if h <= 0 or w <= 0:
            return

        rx = col / max(1, w - 1)
        ry = row / max(1, h - 1)
        
        # Add as a single-point stroke (marker)
        self._strokes.append({
            "points": [(rx, ry)],
            "color": "#ff4d6d",
            "width": 0,  # 0 width means draw as circle marker
            "type": "marker"
        })
        self._refresh()

    # Drawing control API
    def set_pen_color(self, color_hex: str) -> None:
        """Set the pen/marker color (e.g., '#ff4d6d')."""
        self._pen_color = color_hex

    def set_pen_width(self, width: int) -> None:
        """Set the pen stroke width in pixels."""
        self._pen_width = max(1, int(width))

    def clear_strokes(self) -> None:
        """Clear all drawn strokes."""
        self._strokes.clear()
        self._current_stroke.clear()
        self._refresh()

    # Phase 2 / Members 2 and 3: map mouse click on scaled Qt pixmap to true array row/col.
    def _event_to_array_pixel(self, event) -> tuple[int, int] | None:
        """Map mouse event on scaled pixmap back to original array row/col."""
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

        row = max(0, min(arr_h - 1, row))
        col = max(0, min(arr_w - 1, col))
        return row, col

    # ── Internal ──────────────────────────────────────────────────────────────

    def _array_to_qimage(self, arr: np.ndarray) -> QImage:
        """Convert a grayscale/RGB numpy array to QImage."""
        if arr.ndim == 2:
            h, w = arr.shape
            arr8 = arr.astype(np.uint8)
            return QImage(
                arr8.tobytes(),
                w,
                h,
                w,
                QImage.Format.Format_Grayscale8,
            ).copy()

        h, w = arr.shape[:2]
        arr3 = arr[:, :, :3].astype(np.uint8)
        return QImage(
            arr3.tobytes(),
            w,
            h,
            w * 3,
            QImage.Format.Format_RGB888,
        ).copy()

    def _draw_overlays(self, display: QPixmap) -> None:
        """Hook for subclasses to draw additional overlays on the scaled pixmap."""
        return

    def _refresh(self) -> None:
        if self._array is None:
            self._label.setPixmap(QPixmap())
            self._label.setText("Open or drop an image")
            return

        qimg = self._array_to_qimage(self._array)
        pixmap = QPixmap.fromImage(qimg)

        target_w = max(1, int(self.viewport().width() * self._zoom_percent / 100.0))
        target_h = max(1, int(self.viewport().height() * self._zoom_percent / 100.0))
        scaled = pixmap.scaled(
            target_w,
            target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        # Paint annotations and subclass overlays onto a copy of the scaled pixmap.
        display = QPixmap(scaled)

        if self._strokes or self._current_stroke:
            painter = QPainter(display)
            
            # Draw saved strokes (only pen and marker types, skip eraser)
            for stroke in self._strokes:
                if stroke["type"] == "marker":
                    # Draw as small circle
                    painter.setPen(QColor(stroke["color"]))
                    painter.setBrush(QColor(stroke["color"]))
                    for rx, ry in stroke["points"]:
                        x = int(rx * display.width())
                        y = int(ry * display.height())
                        painter.drawEllipse(x - 4, y - 4, 8, 8)
                elif stroke["type"] != "eraser":  # Skip eraser strokes - don't render them
                    # Pen stroke
                    painter.setPen(QPen(QColor(stroke["color"]), stroke["width"], Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
                    if len(stroke["points"]) > 1:
                        for i in range(len(stroke["points"]) - 1):
                            rx1, ry1 = stroke["points"][i]
                            rx2, ry2 = stroke["points"][i + 1]
                            x1 = int(rx1 * display.width())
                            y1 = int(ry1 * display.height())
                            x2 = int(rx2 * display.width())
                            y2 = int(ry2 * display.height())
                            painter.drawLine(x1, y1, x2, y2)
            
            # Draw current in-progress stroke
            if self._current_stroke and self._mode != "eraser":
                # Only draw if NOT eraser mode
                painter.setPen(QPen(QColor(self._pen_color), self._pen_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
                
                if len(self._current_stroke) > 1:
                    for i in range(len(self._current_stroke) - 1):
                        rx1, ry1 = self._current_stroke[i]
                        rx2, ry2 = self._current_stroke[i + 1]
                        x1 = int(rx1 * display.width())
                        y1 = int(ry1 * display.height())
                        x2 = int(rx2 * display.width())
                        y2 = int(ry2 * display.height())
                        painter.drawLine(x1, y1, x2, y2)
            
            painter.end()

        self._draw_overlays(display)

        self._label.setText("")
        self._label.setPixmap(display)

    def _stroke_intersects_point(self, stroke: dict, px: float, py: float, threshold: float = 0.05) -> bool:
        """Check if a stroke passes near a point. px, py in normalized coords (0-1)."""
        for sx, sy in stroke["points"]:
            dist_x = abs(sx - px)
            dist_y = abs(sy - py)
            if dist_x < threshold and dist_y < threshold:
                return True
        return False

    def _apply_eraser_stroke(self, eraser_stroke: dict) -> None:
        """Remove pen strokes that intersect with the eraser stroke."""
        if not eraser_stroke["points"]:
            return
        
        threshold = 0.08  # Proximity threshold for intersection detection (increased for better erasing)
        remaining_strokes = []
        
        for stroke in self._strokes:
            if stroke["type"] == "eraser":
                # Keep eraser strokes (for now, though they won't be rendered)
                remaining_strokes.append(stroke)
            else:
                # Check if this pen stroke intersects with eraser
                intersects = False
                for ex, ey in eraser_stroke["points"]:
                    for sx, sy in stroke["points"]:
                        dist = ((sx - ex) ** 2 + (sy - ey) ** 2) ** 0.5
                        if dist < threshold:
                            intersects = True
                            break
                    if intersects:
                        break
                
                # Only keep stroke if it doesn't intersect with eraser
                if not intersects:
                    remaining_strokes.append(stroke)
        
        self._strokes = remaining_strokes

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    # ----------------- Interaction modes: pan & annotate -----------------

    def set_interaction_mode(self, mode: str) -> None:
        self._mode = mode or "none"

        if self._click_reporting:
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        elif self._mode == "pan":
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        elif self._mode == "marker":
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        elif self._mode == "eraser":
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        elif self._mode == "ruler":
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)

    def mousePressEvent(self, event):
        # Phase 2 / Member 2: Fourier spectrum click selection for notch filtering.
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

        if self._mode in ("marker", "eraser") and event.button() == Qt.MouseButton.LeftButton:
            pixel = self._event_to_array_pixel(event)
            if pixel is not None and self._array is not None:
                row, col = pixel
                h, w = self._array.shape[:2]
                rx = col / max(1, w - 1)
                ry = row / max(1, h - 1)
                self._current_stroke = [(rx, ry)]
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

        # Handle drawing/eraser strokes
        if self._mode in ("marker", "eraser") and self._current_stroke and self._array is not None:
            pixel = self._event_to_array_pixel(event)
            if pixel is not None:
                row, col = pixel
                h, w = self._array.shape[:2]
                rx = col / max(1, w - 1)
                ry = row / max(1, h - 1)
                self._current_stroke.append((rx, ry))
                self._refresh()
            return

        return super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._mode == "pan" and event.button() == Qt.MouseButton.LeftButton:
            self._pan_active = False
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            return

        # Finalize drawing/eraser stroke
        if self._mode in ("marker", "eraser") and event.button() == Qt.MouseButton.LeftButton:
            if self._current_stroke:
                stroke_type = "eraser" if self._mode == "eraser" else "pen"
                eraser_stroke = {
                    "points": self._current_stroke[:],
                    "color": self._pen_color,
                    "width": self._eraser_width if self._mode == "eraser" else self._pen_width,
                    "type": stroke_type
                }
                
                # If eraser, apply the eraser effect to remove intersecting pen strokes
                if self._mode == "eraser":
                    self._apply_eraser_stroke(eraser_stroke)
                    # Don't store eraser strokes - just use them to erase pen strokes
                else:
                    # Store pen strokes normally
                    self._strokes.append(eraser_stroke)
                
                self._current_stroke.clear()
                self._refresh()
            return

        return super().mouseReleaseEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
#  ROI IMAGE CANVAS
# ─────────────────────────────────────────────────────────────────────────────

class ROIImageCanvas(ImageCanvas):
    """ImageCanvas subclass that lets the user drag a rectangular ROI."""

    # Phase 2 / Member 3: x1, y1, x2, y2 in original image pixel coordinates.
    roi_selected = pyqtSignal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._roi_mode = False
        self._roi_start_pixel: tuple[int, int] | None = None  # row, col
        self._roi_end_pixel: tuple[int, int] | None = None    # row, col

    def set_roi_mode(self, active: bool) -> None:
        """Enable/disable ROI drag selection."""
        self._roi_mode = bool(active)
        self._roi_start_pixel = None
        self._roi_end_pixel = None
        self.viewport().setCursor(
            Qt.CursorShape.CrossCursor if self._roi_mode else Qt.CursorShape.ArrowCursor
        )
        self._refresh()

    def _draw_overlays(self, display: QPixmap) -> None:
        """Draw the current ROI rectangle on the scaled display pixmap."""
        if self._roi_start_pixel is None or self._roi_end_pixel is None:
            return
        if self._array is None:
            return

        h, w = self._array.shape[:2]
        if h <= 0 or w <= 0:
            return

        r1, c1 = self._roi_start_pixel
        r2, c2 = self._roi_end_pixel

        x1 = int(c1 * (display.width() - 1) / max(1, w - 1))
        y1 = int(r1 * (display.height() - 1) / max(1, h - 1))
        x2 = int(c2 * (display.width() - 1) / max(1, w - 1))
        y2 = int(r2 * (display.height() - 1) / max(1, h - 1))

        painter = QPainter(display)
        pen = QPen(QColor("#00c8ff"), 2)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        painter.end()

    def mousePressEvent(self, event):
        if self._roi_mode and event.button() == Qt.MouseButton.LeftButton:
            pixel = self._event_to_array_pixel(event)
            if pixel is not None:
                self._roi_start_pixel = pixel
                self._roi_end_pixel = pixel
                self._refresh()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._roi_mode and self._roi_start_pixel is not None:
            pixel = self._event_to_array_pixel(event)
            if pixel is not None:
                self._roi_end_pixel = pixel
                self._refresh()
                return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._roi_mode and self._roi_start_pixel is not None:
            pixel = self._event_to_array_pixel(event)
            if pixel is not None:
                self._roi_end_pixel = pixel

            if self._roi_end_pixel is not None:
                r1, c1 = self._roi_start_pixel
                r2, c2 = self._roi_end_pixel

                x1 = min(c1, c2)
                y1 = min(r1, r2)
                x2 = max(c1, c2)
                y2 = max(r1, r2)

                # Ignore tiny accidental clicks.
                if (x2 - x1) >= 2 and (y2 - y1) >= 2:
                    self.roi_selected.emit(x1, y1, x2, y2)

            self._roi_mode = False
            self._roi_start_pixel = None
            self._roi_end_pixel = None
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
            self._refresh()
            return

        super().mouseReleaseEvent(event)


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

    # Phase 2 / Member 3: directly display a precomputed ROI histogram.
    def set_hist(self, hist: np.ndarray) -> None:
        """Set a precomputed 256-bin histogram and redraw."""
        hist = np.asarray(hist, dtype=np.int64)

        if hist.shape[0] != 256:
            raise ValueError("HistogramWidget.set_hist expects a 256-bin histogram.")

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
        grad = QLinearGradient(0, h, 0, 0)
        grad.setColorAt(0.0, QColor("#00c8ff"))
        grad.setColorAt(1.0, QColor("#0057ff"))
        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.PenStyle.NoPen)

        for i in range(256):
            bh = int((self._hist[i] / mx) * (h - 4))
            x = int(i * bar_w)
            painter.drawRect(x, h - bh, max(1, int(bar_w)), bh)
