"""Top-level MedVision Workbench window built with PyQt6."""

from __future__ import annotations

import os

import numpy as np

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QStatusBar,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..DIP.edge_detection import prewitt_edge, sobel_edge
from ..DIP.histogram_equalization import local_histogram_equalization
from ..DIP.median import median_filter_scratch
from ..DIP.smoothing import apply_linear_filter, make_average_kernel, make_gaussian_kernel
from ..DIP.zoom import bilinear_zoom, nearest_neighbor_zoom
from ..DIP.utils import ensure_gray
from ..core.styles import (
    ACCENT,
    ACCENT_DIM,
    DARK_STYLE,
    GREEN,
    TEXT0,
    TEXT1,
    TEXT2,
    TEXT3,
    TOOLBAR_BTN_STYLE,
)
from ..io.image_io import load_image, save_image
from ..workers.processing_worker import ProcessingWorker
from .panels import MetadataPanel, PipelinePanel
from .sidebar import ToolsSidebar
from .widgets import HistogramWidget, ImageCanvas


class MainWindow(QMainWindow):
    """Main application shell with the HTML-inspired MedVision layout."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MedVision Workbench — Phase 1")
        self.resize(1450, 900)
        self.setStyleSheet(DARK_STYLE)

        self._original: np.ndarray | None = None
        self._current: np.ndarray | None = None
        self._history: list[np.ndarray] = []
        self._history_labels: list[str] = []
        self._redo_stack: list[np.ndarray] = []
        self._redo_labels: list[str] = []
        self._saved_state: np.ndarray | None = None
        self._accumulate: bool = True  # True: apply to current (accumulated); False: apply to original
        self._worker: ProcessingWorker | None = None
        self._current_path: str = ""
        self._loaded_metadata: dict[str, str] = {}

        self._zoom_base: np.ndarray | None = None
        self._zoom_factor: float = 1.0
        self._view_zoom_percent: int = 100

        self._build_actions()
        self.setMenuWidget(self._build_top_bar())
        self._build_central()
        self._build_statusbar()
        self._sync_view_zoom()
        self._set_status("Ready - Open an image to begin", False)

    # ------------------------------------------------------------------ UI --

    def _build_actions(self) -> None:
        self._open_action = QAction("Open", self)
        self._open_action.setShortcut("Ctrl+O")
        self._open_action.triggered.connect(self._open_file)

        self._save_action = QAction("Save", self)
        self._save_action.setShortcut("Ctrl+S")
        self._save_action.triggered.connect(self._save_file)

        self._undo_action = QAction("Undo", self)
        self._undo_action.setShortcut("Ctrl+Z")
        self._undo_action.triggered.connect(self._undo)

        self._redo_action = QAction("Redo", self)
        self._redo_action.setShortcut("Ctrl+Shift+Z")
        self._redo_action.triggered.connect(self._redo)
        self._redo_action.setEnabled(False)

        self._reset_action = QAction("Reset", self)
        self._reset_action.setShortcut("Ctrl+R")
        self._reset_action.triggered.connect(self._reset_to_original)

        self._about_action = QAction("About", self)
        self._about_action.triggered.connect(self._show_about)

        for action in [
            self._open_action,
            self._save_action,
            self._undo_action,
            self._redo_action,
            self._reset_action,
        ]:
            self.addAction(action)

    def _build_top_bar(self) -> QWidget:
        bar = QFrame()
        bar.setFixedHeight(38)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(6)

        logo = QLabel("MedVision <span style='color:%s'>Workbench</span>" % TEXT1)
        logo.setTextFormat(Qt.TextFormat.RichText)
        logo.setStyleSheet(
            f"font-size: 13px; font-weight: 600; color: {ACCENT}; padding: 0 10px 0 4px;"
            f"border-right: 1px solid {ACCENT_DIM};"
        )
        layout.addWidget(logo)

        for title in ["File", "Edit", "View", "Image", "Filters", "Help"]:
            layout.addWidget(self._build_menu_button(title))

        layout.addWidget(self._separator())

        for label, action in [
            ("Open", self._open_action),
            ("Save", self._save_action),
            ("Undo", self._undo_action),
            ("Redo", self._redo_action),
            ("Reset", self._reset_action),
        ]:
            layout.addWidget(self._toolbar_button(label, action))

        layout.addWidget(self._separator())
        layout.addStretch(1)
        return bar

    def _build_menu_button(self, title: str) -> QToolButton:
        button = QToolButton()
        button.setText(title)
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        button.setStyleSheet(
            f"QToolButton {{ padding: 5px 10px; border-radius: 4px; color: {TEXT1}; }}"
            f"QToolButton:hover {{ background: {ACCENT_DIM}; color: {TEXT0}; }}"
        )

        menu = QMenu(button)
        if title == "File":
            menu.addAction(self._open_action)
            menu.addAction(self._save_action)
            menu.addSeparator()
            menu.addAction(self._reset_action)
        elif title == "Edit":
            menu.addAction(self._undo_action)
            menu.addAction(self._redo_action)
        elif title == "View":
            for label, index in [("Single", 0), ("Before / After", 1), ("Edge View", 2)]:
                action = menu.addAction(label)
                action.triggered.connect(lambda _checked=False, i=index: self._switch_view(i))
        elif title == "Help":
            menu.addAction(self._about_action)
        else:
            disabled = menu.addAction("Coming soon")
            disabled.setEnabled(False)
        button.setMenu(menu)
        return button

    def _toolbar_button(self, label: str, action: QAction) -> QPushButton:
        button = QPushButton(label)
        button.setStyleSheet(TOOLBAR_BTN_STYLE)
        button.clicked.connect(action.trigger)
        return button

    @staticmethod
    def _separator() -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFixedHeight(20)
        sep.setStyleSheet(f"background: {ACCENT_DIM}; max-width: 1px;")
        return sep

    def _build_central(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._icon_panel = self._build_icon_panel()
        root.addWidget(self._icon_panel)

        self._sidebar = ToolsSidebar()
        self._sidebar.apply_filter.connect(self._on_apply_filter)
        self._sidebar.apply_zoom.connect(self._on_apply_zoom)
        self._sidebar.apply_edge.connect(self._on_apply_edge)
        self._sidebar.apply_hist_eq.connect(self._on_hist_eq)
        self._sidebar.apply_median.connect(self._on_median)
        self._sidebar.accumulate_toggled.connect(self._on_accumulate_toggled)
        root.addWidget(self._sidebar)

        self._canvas_area = self._build_canvas_area()
        root.addWidget(self._canvas_area, 1)

        self._right_panel = self._build_right_panel()
        root.addWidget(self._right_panel)

    def _build_icon_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(52)
        panel.setStyleSheet(f"background: #0a0c11; border-right: 1px solid {ACCENT_DIM};")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(2)

        self._icon_group = QButtonGroup(self)
        self._icon_group.setExclusive(True)
        icons = [
            ("✥", "Move / Pan"),
            ("▢", "Marquee Select"),
            ("✂", "Crop"),
            ("＋", "Zoom In"),
            ("－", "Zoom Out"),
            ("i", "Measure / Info"),
            ("◌", "Color Picker"),
            ("✎", "Annotations (Phase 2)"),
            ("⌖", "Ruler (Phase 2)"),
        ]

        for index, (symbol, tip) in enumerate(icons):
            if index in (3, 5, 7):
                sep = QFrame()
                sep.setFixedSize(28, 1)
                sep.setStyleSheet("background: #252d42;")
                layout.addWidget(sep)
            button = QToolButton()
            button.setText(symbol)
            button.setToolTip(tip)
            button.setCheckable(True)
            button.setAutoRaise(True)
            button.setStyleSheet(
                f"QToolButton {{ width: 36px; height: 36px; border-radius: 6px; color: {TEXT2};"
                f" font-size: 13px; background: transparent; border: 1px solid transparent; }}"
                f"QToolButton:hover {{ background: {ACCENT_DIM}; color: {ACCENT}; border-color: {ACCENT_DIM}; }}"
                f"QToolButton:checked {{ background: {ACCENT_DIM}; color: {ACCENT}; border-color: {ACCENT}; }}"
            )
            if index == 0:
                button.setChecked(True)
            button.clicked.connect(lambda _checked=False, t=tip, i=index: self._set_interaction_mode(i, t))
            self._icon_group.addButton(button)
            layout.addWidget(button, 0, Qt.AlignmentFlag.AlignHCenter)

        layout.addStretch(1)
        return panel

    def _set_interaction_mode(self, index: int, tip: str) -> None:
        """Set interaction mode based on the icon index and update canvases."""
        # map indices to modes: 0 -> pan, 7 -> annotate, others -> none
        if index == 0:
            mode = "pan"
        elif index == 7:
            mode = "annotate"
        else:
            mode = "none"

        for canvas in self._all_canvases():
            canvas.set_interaction_mode(mode)

        self._set_status(tip, False)

    def _build_canvas_area(self) -> QWidget:
        area = QWidget()
        area.setStyleSheet("background: #0c0d11;")
        layout = QVBoxLayout(area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._canvas_tabs = QTabWidget()
        self._canvas_tabs.currentChanged.connect(self._on_tab_changed)
        self._canvas_tabs.setDocumentMode(True)

        self._canvas_proc = ImageCanvas()
        self._canvas_tabs.addTab(self._canvas_proc, "Processed")
        self._canvas_tabs.addTab(self._build_split_tab(), "Before / After")
        self._canvas_tabs.addTab(self._build_edge_tab(), "Edge View")

        self._canvas_tabs.installEventFilter(self)
        self._canvas_proc.installEventFilter(self)

        layout.addWidget(self._canvas_tabs, 1)
        layout.addWidget(self._build_canvas_footer())
        return area

    def _build_split_tab(self) -> QWidget:
        widget = QWidget()
        widget.setStyleSheet("background: #0c0d11;")
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self._canvas_orig = ImageCanvas()
        self._canvas_proc2 = ImageCanvas()
        self._canvas_orig.installEventFilter(self)
        self._canvas_proc2.installEventFilter(self)

        for heading, subheading, canvas in [
            ("Original", "Source", self._canvas_orig),
            ("Processed", "Current", self._canvas_proc2),
        ]:
            pane = QWidget()
            pane.setStyleSheet(f"background: #080a0d; border: 1px solid {ACCENT_DIM};")
            pane_layout = QVBoxLayout(pane)
            pane_layout.setContentsMargins(0, 0, 0, 0)
            pane_layout.setSpacing(0)

            header = QLabel(f"{heading} <span style='color:{TEXT3};'>{subheading}</span>")
            header.setTextFormat(Qt.TextFormat.RichText)
            header.setStyleSheet(
                f"padding: 5px 10px; background: {TEXT0}; border-bottom: 1px solid {ACCENT_DIM};"
                f"font-size: 10px; color: {TEXT2};"
            )
            pane_layout.addWidget(header)
            pane_layout.addWidget(canvas, 1)
            layout.addWidget(pane)
        return widget

    def _build_edge_tab(self) -> QWidget:
        widget = QWidget()
        widget.setStyleSheet("background: #0c0d11;")
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self._canvas_gx = ImageCanvas()
        self._canvas_gy = ImageCanvas()
        self._canvas_edge = ImageCanvas()
        self._canvas_gx.installEventFilter(self)
        self._canvas_gy.installEventFilter(self)
        self._canvas_edge.installEventFilter(self)

        for heading, canvas in [
            ("Gx — Horizontal gradient", self._canvas_gx),
            ("Gy — Vertical gradient", self._canvas_gy),
            ("Magnitude — sqrt(Gx²+Gy²)", self._canvas_edge),
        ]:
            pane = QWidget()
            pane.setStyleSheet(f"background: #080a0d; border: 1px solid {ACCENT_DIM};")
            pane_layout = QVBoxLayout(pane)
            pane_layout.setContentsMargins(0, 0, 0, 0)
            pane_layout.setSpacing(0)

            header = QLabel(heading)
            header.setStyleSheet(
                f"padding: 5px 10px; background: {TEXT0}; border-bottom: 1px solid {ACCENT_DIM};"
                f"font-size: 10px; color: {TEXT2};"
            )
            pane_layout.addWidget(header)
            pane_layout.addWidget(canvas, 1)
            layout.addWidget(pane)
        return widget

    def _build_canvas_footer(self) -> QWidget:
        footer = QWidget()
        footer.setFixedHeight(26)
        footer.setStyleSheet("background: #11131a; border-top: 1px solid #252d42;")
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(8)

        minus = QPushButton("−")
        minus.setFixedSize(24, 18)
        minus.setStyleSheet(TOOLBAR_BTN_STYLE)
        minus.clicked.connect(lambda: self._change_view_zoom(-1))

        plus = QPushButton("+")
        plus.setFixedSize(24, 18)
        plus.setStyleSheet(TOOLBAR_BTN_STYLE)
        plus.clicked.connect(lambda: self._change_view_zoom(1))

        fit = QPushButton("Fit")
        fit.setStyleSheet(TOOLBAR_BTN_STYLE)
        fit.clicked.connect(self._fit_view)

        self._zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self._zoom_slider.setRange(5, 400)
        self._zoom_slider.setValue(100)
        self._zoom_slider.valueChanged.connect(self._update_view_zoom)
        self._zoom_slider.setFixedWidth(90)

        self._zoom_label = QLabel("100%")
        self._zoom_label.setStyleSheet(
            f"font-family: 'JetBrains Mono', monospace; color: {TEXT2}; min-width: 44px;"
        )

        self._canvas_info = QLabel("No image loaded")
        self._canvas_info.setStyleSheet(f"font-family: 'JetBrains Mono', monospace; color: {TEXT3};")

        layout.addWidget(minus)
        layout.addWidget(self._zoom_slider)
        layout.addWidget(plus)
        layout.addWidget(self._zoom_label)
        layout.addWidget(fit)
        layout.addStretch(1)
        layout.addWidget(self._canvas_info)
        return footer

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(240)
        panel.setStyleSheet("background: #11131a; border-left: 1px solid #252d42;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._right_tabs = QTabWidget()
        self._right_tabs.setDocumentMode(True)
        self._meta_panel = MetadataPanel()
        self._right_tabs.addTab(self._meta_panel, "Info")
        self._right_tabs.addTab(self._build_hist_tab(), "Histogram")

        self._pipeline = PipelinePanel()
        self._pipeline.undo_requested.connect(self._undo)
        self._pipeline.reset_requested.connect(self._reset_to_original)
        self._right_tabs.addTab(self._pipeline, "Pipeline")

        layout.addWidget(self._right_tabs)
        return panel

    def _build_hist_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("Luminance Histogram")
        title.setStyleSheet(
            f"color: {TEXT3}; font-size: 9.5px; text-transform: uppercase;"
            f"letter-spacing: 0.7px; font-weight: 600;"
        )
        layout.addWidget(title)

        self._hist_canvas = HistogramWidget()
        layout.addWidget(self._hist_canvas)

        self._hist_summary = QLabel("Pixels: —   Median: —   Entropy: —")
        self._hist_summary.setWordWrap(True)
        self._hist_summary.setStyleSheet(f"color: {TEXT2}; font-size: 10px; line-height: 1.5;")
        layout.addWidget(self._hist_summary)
        layout.addStretch(1)
        return widget

    def _build_statusbar(self) -> None:
        bar = QStatusBar()
        self.setStatusBar(bar)

        self._status_dot = QLabel()
        self._status_dot.setFixedSize(6, 6)
        self._status_dot.setStyleSheet(f"background: {TEXT3}; border-radius: 3px;")

        self._status_msg = QLabel("Ready - Open an image to begin")
        self._status_msg.setStyleSheet(f"color: {TEXT2};")

        self._cursor_pos = QLabel("—")
        self._cursor_pos.setStyleSheet(f"font-family: 'JetBrains Mono', monospace; color: {TEXT1};")
        self._status_zoom = QLabel("100%")
        self._status_zoom.setStyleSheet(f"font-family: 'JetBrains Mono', monospace; color: {TEXT1};")
        self._status_memory = QLabel("—")
        self._status_memory.setStyleSheet(f"font-family: 'JetBrains Mono', monospace; color: {TEXT1};")
        self._status_footer = QLabel("Team 8 - CUFE - BDE - DIP spring 26")
        self._status_footer.setStyleSheet(f"color: {TEXT3};")

        host = QWidget()
        host_layout = QHBoxLayout(host)
        host_layout.setContentsMargins(10, 0, 10, 0)
        host_layout.setSpacing(16)

        left = QWidget()
        left_layout = QHBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        left_layout.addWidget(self._status_dot)
        left_layout.addWidget(self._status_msg)

        right = QWidget()
        right_layout = QHBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        for label, widget in [
            ("Cursor:", self._cursor_pos),
            ("Zoom:", self._status_zoom),
            ("Memory:", self._status_memory),
        ]:
            segment = QWidget()
            segment_layout = QHBoxLayout(segment)
            segment_layout.setContentsMargins(0, 0, 0, 0)
            segment_layout.setSpacing(5)
            caption = QLabel(label)
            caption.setStyleSheet(f"color: {TEXT3};")
            segment_layout.addWidget(caption)
            segment_layout.addWidget(widget)
            right_layout.addWidget(segment)

        right_layout.addWidget(self._status_footer)

        host_layout.addWidget(left)
        host_layout.addStretch(1)
        host_layout.addWidget(right)

        bar.addPermanentWidget(host, 1)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setFixedSize(120, 10)
        self._progress.setRange(0, 0)
        bar.addPermanentWidget(self._progress)

    # ------------------------------------------------------------- helpers --

    def _set_status(self, msg: str, active: bool) -> None:
        self._status_msg.setText(msg)
        self._status_dot.setStyleSheet(
            f"background: {GREEN if active else TEXT3}; border-radius: 3px;"
        )

    def _sync_dirty_state(self) -> None:
        if self._current is None:
            self.setWindowModified(False)
            return

        is_dirty = self._saved_state is None or not np.array_equal(self._current, self._saved_state)
        self.setWindowModified(is_dirty)

    def _sync_view_zoom(self) -> None:
        for canvas in self._all_canvases():
            canvas.set_display_zoom(self._view_zoom_percent)
        self._zoom_slider.blockSignals(True)
        self._zoom_slider.setValue(self._view_zoom_percent)
        self._zoom_slider.blockSignals(False)
        self._zoom_label.setText(f"{self._view_zoom_percent}%")
        self._status_zoom.setText(f"{self._view_zoom_percent}%")

    def _all_canvases(self) -> list[ImageCanvas]:
        canvases = [self._canvas_proc]
        for canvas in [
            getattr(self, "_canvas_orig", None),
            getattr(self, "_canvas_proc2", None),
            getattr(self, "_canvas_gx", None),
            getattr(self, "_canvas_gy", None),
            getattr(self, "_canvas_edge", None),
        ]:
            if canvas is not None:
                canvases.append(canvas)
        return canvases

    def _update_stats_and_metadata(self, meta: dict | None = None, source_path: str = "") -> None:
        if self._current is None:
            self._meta_panel.clear()
            self._hist_summary.setText("Pixels: —   Median: —   Entropy: —")
            return

        gray = ensure_gray(self._current)
        pixels = int(gray.size)
        mn = int(gray.min())
        mx = int(gray.max())
        mean = float(gray.mean())
        std = float(gray.std())
        median = float(np.median(gray))
        hist = np.bincount(gray.flatten(), minlength=256)
        probabilities = hist[hist > 0] / pixels if pixels else np.array([])
        entropy = float(-(probabilities * np.log2(probabilities)).sum()) if probabilities.size else 0.0

        image_meta = dict(meta or {})
        image_meta.update(
            {
                "Min": str(mn),
                "Max": str(mx),
                "Mean": f"{mean:.1f}",
                "Std Dev": f"{std:.1f}",
            }
        )
        if source_path:
            image_meta.setdefault("File Size", self._format_file_size(source_path))
        self._meta_panel.set_metadata(image_meta)
        self._hist_canvas.set_array(self._current)
        self._hist_summary.setText(
            f"Pixels: {pixels:,}   Median: {median:.0f}   Entropy: {entropy:.2f}"
        )

    @staticmethod
    def _format_file_size(path: str) -> str:
        try:
            size = os.path.getsize(path)
        except OSError:
            return "—"
        units = ["B", "KB", "MB", "GB"]
        value = float(size)
        unit = 0
        while value >= 1024 and unit < len(units) - 1:
            value /= 1024
            unit += 1
        return f"{value:.0f} {units[unit]}" if unit == 0 else f"{value:.1f} {units[unit]}"

    # --------------------------------------------------------------- actions --

    def _switch_view(self, index: int) -> None:
        self._canvas_tabs.setCurrentIndex(index)

    def _update_view_zoom(self, value: int) -> None:
        self._view_zoom_percent = value
        self._sync_view_zoom()

    def _change_view_zoom(self, delta: int) -> None:
        steps = [5, 10, 25, 33, 50, 67, 75, 100, 125, 150, 200, 300, 400]
        current = self._zoom_slider.value()
        index = 0
        for i, value in enumerate(steps):
            if value >= current:
                index = i
                break
        index = max(0, min(len(steps) - 1, index + delta))
        self._zoom_slider.setValue(steps[index])

    def _fit_view(self) -> None:
        self._zoom_slider.setValue(100)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.MouseMove and watched in {
            getattr(self, "_canvas_tabs", None),
            getattr(self, "_canvas_proc", None),
            getattr(self, "_canvas_orig", None),
            getattr(self, "_canvas_proc2", None),
            getattr(self, "_canvas_gx", None),
            getattr(self, "_canvas_gy", None),
            getattr(self, "_canvas_edge", None),
        }:
            pos = event.position().toPoint()
            self._cursor_pos.setText(f"{pos.x()}, {pos.y()}")
        return super().eventFilter(watched, event)

    # --------------------------------------------------------- file actions --

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Medical Image",
            "",
            "All Images (*.dcm *.dicom *.jpg *.jpeg *.bmp *.png *.tif *.tiff);;"
            "DICOM (*.dcm *.dicom);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;All Files (*)",
        )
        if not path:
            return

        try:
            result = load_image(path)
            if result.error:
                QMessageBox.critical(self, "Load Error", result.error)
                return
            if result.pixel_array is None:
                QMessageBox.critical(self, "Load Error", "Could not decode image pixels.")
                return

            self._current_path = path
            self._loaded_metadata = dict(result.metadata)
            self._original = result.pixel_array.copy()
            self._current = result.pixel_array.copy()
            self._history.clear()
            self._history_labels.clear()
            self._pipeline.clear()
            self._redo_stack.clear()
            self._redo_labels.clear()
            self._saved_state = self._current.copy()
            self._zoom_base = None
            self._zoom_factor = 1.0
            self._view_zoom_percent = 100

            self._update_canvases()
            self._update_stats_and_metadata(self._loaded_metadata, path)
            self._sync_view_zoom()
            self._canvas_info.setText(
                f"{result.metadata.get('Width', '?')} × {result.metadata.get('Height', '?')}  |  "
                f"{result.metadata.get('Mode', 'Gray')}  |  {result.format or 'Image'}"
            )
            self._set_status(
                f"Loaded: {os.path.basename(path)}  ({result.metadata.get('Width', '?')} × {result.metadata.get('Height', '?')})  Format: {result.format}",
                True,
            )
            self._sync_dirty_state()
        except Exception as exc:
            QMessageBox.critical(self, "Unexpected Error", str(exc))

    def _save_file(self):
        if self._current is None:
            QMessageBox.information(self, "No Image", "No processed image to save.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Processed Image",
            "",
            "PNG Image (*.png);;JPEG Image (*.jpg);;BMP Image (*.bmp)",
        )
        if not path:
            return

        err = save_image(self._current, path)
        if err:
            QMessageBox.critical(self, "Save Error", err)
        else:
            self._saved_state = self._current.copy()
            self._sync_dirty_state()
            self._set_status(f"Saved: {os.path.basename(path)}", True)

    # ------------------------------------------------------------- state ----

    def _require_image(self) -> bool:
        if self._current is None:
            QMessageBox.information(self, "No Image", "Please open an image first.")
            return False
        return True

    def _push_history(self, label: str) -> None:
        if self._current is not None:
            self._history.append(self._current.copy())
            self._history_labels.append(label)
            # New operation invalidates the redo stack
            self._redo_stack.clear()
            self._redo_labels.clear()
            self._redo_action.setEnabled(False)

    def _undo(self):
        if not self._history:
            self._set_status("Nothing to undo.", False)
            return
        # move current state to redo stack, restore previous
        if self._current is not None:
            self._redo_stack.append(self._current.copy())
            # label for redo describes what will be redone
            self._redo_labels.append(self._history_labels[-1] if self._history_labels else "?")
            self._redo_action.setEnabled(True)

        self._current = self._history.pop()
        label = self._history_labels.pop() if self._history_labels else "?"
        self._pipeline.remove_last()
        self._zoom_base = None
        self._zoom_factor = 1.0
        self._update_canvases()
        self._update_stats_and_metadata(self._loaded_metadata, self._current_path)
        self._sync_view_zoom()
        self._sync_dirty_state()
        self._set_status(f"Undone: {label}", False)

    def _redo(self):
        if not self._redo_stack:
            self._set_status("Nothing to redo.", False)
            return
        # push current to history, restore top of redo stack
        if self._current is not None:
            self._history.append(self._current.copy())
            self._history_labels.append(self._redo_labels[-1] if self._redo_labels else "?")

        self._current = self._redo_stack.pop()
        redo_label = self._redo_labels.pop() if self._redo_labels else "?"
        # update pipeline UI to reflect redo (append step)
        self._pipeline.add_step(redo_label)
        if not self._redo_stack:
            self._redo_action.setEnabled(False)

        self._zoom_base = None
        self._zoom_factor = 1.0
        self._update_canvases()
        self._update_stats_and_metadata(self._loaded_metadata, self._current_path)
        self._sync_view_zoom()
        self._sync_dirty_state()
        self._set_status(f"Redone: {redo_label}", False)

    def _reset_to_original(self):
        if self._original is None:
            return
        self._current = self._original.copy()
        self._history.clear()
        self._history_labels.clear()
        self._pipeline.clear()
        self._zoom_base = None
        self._zoom_factor = 1.0
        self._update_canvases()
        self._update_stats_and_metadata(self._loaded_metadata, self._current_path)
        self._sync_view_zoom()
        self._sync_dirty_state()
        self._set_status("Reset to original.", False)

    def _update_canvases(self):
        if self._current is None:
            return
        self._canvas_proc.set_array(self._current)
        self._canvas_proc2.set_array(self._current)
        if self._original is not None:
            self._canvas_orig.set_array(self._original)

    def _on_tab_changed(self, _index: int):
        self._update_canvases()

    # --------------------------------------------------------- processing --

    def _start_worker(self, func, label: str, *args, **kwargs):
        if self._worker and self._worker.isRunning():
            self._set_status("Processing in progress - please wait…", True)
            return
        self._push_history(label)
        self._progress.setVisible(True)
        self._worker = ProcessingWorker(func, label, *args, **kwargs)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()
        self._set_status(f"Processing: {label}…", True)

    def _on_worker_finished(self, result: np.ndarray, label: str):
        self._current = result
        self._pipeline.add_step(label)
        if "Zoom" not in label:
            self._zoom_base = None
            self._zoom_factor = 1.0
        self._update_canvases()
        self._update_stats_and_metadata(self._loaded_metadata, self._current_path)
        self._progress.setVisible(False)
        self._sync_dirty_state()
        self._set_status(f"Done: {label}  -  {result.shape[1]}×{result.shape[0]} px", True)

    def _on_worker_error(self, msg: str):
        self._progress.setVisible(False)
        if self._history:
            self._current = self._history.pop()
            if self._history_labels:
                self._history_labels.pop()
        self._sync_dirty_state()
        QMessageBox.critical(self, "Processing Error", msg)
        self._set_status("Error during processing.", False)

    def closeEvent(self, event):
        if self.isWindowModified() and self._current is not None:
            choice = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save before closing?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )

            if choice == QMessageBox.StandardButton.Save:
                if not self._save_before_close():
                    event.ignore()
                    return
            elif choice == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return

        event.accept()

    def _save_before_close(self) -> bool:
        if self._current is None:
            return True

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Processed Image",
            self._current_path or "",
            "PNG Image (*.png);;JPEG Image (*.jpg);;BMP Image (*.bmp)",
        )
        if not path:
            return False

        err = save_image(self._current, path)
        if err:
            QMessageBox.critical(self, "Save Error", err)
            return False

        self._saved_state = self._current.copy()
        self._sync_dirty_state()
        self._set_status(f"Saved: {os.path.basename(path)}", True)
        return True

    def _on_accumulate_toggled(self, checked: bool) -> None:
        """Toggle between accumulate (current) and original-based processing."""
        self._accumulate = checked
        mode_text = "Accumulating" if checked else "From Original"
        self._set_status(f"Mode: {mode_text}", False)

    def _on_apply_zoom(self, step: float, method: str):
        if not self._require_image():
            return

        if self._zoom_base is None:
            self._zoom_base = self._current.copy()
            self._zoom_factor = 1.0

        new_factor = max(0.05, min(16.0, self._zoom_factor * step))
        if new_factor == self._zoom_factor:
            self._set_status("Zoom limit reached.", False)
            return

        self._zoom_factor = new_factor
        direction = "In" if step >= 1.0 else "Out"
        mode_mark = "(acc)" if self._accumulate else "(orig)"
        label = f"Zoom {direction} ({method.split('-')[0]}) — {self._zoom_factor:.2f}× {mode_mark}"

        zoom_func = bilinear_zoom if method == "Bilinear" else nearest_neighbor_zoom
        self._start_worker(zoom_func, label, self._zoom_base.copy(), self._zoom_factor)

    def _on_apply_filter(self, filter_name: str, params: dict):
        if not self._require_image():
            return

        kernel_size = params.get("kernel_size", 3)
        if filter_name == "average":
            kernel = make_average_kernel(kernel_size)
            label = f"Average Filter {kernel_size}×{kernel_size}"
        elif filter_name == "gaussian":
            sigma = params.get("sigma", 1.0)
            kernel = make_gaussian_kernel(kernel_size, sigma)
            label = f"Gaussian Filter {kernel_size}×{kernel_size}  σ={sigma:.1f}"
        else:
            return

        mode_mark = "(acc)" if self._accumulate else "(orig)"
        label = f"{label} {mode_mark}"
        base_image = self._current if self._accumulate else self._original
        self._start_worker(apply_linear_filter, label, base_image, kernel)

    def _on_apply_edge(self, operator: str, component: str):
        if not self._require_image():
            return

        base_image = self._current if self._accumulate else self._original
        gray = ensure_gray(base_image)
        if operator == "Sobel":
            gx, gy, magnitude = sobel_edge(gray)
        else:
            gx, gy, magnitude = prewitt_edge(gray)

        self._canvas_gx.set_array(gx)
        self._canvas_gy.set_array(gy)
        self._canvas_edge.set_array(magnitude)
        self._switch_view(2)

        result = {"Magnitude": magnitude, "Horizontal (Gx)": gx, "Vertical (Gy)": gy}.get(
            component, magnitude
        )
        mode_mark = "(acc)" if self._accumulate else "(orig)"
        label = f"{operator} Edge ({component}) {mode_mark}"

        self._push_history(label)
        self._current = result
        self._zoom_base = None
        self._zoom_factor = 1.0
        self._pipeline.add_step(label)
        self._update_stats_and_metadata(self._loaded_metadata, self._current_path)
        self._update_canvases()
        self._set_status(f"Done: {label}", True)

    def _on_hist_eq(self, block_size: int):
        if not self._require_image():
            return
        base_image = self._current if self._accumulate else self._original
        gray = ensure_gray(base_image)
        mode_mark = "(acc)" if self._accumulate else "(orig)"
        label = f"Local Hist. Eq. {block_size}×{block_size} {mode_mark}"
        self._start_worker(local_histogram_equalization, label, gray, block_size)

    def _on_median(self, kernel_size: int):
        if not self._require_image():
            return
        base_image = self._current if self._accumulate else self._original
        mode_mark = "(acc)" if self._accumulate else "(orig)"
        label = f"Median Filter {kernel_size}×{kernel_size} {mode_mark}"
        self._start_worker(median_filter_scratch, label, base_image, kernel_size)

    # --------------------------------------------------------------- about --

    def _show_about(self):
        QMessageBox.about(
            self,
            "About MedVision Workbench",
            "<b style='font-size:14px'>MedVision Workbench</b><br>"
            "<i>Phase 1 — Spatial Domain Operations</i><br><br>"
            "• Multi-format I/O: DICOM, JPEG, BMP, PNG<br>"
            "• Custom 2-D convolution (from scratch)<br>"
            "• Nearest-Neighbor &amp; Bilinear zoom (from scratch)<br>"
            "• Average, Gaussian, Median filtering (from scratch)<br>"
            "• Sobel &amp; Prewitt edge detection (from scratch)<br>"
            "• Local Histogram Equalization (from scratch)<br>"
            "• Sequential Enhancement Pipeline with undo<br><br>"
            "Team 8 - CUFE - BDE - DIP spring 26",
        )
