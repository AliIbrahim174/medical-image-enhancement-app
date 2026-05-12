# -*- coding: utf-8 -*-
"""Main application window - orchestrates UI, state, and event handling."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QWidget, QHBoxLayout

from ..core.styles import DARK_STYLE, GREEN, TEXT3
from ..DIP.frequency_domain import shifted_magnitude_spectrum
from ..DIP.utils import ensure_gray
from ..workers.processing_worker import ProcessingWorker
from .file_operations import FileOperations
from .signal_handlers import SignalHandlers
from .state_manager import ImageStateManager
from .ui_builder import UIBuilder


class MainWindow(QMainWindow):
    """Main application shell - clean architecture with separated concerns."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MedVision Workbench � Phase 2")
        self.resize(1450, 900)
        self.setStyleSheet(DARK_STYLE)

        # Initialize components
        self.state = ImageStateManager()
        self.file_ops = FileOperations()
        self.ui = UIBuilder(self)
        self.handlers = SignalHandlers(self)
        self.current_tool_mode = "pan"  # Track active tool mode

        self._worker: ProcessingWorker | None = None
        self._current_path: str = ""
        self._loaded_metadata: dict[str, str] = {}

        self._build_ui()
        self._connect_signals()

    # ========== UI Construction ==========

    def _build_ui(self) -> None:
        """Build the complete UI."""
        # Build actions first
        self.ui.build_actions()

        # Build top bar
        self.setMenuWidget(self.ui.build_top_bar())

        # Build central widget
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Icon panel
        icon_panel, self._button_data = self.ui.build_icon_panel()
        root.addWidget(icon_panel)

        # Sidebar
        self.sidebar = self.ui.build_sidebar()
        root.addWidget(self.sidebar)

        # Canvas area
        canvas_area, fourier_data = self.ui.build_canvas_area()
        self.ui.panels["canvas_tabs"] = fourier_data["tabs"]
        self._fourier_tab = fourier_data["main"]
        root.addWidget(canvas_area, 1)

        # Right panel
        right_panel = self.ui.build_right_panel()
        root.addWidget(right_panel)

        # Status bar
        statusbar = self.ui.build_statusbar()
        self.setStatusBar(statusbar)

        self.sync_view_zoom()
        self.set_status("Ready - Open an image to begin", False)

    def _connect_signals(self) -> None:
        """Connect all signals to handlers."""
        # Actions
        self.ui.actions["open"].triggered.connect(self._on_open_file)
        self.ui.actions["save"].triggered.connect(self._on_save_file)
        self.ui.actions["undo"].triggered.connect(self._on_undo)
        self.ui.actions["redo"].triggered.connect(self._on_redo)
        self.ui.actions["reset"].triggered.connect(self._on_reset)
        self.ui.actions["about"].triggered.connect(self._on_about)

        # Icon panel - connect each button with its index
        for button, (index, tip) in self._button_data.items():
            button.clicked.connect(
                lambda checked=False, i=index, t=tip: self.handlers.on_set_interaction_mode(i, t)
            )

        # View menu - connect each action to switch canvas view
        for action, index in self.ui.view_menu_actions:
            action.triggered.connect(
                lambda checked=False, i=index: self.handlers.on_switch_view(i)
            )

        # Canvas tabs
        self.ui.panels["canvas_tabs"].currentChanged.connect(self.handlers.on_tab_changed)

        # Zoom controls
        self.ui.sliders["zoom"].valueChanged.connect(self.handlers.on_update_view_zoom)

        # Sidebar signals
        self.sidebar.apply_filter.connect(self.handlers.on_apply_filter)
        self.sidebar.apply_zoom.connect(self.handlers.on_apply_zoom)
        self.sidebar.apply_edge.connect(self.handlers.on_apply_edge)
        self.sidebar.apply_hist_eq.connect(self.handlers.on_hist_eq)
        self.sidebar.apply_median.connect(self.handlers.on_median)
        self.sidebar.apply_noise.connect(self.handlers.on_noise)
        self.sidebar.accumulate_toggled.connect(self.handlers.on_accumulate_toggled)
        self.sidebar.snap_toggled.connect(self.handlers.on_snap_toggled)
        self.sidebar.show_spectrum.connect(self.handlers.on_show_spectrum)
        self.sidebar.apply_notch.connect(self.handlers.on_apply_notch)
        self.sidebar.apply_threshold.connect(self.handlers.on_threshold)
        self.sidebar.apply_morphology.connect(self.handlers.on_morphology)

        # ROI
        if hasattr(self.sidebar, "roi_btn"):
            self.sidebar.roi_btn.toggled.connect(self.ui.canvases["proc"].set_roi_mode)
        self.ui.canvases["proc"].roi_selected.connect(self.handlers.on_roi_selected)

        # Spectrum canvas
        self.ui.canvases["spectrum"].image_clicked.connect(self.handlers.on_spectrum_clicked)

        # Right panel
        self.ui.panels["pipeline"].undo_requested.connect(self._on_undo)
        self.ui.panels["pipeline"].reset_requested.connect(self._on_reset)

        # Install event filters for mouse tracking
        self.ui.panels["canvas_tabs"].installEventFilter(self)
        self.ui.canvases["proc"].installEventFilter(self)
        for canvas_key in ["orig", "proc2", "gx", "gy", "edge", "spectrum"]:
            if canvas_key in self.ui.canvases:
                self.ui.canvases[canvas_key].installEventFilter(self)

    # ========== File Operations ==========

    def _on_open_file(self) -> None:
        """Handle file open dialog."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Medical Image",
            "",
            "All Images (*.dcm *.dicom *.jpg *.jpeg *.bmp *.png *.tif *.tiff);;"
            "DICOM (*.dcm *.dicom);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;All Files (*)",
        )
        if not path:
            return

        result = self.file_ops.load_image_file(path)
        if not result["success"]:
            QMessageBox.critical(self, "Load Error", result["error"])
            return

        self._current_path = path
        self._loaded_metadata = result["metadata"]
        self.state.load_image(result["image"])

        self.update_canvases()
        self.update_stats_and_metadata()
        self.sync_view_zoom()

        self.ui.labels["canvas_info"].setText(
            f"{result['width']} � {result['height']}  |  {result['mode']}  |  {result['format'] or 'Image'}"
        )
        self.set_status(
            f"Loaded: {self.file_ops.get_filename(path)}  ({result['width']} � {result['height']})  Format: {result['format']}",
            True,
        )
        self._sync_dirty_state()

    def _on_save_file(self) -> None:
        """Handle file save dialog."""
        if self.state.current is None:
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

        success, error = self.file_ops.save_image_file(self.state.current, path)
        if not success:
            QMessageBox.critical(self, "Save Error", error)
        else:
            self.state.mark_saved()
            self._sync_dirty_state()
            self.set_status(f"Saved: {self.file_ops.get_filename(path)}", True)

    # ========== History Operations ==========

    def _on_undo(self) -> None:
        """Undo last operation."""
        success, label = self.state.undo()
        if not success:
            self.set_status(label, False)
            return

        self.ui.panels["pipeline"].remove_last()
        self.update_canvases()
        self._refresh_fourier_spectrum_if_active()
        self.update_stats_and_metadata()
        self.sync_view_zoom()
        self._sync_dirty_state()

        # Update redo action availability
        self.ui.actions["redo"].setEnabled(self.state.can_redo())

        self.set_status(f"Undone: {label}", False)

    def _on_redo(self) -> None:
        """Redo last undone operation."""
        success, label = self.state.redo()
        if not success:
            self.set_status(label, False)
            return

        self.ui.panels["pipeline"].add_step(label)
        self.update_canvases()
        self._refresh_fourier_spectrum_if_active()
        self.update_stats_and_metadata()
        self.sync_view_zoom()
        self._sync_dirty_state()

        # Update redo action availability
        self.ui.actions["redo"].setEnabled(self.state.can_redo())

        self.set_status(f"Redone: {label}", False)

    def _on_reset(self) -> None:
        """Reset to original image."""
        if not self.state.reset_to_original():
            return

        self.ui.panels["pipeline"].clear()
        self.update_canvases()
        self._refresh_fourier_spectrum_if_active()
        self.update_stats_and_metadata()
        self.sync_view_zoom()
        self._sync_dirty_state()
        self.set_status("Reset to original.", False)

    # ========== Processing ==========

    def start_worker(self, func, label: str, *args, **kwargs) -> None:
        """Start async image processing worker."""
        if self._worker and self._worker.isRunning():
            self.set_status("Processing in progress - please wait�", True)
            return

        self.state.push_to_history(label)
        self.ui.panels["progress"].setVisible(True)
        self._worker = ProcessingWorker(func, label, *args, **kwargs)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()
        self.set_status(f"Processing: {label}�", True)

    def _on_worker_finished(self, result: np.ndarray, label: str) -> None:
        """Handle worker completion."""
        self.state.current = result
        self.ui.panels["pipeline"].add_step(label)

        if "Zoom" not in label:
            self.state._reset_zoom()

        self.update_canvases()
        self.update_stats_and_metadata()
        self.ui.panels["progress"].setVisible(False)
        self._sync_dirty_state()
        self.set_status(
            f"Done: {label}  -  {result.shape[1]}�{result.shape[0]} px",
            True,
        )

        # Update redo availability
        self.ui.actions["redo"].setEnabled(self.state.can_redo())

    def _on_worker_error(self, msg: str) -> None:
        """Handle worker error."""
        self.ui.panels["progress"].setVisible(False)

        if self.state.history:
            self.state.current = self.state.history.pop()
            if self.state.history_labels:
                self.state.history_labels.pop()

        self._sync_dirty_state()
        QMessageBox.critical(self, "Processing Error", msg)
        self.set_status("Error during processing.", False)

    def commit_phase2_result(self, result: np.ndarray, label: str) -> None:
        """Commit Phase 2 operation result into history."""
        if self.state.current is None:
            return

        self.state.push_to_history(label)
        self.state.current = result.astype(np.uint8)
        self.ui.panels["pipeline"].add_step(label)

        self.update_canvases()
        self.update_stats_and_metadata()
        self._sync_dirty_state()
        self.set_status(f"Applied: {label}", True)

        # Update redo availability
        self.ui.actions["redo"].setEnabled(self.state.can_redo())

    # ========== Canvas Management ==========

    def get_current_tool_mode(self) -> str:
        """Get the currently active tool mode."""
        return self.current_tool_mode

    def get_all_canvases(self) -> list:
        """Get all canvas widgets."""
        canvas_keys = ["proc", "orig", "proc2", "gx", "gy", "edge", "spectrum", "fourier_result"]
        return [self.ui.canvases[key] for key in canvas_keys if key in self.ui.canvases]

    def update_canvases(self) -> None:
        """Update all canvas displays."""
        if self.state.current is None:
            return

        self.ui.canvases["proc"].set_array(self.state.current)
        self.ui.canvases["proc2"].set_array(self.state.current)
        self.ui.canvases["fourier_result"].set_array(self.state.current)

        if self.state.original is not None:
            self.ui.canvases["orig"].set_array(self.state.original)

    def _refresh_fourier_spectrum_if_active(self) -> None:
        """Regenerate spectrum when Fourier workflow is in use.

        This keeps spectrum and notch picks consistent after history operations
        like undo/redo/reset without forcing users to press Show Spectrum again.
        """
        if self.state.current is None:
            return

        spectrum_canvas = self.ui.canvases.get("spectrum")
        tabs = self.ui.panels.get("canvas_tabs")
        if spectrum_canvas is None:
            return

        fourier_tab_active = bool(tabs and tabs.currentIndex() == 3)
        spectrum_was_shown = spectrum_canvas.get_array() is not None
        if not (fourier_tab_active or spectrum_was_shown):
            return

        spectrum = shifted_magnitude_spectrum(self.state.current)
        spectrum_canvas.set_array(spectrum)
        spectrum_canvas.clear_markers()
        self.state.selected_notch_center = None

    def sync_view_zoom(self) -> None:
        """Sync zoom level across all canvases."""
        for canvas in self.get_all_canvases():
            canvas.set_display_zoom(self.state.view_zoom_percent)

        zoom_slider = self.ui.sliders["zoom"]
        zoom_slider.blockSignals(True)
        zoom_slider.setValue(self.state.view_zoom_percent)
        zoom_slider.blockSignals(False)

        self.ui.labels["zoom"].setText(f"{self.state.view_zoom_percent}%")
        self.ui.labels["status_zoom"].setText(f"{self.state.view_zoom_percent}%")

    # ========== Stats and Metadata ==========

    def update_stats_and_metadata(self) -> None:
        """Update statistics panel and histogram."""
        if self.state.current is None:
            self.ui.panels["meta"].clear()
            self.ui.labels["hist_summary"].setText("Pixels: � Median: � Entropy: �")
            return

        gray = ensure_gray(self.state.current)
        pixels = int(gray.size)
        mn = int(gray.min())
        mx = int(gray.max())
        mean = float(gray.mean())
        std = float(gray.std())
        median = float(np.median(gray))

        hist = np.bincount(gray.flatten(), minlength=256)
        probabilities = hist[hist > 0] / pixels if pixels else np.array([])
        entropy = (
            float(-(probabilities * np.log2(probabilities)).sum())
            if probabilities.size
            else 0.0
        )

        metadata = dict(self._loaded_metadata or {})
        metadata.update(
            {
                "Min": str(mn),
                "Max": str(mx),
                "Mean": f"{mean:.1f}",
                "Std Dev": f"{std:.1f}",
            }
        )
        if self._current_path:
            metadata.setdefault("File Size", self.file_ops.format_file_size(self._current_path))

        self.ui.panels["meta"].set_metadata(metadata)
        self.ui.canvases["hist"].set_array(self.state.current)
        self.ui.labels["hist_summary"].setText(
            f"Pixels: {pixels:,}   Median: {median:.0f}   Entropy: {entropy:.2f}"
        )

    # ========== Status Management ==========

    def set_status(self, msg: str, active: bool) -> None:
        """Update status bar message and indicator."""
        self.ui.labels["status_msg"].setText(msg)
        status_color = GREEN if active else TEXT3
        self.ui.labels["status_dot"].setStyleSheet(f"background: {status_color}; border-radius: 3px;")

    def require_image(self) -> bool:
        """Check if an image is loaded."""
        if self.state.current is None:
            QMessageBox.information(self, "No Image", "Please open an image first.")
            return False
        return True

    def _sync_dirty_state(self) -> None:
        """Update window modified state."""
        self.setWindowModified(self.state.is_dirty())

    # ========== Event Handling ==========

    def eventFilter(self, watched, event) -> bool:
        """Handle mouse move events for cursor position tracking."""
        if event.type() == QEvent.Type.MouseMove:
            for canvas in self.get_all_canvases():
                if watched is canvas or watched is self.ui.panels.get("canvas_tabs"):
                    pos = event.position().toPoint()
                    self.handlers.on_mouse_move(pos.x(), pos.y())
                    break

        return super().eventFilter(watched, event)

    def closeEvent(self, event) -> None:
        """Handle window close with unsaved changes dialog."""
        if self.isWindowModified() and self.state.current is not None:
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
        """Save before closing application."""
        if self.state.current is None:
            return True

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Processed Image",
            self._current_path or "",
            "PNG Image (*.png);;JPEG Image (*.jpg);;BMP Image (*.bmp)",
        )
        if not path:
            return False

        success, error = self.file_ops.save_image_file(self.state.current, path)
        if not success:
            QMessageBox.critical(self, "Save Error", error)
            return False

        self.state.mark_saved()
        self._sync_dirty_state()
        return True

    def _on_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About MedVision Workbench",
            "<b style='font-size:14px'>MedVision Workbench</b><br>"
            "<i>Phase 2 � Spatial + Frequency Domain Operations</i><br><br>"
            "� Multi-format I/O: DICOM, JPEG, BMP, PNG<br>"
            "� Custom 2-D convolution, zoom, smoothing, median filtering<br>"
            "� Sobel &amp; Prewitt edge detection<br>"
            "� Local Histogram Equalization<br>"
            "� Fourier spectrum viewer and interactive notch filtering<br>"
            "� Ideal, Butterworth, and Gaussian notch reject filters<br>"
            "� Automatic conjugate notch selection<br>"
            "� Synthetic Gaussian and Uniform noise injection<br>"
            "� ROI selection with local histogram, mean, and variance<br>"
            "� Binary morphology: thresholding, erosion, dilation, opening, closing<br>"
            "� Sequential Enhancement Pipeline with undo/redo/reset<br><br>"
            "Team 8 - CUFE - BDE - DIP spring 26",
        )
