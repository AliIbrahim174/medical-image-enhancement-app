"""Signal and event handlers for the main window."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtWidgets import QMessageBox

from ..DIP.edge_detection import prewitt_edge, sobel_edge
from ..DIP.frequency_domain import (
    apply_notch_filter,
    conjugate_notch_center,
    shifted_magnitude_spectrum,
)
from ..DIP.histogram_equalization import local_histogram_equalization
from ..DIP.median import median_filter_scratch
from ..DIP.morphology import closing, dilate, erode, global_threshold, opening
from ..DIP.noise import inject_gaussian_noise, inject_uniform_noise
from ..DIP.roi_stats import compute_roi_stats
from ..DIP.smoothing import apply_linear_filter, make_average_kernel, make_gaussian_kernel
from ..DIP.utils import ensure_gray
from ..DIP.zoom import bilinear_zoom, nearest_neighbor_zoom
from .roi_dialog import ROIStatsDialog


class SignalHandlers:
    """Handles all signal/slot connections and event callbacks."""

    def __init__(self, main_window):
        self.main = main_window

    # ========== UI Navigation ==========

    def on_switch_view(self, index: int) -> None:
        """Switch canvas tab."""
        self.main.ui.panels["right_tabs"]  # Access through UIBuilder
        tabs = self.main.ui.panels.get("canvas_tabs")
        if tabs:
            tabs.setCurrentIndex(index)

    def on_tab_changed(self, _index: int) -> None:
        """Handle tab change."""
        self.main.update_canvases()

    def on_set_interaction_mode(self, index: int, tip: str) -> None:
        """Set interaction mode based on icon index."""
        mode_map = {0: "pan", 7: "annotate"}
        mode = mode_map.get(index, "none")

        for canvas in self.main.get_all_canvases():
            canvas.set_interaction_mode(mode)

        self.main.set_status(tip, False)

    # ========== Zoom Controls ==========

    def on_update_view_zoom(self, value: int) -> None:
        """Update view zoom from slider."""
        self.main.state.view_zoom_percent = value
        self.main.sync_view_zoom()

    def on_change_view_zoom(self, delta: int) -> None:
        """Change zoom level by delta step."""
        steps = [5, 10, 25, 33, 50, 67, 75, 100, 125, 150, 200, 300, 400]
        zoom_slider = self.main.ui.sliders["zoom"]
        current = zoom_slider.value()

        index = next((i for i, v in enumerate(steps) if v >= current), len(steps) - 1)
        index = max(0, min(len(steps) - 1, index + delta))
        zoom_slider.setValue(steps[index])

    def on_fit_view(self) -> None:
        """Fit view to 100%."""
        self.main.ui.sliders["zoom"].setValue(100)

    # ========== Image Processing (Phase 1) ==========

    def on_apply_zoom(self, step: float, method: str) -> None:
        """Apply zoom operation."""
        if not self.main.require_image():
            return

        state = self.main.state
        if state.zoom_base is None:
            state.zoom_base = state.current.copy()
            state.zoom_factor = 1.0

        new_factor = max(0.05, min(16.0, state.zoom_factor * step))
        if new_factor == state.zoom_factor:
            self.main.set_status("Zoom limit reached.", False)
            return

        state.zoom_factor = new_factor
        direction = "In" if step >= 1.0 else "Out"
        mode_mark = "(acc)" if state.accumulate else "(orig)"
        label = f"Zoom {direction} ({method.split('-')[0]}) — {state.zoom_factor:.2f}× {mode_mark}"

        zoom_func = bilinear_zoom if method == "Bilinear" else nearest_neighbor_zoom
        self.main.start_worker(zoom_func, label, state.zoom_base.copy(), state.zoom_factor)

    def on_apply_filter(self, filter_name: str, params: dict) -> None:
        """Apply smoothing filter."""
        if not self.main.require_image():
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

        mode_mark = "(acc)" if self.main.state.accumulate else "(orig)"
        label = f"{label} {mode_mark}"
        base_image = self.main.state.get_source_image()
        self.main.start_worker(apply_linear_filter, label, base_image, kernel)

    def on_apply_edge(self, operator: str, component: str) -> None:
        """Apply edge detection."""
        if not self.main.require_image():
            return

        source = self.main.state.get_source_image()
        gray = ensure_gray(source)

        if operator == "Sobel":
            gx, gy, magnitude = sobel_edge(gray)
        else:
            gx, gy, magnitude = prewitt_edge(gray)

        self.main.ui.canvases["gx"].set_array(gx)
        self.main.ui.canvases["gy"].set_array(gy)
        self.main.ui.canvases["edge"].set_array(magnitude)

        # Switch to edge view tab
        canvas_tabs = self.main.ui.panels.get("canvas_tabs")
        if canvas_tabs:
            canvas_tabs.setCurrentIndex(2)

        result = {"Magnitude": magnitude, "Horizontal (Gx)": gx, "Vertical (Gy)": gy}.get(
            component, magnitude
        )
        mode_mark = "(acc)" if self.main.state.accumulate else "(orig)"
        label = f"{operator} Edge ({component}) {mode_mark}"

        self.main.state.push_to_history(label)
        self.main.state.current = result
        self.main.state._reset_zoom()
        self.main.ui.panels["pipeline"].add_step(label)
        self.main.update_stats_and_metadata()
        self.main.update_canvases()
        self.main.set_status(f"Done: {label}", True)

    def on_hist_eq(self, block_size: int) -> None:
        """Apply local histogram equalization."""
        if not self.main.require_image():
            return

        source = self.main.state.get_source_image()
        gray = ensure_gray(source)
        mode_mark = "(acc)" if self.main.state.accumulate else "(orig)"
        label = f"Local Hist. Eq. {block_size}×{block_size} {mode_mark}"
        self.main.start_worker(local_histogram_equalization, label, gray, block_size)

    def on_median(self, kernel_size: int) -> None:
        """Apply median filter."""
        if not self.main.require_image():
            return

        source = self.main.state.get_source_image()
        mode_mark = "(acc)" if self.main.state.accumulate else "(orig)"
        label = f"Median Filter {kernel_size}×{kernel_size} {mode_mark}"
        self.main.start_worker(median_filter_scratch, label, source, kernel_size)

    def on_noise(self, noise_type: str, params: dict) -> None:
        """Apply synthetic noise."""
        if not self.main.require_image():
            return

        source = self.main.state.get_source_image()
        param = float(params.get("param", 1.0))
        mode_mark = "(acc)" if self.main.state.accumulate else "(orig)"

        if noise_type == "Gaussian":
            label = f"Gaussian Noise σ={param:.0f} {mode_mark}"
            self.main.start_worker(inject_gaussian_noise, label, source, 0.0, param)
        else:
            label = f"Uniform Noise ±{param:.0f} {mode_mark}"
            self.main.start_worker(inject_uniform_noise, label, source, -param, param)

    def on_accumulate_toggled(self, checked: bool) -> None:
        """Toggle accumulate mode."""
        self.main.state.set_accumulate_mode(checked)
        mode_text = "Accumulating" if checked else "From Original"
        self.main.set_status(f"Mode: {mode_text}", False)

    # ========== Phase 2: Frequency Domain ==========

    def on_show_spectrum(self) -> None:
        """Display Fourier spectrum."""
        if not self.main.require_image():
            return

        source = self.main.state.get_source_image()
        if source is None:
            return

        spectrum = shifted_magnitude_spectrum(source)
        self.main.state.selected_notch_center = None
        self.main.ui.canvases["spectrum"].clear_markers()
        self.main.ui.canvases["spectrum"].set_array(spectrum)

        canvas_tabs = self.main.ui.panels.get("canvas_tabs")
        if canvas_tabs:
            fourier_index = 3  # Fourier tab is at index 3
            canvas_tabs.setCurrentIndex(fourier_index)

        self.main.set_status(
            "Fourier spectrum displayed. Click a bright spike away from the center.",
            True,
        )

    def on_spectrum_clicked(self, row: int, col: int) -> None:
        """Handle spectrum click for notch selection."""
        spectrum_canvas = self.main.ui.canvases["spectrum"]
        if spectrum_canvas.get_array() is None:
            return

        spectrum = spectrum_canvas.get_array()
        h, w = spectrum.shape[:2]

        self.main.state.selected_notch_center = (row, col)
        mirror_row, mirror_col = conjugate_notch_center((h, w), (row, col))

        spectrum_canvas.clear_markers()
        spectrum_canvas.add_marker_at_pixel(row, col)
        spectrum_canvas.add_marker_at_pixel(mirror_row, mirror_col)

        self.main.set_status(
            f"Selected notch: ({row}, {col}); conjugate: ({mirror_row}, {mirror_col})",
            True,
        )

    def on_apply_notch(self, filter_type: str, radius: float, order: int) -> None:
        """Apply notch filter."""
        if not self.main.require_image():
            return

        if self.main.state.selected_notch_center is None:
            QMessageBox.information(
                self.main,
                "No Notch Selected",
                "Show the Fourier spectrum first, then click a bright noise spike.",
            )
            return

        source = self.main.state.get_source_image()
        if source is None:
            return

        try:
            result = apply_notch_filter(
                source,
                center=self.main.state.selected_notch_center,
                radius=radius,
                filter_type=filter_type,
                order=order,
            )
        except Exception as exc:
            QMessageBox.critical(self.main, "Notch Filter Error", str(exc))
            return

        label = f"{filter_type} notch r={radius:.0f}, n={order}"
        self.main.commit_phase2_result(result, label)

        # Refresh spectrum
        spectrum = shifted_magnitude_spectrum(self.main.state.current)
        self.main.ui.canvases["spectrum"].set_array(spectrum)

    def on_threshold(self, threshold: int) -> None:
        """Apply global threshold."""
        if not self.main.require_image():
            return

        source = self.main.state.get_source_image()
        if source is None:
            return

        result = global_threshold(source, threshold)
        self.main.commit_phase2_result(result, f"Global threshold T={threshold}")

    def on_morphology(self, operation: str, size: int, shape: str) -> None:
        """Apply morphology operation."""
        if not self.main.require_image():
            return

        source = self.main.state.get_source_image()
        if source is None:
            return

        try:
            if operation == "erosion":
                result = erode(source, size, shape)
                label = f"Erosion {shape} {size}×{size}"
            elif operation == "dilation":
                result = dilate(source, size, shape)
                label = f"Dilation {shape} {size}×{size}"
            elif operation == "opening":
                result = opening(source, size, shape)
                label = f"Opening {shape} {size}×{size}"
            elif operation == "closing":
                result = closing(source, size, shape)
                label = f"Closing {shape} {size}×{size}"
            else:
                raise ValueError("Unknown morphology operation.")
        except Exception as exc:
            QMessageBox.critical(self.main, "Morphology Error", str(exc))
            return

        self.main.commit_phase2_result(result, label)

    # ========== ROI and Statistics ==========

    def on_roi_selected(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Handle ROI selection."""
        if self.main.state.current is None:
            return

        sidebar = self.main.sidebar
        if hasattr(sidebar, "roi_btn"):
            sidebar.roi_btn.setChecked(False)

        gray = ensure_gray(self.main.state.current)
        hist, mean, var = compute_roi_stats(gray, x1, y1, x2, y2)
        dlg = ROIStatsDialog(hist, mean, var, parent=self.main)
        dlg.exec()

    # ========== Event Filter ==========

    def on_mouse_move(self, pos_x: int, pos_y: int) -> None:
        """Handle mouse move event."""
        self.main.ui.labels["cursor_pos"].setText(f"{pos_x}, {pos_y}")
