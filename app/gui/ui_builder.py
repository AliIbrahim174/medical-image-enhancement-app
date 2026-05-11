# -*- coding: utf-8 -*-
"""UI component builder for the main window."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QProgressBar,
    QPushButton,
    QSlider,
    QStatusBar,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core.styles import (
    ACCENT,
    ACCENT_DIM,
    GREEN,
    TEXT0,
    TEXT1,
    TEXT2,
    TEXT3,
    TOOLBAR_BTN_STYLE,
)
from .panels import MetadataPanel, PipelinePanel
from .sidebar import ToolsSidebar
from .widgets import HistogramWidget, ImageCanvas, ROIImageCanvas


class UIBuilder:
    """Constructs and manages UI components for the main window."""

    def __init__(self, parent: QWidget):
        self.parent = parent
        self.actions = {}
        self.canvases = {}
        self.panels = {}
        self.labels = {}
        self.sliders = {}
        self.view_menu_actions = []

    def build_actions(self) -> dict:
        """Build all menu actions."""
        self.actions = {
            "open": self._create_action("Open", "Ctrl+O"),
            "save": self._create_action("Save", "Ctrl+S"),
            "undo": self._create_action("Undo", "Ctrl+Z"),
            "redo": self._create_action("Redo", "Ctrl+Shift+Z"),
            "reset": self._create_action("Reset", "Ctrl+R"),
            "about": self._create_action("About", ""),
        }
        self.actions["redo"].setEnabled(False)
        return self.actions

    @staticmethod
    def _create_action(text: str, shortcut: str) -> QAction:
        """Create a single action."""
        action = QAction(text)
        if shortcut:
            action.setShortcut(shortcut)
        return action

    def build_top_bar(self) -> QWidget:
        """Build the top menu and toolbar bar."""
        bar = QFrame()
        bar.setFixedHeight(38)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(6)

        logo = QLabel(f"MedVision <span style='color:{TEXT1}'>Workbench</span>")
        logo.setTextFormat(Qt.TextFormat.RichText)
        logo.setStyleSheet(
            f"font-size: 13px; font-weight: 600; color: {ACCENT}; padding: 0 10px 0 4px;"
            f"border-right: 1px solid {ACCENT_DIM};"
        )
        layout.addWidget(logo)

        for title in ["File", "Edit", "View", "Image", "Filters", "Help"]:
            layout.addWidget(self._build_menu_button(title))

        layout.addWidget(self._separator())

        for label, action_key in [
            ("Open", "open"),
            ("Save", "save"),
            ("Undo", "undo"),
            ("Redo", "redo"),
            ("Reset", "reset"),
        ]:
            layout.addWidget(self._toolbar_button(label, self.actions[action_key]))

        layout.addWidget(self._separator())
        layout.addStretch(1)
        return bar

    def _build_menu_button(self, title: str) -> QToolButton:
        """Build a dropdown menu button."""
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
            menu.addAction(self.actions["open"])
            menu.addAction(self.actions["save"])
            menu.addSeparator()
            menu.addAction(self.actions["reset"])
        elif title == "Edit":
            menu.addAction(self.actions["undo"])
            menu.addAction(self.actions["redo"])
        elif title == "View":
            # View actions will be connected by main window
            # Store view menu for later access
            self.view_menu_actions = []
            for label, index in [
                ("Single", 0),
                ("Before / After", 1),
                ("Edge View", 2),
                ("Fourier", 3),
            ]:
                action = menu.addAction(label)
                self.view_menu_actions.append((action, index))
        elif title == "Help":
            menu.addAction(self.actions["about"])
        else:
            disabled = menu.addAction("Coming soon")
            disabled.setEnabled(False)

        button.setMenu(menu)
        return button

    @staticmethod
    def _toolbar_button(label: str, action: QAction) -> QPushButton:
        """Build a toolbar button."""
        button = QPushButton(label)
        button.setStyleSheet(TOOLBAR_BTN_STYLE)
        button.clicked.connect(action.trigger)
        return button

    @staticmethod
    def _separator() -> QFrame:
        """Build a vertical separator."""
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFixedHeight(20)
        sep.setStyleSheet(f"background: {ACCENT_DIM}; max-width: 1px;")
        return sep

    def build_icon_panel(self) -> tuple[QWidget, dict]:
        """Build the left icon panel. Returns (panel, button_data_map)."""
        panel = QWidget()
        panel.setFixedWidth(52)
        panel.setStyleSheet(f"background: #0a0c11; border-right: 1px solid {ACCENT_DIM};")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(2)

        icon_group = QButtonGroup()
        icon_group.setExclusive(True)
        button_data = {}  # Map button -> (index, tip)
        
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

            button_data[button] = (index, tip)
            icon_group.addButton(button)
            layout.addWidget(button, 0, Qt.AlignmentFlag.AlignHCenter)

        layout.addStretch(1)
        return panel, button_data

    def build_canvas_area(self) -> tuple[QWidget, dict]:
        """Build the central canvas area with tabs."""
        area = QWidget()
        area.setStyleSheet("background: #0c0d11;")
        layout = QVBoxLayout(area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        canvas_tabs = QTabWidget()
        canvas_tabs.setDocumentMode(True)

        # Processed canvas (main)
        self.canvases["proc"] = ROIImageCanvas()
        canvas_tabs.addTab(self.canvases["proc"], "Processed")

        # Before / After tab
        canvas_tabs.addTab(self._build_split_tab(), "Before / After")

        # Edge View tab
        canvas_tabs.addTab(self._build_edge_tab(), "Edge View")

        # Fourier tab
        fourier_tab = self._build_fourier_tab()
        canvas_tabs.addTab(fourier_tab, "Fourier")

        layout.addWidget(canvas_tabs, 1)
        layout.addWidget(self._build_canvas_footer())

        return area, {"tabs": canvas_tabs, "main": fourier_tab}

    def _build_split_tab(self) -> QWidget:
        """Build the Before/After comparison tab."""
        widget = QWidget()
        widget.setStyleSheet("background: #0c0d11;")
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self.canvases["orig"] = ImageCanvas()
        self.canvases["proc2"] = ImageCanvas()

        for heading, subheading, canvas in [
            ("Original", "Source", self.canvases["orig"]),
            ("Processed", "Current", self.canvases["proc2"]),
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

    def _build_fourier_tab(self) -> QWidget:
        """Build the Fourier spectrum tab with both spectrum and filtered result."""
        widget = QWidget()
        widget.setStyleSheet("background: #0c0d11;")
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # Left pane: Spectrum (editable)
        spectrum_pane = QWidget()
        spectrum_pane.setStyleSheet(f"background: #080a0d; border: 1px solid {ACCENT_DIM};")
        spectrum_layout = QVBoxLayout(spectrum_pane)
        spectrum_layout.setContentsMargins(0, 0, 0, 0)
        spectrum_layout.setSpacing(0)

        spectrum_header = QLabel("Fourier Spectrum <span style='color:{};'>click to select notch</span>".format(TEXT3))
        spectrum_header.setTextFormat(Qt.TextFormat.RichText)
        spectrum_header.setStyleSheet(
            f"padding: 5px 10px; background: {TEXT0}; border-bottom: 1px solid {ACCENT_DIM};"
            f"font-size: 10px; color: {TEXT2};"
        )
        spectrum_layout.addWidget(spectrum_header)

        self.canvases["spectrum"] = ImageCanvas()
        self.canvases["spectrum"].set_click_reporting(True)
        spectrum_layout.addWidget(self.canvases["spectrum"], 1)
        layout.addWidget(spectrum_pane)

        # Right pane: Filtered result
        result_pane = QWidget()
        result_pane.setStyleSheet(f"background: #080a0d; border: 1px solid {ACCENT_DIM};")
        result_layout = QVBoxLayout(result_pane)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.setSpacing(0)

        result_header = QLabel("Filtered Result <span style='color:{};'>notch-filtered image</span>".format(TEXT3))
        result_header.setTextFormat(Qt.TextFormat.RichText)
        result_header.setStyleSheet(
            f"padding: 5px 10px; background: {TEXT0}; border-bottom: 1px solid {ACCENT_DIM};"
            f"font-size: 10px; color: {TEXT2};"
        )
        result_layout.addWidget(result_header)

        self.canvases["fourier_result"] = ImageCanvas()
        result_layout.addWidget(self.canvases["fourier_result"], 1)
        layout.addWidget(result_pane)

        return widget

    def _build_edge_tab(self) -> QWidget:
        """Build the Edge View tab."""
        widget = QWidget()
        widget.setStyleSheet("background: #0c0d11;")
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self.canvases["gx"] = ImageCanvas()
        self.canvases["gy"] = ImageCanvas()
        self.canvases["edge"] = ImageCanvas()

        for heading, canvas_key in [
            ("Gx — Horizontal gradient", "gx"),
            ("Gy — Vertical gradient", "gy"),
            ("Magnitude — sqrt(Gx²+Gy²)", "edge"),
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
            pane_layout.addWidget(self.canvases[canvas_key], 1)
            layout.addWidget(pane)

        return widget

    def _build_canvas_footer(self) -> QWidget:
        """Build the canvas footer with zoom controls."""
        footer = QWidget()
        footer.setFixedHeight(26)
        footer.setStyleSheet("background: #11131a; border-top: 1px solid #252d42;")
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(8)

        minus = QPushButton("−")
        minus.setFixedSize(24, 18)
        minus.setStyleSheet(TOOLBAR_BTN_STYLE)

        plus = QPushButton("+")
        plus.setFixedSize(24, 18)
        plus.setStyleSheet(TOOLBAR_BTN_STYLE)

        fit = QPushButton("Fit")
        fit.setStyleSheet(TOOLBAR_BTN_STYLE)

        self.sliders["zoom"] = QSlider(Qt.Orientation.Horizontal)
        self.sliders["zoom"].setRange(5, 400)
        self.sliders["zoom"].setValue(100)
        self.sliders["zoom"].setFixedWidth(90)

        self.labels["zoom"] = QLabel("100%")
        self.labels["zoom"].setStyleSheet(
            f"font-family: 'JetBrains Mono', monospace; color: {TEXT2}; min-width: 44px;"
        )

        self.labels["canvas_info"] = QLabel("No image loaded")
        self.labels["canvas_info"].setStyleSheet(
            f"font-family: 'JetBrains Mono', monospace; color: {TEXT3};"
        )

        layout.addWidget(minus)
        layout.addWidget(self.sliders["zoom"])
        layout.addWidget(plus)
        layout.addWidget(self.labels["zoom"])
        layout.addWidget(fit)
        layout.addStretch(1)
        layout.addWidget(self.labels["canvas_info"])

        return footer

    def build_right_panel(self) -> QWidget:
        """Build the right sidebar panel."""
        panel = QWidget()
        panel.setFixedWidth(240)
        panel.setStyleSheet("background: #11131a; border-left: 1px solid #252d42;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.panels["right_tabs"] = QTabWidget()
        self.panels["right_tabs"].setDocumentMode(True)

        self.panels["meta"] = MetadataPanel()
        self.panels["right_tabs"].addTab(self.panels["meta"], "Info")

        hist_tab = self._build_hist_tab()
        self.panels["right_tabs"].addTab(hist_tab, "Histogram")

        self.panels["pipeline"] = PipelinePanel()
        self.panels["right_tabs"].addTab(self.panels["pipeline"], "Pipeline")

        layout.addWidget(self.panels["right_tabs"])
        return panel

    def _build_hist_tab(self) -> QWidget:
        """Build the histogram display tab."""
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

        self.canvases["hist"] = HistogramWidget()
        layout.addWidget(self.canvases["hist"])

        self.labels["hist_summary"] = QLabel("Pixels: —   Median: —   Entropy: —")
        self.labels["hist_summary"].setWordWrap(True)
        self.labels["hist_summary"].setStyleSheet(
            f"color: {TEXT2}; font-size: 10px; line-height: 1.5;"
        )
        layout.addWidget(self.labels["hist_summary"])
        layout.addStretch(1)

        return widget

    def build_statusbar(self) -> QStatusBar:
        """Build the status bar."""
        bar = QStatusBar()

        self.labels["status_dot"] = QLabel()
        self.labels["status_dot"].setFixedSize(6, 6)
        self.labels["status_dot"].setStyleSheet(f"background: {TEXT3}; border-radius: 3px;")

        self.labels["status_msg"] = QLabel("Ready - Open an image to begin")
        self.labels["status_msg"].setStyleSheet(f"color: {TEXT2};")

        self.labels["cursor_pos"] = QLabel("—")
        self.labels["cursor_pos"].setStyleSheet(
            f"font-family: 'JetBrains Mono', monospace; color: {TEXT1};"
        )

        self.labels["status_zoom"] = QLabel("100%")
        self.labels["status_zoom"].setStyleSheet(
            f"font-family: 'JetBrains Mono', monospace; color: {TEXT1};"
        )

        self.labels["status_memory"] = QLabel("—")
        self.labels["status_memory"].setStyleSheet(
            f"font-family: 'JetBrains Mono', monospace; color: {TEXT1};"
        )

        self.labels["status_footer"] = QLabel("Team 8 - CUFE - BDE - DIP spring 26")
        self.labels["status_footer"].setStyleSheet(f"color: {TEXT3};")

        host = QWidget()
        host_layout = QHBoxLayout(host)
        host_layout.setContentsMargins(10, 0, 10, 0)
        host_layout.setSpacing(16)

        left = QWidget()
        left_layout = QHBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        left_layout.addWidget(self.labels["status_dot"])
        left_layout.addWidget(self.labels["status_msg"])

        right = QWidget()
        right_layout = QHBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        for label_text, label_key in [
            ("Cursor:", "cursor_pos"),
            ("Zoom:", "status_zoom"),
            ("Memory:", "status_memory"),
        ]:
            segment = QWidget()
            segment_layout = QHBoxLayout(segment)
            segment_layout.setContentsMargins(0, 0, 0, 0)
            segment_layout.setSpacing(5)
            caption = QLabel(label_text)
            caption.setStyleSheet(f"color: {TEXT3};")
            segment_layout.addWidget(caption)
            segment_layout.addWidget(self.labels[label_key])
            right_layout.addWidget(segment)

        right_layout.addWidget(self.labels["status_footer"])

        host_layout.addWidget(left)
        host_layout.addStretch(1)
        host_layout.addWidget(right)

        bar.addPermanentWidget(host, 1)

        self.panels["progress"] = QProgressBar()
        self.panels["progress"].setVisible(False)
        self.panels["progress"].setFixedSize(120, 10)
        self.panels["progress"].setRange(0, 0)
        bar.addPermanentWidget(self.panels["progress"])

        return bar

    def build_sidebar(self) -> ToolsSidebar:
        """Build the tools sidebar."""
        sidebar = ToolsSidebar()
        return sidebar
