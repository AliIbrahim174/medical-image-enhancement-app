"""Right-side dock panels for image metadata and the enhancement pipeline."""

from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QLabel, QTextEdit,
        QPushButton, QListWidget, QListWidgetItem, QHBoxLayout
)
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor

from .widgets import HistogramWidget
from ..core.styles import (
    ACCENT, PIPELINE_LIST_STYLE, PIPELINE_UNDO_STYLE, METADATA_TEXT_STYLE,
    BG2, BG3, BORDER, TEXT2, TEXT3, TEXT0, ACCENT_DIM
)


# ─────────────────────────────────────────────────────────────────────────────
#  METADATA PANEL
# ─────────────────────────────────────────────────────────────────────────────

class MetadataPanel(QWidget):
    """
    Displays image metadata (width, height, bit-depth, DICOM tags, …)
    as a styled HTML list, plus a live luminance histogram below it.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumWidth(240)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        title = QLabel("Image Info")
        title.setStyleSheet(
            f"padding: 8px 10px; border-bottom: 1px solid {BORDER};"
            f"color: {TEXT2}; font-size: 10px; text-transform: uppercase;"
            f"letter-spacing: 0.7px; font-weight: 600; background: {BG2};"
        )
        layout.addWidget(title)

        section = QWidget()
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(8, 8, 8, 8)
        section_layout.setSpacing(6)

        props = QLabel("Image Properties")
        props.setStyleSheet(
            f"color: {TEXT3}; font-size: 9.5px; text-transform: uppercase;"
            f"letter-spacing: 0.7px; font-weight: 600;"
        )
        section_layout.addWidget(props)

        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setStyleSheet(METADATA_TEXT_STYLE)
        self._text.setMinimumHeight(230)
        section_layout.addWidget(self._text)

        hist_title = QLabel("Histogram")
        hist_title.setStyleSheet(
            f"color: {TEXT3}; font-size: 9.5px; text-transform: uppercase;"
            f"letter-spacing: 0.7px; font-weight: 600; margin-top: 6px;"
        )
        section_layout.addWidget(hist_title)

        self.histogram = HistogramWidget()
        section_layout.addWidget(self.histogram)

        layout.addWidget(section)

    def set_metadata(self, meta: dict) -> None:
        """Render a dict of {label: value} as colour-coded HTML."""
        lines = [
            f"<span style='color:{ACCENT}'>{k}</span><span style='color:{TEXT0}'>: {v}</span>"
            for k, v in meta.items()
        ]
        self._text.setHtml(
            "<div style='line-height:1.7; font-family: JetBrains Mono, Consolas, monospace;'>"
            + "<br>".join(lines)
            + "</div>"
        )

    def clear(self) -> None:
        self._text.clear()


# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE PANEL
# ─────────────────────────────────────────────────────────────────────────────

class PipelinePanel(QWidget):
    """
    Numbered list of applied operations representing the sequential
    enhancement pipeline.  Emits undo_requested when the Undo button
    is clicked.
    """

    undo_requested = pyqtSignal()
    reset_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumWidth(240)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        title_row = QWidget()
        title_layout = QHBoxLayout(title_row)
        title_layout.setContentsMargins(10, 8, 10, 8)
        title_layout.setSpacing(6)

        title = QLabel("Enhancement Pipeline")
        title.setStyleSheet(
            f"color: {TEXT2}; font-size: 10px; text-transform: uppercase;"
            f"letter-spacing: 0.7px; font-weight: 600;"
        )
        self._count = QLabel("0")
        self._count.setStyleSheet(
            f"background: {ACCENT_DIM}; color: {ACCENT}; padding: 1px 5px;"
            f"border-radius: 3px; font-size: 9px;"
        )
        title_layout.addWidget(title)
        title_layout.addStretch()
        title_layout.addWidget(self._count)
        layout.addWidget(title_row)

        self._list = QListWidget()
        self._list.setStyleSheet(PIPELINE_LIST_STYLE)
        self._list.setMinimumHeight(180)
        layout.addWidget(self._list)

        action_row = QWidget()
        action_layout = QHBoxLayout(action_row)
        action_layout.setContentsMargins(6, 6, 6, 6)
        action_layout.setSpacing(4)

        self._undo_btn = QPushButton("Undo")
        self._undo_btn.setStyleSheet(PIPELINE_UNDO_STYLE)
        self._undo_btn.clicked.connect(self.undo_requested)
        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setStyleSheet(PIPELINE_UNDO_STYLE)
        self._reset_btn.clicked.connect(self.reset_requested)
        action_layout.addWidget(self._undo_btn)
        action_layout.addWidget(self._reset_btn)
        layout.addWidget(action_row)

    # ── Public API ────────────────────────────────────────────────────────────

    def add_step(self, label: str) -> None:
        """Append a numbered step entry with color-coding for acc vs orig mode."""
        n    = self._list.count() + 1
        item = QListWidgetItem(f"{n}. {label}")
        
        # Color-code based on accumulate mode
        if "(acc)" in label:
            item.setForeground(QColor("#4ade80"))  # GREEN for accumulate
            item.setBackground(QColor("#1a3d1a"))  # Dark green background
        elif "(orig)" in label:
            item.setForeground(QColor("#f87171"))  # RED for original
            item.setBackground(QColor("#3d1a1a"))  # Dark red background
        
        self._list.addItem(item)
        self._list.scrollToBottom()
        self._count.setText(str(self._list.count()))

    def remove_last(self) -> None:
        """Remove the most recently added step (used by undo)."""
        row = self._list.count() - 1
        if row >= 0:
            self._list.takeItem(row)
        self._count.setText(str(self._list.count()))

    def clear(self) -> None:
        """Remove all steps."""
        self._list.clear()
        self._count.setText("0")