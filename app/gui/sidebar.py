"""Left-side tools sidebar containing the image processing controls."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QPushButton, QLabel, QComboBox, QDoubleSpinBox, QScrollArea,
    QToolButton, QFrame, QCheckBox
)
from PyQt6.QtCore import pyqtSignal, Qt

from ..core.styles import SIDEBAR_STYLE, GREEN, RED


class CollapsibleSection(QWidget):
    def __init__(self, title: str, open_by_default: bool = False, parent=None):
        super().__init__(parent)
        self._button = QToolButton()
        self._button.setText(title)
        self._button.setCheckable(True)
        self._button.setChecked(open_by_default)
        self._button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._button.setArrowType(
            Qt.ArrowType.DownArrow if open_by_default else Qt.ArrowType.RightArrow
        )
        self._button.clicked.connect(self._toggle)
        self._button.setStyleSheet(
            "QToolButton { text-align: left; border: none; padding: 6px 8px; }"
        )

        self._body = QFrame()
        self._body.setFrameShape(QFrame.Shape.NoFrame)
        self._body.setVisible(open_by_default)
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(8, 8, 8, 8)
        self._body_layout.setSpacing(6)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._button)
        layout.addWidget(self._body)

    def _toggle(self, checked: bool) -> None:
        self._body.setVisible(checked)
        self._button.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )

    def bodyLayout(self) -> QVBoxLayout:
        return self._body_layout


class ToolsSidebar(QWidget):
    """
    Emits signals whenever the user requests an operation.
    The MainWindow connects these to the processing dispatch layer,
    keeping UI logic cleanly separated from business logic.

    Signals
    -------
    apply_filter(str, dict)     filter_name, params dict
    apply_zoom(float, str)      step multiplier, interpolation method name
    apply_edge(str, str)        operator name, component name
    apply_hist_eq(int)          block size
    apply_median(int)           kernel size
    """

    apply_filter  = pyqtSignal(str, dict)
    apply_zoom    = pyqtSignal(float, str)
    apply_edge    = pyqtSignal(str, str)
    apply_hist_eq = pyqtSignal(int)
    apply_median  = pyqtSignal(int)
    accumulate_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(220)
        self.setStyleSheet(SIDEBAR_STYLE)

        # Wrap everything in a scroll area so it never clips on small screens
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")

        container = QWidget()
        layout    = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(5)

        # State mode checkbox with theme styling
        mode_section = QWidget()
        mode_layout = QHBoxLayout(mode_section)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(4)
        mode_checkbox = QCheckBox("Accumulate Mode")
        mode_checkbox.setChecked(True)
        mode_checkbox.stateChanged.connect(
            lambda state: self.accumulate_toggled.emit(mode_checkbox.isChecked())
        )
        mode_checkbox.setStyleSheet(
            "QCheckBox { color: #4ade80; font-size: 10px; font-weight: 600; }"
            "QCheckBox::indicator { width: 16px; height: 16px; }"
            "QCheckBox::indicator:unchecked { background: #1c2030; border: 1px solid #252d42; border-radius: 3px; }"
            "QCheckBox::indicator:checked { background: #4ade80; border: 1px solid #4ade80; border-radius: 3px; }"
        )
        mode_label = QLabel("apply to current")
        mode_label.setStyleSheet("color: #4ade80; font-size: 9px; margin-left: 2px;")
        mode_layout.addWidget(mode_checkbox)
        mode_layout.addWidget(mode_label)
        mode_layout.addStretch()
        layout.addWidget(mode_section)
        layout.addWidget(self._separator())

        layout.addWidget(self._build_zoom_group())
        layout.addWidget(self._build_smoothing_group())
        layout.addWidget(self._build_edge_group())
        layout.addWidget(self._build_median_group())
        layout.addWidget(self._build_histeq_group())
        layout.addStretch()

        scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ── Group builders ────────────────────────────────────────────────────────

    def _build_zoom_group(self) -> QWidget:
        section = CollapsibleSection("Zoom / Interpolation", True)
        gl = section.bodyLayout()

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Method"))
        self._zoom_method = QComboBox()
        self._zoom_method.addItems(["Nearest-Neighbor", "Bilinear"])
        r1.addWidget(self._zoom_method)
        gl.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Step ×"))
        self._zoom_scale = QDoubleSpinBox()
        self._zoom_scale.setRange(1.1, 8.0)
        self._zoom_scale.setSingleStep(0.25)
        self._zoom_scale.setValue(1.5)
        r2.addWidget(self._zoom_scale)
        gl.addLayout(r2)

        btn_row = QHBoxLayout()
        btn_in  = QPushButton(" ⊖ ZoomOut")
        btn_out = QPushButton(" ⊕ Zoom In")
        btn_in .clicked.connect(
            lambda: self.apply_zoom.emit(self._zoom_scale.value(),
                                         self._zoom_method.currentText()))
        btn_out.clicked.connect(
            lambda: self.apply_zoom.emit(1.0 / self._zoom_scale.value(),
                                         self._zoom_method.currentText()))
        btn_row.addWidget(btn_in)
        btn_row.addWidget(btn_out)
        gl.addLayout(btn_row)
        return section

    def _build_smoothing_group(self) -> QWidget:
        section = CollapsibleSection("Smoothing Filters", True)
        sl = section.bodyLayout()

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Kernel"))
        self._smooth_kernel = QComboBox()
        self._smooth_kernel.addItems(["3×3", "5×5", "7×7", "9×9", "11×11"])
        r1.addWidget(self._smooth_kernel)
        sl.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Gauss σ"))
        self._gauss_sigma = QDoubleSpinBox()
        self._gauss_sigma.setRange(0.1, 20.0)
        self._gauss_sigma.setSingleStep(0.1)
        self._gauss_sigma.setValue(1.0)
        r2.addWidget(self._gauss_sigma)
        sl.addLayout(r2)

        grid = QGridLayout()
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(4)
        btn_avg   = QPushButton("Average")
        btn_gauss = QPushButton("Gaussian")
        btn_avg.clicked.connect(self._emit_average)
        btn_gauss.clicked.connect(self._emit_gaussian)
        grid.addWidget(btn_avg, 0, 0)
        grid.addWidget(btn_gauss, 0, 1)
        sl.addLayout(grid)
        return section

    def _build_edge_group(self) -> QWidget:
        section = CollapsibleSection("Edge Detection")
        el = section.bodyLayout()

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Operator"))
        self._edge_op = QComboBox()
        self._edge_op.addItems(["Sobel", "Prewitt"])
        r1.addWidget(self._edge_op)
        el.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Show"))
        self._edge_component = QComboBox()
        self._edge_component.addItems(["Magnitude", "Horizontal (Gx)", "Vertical (Gy)"])
        r2.addWidget(self._edge_component)
        el.addLayout(r2)

        btn = QPushButton("⟁ Detect Edges")
        btn.clicked.connect(lambda: self.apply_edge.emit(
            self._edge_op.currentText(),
            self._edge_component.currentText()
        ))
        el.addWidget(btn)
        return section

    def _build_median_group(self) -> QWidget:
        section = CollapsibleSection("Median Filter")
        ml = section.bodyLayout()

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Kernel"))
        self._median_kernel = QComboBox()
        self._median_kernel.addItems(["3×3", "5×5", "7×7"])
        r1.addWidget(self._median_kernel)
        ml.addLayout(r1)

        btn = QPushButton("⊡ Apply Median")
        btn.clicked.connect(self._emit_median)
        ml.addWidget(btn)
        return section

    def _build_histeq_group(self) -> QWidget:
        section = CollapsibleSection("Local Histogram Equalization")
        hl = section.bodyLayout()

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Block"))
        self._block_size = QComboBox()
        self._block_size.addItems(["8×8", "16×16", "32×32", "64×64"])
        r1.addWidget(self._block_size)
        hl.addLayout(r1)

        btn = QPushButton("◑ Local Equalize")
        btn.clicked.connect(self._emit_hist_eq)
        hl.addWidget(btn)
        return section

    # ── Signal emitters ───────────────────────────────────────────────────────

    @staticmethod
    def _separator() -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: #252d42;")
        return sep

    @staticmethod
    def _combo_size(combo: QComboBox) -> int:
        """Parse '3×3' → 3, '11×11' → 11, etc."""
        return int(combo.currentText().split("×")[0])

    def _emit_average(self):
        self.apply_filter.emit("average", {
            "kernel_size": self._combo_size(self._smooth_kernel)
        })

    def _emit_gaussian(self):
        self.apply_filter.emit("gaussian", {
            "kernel_size": self._combo_size(self._smooth_kernel),
            "sigma":       self._gauss_sigma.value(),
        })

    def _emit_median(self):
        self.apply_median.emit(self._combo_size(self._median_kernel))

    def _emit_hist_eq(self):
        self.apply_hist_eq.emit(self._combo_size(self._block_size))