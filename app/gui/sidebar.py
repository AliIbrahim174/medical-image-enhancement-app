"""Left-side tools sidebar containing the image processing controls."""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QDoubleSpinBox,
    QScrollArea,
    QToolButton,
    QFrame,
    QCheckBox,
    QSpinBox,
    QSlider,
)
from PyQt6.QtCore import pyqtSignal, Qt

from ..core.styles import SIDEBAR_STYLE


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
    apply_filter(str, dict)         filter_name, params dict
    apply_zoom(float, str)          step multiplier, interpolation method name
    apply_edge(str, str)            operator name, component name
    apply_hist_eq(int)              block size
    apply_median(int)               kernel size
    apply_noise(str, dict)          noise type, params dict
    show_spectrum()                 request Fourier spectrum display
    apply_notch(str, float, int)    notch type, radius, Butterworth order
    apply_threshold(int)            binary threshold value
    apply_morphology(str, int, str) operation, SE size, SE shape
    """

    # Phase 1 signals.
    apply_filter = pyqtSignal(str, dict)
    apply_zoom = pyqtSignal(float, str)
    apply_edge = pyqtSignal(str, str)
    apply_hist_eq = pyqtSignal(int)
    apply_median = pyqtSignal(int)
    accumulate_toggled = pyqtSignal(bool)

    # Phase 2 / Member 3: ROI and noise modeling signal.
    apply_noise = pyqtSignal(str, dict)

    # Phase 2 / Members 1 and 2: frequency-domain notch filter signals.
    show_spectrum = pyqtSignal()
    apply_notch = pyqtSignal(str, float, int)  # filter_type, radius, order

    # Phase 2 / Member 4: bonus morphology signals.
    apply_threshold = pyqtSignal(int)             # threshold value
    apply_morphology = pyqtSignal(str, int, str)  # operation, SE size, SE shape

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(220)
        self.setStyleSheet(SIDEBAR_STYLE)

        # Wrap everything in a scroll area so it never clips on small screens.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(5)

        # State mode checkbox with theme styling.
        mode_section = QWidget()
        mode_layout = QHBoxLayout(mode_section)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(4)

        mode_checkbox = QCheckBox("Accumulate Mode")
        mode_checkbox.setChecked(True)
        mode_checkbox.stateChanged.connect(
            lambda _state: self.accumulate_toggled.emit(mode_checkbox.isChecked())
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

        # Phase 2 / Member 3: noise injection and ROI toggle controls.
        layout.addWidget(self._build_noise_group())

        layout.addWidget(self._build_edge_group())
        layout.addWidget(self._build_median_group())
        layout.addWidget(self._build_histeq_group())

        # Phase 2 / Members 1 and 2: frequency-domain notch filtering controls.
        layout.addWidget(self._build_frequency_group())

        # Phase 2 / Member 4: bonus morphology controls.
        layout.addWidget(self._build_morphology_group())

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

        # Phase 2 cleanup: correct Zoom Out / Zoom In emitted scale direction.
        btn_out = QPushButton("⊖ Zoom Out")
        btn_in = QPushButton("⊕ Zoom In")

        btn_out.clicked.connect(
            lambda: self.apply_zoom.emit(
                1.0 / self._zoom_scale.value(),
                self._zoom_method.currentText(),
            )
        )
        btn_in.clicked.connect(
            lambda: self.apply_zoom.emit(
                self._zoom_scale.value(),
                self._zoom_method.currentText(),
            )
        )

        btn_row.addWidget(btn_out)
        btn_row.addWidget(btn_in)
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

        btn_avg = QPushButton("Average")
        btn_gauss = QPushButton("Gaussian")
        btn_avg.clicked.connect(self._emit_average)
        btn_gauss.clicked.connect(self._emit_gaussian)

        grid.addWidget(btn_avg, 0, 0)
        grid.addWidget(btn_gauss, 0, 1)
        sl.addLayout(grid)
        return section

    # Phase 2 / Member 3: GUI controls for Gaussian/Uniform noise and ROI selection.
    def _build_noise_group(self) -> QWidget:
        section = CollapsibleSection("Noise Injection / ROI", False)
        nl = section.bodyLayout()

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Type"))
        self._noise_type = QComboBox()
        self._noise_type.addItems(["Gaussian", "Uniform"])
        r1.addWidget(self._noise_type)
        nl.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("σ / Range"))
        self._noise_param = QDoubleSpinBox()
        self._noise_param.setRange(1.0, 150.0)
        self._noise_param.setSingleStep(1.0)
        self._noise_param.setValue(25.0)
        r2.addWidget(self._noise_param)
        nl.addLayout(r2)

        btn = QPushButton("⚡ Inject Noise")
        btn.clicked.connect(self._emit_noise)
        nl.addWidget(btn)

        # Phase 2 / Member 3: MainWindow connects this toggle to ROIImageCanvas.set_roi_mode.
        self.roi_btn = QPushButton("▢ Draw ROI")
        self.roi_btn.setCheckable(True)
        nl.addWidget(self.roi_btn)

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
        btn.clicked.connect(
            lambda: self.apply_edge.emit(
                self._edge_op.currentText(),
                self._edge_component.currentText(),
            )
        )
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

    # Phase 2 / Members 1 and 2: GUI controls for Fourier spectrum display and notch filtering.
    def _build_frequency_group(self) -> QWidget:
        section = CollapsibleSection("Frequency / Notch Filter", False)
        fl = section.bodyLayout()

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Type"))
        self._notch_type = QComboBox()
        self._notch_type.addItems(["Ideal", "Butterworth", "Gaussian"])
        self._notch_type.setCurrentText("Gaussian")
        r1.addWidget(self._notch_type)
        fl.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Radius"))
        self._notch_radius = QDoubleSpinBox()
        self._notch_radius.setRange(1.0, 100.0)
        self._notch_radius.setSingleStep(1.0)
        self._notch_radius.setValue(10.0)
        r2.addWidget(self._notch_radius)
        fl.addLayout(r2)

        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Order"))
        self._notch_order = QSpinBox()
        self._notch_order.setRange(1, 10)
        self._notch_order.setValue(2)
        r3.addWidget(self._notch_order)
        fl.addLayout(r3)

        btn_show = QPushButton("Show Fourier Spectrum")
        btn_show.clicked.connect(self.show_spectrum.emit)
        fl.addWidget(btn_show)

        btn_apply = QPushButton("Apply Selected Notch")
        btn_apply.clicked.connect(
            lambda: self.apply_notch.emit(
                self._notch_type.currentText(),
                self._notch_radius.value(),
                self._notch_order.value(),
            )
        )
        fl.addWidget(btn_apply)

        hint = QLabel("1) Show spectrum  2) Click bright spike  3) Apply notch")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #94a3b8; font-size: 9px;")
        fl.addWidget(hint)

        return section

    # Phase 2 / Member 4: GUI controls for the bonus binary morphology engine.
    def _build_morphology_group(self) -> QWidget:
        section = CollapsibleSection("Clinical Morphology Bonus", False)
        ml = section.bodyLayout()

        self._threshold_label = QLabel("Threshold: 128")
        ml.addWidget(self._threshold_label)

        self._threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._threshold_slider.setRange(0, 255)
        self._threshold_slider.setValue(128)
        self._threshold_slider.valueChanged.connect(
            lambda value: self._threshold_label.setText(f"Threshold: {value}")
        )
        ml.addWidget(self._threshold_slider)

        btn_threshold = QPushButton("Binarize")
        btn_threshold.clicked.connect(
            lambda: self.apply_threshold.emit(self._threshold_slider.value())
        )
        ml.addWidget(btn_threshold)

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("SE Size"))
        self._se_size = QComboBox()
        self._se_size.addItems(["3×3", "5×5", "7×7", "9×9", "11×11"])
        r1.addWidget(self._se_size)
        ml.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("SE Shape"))
        self._se_shape = QComboBox()
        self._se_shape.addItems(["Square", "Cross"])
        r2.addWidget(self._se_shape)
        ml.addLayout(r2)

        grid = QGridLayout()
        buttons = [
            ("Erosion", "erosion"),
            ("Dilation", "dilation"),
            ("Opening", "opening"),
            ("Closing", "closing"),
        ]

        for index, (label, operation) in enumerate(buttons):
            btn = QPushButton(label)
            btn.clicked.connect(
                lambda _checked=False, op=operation: self.apply_morphology.emit(
                    op,
                    self._combo_size(self._se_size),
                    self._se_shape.currentText(),
                )
            )
            grid.addWidget(btn, index // 2, index % 2)

        ml.addLayout(grid)
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
        self.apply_filter.emit(
            "average",
            {"kernel_size": self._combo_size(self._smooth_kernel)},
        )

    def _emit_gaussian(self):
        self.apply_filter.emit(
            "gaussian",
            {
                "kernel_size": self._combo_size(self._smooth_kernel),
                "sigma": self._gauss_sigma.value(),
            },
        )

    def _emit_median(self):
        self.apply_median.emit(self._combo_size(self._median_kernel))

    # Phase 2 / Member 3: emit selected noise type and parameter.
    def _emit_noise(self):
        self.apply_noise.emit(
            self._noise_type.currentText(),
            {"param": self._noise_param.value()},
        )

    def _emit_hist_eq(self):
        self.apply_hist_eq.emit(self._combo_size(self._block_size))
