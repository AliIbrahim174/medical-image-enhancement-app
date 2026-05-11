from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QHBoxLayout
import numpy as np
from .widgets import HistogramWidget

class ROIStatsDialog(QDialog):
    def __init__(self, hist: np.ndarray, mean: float, variance: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ROI — Local Statistics")
        self.setMinimumSize(340, 320)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        title = QLabel("Local ROI Histogram")
        title.setStyleSheet("font-weight: 600; font-size: 11px;")
        layout.addWidget(title)

        self._hist_widget = HistogramWidget()
        # Feed raw hist array directly
        self._hist_widget.set_hist(hist)   # add this small method to HistogramWidget
        layout.addWidget(self._hist_widget)

        stats_row = QHBoxLayout()
        stats_row.addWidget(QLabel(f"Mean:  {mean:.4f}"))
        stats_row.addWidget(QLabel(f"Variance:  {variance:.4f}"))
        layout.addLayout(stats_row)