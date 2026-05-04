"""
Background QThread worker that runs any processing function off the main
thread, preventing the GUI from freezing during heavy operations.
"""

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


class ProcessingWorker(QThread):
    """
    Runs an arbitrary callable in a background thread.

    Signals
    -------
    finished(np.ndarray, str)
        Emitted on success with (result_array, operation_label).
    error(str)
        Emitted on any exception with the error message.
    """

    finished = pyqtSignal(np.ndarray, str)
    error    = pyqtSignal(str)

    def __init__(self, func, label: str, *args, **kwargs):
        super().__init__()
        self._func   = func
        self._label  = label
        self._args   = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._func(*self._args, **self._kwargs)
            self.finished.emit(result, self._label)
        except Exception as exc:
            self.error.emit(str(exc))