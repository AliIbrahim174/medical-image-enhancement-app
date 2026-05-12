"""Image state management and history tracking."""

from __future__ import annotations

import numpy as np


class ImageStateManager:
    """Manages image state, history, undo/redo stacks."""

    def __init__(self):
        self.original: np.ndarray | None = None
        self.current: np.ndarray | None = None
        self.saved_state: np.ndarray | None = None

        self.history: list[np.ndarray] = []
        self.history_labels: list[str] = []

        self.redo_stack: list[np.ndarray] = []
        self.redo_labels: list[str] = []

        # Zoom state
        self.zoom_base: np.ndarray | None = None
        self.zoom_factor: float = 1.0
        self.view_zoom_percent: int = 100

        # Processing mode
        self.accumulate: bool = True  # True: apply to current; False: apply to original

        # Phase 2: Fourier notch center
        self.selected_notch_center: tuple[int, int] | None = None
        self.snap_to_bright_peak: bool = True

    def load_image(self, image_array: np.ndarray) -> None:
        """Load a new image."""
        self.original = image_array.copy()
        self.current = image_array.copy()
        self.saved_state = self.current.copy()
        self._clear_processing_state()

    def _clear_processing_state(self) -> None:
        """Clear history and redo stack."""
        self.history.clear()
        self.history_labels.clear()
        self.redo_stack.clear()
        self.redo_labels.clear()
        self.zoom_base = None
        self.zoom_factor = 1.0
        self.view_zoom_percent = 100
        self.selected_notch_center = None

    def push_to_history(self, label: str) -> None:
        """Push current state to history."""
        if self.current is not None:
            self.history.append(self.current.copy())
            self.history_labels.append(label)
            # New operation invalidates redo stack
            self.redo_stack.clear()
            self.redo_labels.clear()

    def undo(self) -> tuple[bool, str]:
        """Undo last operation. Returns (success, label)."""
        if not self.history:
            return False, "Nothing to undo."

        # Move current to redo stack
        if self.current is not None:
            self.redo_stack.append(self.current.copy())
            redo_label = self.history_labels[-1] if self.history_labels else "?"
            self.redo_labels.append(redo_label)

        # Restore previous
        self.current = self.history.pop()
        label = self.history_labels.pop() if self.history_labels else "?"

        self._reset_zoom()
        return True, label

    def redo(self) -> tuple[bool, str]:
        """Redo last undone operation. Returns (success, label)."""
        if not self.redo_stack:
            return False, "Nothing to redo."

        # Move current to history
        if self.current is not None:
            self.history.append(self.current.copy())
            redo_label = self.redo_labels[-1] if self.redo_labels else "?"
            self.history_labels.append(redo_label)

        # Restore redo
        self.current = self.redo_stack.pop()
        redo_label = self.redo_labels.pop() if self.redo_labels else "?"

        self._reset_zoom()
        return True, redo_label

    def reset_to_original(self) -> bool:
        """Reset to original image."""
        if self.original is None:
            return False

        self.current = self.original.copy()
        self._clear_processing_state()
        return True

    def _reset_zoom(self) -> None:
        """Reset zoom state."""
        self.zoom_base = None
        self.zoom_factor = 1.0

    def is_dirty(self) -> bool:
        """Check if current state differs from saved state."""
        if self.current is None:
            return False
        if self.saved_state is None:
            return True
        return not np.array_equal(self.current, self.saved_state)

    def mark_saved(self) -> None:
        """Mark current state as saved."""
        if self.current is not None:
            self.saved_state = self.current.copy()

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self.history) > 0

    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(self.redo_stack) > 0

    def get_source_image(self) -> np.ndarray | None:
        """Get processing source based on accumulate mode."""
        if self.current is None:
            return None

        if self.accumulate:
            return self.current.copy()

        if self.original is None:
            return self.current.copy()

        return self.original.copy()

    def set_accumulate_mode(self, enabled: bool) -> None:
        """Set accumulate mode."""
        self.accumulate = enabled
