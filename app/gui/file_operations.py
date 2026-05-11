"""File I/O operations for the image editor."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..io.image_io import load_image, save_image


class FileOperations:
    """Handles file loading and saving."""

    @staticmethod
    def load_image_file(path: str) -> dict:
        """
        Load an image file.
        Returns dict with keys: success, error, image, metadata, format, width, height.
        """
        try:
            result = load_image(path)
            if result.error:
                return {
                    "success": False,
                    "error": result.error,
                    "image": None,
                    "metadata": {},
                }

            if result.pixel_array is None:
                return {
                    "success": False,
                    "error": "Could not decode image pixels.",
                    "image": None,
                    "metadata": {},
                }

            return {
                "success": True,
                "error": None,
                "image": result.pixel_array.copy(),
                "metadata": dict(result.metadata),
                "format": result.format,
                "width": result.metadata.get("Width", "?"),
                "height": result.metadata.get("Height", "?"),
                "mode": result.metadata.get("Mode", "Gray"),
            }
        except Exception as exc:
            return {
                "success": False,
                "error": f"Unexpected error loading image: {str(exc)}",
                "image": None,
                "metadata": {},
            }

    @staticmethod
    def save_image_file(image: np.ndarray, path: str) -> tuple[bool, str]:
        """Save an image file. Returns (success, error_message)."""
        try:
            err = save_image(image, path)
            if err:
                return False, err
            return True, ""
        except Exception as exc:
            return False, f"Unexpected error saving image: {str(exc)}"

    @staticmethod
    def format_file_size(path: str) -> str:
        """Format file size in human-readable format."""
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

        if unit == 0:
            return f"{value:.0f} {units[unit]}"
        return f"{value:.1f} {units[unit]}"

    @staticmethod
    def get_filename(path: str) -> str:
        """Get filename from path."""
        return os.path.basename(path)
