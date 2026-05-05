"""Utility functions for image processing."""

import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert RGB/RGBA image to grayscale without built-in colour functions."""
    if image.ndim == 2:
        return image.astype(np.uint8)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    # ITU-R BT.601 weights
    gray = (0.299 * image[:, :, 0] +
            0.587 * image[:, :, 1] +
            0.114 * image[:, :, 2])
    return np.clip(gray, 0, 255).astype(np.uint8)


def ensure_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return to_gray(image)
    return image.astype(np.uint8)
