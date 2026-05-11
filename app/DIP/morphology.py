"""Binary morphological operations implemented from scratch.

No cv2 morphology, skimage morphology, scipy morphology, or bwmorph-style
helpers are used here.
"""

from __future__ import annotations

import numpy as np

from .utils import ensure_gray


def global_threshold(image: np.ndarray, threshold: int) -> np.ndarray:
    """Convert grayscale image to binary mask using a global threshold."""
    gray = ensure_gray(image)
    t = int(np.clip(threshold, 0, 255))

    output = np.zeros(gray.shape, dtype=np.uint8)
    output[gray >= t] = 255
    return output


def make_structuring_element(size: int, shape: str = "Square") -> np.ndarray:
    """Create a binary structuring element.

    Supported shapes:
    - Square
    - Cross
    """
    size = int(size)

    if size < 3 or size % 2 == 0:
        raise ValueError("Structuring element size must be an odd integer >= 3.")

    kind = shape.strip().lower()

    if kind == "square":
        return np.ones((size, size), dtype=np.uint8)

    if kind == "cross":
        se = np.zeros((size, size), dtype=np.uint8)
        center = size // 2
        se[center, :] = 1
        se[:, center] = 1
        return se

    raise ValueError("Structuring element shape must be 'Square' or 'Cross'.")


def _as_binary(image: np.ndarray) -> np.ndarray:
    """Force any grayscale image into a clean 0/255 binary mask."""
    gray = ensure_gray(image)
    binary = np.zeros(gray.shape, dtype=np.uint8)
    binary[gray > 0] = 255
    return binary


def erode(binary_image: np.ndarray, size: int = 3, shape: str = "Square") -> np.ndarray:
    """Binary erosion from scratch.

    Output pixel is white only if all active SE positions fit inside white
    foreground pixels.
    """
    binary = _as_binary(binary_image)
    se = make_structuring_element(size, shape)

    h, w = binary.shape
    pad = size // 2
    padded = np.pad(binary, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)

    output = np.zeros((h, w), dtype=np.uint8)

    active_positions = []
    for r in range(size):
        for c in range(size):
            if se[r, c] == 1:
                active_positions.append((r, c))

    for row in range(h):
        for col in range(w):
            keep = True
            for dr, dc in active_positions:
                if padded[row + dr, col + dc] != 255:
                    keep = False
                    break
            if keep:
                output[row, col] = 255

    return output


def dilate(binary_image: np.ndarray, size: int = 3, shape: str = "Square") -> np.ndarray:
    """Binary dilation from scratch.

    Output pixel is white if any active SE position overlaps foreground.
    """
    binary = _as_binary(binary_image)
    se = make_structuring_element(size, shape)

    h, w = binary.shape
    pad = size // 2
    padded = np.pad(binary, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)

    output = np.zeros((h, w), dtype=np.uint8)

    active_positions = []
    for r in range(size):
        for c in range(size):
            if se[r, c] == 1:
                active_positions.append((r, c))

    for row in range(h):
        for col in range(w):
            hit = False
            for dr, dc in active_positions:
                if padded[row + dr, col + dc] == 255:
                    hit = True
                    break
            if hit:
                output[row, col] = 255

    return output


def opening(binary_image: np.ndarray, size: int = 3, shape: str = "Square") -> np.ndarray:
    """Opening = erosion followed by dilation.

    Clinical use: removes small foreground noise.
    """
    return dilate(erode(binary_image, size, shape), size, shape)


def closing(binary_image: np.ndarray, size: int = 3, shape: str = "Square") -> np.ndarray:
    """Closing = dilation followed by erosion.

    Clinical use: fills small gaps and holes in segmented structures.
    """
    return erode(dilate(binary_image, size, shape), size, shape)