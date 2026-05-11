"""Frequency-domain processing and interactive notch filtering.

This module contains only algorithmic logic. GUI code should call these
functions but should not implement Fourier masks directly.
"""

from __future__ import annotations

import numpy as np

from .utils import ensure_gray


def _normalize_to_uint8(values: np.ndarray) -> np.ndarray:
    """Normalize a numeric array to uint8 range [0, 255]."""
    arr = values.astype(np.float64)
    mn = float(np.min(arr))
    mx = float(np.max(arr))

    if mx <= mn:
        return np.zeros(arr.shape, dtype=np.uint8)

    normalized = (arr - mn) / (mx - mn) * 255.0
    return np.clip(normalized, 0, 255).astype(np.uint8)


def shifted_magnitude_spectrum(image: np.ndarray) -> np.ndarray:
    """Return log-scaled shifted Fourier magnitude spectrum as uint8 image.

    Steps:
    1. Convert image to grayscale.
    2. Compute 2-D Fourier transform.
    3. Shift low frequencies to the center.
    4. Use log scaling for visibility.
    """
    gray = ensure_gray(image).astype(np.float64)
    fourier = np.fft.fft2(gray)
    shifted = np.fft.fftshift(fourier)
    magnitude = np.log1p(np.abs(shifted))
    return _normalize_to_uint8(magnitude)


def conjugate_notch_center(shape: tuple[int, int], center: tuple[int, int]) -> tuple[int, int]:
    """Return symmetric conjugate notch position in the shifted spectrum.

    Parameters
    ----------
    shape:
        Spectrum shape as (height, width).
    center:
        Clicked position as (row, col).

    In a shifted Fourier spectrum, the DC point is around (H//2, W//2).
    The conjugate point is mirrored around this center.
    """
    h, w = shape
    row, col = center
    crow, ccol = h // 2, w // 2

    mirror_row = (2 * crow - row) % h
    mirror_col = (2 * ccol - col) % w
    return int(mirror_row), int(mirror_col)


def _distance_grid(shape: tuple[int, int], center: tuple[int, int]) -> np.ndarray:
    """Compute Euclidean distance from every frequency pixel to center."""
    h, w = shape
    row0, col0 = center

    rows, cols = np.indices((h, w))
    return np.sqrt((rows - row0) ** 2 + (cols - col0) ** 2)


def _single_notch_mask(
    shape: tuple[int, int],
    center: tuple[int, int],
    radius: float,
    filter_type: str,
    order: int,
) -> np.ndarray:
    """Generate one notch reject component centered at one spike."""
    radius = max(float(radius), 1e-6)
    order = max(int(order), 1)

    distance = _distance_grid(shape, center)
    kind = filter_type.strip().lower()

    if kind == "ideal":
        mask = np.ones(shape, dtype=np.float64)
        mask[distance <= radius] = 0.0
        return mask

    if kind == "butterworth":
        # Butterworth high-pass shape centered at the noise spike.
        # At D = 0 -> reject, far from center -> pass.
        safe_distance = np.maximum(distance, 1e-6)
        return 1.0 / (1.0 + (radius / safe_distance) ** (2 * order))

    if kind == "gaussian":
        # Gaussian notch reject: zero at the spike center, approaches one away.
        return 1.0 - np.exp(-(distance ** 2) / (2.0 * radius ** 2))

    raise ValueError("filter_type must be 'Ideal', 'Butterworth', or 'Gaussian'.")


def make_notch_reject_mask(
    shape: tuple[int, int],
    center: tuple[int, int],
    radius: float,
    filter_type: str = "Gaussian",
    order: int = 2,
    include_conjugate: bool = True,
) -> np.ndarray:
    """Generate a notch reject mask and its conjugate mirror notch.

    The conjugate notch is essential because real spatial images have
    conjugate-symmetric Fourier spectra. Removing only one spike can produce
    complex artifacts after inverse FFT.
    """
    h, w = shape
    row, col = center

    if not (0 <= row < h and 0 <= col < w):
        raise ValueError("Notch center is outside the spectrum dimensions.")

    centers = [(int(row), int(col))]

    if include_conjugate:
        mirror = conjugate_notch_center(shape, center)
        if mirror not in centers:
            centers.append(mirror)

    mask = np.ones(shape, dtype=np.float64)
    for notch_center in centers:
        mask *= _single_notch_mask(shape, notch_center, radius, filter_type, order)

    return np.clip(mask, 0.0, 1.0)


def _apply_mask_to_channel(channel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply shifted notch mask to one 2-D image channel."""
    channel_float = channel.astype(np.float64)

    fourier = np.fft.fft2(channel_float)
    shifted = np.fft.fftshift(fourier)

    filtered_shifted = shifted * mask

    unshifted = np.fft.ifftshift(filtered_shifted)
    restored = np.fft.ifft2(unshifted)

    return np.real(restored)


def apply_notch_filter(
    image: np.ndarray,
    center: tuple[int, int],
    radius: float,
    filter_type: str = "Gaussian",
    order: int = 2,
) -> np.ndarray:
    """Apply notch reject filter to grayscale or RGB image.

    For RGB images, the same Fourier mask is applied independently to each
    channel to preserve color structure.
    """
    if image.ndim == 2:
        shape = image.shape
        mask = make_notch_reject_mask(shape, center, radius, filter_type, order)
        restored = _apply_mask_to_channel(image, mask)
        return np.clip(restored, 0, 255).astype(np.uint8)

    if image.ndim == 3:
        shape = image.shape[:2]
        mask = make_notch_reject_mask(shape, center, radius, filter_type, order)

        channels = []
        for ch in range(image.shape[2]):
            channels.append(_apply_mask_to_channel(image[:, :, ch], mask))

        restored = np.stack(channels, axis=2)
        return np.clip(restored, 0, 255).astype(np.uint8)

    raise ValueError("Unsupported image shape for notch filtering.")


def add_sinusoidal_noise(
    image: np.ndarray,
    amplitude: float = 35.0,
    frequency_x: float = 12.0,
    frequency_y: float = 0.0,
) -> np.ndarray:
    """Optional helper for manual testing: add synthetic periodic noise.

    This is not required by your specific task, but it helps you demonstrate
    the notch filter during the defense.
    """
    gray_or_color = image.astype(np.float64)
    h, w = image.shape[:2]

    rows, cols = np.indices((h, w))
    pattern = amplitude * np.sin(
        2.0 * np.pi * ((frequency_x * cols / w) + (frequency_y * rows / h))
    )

    if image.ndim == 2:
        noisy = gray_or_color + pattern
    else:
        noisy = gray_or_color + pattern[:, :, None]

    return np.clip(noisy, 0, 255).astype(np.uint8)