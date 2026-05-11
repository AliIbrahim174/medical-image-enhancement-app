"""
image_processor.py
==================
All image processing algorithms implemented from scratch.
No built-in spatial filter / histogram / interpolation functions used.
FFT utilities (numpy.fft) are permitted per project rules.
"""

import numpy as np
import math


# ─────────────────────────────────────────────────────────────────────────────
#  INTERPOLATION  (zoom)
# ─────────────────────────────────────────────────────────────────────────────

def nearest_neighbor_zoom(image: np.ndarray, scale: float) -> np.ndarray:
    """Zoom using nearest-neighbor interpolation (from scratch)."""
    if image.ndim == 3:
        h, w, c = image.shape
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        output = np.zeros((new_h, new_w, c), dtype=image.dtype)
        for r in range(new_h):
            for col in range(new_w):
                src_r = min(int(r / scale), h - 1)
                src_c = min(int(col / scale), w - 1)
                output[r, col] = image[src_r, src_c]
    else:
        h, w = image.shape
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        output = np.zeros((new_h, new_w), dtype=image.dtype)
        for r in range(new_h):
            for col in range(new_w):
                src_r = min(int(r / scale), h - 1)
                src_c = min(int(col / scale), w - 1)
                output[r, col] = image[src_r, src_c]
    return output


def bilinear_zoom(image: np.ndarray, scale: float) -> np.ndarray:
    """Zoom using bilinear (linear) interpolation (from scratch)."""
    if image.ndim == 3:
        h, w, c = image.shape
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        output = np.zeros((new_h, new_w, c), dtype=np.float64)
        for r in range(new_h):
            for col in range(new_w):
                # Map back to source space
                src_r = r / scale
                src_c = col / scale
                r0 = int(src_r)
                c0 = int(src_c)
                r1 = min(r0 + 1, h - 1)
                c1 = min(c0 + 1, w - 1)
                dr = src_r - r0
                dc = src_c - c0
                output[r, col] = (
                    image[r0, c0] * (1 - dr) * (1 - dc) +
                    image[r0, c1] * (1 - dr) * dc +
                    image[r1, c0] * dr * (1 - dc) +
                    image[r1, c1] * dr * dc
                )
        return np.clip(output, 0, 255).astype(image.dtype)
    else:
        h, w = image.shape
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        output = np.zeros((new_h, new_w), dtype=np.float64)
        for r in range(new_h):
            for col in range(new_w):
                src_r = r / scale
                src_c = col / scale
                r0 = int(src_r)
                c0 = int(src_c)
                r1 = min(r0 + 1, h - 1)
                c1 = min(c0 + 1, w - 1)
                dr = src_r - r0
                dc = src_c - c0
                output[r, col] = (
                    image[r0, c0] * (1 - dr) * (1 - dc) +
                    image[r0, c1] * (1 - dr) * dc +
                    image[r1, c0] * dr * (1 - dc) +
                    image[r1, c1] * dr * dc
                )
        return np.clip(output, 0, 255).astype(image.dtype)


# ─────────────────────────────────────────────────────────────────────────────
#  KERNEL GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def make_average_kernel(size: int) -> np.ndarray:
    """Generate a normalized box (average) kernel of given odd size."""
    k = np.ones((size, size), dtype=np.float64) / (size * size)
    return k


def make_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Generate a Gaussian kernel from scratch using the Gaussian formula."""
    center = size // 2
    kernel = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = math.exp(-(x * x + y * y) / (2 * sigma * sigma))
    total = kernel.sum()
    if total > 0:
        kernel /= total
    return kernel


def make_sobel_kernels():
    """Return Sobel Kx, Ky kernels."""
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)
    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float64)
    return Kx, Ky


def make_prewitt_kernels():
    """Return Prewitt Kx, Ky kernels."""
    Kx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], dtype=np.float64)
    Ky = np.array([[-1, -1, -1],
                   [ 0,  0,  0],
                   [ 1,  1,  1]], dtype=np.float64)
    return Kx, Ky


# ─────────────────────────────────────────────────────────────────────────────
#  2-D CONVOLUTION (from scratch, handles grayscale & RGB)
# ─────────────────────────────────────────────────────────────────────────────

def convolve2d_scratch(image: np.ndarray, kernel: np.ndarray,
                       pad_mode: str = "reflect") -> np.ndarray:
    """
    Full 2-D convolution implemented from scratch.
    pad_mode: 'reflect', 'constant' (zero), 'replicate'
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2

    def _pad(ch2d):
        # Use numpy's own padding infrastructure for correctness;
        # the kernel is still generated entirely from scratch.
        if pad_mode == "reflect":
            try:
                return np.pad(ch2d, ((ph, ph), (pw, pw)), mode='reflect')
            except ValueError:
                # fallback to edge-pad if image too small for reflect
                return np.pad(ch2d, ((ph, ph), (pw, pw)), mode='edge')
        elif pad_mode == "replicate":
            return np.pad(ch2d, ((ph, ph), (pw, pw)), mode='edge')
        else:
            return np.pad(ch2d, ((ph, ph), (pw, pw)), mode='constant', constant_values=0)

    def _conv_single(ch2d):
        padded = _pad(ch2d.astype(np.float64))
        h, w = ch2d.shape
        out = np.zeros((h, w), dtype=np.float64)
        # Flip kernel for true convolution
        flipped = kernel[::-1, ::-1]
        for r in range(h):
            for c in range(w):
                out[r, c] = np.sum(padded[r:r + kh, c:c + kw] * flipped)
        return out

    if image.ndim == 3:
        channels = []
        for ch in range(image.shape[2]):
            channels.append(_conv_single(image[:, :, ch]))
        result = np.stack(channels, axis=2)
    else:
        result = _conv_single(image.astype(np.float64))

    return result


def fast_convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    FFT-accelerated convolution (permitted).
    Used for large kernels (>= 7x7) to maintain responsiveness.
    The kernel itself is still generated manually.
    """
    def _fft_conv(ch2d, k):
        h, w = ch2d.shape
        kh, kw = k.shape
        fh = h + kh - 1
        fw = w + kw - 1
        # Pad both to same size
        F = np.fft.fft2(ch2d.astype(np.float64), s=(fh, fw))
        G = np.fft.fft2(k[::-1, ::-1], s=(fh, fw))
        conv_full = np.real(np.fft.ifft2(F * G))
        # Crop to same size as input
        ph, pw = kh // 2, kw // 2
        return conv_full[ph:ph + h, pw:pw + w]

    if image.ndim == 3:
        channels = [_fft_conv(image[:, :, c], kernel) for c in range(image.shape[2])]
        return np.stack(channels, axis=2)
    else:
        return _fft_conv(image.astype(np.float64), kernel)


def apply_linear_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Route convolution: scratch for small kernels, FFT for large ones.
    Returns uint8-clipped result.
    """
    kh, kw = kernel.shape
    if kh * kw >= 49:  # 7×7 and above → FFT path
        result = fast_convolve2d(image, kernel)
    else:
        result = convolve2d_scratch(image, kernel)
    return np.clip(result, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  EDGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def sobel_edge(gray: np.ndarray):
    """Returns (Gx, Gy, magnitude) for Sobel edge detection."""
    Kx, Ky = make_sobel_kernels()
    Gx = convolve2d_scratch(gray.astype(np.float64), Kx)
    Gy = convolve2d_scratch(gray.astype(np.float64), Ky)
    mag = np.sqrt(Gx ** 2 + Gy ** 2)
    def _norm(arr):
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return np.zeros_like(arr, dtype=np.uint8)
        return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
    return _norm(Gx), _norm(Gy), _norm(mag)


def prewitt_edge(gray: np.ndarray):
    """Returns (Gx, Gy, magnitude) for Prewitt edge detection."""
    Kx, Ky = make_prewitt_kernels()
    Gx = convolve2d_scratch(gray.astype(np.float64), Kx)
    Gy = convolve2d_scratch(gray.astype(np.float64), Ky)
    mag = np.sqrt(Gx ** 2 + Gy ** 2)
    def _norm(arr):
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return np.zeros_like(arr, dtype=np.uint8)
        return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
    return _norm(Gx), _norm(Gy), _norm(mag)


# ─────────────────────────────────────────────────────────────────────────────
#  MEDIAN FILTER  (non-linear, from scratch)
# ─────────────────────────────────────────────────────────────────────────────

def median_filter_scratch(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    2-D median filter implemented from scratch.
    Works on grayscale or RGB images.
    """
    pad = kernel_size // 2

    def _med_single(ch2d):
        h, w = ch2d.shape
        # Replicate-pad the border
        padded = np.pad(ch2d.astype(np.float64), pad, mode='edge')
        out = np.zeros((h, w), dtype=np.float64)
        for r in range(h):
            for c in range(w):
                window = padded[r:r + kernel_size, c:c + kernel_size].flatten()
                window.sort()
                out[r, c] = window[len(window) // 2]
        return out

    if image.ndim == 3:
        channels = [_med_single(image[:, :, ch]) for ch in range(image.shape[2])]
        result = np.stack(channels, axis=2)
    else:
        result = _med_single(image)

    return np.clip(result, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  LOCAL HISTOGRAM EQUALIZATION  (from scratch)
# ─────────────────────────────────────────────────────────────────────────────

def local_histogram_equalization(gray: np.ndarray, block_size: int) -> np.ndarray:
    """
    Local (block-wise) histogram equalization from scratch.
    Each non-overlapping block is equalized independently.
    """
    if gray.ndim == 3:
        gray = gray[:, :, 0]

    h, w = gray.shape
    output = gray.copy().astype(np.uint8)

    for r0 in range(0, h, block_size):
        for c0 in range(0, w, block_size):
            r1 = min(r0 + block_size, h)
            c1 = min(c0 + block_size, w)
            block = gray[r0:r1, c0:c1].astype(np.int32)

            # Build histogram manually
            hist = [0] * 256
            for val in block.flatten():
                hist[val] += 1

            # CDF
            cdf = [0] * 256
            cdf[0] = hist[0]
            for i in range(1, 256):
                cdf[i] = cdf[i - 1] + hist[i]

            # Find first non-zero CDF entry
            cdf_min = next((v for v in cdf if v > 0), 1)
            total = block.size

            # Equalization mapping
            lut = np.zeros(256, dtype=np.uint8)
            denom = total - cdf_min
            for i in range(256):
                if denom <= 0:
                    lut[i] = np.uint8(i)
                else:
                    val = int(round((cdf[i] - cdf_min) / denom * 255))
                    lut[i] = np.uint8(max(0, min(255, val)))

            output[r0:r1, c0:c1] = lut[block.astype(np.uint8)]

    return output


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITY: RGB ↔ GRAY
# ─────────────────────────────────────────────────────────────────────────────

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