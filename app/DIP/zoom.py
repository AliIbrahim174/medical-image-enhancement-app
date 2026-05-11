"""Zoom and interpolation filters."""

import numpy as np


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
