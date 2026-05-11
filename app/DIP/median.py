"""Median filter implementation."""

import numpy as np


def median_filter_scratch(image: np.ndarray, kernel_size: int) -> np.ndarray:
	"""
	2-D median filter implemented from scratch.
	Works on grayscale or RGB images.
	"""
	pad = kernel_size // 2

	def _med_single(ch2d):
		h, w = ch2d.shape
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
