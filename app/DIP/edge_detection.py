"""Edge detection filters."""

import numpy as np


def make_sobel_kernels():
	Kx = np.array([[-1, 0, 1],
				   [-2, 0, 2],
				   [-1, 0, 1]], dtype=np.float64)
	Ky = np.array([[-1, -2, -1],
				   [ 0,  0,  0],
				   [ 1,  2,  1]], dtype=np.float64)
	return Kx, Ky


def make_prewitt_kernels():
	Kx = np.array([[-1, 0, 1],
				   [-1, 0, 1],
				   [-1, 0, 1]], dtype=np.float64)
	Ky = np.array([[-1, -1, -1],
				   [ 0,  0,  0],
				   [ 1,  1,  1]], dtype=np.float64)
	return Kx, Ky


def _convolve2d_scratch(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
	kh, kw = kernel.shape
	ph, pw = kh // 2, kw // 2

	def _pad(ch2d):
		try:
			return np.pad(ch2d, ((ph, ph), (pw, pw)), mode='reflect')
		except ValueError:
			return np.pad(ch2d, ((ph, ph), (pw, pw)), mode='edge')

	padded = _pad(image.astype(np.float64))
	h, w = image.shape
	out = np.zeros((h, w), dtype=np.float64)
	flipped = kernel[::-1, ::-1]
	for r in range(h):
		for c in range(w):
			out[r, c] = np.sum(padded[r:r + kh, c:c + kw] * flipped)
	return out


def sobel_edge(gray: np.ndarray):
	"""Returns (Gx, Gy, magnitude) for Sobel edge detection."""
	Kx, Ky = make_sobel_kernels()
	Gx = _convolve2d_scratch(gray.astype(np.float64), Kx)
	Gy = _convolve2d_scratch(gray.astype(np.float64), Ky)
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
	Gx = _convolve2d_scratch(gray.astype(np.float64), Kx)
	Gy = _convolve2d_scratch(gray.astype(np.float64), Ky)
	mag = np.sqrt(Gx ** 2 + Gy ** 2)

	def _norm(arr):
		mn, mx = arr.min(), arr.max()
		if mx == mn:
			return np.zeros_like(arr, dtype=np.uint8)
		return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)

	return _norm(Gx), _norm(Gy), _norm(mag)
