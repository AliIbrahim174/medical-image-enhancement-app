"""Linear smoothing filters."""

import math
import numpy as np


def make_average_kernel(size: int) -> np.ndarray:
	"""Generate a normalized box (average) kernel of given odd size."""
	return np.ones((size, size), dtype=np.float64) / (size * size)


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


def _convolve2d_scratch(image: np.ndarray, kernel: np.ndarray,
						pad_mode: str = "reflect") -> np.ndarray:
	kh, kw = kernel.shape
	ph, pw = kh // 2, kw // 2

	def _pad(ch2d):
		if pad_mode == "reflect":
			try:
				return np.pad(ch2d, ((ph, ph), (pw, pw)), mode='reflect')
			except ValueError:
				return np.pad(ch2d, ((ph, ph), (pw, pw)), mode='edge')
		elif pad_mode == "replicate":
			return np.pad(ch2d, ((ph, ph), (pw, pw)), mode='edge')
		return np.pad(ch2d, ((ph, ph), (pw, pw)), mode='constant', constant_values=0)

	def _conv_single(ch2d):
		padded = _pad(ch2d.astype(np.float64))
		h, w = ch2d.shape
		out = np.zeros((h, w), dtype=np.float64)
		flipped = kernel[::-1, ::-1]
		for r in range(h):
			for c in range(w):
				out[r, c] = np.sum(padded[r:r + kh, c:c + kw] * flipped)
		return out

	if image.ndim == 3:
		return np.stack([_conv_single(image[:, :, ch]) for ch in range(image.shape[2])], axis=2)
	return _conv_single(image.astype(np.float64))


def _fast_convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
	def _fft_conv(ch2d, k):
		h, w = ch2d.shape
		kh, kw = k.shape
		fh = h + kh - 1
		fw = w + kw - 1
		F = np.fft.fft2(ch2d.astype(np.float64), s=(fh, fw))
		G = np.fft.fft2(k[::-1, ::-1], s=(fh, fw))
		conv_full = np.real(np.fft.ifft2(F * G))
		ph, pw = kh // 2, kw // 2
		return conv_full[ph:ph + h, pw:pw + w]

	if image.ndim == 3:
		return np.stack([_fft_conv(image[:, :, c], kernel) for c in range(image.shape[2])], axis=2)
	return _fft_conv(image.astype(np.float64), kernel)


def apply_linear_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
	"""Route convolution: scratch for small kernels, FFT for large ones."""
	kh, kw = kernel.shape
	if kh * kw >= 49:
		result = _fast_convolve2d(image, kernel)
	else:
		result = _convolve2d_scratch(image, kernel)
	return np.clip(result, 0, 255).astype(np.uint8)
