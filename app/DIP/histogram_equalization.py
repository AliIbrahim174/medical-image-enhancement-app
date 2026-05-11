"""Local histogram equalization filter."""

import numpy as np


def local_histogram_equalization(gray: np.ndarray, block_size: int) -> np.ndarray:
	"""
	Local (block-wise) histogram equalization from scratch.
	Each non-overlapping block is equalized independently.
	"""
	if gray.ndim == 3:
		gray = (0.299 * gray[:, :, 0] +
        0.587 * gray[:, :, 1] +
        0.114 * gray[:, :, 2]).astype(np.uint8)
	h, w = gray.shape
	output = gray.copy().astype(np.uint8)

	for r0 in range(0, h, block_size):
		for c0 in range(0, w, block_size):
			r1 = min(r0 + block_size, h)
			c1 = min(c0 + block_size, w)
			block = gray[r0:r1, c0:c1].astype(np.int32)

			hist = [0] * 256
			for val in block.flatten():
				hist[val] += 1

			cdf = [0] * 256
			cdf[0] = hist[0]
			for i in range(1, 256):
				cdf[i] = cdf[i - 1] + hist[i]

			cdf_min = next((v for v in cdf if v > 0), 1)
			total = block.size

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