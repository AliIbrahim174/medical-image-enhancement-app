import numpy as np

def compute_roi_stats(img: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    """
    Isolate the ROI pixels, compute histogram from scratch, return hist/mean/variance.
    Returns: (hist: np.ndarray shape [256], mean: float, variance: float)
    """
    # Ensure coords are ordered
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    roi = img[y1:y2, x1:x2]
    if roi.ndim == 3:
        # Convert to grayscale via luminance weights (from scratch)
        roi = (0.299 * roi[:,:,0] + 0.587 * roi[:,:,1] + 0.114 * roi[:,:,2]).astype(np.uint8)

    flat = roi.flatten().astype(np.int32)
    # Histogram from scratch — no np.histogram
    hist = np.zeros(256, dtype=np.int64)
    for px in flat:
        hist[px] += 1          # or use np.bincount for speed

    mean = float(flat.mean())
    variance = float(flat.var())
    return hist, mean, variance