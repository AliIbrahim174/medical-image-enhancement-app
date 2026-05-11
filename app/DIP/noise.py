import numpy as np

def inject_gaussian_noise(img: np.ndarray, mean: float = 0.0, std: float = 25.0) -> np.ndarray:
    """Add Gaussian noise from scratch using Box-Muller or numpy normal."""
    noise = np.random.normal(mean, std, img.shape)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def inject_uniform_noise(img: np.ndarray, low: float = -40.0, high: float = 40.0) -> np.ndarray:
    """Add Uniform noise from scratch."""
    noise = np.random.uniform(low, high, img.shape)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)