"""
Image loading (DICOM / JPEG / BMP / PNG) and metadata extraction.
"""

import os
import numpy as np
from PIL import Image

try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False


class ImageLoadResult:
    def __init__(self):
        self.pixel_array: np.ndarray | None = None
        self.metadata: dict = {}
        self.error: str = ""
        self.file_path: str = ""
        self.format: str = ""


def load_image(file_path: str) -> ImageLoadResult:
    result = ImageLoadResult()
    result.file_path = file_path

    if not os.path.isfile(file_path):
        result.error = f"File not found: {file_path}"
        return result

    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext in (".dcm", ".dicom", ""):
            return _load_dicom(file_path, result)
        elif ext in (".jpg", ".jpeg", ".bmp", ".png", ".tif", ".tiff"):
            return _load_standard(file_path, result)
        else:
            # Try DICOM first, fallback to Pillow
            try:
                return _load_dicom(file_path, result)
            except Exception:
                return _load_standard(file_path, result)
    except Exception as e:
        result.error = str(e)
        return result


def _load_dicom(file_path: str, result: ImageLoadResult) -> ImageLoadResult:
    if not DICOM_AVAILABLE:
        result.error = "pydicom not installed. Cannot open DICOM file."
        return result

    ds = pydicom.dcmread(file_path, force=True)
    pixels = ds.pixel_array.astype(np.float64)

    # Normalise to 8-bit
    mn, mx = pixels.min(), pixels.max()
    if mx > mn:
        pixels = (pixels - mn) / (mx - mn) * 255.0
    pixels = pixels.astype(np.uint8)

    # Ensure 2D or 3-channel
    if pixels.ndim == 2:
        pass  # grayscale
    elif pixels.ndim == 3 and pixels.shape[0] < 5:
        # (slices, H, W) → take first slice
        pixels = pixels[0]

    result.pixel_array = pixels
    result.format = "DICOM"

    # Metadata extraction
    meta = {}
    meta["Width"] = str(getattr(ds, "Columns", pixels.shape[1] if pixels.ndim >= 2 else "N/A"))
    meta["Height"] = str(getattr(ds, "Rows", pixels.shape[0] if pixels.ndim >= 1 else "N/A"))
    meta["Bit Depth"] = str(getattr(ds, "BitsStored", "N/A"))
    meta["Modality"] = str(getattr(ds, "Modality", "N/A"))
    meta["Patient Name"] = str(getattr(ds, "PatientName", "N/A"))
    meta["Patient Age"] = str(getattr(ds, "PatientAge", "N/A"))
    meta["Body Part"] = str(getattr(ds, "BodyPartExamined", "N/A"))
    meta["Study Date"] = str(getattr(ds, "StudyDate", "N/A"))
    meta["Institution"] = str(getattr(ds, "InstitutionName", "N/A"))
    meta["Manufacturer"] = str(getattr(ds, "Manufacturer", "N/A"))
    result.metadata = meta
    return result


def _load_standard(file_path: str, result: ImageLoadResult) -> ImageLoadResult:
    img = Image.open(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    result.format = img.format or ext.upper().lstrip(".")

    # Convert to RGB or keep grayscale
    if img.mode == "RGBA":
        img = img.convert("RGB")
    elif img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    pixels = np.array(img)
    result.pixel_array = pixels

    h, w = pixels.shape[:2]
    bit_depth = 8 if img.mode == "L" else 24
    meta = {
        "Width": str(w),
        "Height": str(h),
        "Bit Depth": str(bit_depth),
        "Mode": img.mode,
        "Format": result.format,
    }
    result.metadata = meta
    return result


def save_image(pixel_array: np.ndarray, file_path: str) -> str:
    """Save a numpy array as image. Returns error string or empty string on success."""
    try:
        if pixel_array.ndim == 2:
            img = Image.fromarray(pixel_array.astype(np.uint8), mode="L")
        elif pixel_array.ndim == 3:
            img = Image.fromarray(pixel_array.astype(np.uint8), mode="RGB")
        else:
            return "Unsupported array shape for saving."
        img.save(file_path)
        return ""
    except Exception as e:
        return str(e)