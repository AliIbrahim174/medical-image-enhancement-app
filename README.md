# MedVision Workbench (DIP Project)

MedVision Workbench is a desktop application for medical and standard image enhancement, built with PyQt5 and NumPy.

The project focuses on spatial-domain digital image processing with core algorithms implemented from scratch, including interpolation, filtering, edge detection, and local histogram equalization.

## Highlights

- GUI desktop app using PyQt5
- Multi-format image I/O:
  - DICOM (.dcm, .dicom)
  - JPEG, BMP, PNG, TIFF
- From-scratch implementations for:
  - Nearest-neighbor and bilinear zoom
  - Average and Gaussian smoothing filters
  - Median filtering
  - Sobel and Prewitt edge detection
  - Block-wise local histogram equalization
- Sequential enhancement pipeline with undo support
- Live metadata panel and histogram visualization
- Background worker thread for long operations (responsive UI)

## Tech Stack

- Python 3.10+ (recommended)
- PyQt5
- NumPy
- Pillow
- pydicom

## Project Structure

```text
DIP-Project/
  main.py
  requirements.txt
  app/
    core/
      __init__.py
      image_processor.py
      styles.py   ← all Qt stylesheets and colour tokens
    DIP/   ← all DSP algorithms (from scratch)
      __init__.py
      edge_detection.py
      histogram_equalization.py
      median.py
      smoothing.py
      zoom.py
    gui/
      __init__.py
      main_window.py   ← MainWindow (layout, menus, signal wiring)
      panels.py   ← MetadataPanel + PipelinePanel
      sidebar.py  ← ToolsSidebar (all filter/zoom controls)
      widgets.py  ← ImageCanvas + HistogramWidget
    io/
      __init__.py
      image_io.py   ← DICOM / JPEG / BMP / PNG I/O
    workers/   ← ProcessingWorker (background QThread)
      __init__.py
      processing_worker.py
```

## Installation

1. Open a terminal in the project root.
2. Create a virtual environment.
3. Activate it.
4. Install dependencies.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## How to Use

1. Open an image using File > Open Image or the toolbar.
2. Choose an operation from the left sidebar:
   - Zoom (nearest-neighbor or bilinear)
   - Smoothing (average or Gaussian)
   - Edge detection (Sobel or Prewitt)
   - Median filter
   - Local histogram equalization
3. Inspect outputs in tabs:
   - Processed
   - Before / After
   - Edge View (Gx, Gy, Magnitude)
4. Track applied operations in the Pipeline panel.
5. Undo the latest step or reset to original as needed.
6. Save the processed result from File > Save Processed Image.

## Processing Notes

- Zooming is applied from a clean zoom base to avoid cumulative interpolation artifacts.
- Linear filtering routes to FFT-based convolution for larger kernels to keep the UI responsive.
- Edge operators output normalized Gx, Gy, and gradient magnitude.
- Local histogram equalization is block-wise and non-overlapping.

## Supported Inputs and Outputs

- Input:
  - DICOM (.dcm, .dicom)
  - .jpg, .jpeg, .bmp, .png, .tif, .tiff
- Output:
  - PNG
  - JPEG
  - BMP

## Course Context

This repository appears to be developed for a Medical Image Processing and Computer Vision course project at Cairo University (Biomedical Engineering).

## License

No license file is currently included in this repository.
If you plan to distribute this project, add a LICENSE file (for example: MIT, BSD-3-Clause, or GPL-3.0).
