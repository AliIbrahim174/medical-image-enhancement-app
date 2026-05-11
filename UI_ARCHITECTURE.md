# MedVision Workbench — UI Architecture & Design

## Overview

The **MedVision Workbench** is a PyQt6-based image processing application built with a **clean separation of concerns** between UI logic and business logic. The UI follows a **command-dispatch pattern** with signals that communicate user intent to processing handlers, keeping presentation decoupled from computation.

### Key Design Principle
- **UI emits signals** → **MainWindow handles & dispatches** → **DIP modules compute** → **Worker threads prevent freezing**
- No direct processing calls from UI; all routing through `MainWindow` handlers

---

## Project Structure

```
app/gui/
├── main_window.py          # Main application shell & orchestration
├── sidebar.py              # Left-side tools panel with collapsible sections
├── panels.py               # Right-side metadata, histogram, pipeline tracking
├── widgets.py              # Reusable components (ImageCanvas, HistogramWidget)
└── __init__.py
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          MedVision Workbench (QMainWindow)          │
├──────────┬──────────────────────────────────────────────────────────┤
│          │                                                           │
│  Icon    │  Canvas Area (QTabWidget)                    │ Right Panel
│  Panel   │  ┌──────────────────────────────┐            │ ┌─────────
│  (Tool   │  │ Single View                  │            │ │ Info Tab
│  Palette)│  │ (ImageCanvas)                │            │ │ ├─ Metadata
│  ├─ Pan  │  └──────────────────────────────┘            │ │ └─ Histogram
│  ├─ Zoom │  ┌──────────────────────────────┐            │ │
│  ├─ Crop │  │ Before/After Split View      │            │ │ Histogram Tab
│  ├─ etc. │  │ (ImageCanvas + ImageCanvas)  │            │ │ ├─ Visual
│  │        │  └──────────────────────────────┘            │ │ └─ Stats
│  │        │  ┌──────────────────────────────┐            │ │
│  │ Sidebar│  │ Edge View (3-pane)           │            │ │ Pipeline Tab
│  │ ┌──────┤  │ Gx | Gy | Magnitude          │            │ │ ├─ Op List
│  │ │Zoom  │  │ (3× ImageCanvas)             │            │ │ ├─ Undo
│  │ │Smooth│  └──────────────────────────────┘            │ │ └─ Reset
│  │ │Edge  │  [Zoom footer: − slider + %]                 │ │
│  │ │Median│                                              │ │
│  │ │HistEq│                                              │ │
│  │ └──────┤                                              │ │
│  │        │                                              │ │
│  └────────┴──────────────────────────────────────────────┴─┘
│                                                           
└────────────────────────────────────────────────────────────
                    Top Bar (Menu + Toolbar)
                    Status Bar (Progress, Cursor, etc.)
```

---

## Core Components

### 1. **MainWindow** (`main_window.py`)

**Role:** Central orchestrator and state manager

#### Responsibilities:
- Manage image state (`_original`, `_current`, `_history`)
- Build and layout all UI components
- Connect sidebar signals to processing handlers
- Update all views when image changes
- Handle file I/O and undo/redo stack

#### Key State Variables:
```python
_original         # Original unmodified image (np.ndarray)
_current          # Currently displayed image (accumulated or from original)
_history          # Stack of all applied operations (for undo)
_history_labels   # Labels describing each operation
_redo_stack       # Stack of undone operations (for redo)
_saved_state      # Last saved checkpoint for dirty state tracking
_accumulate       # Mode flag: True = apply to current; False = apply to original
_worker           # Background ProcessingWorker thread
_zoom_factor      # Zoom scale for image zoom operations
_view_zoom_percent# Display zoom percentage (pan/zoom in viewport)
```

#### Canvas Management:
```python
_canvas_proc      # Main single-view canvas (always present)
_canvas_orig      # Before/After split view (left pane)
_canvas_proc2     # Before/After split view (right pane)
_canvas_gx        # Edge View: Horizontal gradients
_canvas_gy        # Edge View: Vertical gradients
_canvas_edge      # Edge View: Magnitude (sqrt(Gx²+Gy²))
```

#### View Modes (Tab switching):
1. **Single** – Shows processed image only
2. **Before/After** – Side-by-side comparison
3. **Edge View** – Three-pane gradient/magnitude display

#### Event Handlers (Signal-Slot Connections):
| Sidebar Signal | Handler | Action |
|---|---|---|
| `apply_filter(str, dict)` | `_on_apply_filter()` | Dispatch to smoothing (Gaussian/Average) |
| `apply_zoom(float, str)` | `_on_apply_zoom()` | Zoom in/out with selected interpolation |
| `apply_edge(str, str)` | `_on_apply_edge()` | Run Sobel/Prewitt edge detection |
| `apply_hist_eq(int)` | `_on_hist_eq()` | Apply local histogram equalization |
| `apply_median(int)` | `_on_median()` | Apply median filter |
| `accumulate_toggled(bool)` | `_on_accumulate_toggled()` | Switch mode (accumulated vs fresh) |

#### File Operations:
- **Open**: `Ctrl+O` → Load image, reset history, update all panels
- **Save**: `Ctrl+S` → Save current image to disk
- **Undo**: `Ctrl+Z` → Pop from history, display previous
- **Redo**: `Ctrl+Shift+Z` → Pop from redo stack, redisplay
- **Reset**: `Ctrl+R` → Revert to original, clear history

#### Processing Pipeline:
```python
def _on_apply_filter(self, filter_name: str, params: dict):
    # 1. Show progress indicator
    self._set_status(f"Applying {filter_name}...", True)
    self._progress.setVisible(True)
    
    # 2. Select base image (current if accumulate, else original)
    base = self._current if self._accumulate else self._original
    
    # 3. Choose processing function based on filter_name
    # 4. Spawn background worker thread
    self._worker = ProcessingWorker(func, label, base, **params)
    self._worker.finished.connect(self._on_processing_done)
    self._worker.error.connect(self._on_processing_error)
    self._worker.start()
    
    # 5. Main thread remains responsive
```

#### Synchronization Methods:
- `_sync_dirty_state()` – Update window title with "*" if modified
- `_sync_view_zoom()` – Distribute zoom value to all canvases
- `_update_stats_and_metadata()` – Compute histogram, entropy, min/max
- `_all_canvases()` – Return list of all active ImageCanvas instances

---

### 2. **ToolsSidebar** (`sidebar.py`)

**Role:** Parameter input & operation request interface

#### Architecture:
- **Fixed width**: 220px
- **Scrollable container** to prevent clipping on small screens
- **Collapsible sections** with toggle buttons (▶/▼)

#### Signal Definitions:
```python
apply_filter     = pyqtSignal(str, dict)      # filter_name, params dict
apply_zoom       = pyqtSignal(float, str)     # scale_factor, interpolation_method
apply_edge       = pyqtSignal(str, str)       # operator, component_name
apply_hist_eq    = pyqtSignal(int)            # block_size
apply_median     = pyqtSignal(int)            # kernel_size
accumulate_toggled = pyqtSignal(bool)         # mode_enabled
```

#### Collapsible Sections:

##### **1. Accumulate Mode Checkbox**
```python
"Accumulate Mode" (green when enabled)
├─ When enabled: operations apply to current image (layers effect)
└─ When disabled: operations apply to original each time (fresh base)
```

##### **2. Zoom/Interpolation Section**
```python
Method: [Nearest-Neighbor ▼ | Bilinear]
Step ×: [1.5 ▼]
⊖ Zoom Out | ⊕ Zoom In
```
- Emits: `apply_zoom(1/factor, method)` for zoom-out
- Emits: `apply_zoom(factor, method)` for zoom-in

##### **3. Smoothing Filters Section**
```python
Kernel: [3×3 ▼ | 5×5 | 7×7 | 9×9 | 11×11]
Gauss σ: [1.0 ▼]
┌─────────────┬──────────┐
│  Average    │ Gaussian │
└─────────────┴──────────┘
```
- **Average**: Emits `apply_filter("average", {"kernel_size": N})`
- **Gaussian**: Emits `apply_filter("gaussian", {"kernel_size": N, "sigma": σ})`

##### **4. Edge Detection Section**
```python
Operator: [Sobel ▼ | Prewitt]
Show: [Magnitude ▼ | Horizontal (Gx) | Vertical (Gy)]
[⟁ Detect Edges]
```
- Emits: `apply_edge(operator, component)`
- Sets displayed tabs to Edge View automatically

##### **5. Median Filter Section**
```python
Kernel: [3×3 ▼ | 5×5 | 7×7]
[⊡ Apply Median]
```
- Emits: `apply_median(kernel_size)`

##### **6. Histogram Equalization Section**
```python
Block: [8×8 ▼ | 16×16 | 32×32 | 64×64]
[◑ Local Equalize]
```
- Emits: `apply_hist_eq(block_size)`

#### Helper Component: CollapsibleSection Class
```python
class CollapsibleSection(QWidget):
    """
    Reusable collapsible group with arrow toggle button and frame body.
    
    Methods:
    - _toggle(bool): Show/hide body and rotate arrow
    - bodyLayout(): Returns QVBoxLayout for adding child widgets
    """
```

---

### 3. **Right Panels** (`panels.py`)

#### **MetadataPanel** (Info Tab)

**Purpose:** Display image statistics and luminance histogram

**Components:**
```
┌──────────────────────────────────┐
│ Image Info                       │
├──────────────────────────────────┤
│ Image Properties                 │
│                                  │
│ Width: 512                       │  (HTML formatted, color-coded)
│ Height: 512                      │
│ Min: 0       Max: 255           │
│ Mean: 127.5  Std Dev: 64.2      │
│                                  │
│ Histogram                        │
│ ┌────────────────────────────┐  │
│ │    (256-bin gradient)      │  │
│ └────────────────────────────┘  │
└──────────────────────────────────┘
```

**Key Method:**
```python
set_metadata(meta: dict) -> None
    # Renders dict as HTML with color-coding
    # ACCENT color for keys, TEXT0 for values
    # Monospace font for technical appearance
```

#### **PipelinePanel** (Pipeline Tab)

**Purpose:** Track applied operations and provide quick undo/reset

**Components:**
```
┌──────────────────────────────────┐
│ Enhancement Pipeline        [5]  │  (numbered badge)
├──────────────────────────────────┤
│ 1. Average Filter (3×3) [acc]   │ ← Green for accumulate mode
│ 2. Gaussian Blur σ=1.5 [orig]   │ ← Red for original-base mode
│ 3. Sobel Edge (Magnitude) [acc]  │
│                                  │
├──────────────────────────────────┤
│ [Undo]  [Reset]                  │
└──────────────────────────────────┘
```

**Key Methods:**
```python
add_step(label: str) -> None
    # Append numbered entry, auto-color based on (acc)/(orig) tags
    # Scroll to bottom to show latest operation
    
remove_last() -> None
    # Called by undo handler, removes most recent step
    
clear() -> None
    # Clear all steps when resetting to original
```

**Signal:**
- `undo_requested` – Emitted when Undo button clicked
- `reset_requested` – Emitted when Reset button clicked

---

### 4. **Image Canvas Widget** (`widgets.py`)

#### **ImageCanvas** (Scrollable Image Display)

**Purpose:** Render numpy image arrays with support for pan, zoom, and annotations

**Key Features:**

##### Display Modes:
```python
set_array(arr: np.ndarray)      # Display new image
get_array() -> np.ndarray       # Retrieve current array
set_display_zoom(percent: int)  # Set 5-400% viewport zoom
fit_to_window()                 # Reset zoom to 100%
clear()                         # Clear and show placeholder
```

##### Interaction Modes:
```python
set_interaction_mode(mode: str)
    # "none"      – No interaction (default cursor)
    # "pan"       – Click-drag to pan (hand cursor)
    # "annotate"  – Click to place annotations (crosshair cursor)
```

##### Pan Interaction:
```
Mouse Press (LButton) in pan mode:
  1. Save scroll bar position
  2. Record mouse start position
  3. Change cursor to ClosedHandCursor
  
Mouse Move (LButton down):
  1. Calculate delta from start
  2. Update scroll bar values in opposite direction
  
Mouse Release:
  1. Set pan_active = False
  2. Restore OpenHandCursor
```

##### Annotation Interaction:
```
Mouse Press (LButton) in annotate mode:
  1. Map viewport click to pixmap coordinates
  2. Account for center-alignment offset
  3. Convert absolute coords to relative (0-1 ratio)
  4. Store in _annotations list
  5. Repaint with 4px red circles
```

##### Rendering:
```python
def _refresh():
    if arr.ndim == 2:  # Grayscale
        QImage with Format_Grayscale8
    else:  # RGB
        QImage with Format_RGB888 (first 3 channels)
    
    # Scale to display zoom percentage
    # Maintain aspect ratio
    # Apply smooth transformation
    # Paint annotations on top (if any)
```

**Display Properties:**
- **Background**: Dark (#080a0d) for contrast
- **Placeholder text**: "Open or drop an image"
- **Scaling**: Maintains aspect ratio, fills available space

#### **HistogramWidget** (256-bin Histogram Display)

**Purpose:** Real-time visualization of image luminance distribution

**Key Features:**
```python
set_array(arr: np.ndarray) -> None
    # Convert to grayscale if needed
    # Compute 256-bin histogram
    # Trigger repaint with gradient fill
    
def paintEvent(event):
    # Draw horizontal axis (0-255)
    # Draw vertical bars with blue gradient
    # Normalize height to widget height
    # Add grid lines for reference
```

**Visual Design:**
- **Height**: Fixed 80-100px
- **Color**: Blue gradient fill (visual consistency)
- **Statistics**: Displayed below (pixel count, median, entropy)

---

## UI Layout Flow

### Top-Level Structure
```
QMainWindow
├─ MenuBar (custom toolbar-style)
│  ├─ Logo "MedVision Workbench"
│  ├─ Menu Buttons: File | Edit | View | Image | Filters | Help
│  └─ Toolbar: [Open] [Save] [Undo] [Redo] [Reset]
│
├─ CentralWidget (QHBoxLayout)
│  ├─ IconPanel (52px fixed)
│  │  └─ Tool palette buttons (pan, zoom, crop, annotate, etc.)
│  │
│  ├─ Sidebar (220px fixed, scrollable)
│  │  └─ CollapsibleSections × 5 (Zoom, Smoothing, Edge, Median, HistEq)
│  │
│  ├─ CanvasArea (stretch)
│  │  └─ QTabWidget (3 tabs)
│  │     ├─ Tab 0: Single view (1× ImageCanvas)
│  │     ├─ Tab 1: Before/After (2× ImageCanvas split)
│  │     └─ Tab 2: Edge View (3× ImageCanvas for Gx/Gy/Magnitude)
│  │  └─ Footer: [−] [slider] [+] [Fit] [%] [info]
│  │
│  └─ RightPanel (240px fixed)
│     └─ QTabWidget (3 tabs)
│        ├─ Info: MetadataPanel
│        ├─ Histogram: HistogramWidget + stats
│        └─ Pipeline: PipelinePanel + Undo/Reset buttons
│
└─ StatusBar
   ├─ Status dot + message
   ├─ Cursor position (x, y)
   ├─ Zoom percentage
   ├─ Memory usage estimate
   └─ Team attribution
```

---

## Signal-Slot Architecture

### Data Flow Diagram

```
User clicks button in Sidebar
       ↓
Sidebar emits signal (e.g., apply_filter)
       ↓
MainWindow._on_apply_filter handler fires
       ↓
1. Prepare image (select base: _current or _original)
2. Create lambda with all parameters
3. Spawn ProcessingWorker thread
       ↓
ProcessingWorker.run() (background thread)
       ↓
Calls DIP module function (e.g., apply_linear_filter)
       ↓
Returns result (np.ndarray)
       ↓
Emits finished signal with (result, label)
       ↓
MainWindow._on_processing_done handler fires
       ↓
1. Stop progress indicator
2. Update _current = result
3. Push to _history and _history_labels
4. Clear _redo_stack
5. Update all canvases with new image
6. Update metadata panel
7. Add entry to pipeline panel
8. Sync UI state (dirty flag, undo/redo enablement)
```

### Key Connections (in __init__)

```python
# File operations
self._open_action.triggered.connect(self._open_file)
self._save_action.triggered.connect(self._save_file)
self._undo_action.triggered.connect(self._undo)
self._redo_action.triggered.connect(self._redo)
self._reset_action.triggered.connect(self._reset_to_original)

# Sidebar signals
self._sidebar.apply_filter.connect(self._on_apply_filter)
self._sidebar.apply_zoom.connect(self._on_apply_zoom)
self._sidebar.apply_edge.connect(self._on_apply_edge)
self._sidebar.apply_hist_eq.connect(self._on_hist_eq)
self._sidebar.apply_median.connect(self._on_median)
self._sidebar.accumulate_toggled.connect(self._on_accumulate_toggled)

# Right panel signals
self._pipeline.undo_requested.connect(self._undo)
self._pipeline.reset_requested.connect(self._reset_to_original)

# Canvas tabs
self._canvas_tabs.currentChanged.connect(self._on_tab_changed)

# View zoom controls
self._zoom_slider.valueChanged.connect(self._update_view_zoom)

# Worker completion
self._worker.finished.connect(self._on_processing_done)
self._worker.error.connect(self._on_processing_error)
```

---

## State Management & Undo/Redo

### History Stack Mechanics

```
Initial state: 
    _history = []
    _redo_stack = []
    _current = original image

After Op1 (Gaussian blur):
    _history = [original]
    _history_labels = ["Gaussian (3×3, σ=1.0) [acc]"]
    _current = blurred
    _redo_stack = []

After Op2 (Sobel edge):
    _history = [original, blurred]
    _history_labels = ["...", "Sobel Edge (Magnitude) [acc]"]
    _current = edges
    _redo_stack = []

User presses Undo:
    _redo_stack.append(_current)      # Save current to redo
    _current = _history.pop()         # Restore from history
    _history_labels.pop()
    → Display: blurred image

User presses Redo:
    _history.append(_current)         # Save to history
    _history_labels.append(saved_label)
    _current = _redo_stack.pop()      # Restore from redo
    → Display: edges again
```

### Accumulate Mode

**When enabled** (green checkbox):
- Operations apply to `_current` (accumulated effect)
- Results stack up: blur → sharpen → edge detect
- Pipeline shows "(acc)" tag

**When disabled**:
- Each operation starts from `_original`
- No accumulation, just variations
- Pipeline shows "(orig)" tag
- Useful for comparing different filters on same base

---

## Style System

### Color Palette (`app/core/styles.py`)

| Role | Variable | Value |
|------|----------|-------|
| **Accent** | `ACCENT` | `#3b82f6` (bright blue) |
| **Accent Dim** | `ACCENT_DIM` | `#1e40af` (darker blue) |
| **Background Primary** | `BG0` | `#0a0c11` |
| **Background Secondary** | `BG1` | `#11131a` |
| **Background Tertiary** | `BG2` | `#1a202c` |
| **Text Primary** | `TEXT0` | `#ffffff` |
| **Text Secondary** | `TEXT1` | `#cbd5e1` |
| **Text Tertiary** | `TEXT2` | `#94a3b8` |
| **Text Quaternary** | `TEXT3` | `#64748b` |
| **Success** | `GREEN` | `#4ade80` |
| **Error** | `RED` | `#f87171` |
| **Border** | `BORDER` | `#252d42` |

### Typography
- **Fonts**: JetBrains Mono (primary), Consolas (fallback)
- **Sizes**: 9px–13px (small compact interface)
- **Weights**: 400 (normal), 600 (bold for headers)

### Component Styling
```python
TOOLBAR_BTN_STYLE = "QPushButton { ... }"      # Toolbar buttons
SIDEBAR_STYLE = "QWidget { ... }"              # Sidebar background
PIPELINE_LIST_STYLE = "QListWidget { ... }"    # Pipeline list
METADATA_TEXT_STYLE = "QTextEdit { ... }"      # Metadata editor
```

---

## Processing Pipeline Execution

### Example: Gaussian Blur Flow

```python
# 1. User adjusts kernel size to 5×5, sigma to 1.5, clicks "Gaussian"
sidebar._emit_gaussian()
    ↓
apply_filter.emit("gaussian", {"kernel_size": 5, "sigma": 1.5})
    ↓
# 2. MainWindow handler receives signal
main_window._on_apply_filter("gaussian", {"kernel_size": 5, "sigma": 1.5})
    ↓
# 3. Select base image
base = self._current if self._accumulate else self._original
    ↓
# 4. Prepare lambda with all needed parameters
func = lambda img: make_gaussian_kernel(5, 1.5) 
                   → apply_linear_filter(img, kernel)
    ↓
# 5. Show progress and spawn worker
self._progress.setVisible(True)
worker = ProcessingWorker(func, "Gaussian (5×5, σ=1.5) [acc]", base)
worker.finished.connect(self._on_processing_done)
worker.start()
    ↓
# 6. Main thread continues, worker runs in background
# DIP.smoothing.apply_linear_filter executes:
#   - Create kernel via make_gaussian_kernel
#   - Route to scratch or FFT convolution
#   - Return blurred image
    ↓
# 7. Worker emits finished signal
worker.finished.emit(blurred_array, "Gaussian (5×5, σ=1.5) [acc]")
    ↓
# 8. MainWindow handler processes result
_on_processing_done(blurred_array, label)
    ├─ self._current = blurred_array
    ├─ self._history.append(old_current)
    ├─ self._history_labels.append(label)
    ├─ self._redo_stack.clear()
    ├─ Update all canvases
    ├─ Update metadata/histogram
    ├─ Add to pipeline list
    └─ Hide progress
```

---

## Threading Model

### Why Background Threads?

**Problem**: Large image processing (5MP edge detection) can take 2–5 seconds
- Blocking main thread → UI freezes → "Not Responding"

**Solution**: ProcessingWorker

```python
class ProcessingWorker(QThread):
    finished = pyqtSignal(np.ndarray, str)  # Safe signal emission
    error = pyqtSignal(str)
    
    def run(self):  # Executes in background thread
        try:
            result = self._func(*self._args, **self._kwargs)
            self.finished.emit(result, self._label)  # Thread-safe
        except Exception as exc:
            self.error.emit(str(exc))
```

**Benefits:**
- Main thread remains responsive
- Progress indicator can animate
- User can cancel (Phase 2 enhancement)
- Smooth interaction even during heavy processing

---

## Multi-Tab Canvas System

### Tab 0: Single View
```
┌────────────────────────────┐
│  ImageCanvas (full)        │
│  Displays: _current        │
│  Pan/Zoom supported        │
└────────────────────────────┘
```

### Tab 1: Before/After Split
```
┌──────────────────┬──────────────────┐
│ Original         │ Processed        │
│ ImageCanvas      │ ImageCanvas      │
│ (_original)      │ (_current)       │
└──────────────────┴──────────────────┘
```

### Tab 2: Edge View
```
┌──────────────┬──────────────┬──────────────┐
│ Gx           │ Gy           │ Magnitude    │
│ (Horizontal) │ (Vertical)   │ (sqrt...)    │
│ ImageCanvas  │ ImageCanvas  │ ImageCanvas  │
└──────────────┴──────────────┴──────────────┘
```

### Synchronization
- All canvases sync zoom level via `_sync_view_zoom()`
- All canvases set interaction mode via `_set_interaction_mode()`
- Tab change tracked by `_on_tab_changed()` to update metadata context

---

## User Interaction Flows

### Loading an Image
```
1. User: Ctrl+O or File → Open
2. MainWindow._open_file()
   ├─ Show QFileDialog
   ├─ Load via image_io.load_image(path)
   ├─ Reset history/redo stacks
   ├─ Set _original = loaded_array
   ├─ Set _current = _original.copy()
   ├─ Update all canvases
   ├─ Compute and display metadata
   ├─ Update status: "Loaded: {filename}"
   └─ Enable undo/redo (as needed)
```

### Applying a Filter
```
1. User: Select params in sidebar, click button
2. Sidebar emits signal with parameters
3. MainWindow handler:
   ├─ Show progress
   ├─ Select base image (accumulate mode)
   ├─ Spawn background worker
4. Worker processes (non-blocking)
5. MainWindow._on_processing_done:
   ├─ Update _current
   ├─ Push to history
   ├─ Refresh all UI panels
   └─ Hide progress
```

### Undoing an Operation
```
1. User: Ctrl+Z or click Undo button
2. MainWindow._undo():
   ├─ Pop _current to _redo_stack
   ├─ Pop _history to _current
   ├─ Pop label from _history_labels
   ├─ Remove from pipeline list
   ├─ Refresh all canvases
   └─ Update undo/redo button states
```

### Switching Interaction Mode
```
1. User: Click pan icon (hand) in left icon panel
2. MainWindow._set_interaction_mode(0, "Move / Pan"):
   ├─ Set mode = "pan"
   ├─ Call canvas.set_interaction_mode("pan") for each canvas
   └─ Canvas sets cursor to OpenHandCursor
3. User can now click-drag to pan the image
```

---

## Performance Considerations

### Image Size Handling
- **Small images** (< 1MP): Scratch convolution acceptable
- **Medium images** (1–10MP): Hybrid (FFT for 7×7+ kernels)
- **Large images** (> 10MP): FFT convolution mandatory

### Routing Logic (in `apply_linear_filter`):
```python
if kernel.shape[0] * kernel.shape[1] >= 49:  # 7×7+
    use FFT convolution
else:
    use scratch convolution
```

### Memory Management
- Status bar displays estimated memory usage
- Images kept in memory: _original, _current, _history (up to ~10 levels)
- No aggressive GC; Python handles cleanup

---

## Testing & Validation

### UI Testing Scenarios

| Scenario | Expected Behavior |
|----------|---|
| Load 8-bit RGB image | Displays correctly, metadata shows dimensions/channels |
| Apply Gaussian then Sobel | Pipeline shows both ops, tab switches to Edge View |
| Undo after edge detection | Returns to pre-edge state, pipeline removes last entry |
| Switch to Before/After tab | Shows original vs processed side-by-side, both pan-able |
| Zoom slider to 50% | All canvases scale to 50%, maintains aspect ratio |
| Enable accumulate mode | Operations stack; disable = fresh from original |
| Pan in single view | Can scroll large images smoothly |
| Annotate mode | Click places red circles, visible on repaint |

---

## Known Limitations & Future Work (Phase 2)

### Current Phase 1 Limitations:
- ✗ No crop tool (icon present but disabled)
- ✗ No measurement tool
- ✗ No color picker
- ✗ No ruler or guides
- ✗ No operation cancellation mid-process
- ✗ No batch processing

### Planned Phase 2 Enhancements:
- Advanced tools (crop, measure, color picker)
- Annotations & markup persistence
- Filters history export/import
- Multi-image comparison
- Real-time parameter preview

---

## Conclusion

The MedVision Workbench UI is built on a **clean separation of concerns**:
- **Sidebar** emits what user wants
- **MainWindow** decides how to do it
- **Workers** compute without freezing UI
- **Panels** reflect results

This architecture allows easy addition of new filters (just add a sidebar section and handler) while keeping the codebase maintainable and testable.

