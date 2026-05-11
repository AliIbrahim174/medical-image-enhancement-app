# Fourier and Notch Filtering Technical Notes

## Purpose

This document explains everything implemented so far for the Fourier workflow and notch filtering in MedVision Workbench. It covers:

- the scientific basis for the Fourier spectrum and notch reject filtering
- the UI flow from widgets to handlers
- the math and algorithmic steps used to apply a notch filter
- the current usability improvements, including auto-snap selection and history-driven spectrum refresh

## Scientific Background

### Why Fourier-domain filtering is useful

Periodic noise often appears in images as repeated bright spikes in the frequency domain. In the spatial domain this noise may be hard to isolate, but after a Fourier transform it becomes much easier to identify as concentrated peaks in the shifted magnitude spectrum.

The idea is simple:

- transform the image from the spatial domain to the frequency domain
- identify the unwanted periodic components as bright spikes
- suppress those spikes with a notch reject filter
- transform back to the spatial domain

### The Fourier transform used here

For a grayscale image $f(x, y)$, the implementation uses the 2-D discrete Fourier transform:

$$
F(u, v) = \sum_x \sum_y f(x, y)e^{-j2\pi(ux/M + vy/N)}
$$

Then the spectrum is shifted so the DC component moves to the center:

$$
F_{shifted}(u, v) = \text{fftshift}(F(u, v))
$$

The displayed spectrum is the log magnitude:

$$
S(u, v) = \log(1 + |F_{shifted}(u, v)|)
$$

This makes faint peaks easier to see.

### Why the conjugate notch matters

For real-valued spatial images, the Fourier spectrum is conjugate symmetric. That means if a periodic artifact appears at one point in frequency space, a mirrored counterpart exists at the conjugate location.

If only one spike is removed, the inverse transform can produce artifacts or complex-valued residuals. For that reason the implementation always applies the notch to both the selected peak and its conjugate mirror.

## User Experience Design

The Fourier workflow now follows a low-friction process:

1. Open an image.
2. Click Show Fourier Spectrum.
3. Click near a bright spike instead of needing exact pixel precision.
4. The system auto-snaps to the nearest bright peak.
5. The selected peak and its conjugate are marked.
6. Apply the notch filter.
7. Compare the filtered result in the Fourier tab.

This design reduces the precision burden on the user while keeping the underlying math correct.

## UI to Math Flow

The Fourier workflow is intentionally split across the UI and processing layers.

### 1. Sidebar widgets

The frequency controls live in [app/gui/sidebar.py](../app/gui/sidebar.py):

- `Show Fourier Spectrum`
- notch filter type selector
- notch radius selector
- Butterworth order selector
- `Apply Selected Notch`

The sidebar does not perform any Fourier computation itself. It only emits signals.

### 2. Signal emission

The sidebar emits these signals:

- `show_spectrum`
- `apply_notch(str, float, int)`

These are connected in [app/gui/main_window.py](../app/gui/main_window.py) to methods inside `SignalHandlers`.

### 3. MainWindow orchestration

`MainWindow` only coordinates the UI and application state. It does not directly implement the Fourier math. Its role is to:

- connect the sidebar signals to the handlers
- keep canvas widgets up to date
- manage history, undo, redo, and reset
- refresh the spectrum when the image changes

### 4. SignalHandlers dispatch

The actual behavior is implemented in [app/gui/signal_handlers.py](../app/gui/signal_handlers.py):

- `on_show_spectrum()` computes the spectrum from the current source image
- `on_spectrum_clicked()` snaps the click to a nearby bright peak and stores the selected notch center
- `on_apply_notch()` applies the filter and commits the result back into history

### 5. Math layer

The Fourier math lives in [app/DIP/frequency_domain.py](../app/DIP/frequency_domain.py):

- `shifted_magnitude_spectrum()` computes the display image for the spectrum tab
- `snap_to_bright_peak()` chooses a nearby strong frequency peak
- `conjugate_notch_center()` computes the mirrored companion location
- `make_notch_reject_mask()` builds the filter mask
- `apply_notch_filter()` applies the mask and restores the filtered image

## Current Fourier Tab Layout

The Fourier tab is built in [app/gui/ui_builder.py](../app/gui/ui_builder.py).

It contains two panes:

- left: editable spectrum view
- right: filtered result preview

This was added so the user can see the effect of the notch selection immediately without switching tabs.

## Detailed Processing Pipeline

### A. Showing the spectrum

When the user clicks `Show Fourier Spectrum`:

1. `ToolsSidebar` emits `show_spectrum`.
2. `SignalHandlers.on_show_spectrum()` receives the request.
3. The current source image is selected from `ImageStateManager`.
4. `shifted_magnitude_spectrum()` computes the display spectrum.
5. The spectrum canvas is populated with the generated spectrum image.
6. The Fourier tab becomes active.

### B. Clicking the spectrum

When the user clicks on the spectrum canvas:

1. `ImageCanvas` emits `image_clicked(row, col)`.
2. `SignalHandlers.on_spectrum_clicked()` receives the click.
3. `snap_to_bright_peak()` searches a small local neighborhood.
4. The click is snapped to the nearest strong bright peak.
5. `conjugate_notch_center()` computes the mirrored peak.
6. Markers are drawn on both points.
7. The selected notch center is stored in application state.

### C. Applying the notch filter

When the user clicks `Apply Selected Notch`:

1. `ToolsSidebar` emits `apply_notch(filter_type, radius, order)`.
2. `SignalHandlers.on_apply_notch()` checks that a notch center exists.
3. `apply_notch_filter()` is called with the selected coordinates.
4. Inside the frequency module:
   - a notch reject mask is created
   - the selected peak and its conjugate are both suppressed
   - the image is transformed back using inverse FFT
5. The filtered image is committed into history.
6. The Fourier spectrum is refreshed from the new current image.

## Notch Mask Construction

The mask creation happens in `make_notch_reject_mask()`.

### Supported filter types

- Ideal
- Butterworth
- Gaussian

### How the mask works

For a selected center $(r, c)$, the code generates a distance grid:

$$
D(i, j) = \sqrt{(i-r)^2 + (j-c)^2}
$$

Then it builds a reject mask around that center and its conjugate.

#### Ideal notch reject

A binary mask that zeros values inside the radius and keeps the rest unchanged.

#### Butterworth notch reject

A smooth transition mask controlled by the order parameter.

#### Gaussian notch reject

A smooth Gaussian-shaped suppression around the notch center.

The final mask is applied multiplicatively in the shifted Fourier domain.

## Auto-Snap Selection Logic

The current implementation intentionally avoids exact-dot clicking.

### Why auto-snap was added

Periodic spikes are often small and difficult to click exactly. Users were spending too much time trying to hit the exact bright point.

### How snapping works

`snap_to_bright_peak()` scans a local square neighborhood around the click and chooses the best candidate using a score that favors:

- high intensity
- small distance from the click

The DC neighborhood is excluded because the central low-frequency region is usually not where periodic noise spikes are selected.

### Result

The user can click approximately near a spike, and the selection will lock onto the nearest bright peak.

## Spectrum Refresh After History Actions

Undo, redo, and reset now refresh the Fourier spectrum when relevant.

This behavior is implemented in [app/gui/main_window.py](../app/gui/main_window.py).

### Why this matters

If the current image changes through history operations, the old spectrum can become stale. That would make notch selection inconsistent with the actual displayed image.

### Current behavior

- Undo recomputes the spectrum if the Fourier workflow is active or already visible
- Redo recomputes the spectrum under the same conditions
- Reset recomputes the spectrum and clears the selection

This keeps the frequency-domain view synchronized with the current spatial image.

## State and History Interaction

The Fourier selection is stored in the application state as `selected_notch_center`.

`ImageStateManager` also tracks:

- `current`
- `original`
- `history`
- `redo_stack`
- zoom state
- accumulate mode

When an image is reset or when history changes, the selected notch center is cleared so a stale selection cannot be reused accidentally.

## File-Level Responsibility Summary

### [app/gui/sidebar.py](../app/gui/sidebar.py)
- builds the Fourier controls in the sidebar
- emits show-spectrum and apply-notch signals

### [app/gui/signal_handlers.py](../app/gui/signal_handlers.py)
- handles spectrum display
- handles spectrum clicks
- handles notch application
- refreshes the spectrum after filtering

### [app/gui/main_window.py](../app/gui/main_window.py)
- connects the UI flow
- manages undo/redo/reset refresh behavior
- keeps the Fourier canvas synchronized with image state

### [app/gui/widgets.py](../app/gui/widgets.py)
- provides the spectrum canvas
- emits click coordinates
- paints the selection markers
- supports zoom, pan, and overlay rendering

### [app/DIP/frequency_domain.py](../app/DIP/frequency_domain.py)
- computes the Fourier magnitude spectrum
- finds conjugate points
- snaps clicks to peaks
- builds notch masks
- applies notch filtering with FFT and inverse FFT

## Limitations and Notes

- Auto-snap improves usability, but it still depends on the spectrum being reasonably clear.
- The notch selection is best for strong periodic peaks, not broad or diffuse artifacts.
- The current implementation uses a local neighborhood search; it is intentionally simple and fast.
- The selected notch preview is shown as markers, not as a full mask overlay.

## Practical Interpretation

Scientifically, the implementation follows a standard frequency-domain denoising pipeline:

- identify periodic energy concentration
- suppress the corresponding frequencies
- preserve the rest of the spectrum
- transform back to the spatial image

Practically, the UI is designed so the user can do this with minimal precision burden and immediate visual feedback.

## Conclusion

The Fourier and notch subsystem is now a complete workflow, not just a filter button. It includes:

- spectrum generation
- interactive peak selection with auto-snap
- conjugate-aware notch placement
- notch reject filtering in the Fourier domain
- synchronized filtered-result preview
- history-aware spectrum refresh on undo/redo/reset

That combination is what makes the current implementation usable for real periodic-noise removal rather than only being a mathematical demo.
