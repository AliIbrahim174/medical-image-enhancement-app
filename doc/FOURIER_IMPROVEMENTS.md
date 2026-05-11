# Fourier Workflow Improvements

## Overview
This document describes the recent usability improvements to Fourier-domain notch filtering.

## Problems Addressed
- Selecting periodic-noise spikes required very precise clicking.
- Users often clicked near spikes and selected weak or wrong points.
- Iterative notch editing was slower than necessary.

## Implemented Improvements

### 1. Near-Click Peak Auto-Snap
- Clicks on the Fourier spectrum now snap to a nearby bright peak.
- A local search window is evaluated around the click point.
- Candidate scoring balances brightness and distance to the click.
- The DC-center neighborhood is excluded to reduce false selections.

### 2. Better Selection Feedback
- The selected notch and its conjugate are both marked automatically.
- Status text reports:
  - clicked coordinates
  - snapped coordinates
  - selected peak intensity
  - conjugate coordinates

### 3. Clearer Fourier Instructions
- The Fourier tab hint now explains that auto-snap is enabled.
- The spectrum status message also guides users to click near spikes.

### 4. Side-by-Side Fourier Workspace
- Fourier tab includes:
  - editable spectrum view (left)
  - processed/filtered image preview (right)
- This supports faster trial-and-adjust workflows.

## Technical Notes
- Added `snap_to_bright_peak(...)` in `app/DIP/frequency_domain.py`.
- Integrated snapping inside spectrum click handling in `app/gui/signal_handlers.py`.
- Updated user-facing hint text in `app/gui/ui_builder.py`.

## Result
The Fourier notch workflow is now more forgiving, faster, and easier to use in practical noisy-image cleanup tasks.
