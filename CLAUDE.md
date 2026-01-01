# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a stereo calibration check tool for fisheye camera systems, specifically designed to verify and re-rectify stereo camera setups. The tool handles the complete pipeline: loading calibration data, rectifying fisheye images, verifying calibration quality through epipolar line visualization, and re-distorting images back to the original lens geometry.

## Core Architecture

### Fisheye Stereo Rectification Pipeline

The project implements a bidirectional transformation pipeline:

1. **Distorted (Original) → Rectified**: Uses `cv2.fisheye.initUndistortRectifyMap()` and `cv2.remap()` to undistort and rectify fisheye images to a common epipolar plane
2. **Rectified → Distorted**: Implements manual ray-tracing by unprojecting rectified pixels to 3D rays, rotating them back to the original camera frame, and projecting through the fisheye distortion model
3. **Epipolar Validation**: Uses SIFT feature matching and fundamental matrix computation to draw epipolar lines for visual quality assessment

The key insight is that the calibration YAML files store `R` (rectification_matrix) which represents rotation from the original frame to the rectified frame. The distortion process reverses this transformation.

### Package Structure

```
src/stereo_calibration_check/
├── calibrate/
│   ├── stereo_calibrate.py  # Computes R1, R2, P1, P2 using cv2.fisheye.stereoRectify()
│   ├── undistort.py         # Rectifies fisheye images using calibration data
│   └── distort.py           # Reverse transformation: rectified → distorted
├── epipolar_calibration_check/
│   └── epipolar_line.py     # Draws epipolar lines using SIFT or corner detection
├── feature_detection/
│   ├── sift.py              # SIFT feature matching with FLANN matcher
│   └── corners.py           # Corner-based feature detection
└── utils/
    ├── file_utils.py        # Loads calibration YAML files
    └── visualization.py     # Side-by-side image display
```

### Calibration File Format

Calibration files (e.g., `data/calibration/thermal_left.yaml`) contain:
- `K`: 3x3 camera matrix (intrinsics)
- `D`: Distortion coefficients (4 coefficients for equidistant/fisheye model)
- `R`: 3x3 rectification matrix (rotation from original → rectified frame)
- `P`: 3x4 projection matrix (rectified camera parameters)
- `distortion_model`: Must be "equidistant" for fisheye cameras

## Development Commands

### Installation

```bash
# Install the package in development mode (uses uv by default)
pip install -e .
```

### Running the Tool

```bash
# Run the installed CLI tool with default parameters
uv run stereo-calibration-check

# Run with custom images and calibration files
uv run stereo-calibration-check \
  --left_image data/images/sync_left.png \
  --right_image data/images/sync_right.png \
  --left_calib data/calibration/thermal_left.yaml \
  --right_calib data/calibration/thermal_right.yaml

# Run with custom baseline (in meters)
uv run stereo-calibration-check \
  --left_image data/images/sync_left.png \
  --right_image data/images/sync_right.png \
  --left_calib data/calibration/thermal_left.yaml \
  --right_calib data/calibration/thermal_right.yaml \
  --baseline 0.15
```

### Running Example Scripts

```bash
# Example scripts in the root directory demonstrate various use cases
uv run example_stereo.py        # Full stereo pipeline with overlay comparisons
uv run example_recalibrate.py   # Recalibration workflow
uv run example_chessboard.py    # Chessboard-based calibration check
uv run epipolar.py              # Basic epipolar line visualization
```

### Building and Distribution

```bash
# Build the package
uv sync

# The package uses setuptools backend (specified in pyproject.toml)
```

### Testing

```bash
# Install development dependencies
uv add --dev pytest pytest-cov
uv sync

# Run all tests
uv run pytest

# Run tests with coverage report
uv run pytest --cov=stereo_calibration_check --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_baseline.py

# Run tests matching a pattern
uv run pytest -k "baseline"
```

## Important Implementation Details

### Stereo Calibration Baseline

The stereo baseline (translation vector `T`) can be configured via the `--baseline` parameter in the CLI tool or by passing the `baseline` argument to `get_projection_matrix()`. The default value is 0.12 meters (120mm), assuming cameras are horizontally aligned with 12cm separation.

**Note:** In `calibrate/stereo_calibrate.py`, the baseline is applied in the X direction `[baseline, 0, 0]`, while in `recalibrate.py`, it's applied in the Y direction `[0, baseline, 0]` due to different camera orientations in those scripts. Adjust based on your physical camera setup.

### Feature Matching for Epipolar Validation

- `draw_epilines_sift()` in `epipolar_calibration_check/epipolar_line.py:44-69` applies a 100-pixel margin filter before computing the fundamental matrix to exclude edge regions where SIFT features may be unreliable
- Uses Lowe's ratio test (0.8 threshold) for robust SIFT matching
- Fundamental matrix is computed with `cv.FM_LMEDS` for outlier rejection

### Distortion Implementation

The `distort_image()` function in `calibrate/distort.py` implements the reverse rectification by:
1. Creating a pixel grid for the destination (distorted) image
2. Unprojecting each pixel through the fisheye model to get 3D rays
3. Rotating rays from original frame to rectified frame using `R`
4. Projecting onto the rectified image plane using `P[:3, :3]`
5. Remapping the rectified image to create the distorted output

This is conceptually opposite to standard undistortion and requires careful handling of the rotation matrix direction.

## Data Organization

```
data/
├── calibration/              # Camera calibration YAML files
│   ├── thermal_left.yaml
│   └── thermal_right.yaml
└── images/                   # Test images (not tracked in git)
    ├── sync_left.png
    └── sync_right.png
```

Image files and outputs (*.png in root directory) are gitignored to keep the repository clean.
