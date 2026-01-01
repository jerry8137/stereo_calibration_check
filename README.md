# Stereo Calibration Check

A tool for verifying and re-rectifying fisheye stereo camera calibrations. This package implements a complete bidirectional transformation pipeline for fisheye stereo systems, including rectification, distortion, and epipolar geometry validation.

## Features

- **Fisheye Stereo Rectification**: Undistort and rectify fisheye images using OpenCV's fisheye model
- **Reverse Distortion**: Transform rectified images back to the original fisheye geometry
- **Epipolar Validation**: Visualize epipolar lines using SIFT feature matching
- **Configurable Baseline**: Adjust stereo baseline distance for different camera setups
- **CLI Tool**: Easy-to-use command-line interface
- **Comprehensive Tests**: Automated testing with pytest

## Installation

### Prerequisites

- Python >= 3.12
- uv (recommended) or pip

### Install from source

```bash
# Clone the repository
git clone <repository-url>
cd stereo_calibration_check

# Install with uv (recommended)
uv sync
pip install -e .

# Or install with pip
pip install -e .
```

### Install development dependencies

```bash
# With uv
uv add --dev pytest pytest-cov
uv sync

# Or with pip
pip install -e ".[dev]"
```

## Usage

### Command-Line Interface

```bash
# Run with default parameters (baseline: 0.12m)
uv run stereo-calibration-check

# Specify custom images and calibration files
uv run stereo-calibration-check \
  --left_image data/images/sync_left.png \
  --right_image data/images/sync_right.png \
  --left_calib data/calibration/thermal_left.yaml \
  --right_calib data/calibration/thermal_right.yaml

# Use custom baseline (in meters)
uv run stereo-calibration-check \
  --left_image data/images/sync_left.png \
  --right_image data/images/sync_right.png \
  --left_calib data/calibration/thermal_left.yaml \
  --right_calib data/calibration/thermal_right.yaml \
  --baseline 0.15
```

### CLI Options

- `--left_image`: Path to the left image (default: `data/images/sync_left.png`)
- `--right_image`: Path to the right image (default: `data/images/sync_right.png`)
- `--left_calib`: Path to left camera calibration YAML (default: `data/calibration/thermal_left.yaml`)
- `--right_calib`: Path to right camera calibration YAML (default: `data/calibration/thermal_right.yaml`)
- `--baseline`: Stereo baseline distance in meters (default: `0.12`)

### Example Scripts

The repository includes several example scripts demonstrating different use cases:

```bash
# Full stereo pipeline with overlay comparisons
uv run example_stereo.py

# Recalibration workflow
uv run example_recalibrate.py

# Chessboard-based calibration check
uv run example_chessboard.py

# Basic epipolar line visualization
uv run epipolar.py
```

### Programmatic Usage

```python
from stereo_calibration_check.calibrate.stereo_calibrate import get_projection_matrix
from stereo_calibration_check.calibrate.undistort import rectify_image
from stereo_calibration_check.utils.file_utils import load_intrinsics
import cv2

# Load calibration data
intrinsics1 = load_intrinsics('data/calibration/thermal_left.yaml')
intrinsics2 = load_intrinsics('data/calibration/thermal_right.yaml')

# Get projection matrices with custom baseline
R1, R2, P1, P2 = get_projection_matrix(intrinsics1, intrinsics2, baseline=0.15)

# Rectify images
image1 = cv2.imread('data/images/sync_left.png', cv2.IMREAD_GRAYSCALE)
rect_image1 = rectify_image(image1, None, intrinsics1, R1, P1)
```

## Calibration File Format

Calibration files should be in YAML format with the following structure:

```yaml
camera_matrix:
  data: [fx, 0, cx, 0, fy, cy, 0, 0, 1]  # 3x3 camera matrix (row-major)
distortion_coefficients:
  data: [k1, k2, k3, k4]  # 4 fisheye distortion coefficients
distortion_model: "equidistant"  # Must be equidistant for fisheye
rectification_matrix:
  data: [...]  # 3x3 rectification matrix
projection_matrix:
  data: [...]  # 3x4 projection matrix
width: 640
height: 512
```

## Testing

Run the automated test suite:

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=stereo_calibration_check --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_baseline.py

# Run tests matching a pattern
uv run pytest -k "baseline"
```

### Test Coverage

- Unit tests for baseline parameter functionality
- CLI integration tests
- Edge case validation (zero, negative, large baselines)
- Backward compatibility tests

## Project Structure

```
stereo_calibration_check/
├── src/stereo_calibration_check/
│   ├── calibrate/
│   │   ├── stereo_calibrate.py    # Stereo rectification matrices
│   │   ├── undistort.py           # Image rectification
│   │   └── distort.py             # Reverse distortion
│   ├── epipolar_calibration_check/
│   │   └── epipolar_line.py       # Epipolar line visualization
│   ├── feature_detection/
│   │   ├── sift.py                # SIFT feature matching
│   │   └── corners.py             # Corner detection
│   ├── utils/
│   │   ├── file_utils.py          # YAML calibration loading
│   │   └── visualization.py       # Image display utilities
│   └── main.py                    # CLI entry point
├── tests/
│   ├── test_baseline.py           # Baseline parameter tests
│   └── test_cli.py                # CLI integration tests
├── data/
│   ├── calibration/               # Calibration YAML files
│   └── images/                    # Test images
└── example_*.py                   # Example scripts
```

## Configuration

### Baseline Distance

The stereo baseline (distance between cameras) can be configured:

- **CLI**: Use the `--baseline` parameter
- **Programmatic**: Pass `baseline` argument to `get_projection_matrix()`
- **Default**: 0.12 meters (120mm)

Note: In `calibrate/stereo_calibrate.py`, the baseline is applied in the X direction `[baseline, 0, 0]`. Adjust based on your physical camera setup.

## Development

### Contributing

See [CLAUDE.md](CLAUDE.md) for detailed development guidelines and architecture documentation.

### Building

```bash
# Sync dependencies
uv sync

# The package uses setuptools backend
```

## License

[Add your license here]

## Acknowledgments

This tool uses OpenCV's fisheye camera model for stereo calibration and rectification.
