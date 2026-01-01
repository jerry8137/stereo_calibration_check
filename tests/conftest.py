"""Pytest configuration and fixtures."""

import pytest
import os
from pathlib import Path


@pytest.fixture
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent.parent / "data"


@pytest.fixture
def calibration_files(test_data_dir):
    """Return paths to calibration files."""
    return {
        'left': test_data_dir / "calibration" / "thermal_left.yaml",
        'right': test_data_dir / "calibration" / "thermal_right.yaml"
    }


@pytest.fixture
def test_images(test_data_dir):
    """Return paths to test images."""
    return {
        'left': test_data_dir / "images" / "sync_left.png",
        'right': test_data_dir / "images" / "sync_right.png"
    }


@pytest.fixture
def sample_intrinsics(calibration_files):
    """Load sample intrinsics for testing."""
    from stereo_calibration_check.utils.file_utils import load_intrinsics

    intrinsics1 = load_intrinsics(str(calibration_files['left']))
    intrinsics2 = load_intrinsics(str(calibration_files['right']))

    return intrinsics1, intrinsics2
