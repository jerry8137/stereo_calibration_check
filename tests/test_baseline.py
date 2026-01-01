"""Tests for baseline parameter functionality."""

import pytest
import numpy as np
from stereo_calibration_check.calibrate.stereo_calibrate import get_projection_matrix


class TestBaselineParameter:
    """Test suite for baseline parameter in get_projection_matrix."""

    def test_default_baseline(self, sample_intrinsics):
        """Test that get_projection_matrix works with default baseline."""
        intrinsics1, intrinsics2 = sample_intrinsics

        R1, R2, P1, P2 = get_projection_matrix(intrinsics1, intrinsics2)

        # Verify output shapes
        assert R1.shape == (3, 3), "R1 should be 3x3 matrix"
        assert R2.shape == (3, 3), "R2 should be 3x3 matrix"
        assert P1.shape == (3, 4), "P1 should be 3x4 matrix"
        assert P2.shape == (3, 4), "P2 should be 3x4 matrix"

        # Verify matrices are not all zeros
        assert not np.allclose(R1, 0), "R1 should not be all zeros"
        assert not np.allclose(R2, 0), "R2 should not be all zeros"
        assert not np.allclose(P1, 0), "P1 should not be all zeros"
        assert not np.allclose(P2, 0), "P2 should not be all zeros"

    def test_custom_baseline(self, sample_intrinsics):
        """Test that get_projection_matrix works with custom baseline."""
        intrinsics1, intrinsics2 = sample_intrinsics

        baseline = 0.15  # 150mm
        R1, R2, P1, P2 = get_projection_matrix(intrinsics1, intrinsics2, baseline=baseline)

        # Verify output shapes
        assert R1.shape == (3, 3)
        assert R2.shape == (3, 3)
        assert P1.shape == (3, 4)
        assert P2.shape == (3, 4)

    @pytest.mark.parametrize("baseline", [0.05, 0.10, 0.12, 0.15, 0.20, 0.30])
    def test_various_baselines(self, sample_intrinsics, baseline):
        """Test that function works with various baseline values."""
        intrinsics1, intrinsics2 = sample_intrinsics

        R1, R2, P1, P2 = get_projection_matrix(intrinsics1, intrinsics2, baseline=baseline)

        # All outputs should be valid matrices
        assert R1.shape == (3, 3)
        assert R2.shape == (3, 3)
        assert P1.shape == (3, 4)
        assert P2.shape == (3, 4)
        assert not np.any(np.isnan(R1))
        assert not np.any(np.isnan(R2))
        assert not np.any(np.isnan(P1))
        assert not np.any(np.isnan(P2))

    def test_different_baselines_produce_different_results(self, sample_intrinsics):
        """Test that different baselines produce different projection matrices."""
        intrinsics1, intrinsics2 = sample_intrinsics

        # Get results with two different baselines
        R1_a, R2_a, P1_a, P2_a = get_projection_matrix(intrinsics1, intrinsics2, baseline=0.12)
        R1_b, R2_b, P1_b, P2_b = get_projection_matrix(intrinsics1, intrinsics2, baseline=0.20)

        # At least one of the projection matrices should be different
        # (P2 typically changes with baseline in stereo rectification)
        matrices_differ = (
            not np.array_equal(P1_a, P1_b) or
            not np.array_equal(P2_a, P2_b) or
            not np.array_equal(R1_a, R1_b) or
            not np.array_equal(R2_a, R2_b)
        )

        assert matrices_differ, "Different baselines should produce different results"

    def test_baseline_affects_P2(self, sample_intrinsics):
        """Test that baseline specifically affects P2 matrix."""
        intrinsics1, intrinsics2 = sample_intrinsics

        # Get results with two different baselines
        _, _, _, P2_small = get_projection_matrix(intrinsics1, intrinsics2, baseline=0.08)
        _, _, _, P2_large = get_projection_matrix(intrinsics1, intrinsics2, baseline=0.20)

        # P2 should be different for different baselines
        assert not np.array_equal(P2_small, P2_large), \
            "P2 matrix should differ with different baselines"

    def test_zero_baseline(self, sample_intrinsics):
        """Test edge case with zero baseline."""
        intrinsics1, intrinsics2 = sample_intrinsics

        R1, R2, P1, P2 = get_projection_matrix(intrinsics1, intrinsics2, baseline=0.0)

        # Should still produce valid matrices
        assert R1.shape == (3, 3)
        assert R2.shape == (3, 3)
        assert P1.shape == (3, 4)
        assert P2.shape == (3, 4)

    def test_negative_baseline(self, sample_intrinsics):
        """Test that negative baseline produces valid results (cameras in reverse order)."""
        intrinsics1, intrinsics2 = sample_intrinsics

        R1, R2, P1, P2 = get_projection_matrix(intrinsics1, intrinsics2, baseline=-0.12)

        # Should still produce valid matrices
        assert R1.shape == (3, 3)
        assert R2.shape == (3, 3)
        assert P1.shape == (3, 4)
        assert P2.shape == (3, 4)
        assert not np.any(np.isnan(P1))
        assert not np.any(np.isnan(P2))

    def test_backward_compatibility(self, sample_intrinsics):
        """Test that calling without baseline parameter uses default value."""
        intrinsics1, intrinsics2 = sample_intrinsics

        # Call without baseline parameter (should use default 0.12)
        R1_no_param, R2_no_param, P1_no_param, P2_no_param = get_projection_matrix(
            intrinsics1, intrinsics2
        )

        # Call with explicit default baseline
        R1_explicit, R2_explicit, P1_explicit, P2_explicit = get_projection_matrix(
            intrinsics1, intrinsics2, baseline=0.12
        )

        # Results should be identical
        np.testing.assert_array_equal(R1_no_param, R1_explicit)
        np.testing.assert_array_equal(R2_no_param, R2_explicit)
        np.testing.assert_array_equal(P1_no_param, P1_explicit)
        np.testing.assert_array_equal(P2_no_param, P2_explicit)

    def test_very_large_baseline(self, sample_intrinsics):
        """Test with unrealistically large baseline."""
        intrinsics1, intrinsics2 = sample_intrinsics

        R1, R2, P1, P2 = get_projection_matrix(intrinsics1, intrinsics2, baseline=1.0)

        # Should still produce valid matrices even with large baseline
        assert R1.shape == (3, 3)
        assert R2.shape == (3, 3)
        assert P1.shape == (3, 4)
        assert P2.shape == (3, 4)
        assert not np.any(np.isnan(P1))
        assert not np.any(np.isnan(P2))
        assert not np.any(np.isinf(P1))
        assert not np.any(np.isinf(P2))
