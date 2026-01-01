"""Tests for CLI interface."""

import pytest
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestCLI:
    """Test suite for command-line interface."""

    def test_cli_help_includes_baseline(self, capsys):
        """Test that --help output includes baseline parameter."""
        from stereo_calibration_check.main import main

        with pytest.raises(SystemExit) as exc_info:
            with patch('sys.argv', ['stereo-calibration-check', '--help']):
                main()

        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert '--baseline' in captured.out
        assert 'Stereo baseline distance in meters' in captured.out

    def test_cli_default_baseline_argument(self):
        """Test that CLI accepts baseline argument with default value."""
        from stereo_calibration_check.main import main
        import argparse

        # Mock display functions and image processing to prevent GUI and SIFT issues
        with patch('stereo_calibration_check.main.display_images_side_by_side'):
            with patch('stereo_calibration_check.main.draw_epilines') as mock_epilines:
                with patch('stereo_calibration_check.main.cv2.imread') as mock_imread:
                    with patch('stereo_calibration_check.main.distort_image') as mock_distort:
                        # Mock imread to return dummy images
                        import numpy as np
                        dummy_image = np.zeros((100, 100), dtype=np.uint8)
                        mock_imread.return_value = dummy_image
                        mock_distort.return_value = dummy_image
                        mock_epilines.return_value = (dummy_image, dummy_image)

                        # Test with default arguments
                        with patch('sys.argv', [
                            'stereo-calibration-check',
                            '--left_image', 'data/images/sync_left.png',
                            '--right_image', 'data/images/sync_right.png',
                            '--left_calib', 'data/calibration/thermal_left.yaml',
                            '--right_calib', 'data/calibration/thermal_right.yaml'
                        ]):
                            # Should not raise any errors
                            main()

    def test_cli_custom_baseline_argument(self):
        """Test that CLI accepts custom baseline value."""
        from stereo_calibration_check.main import main
        import numpy as np

        # Mock display functions and image processing
        with patch('stereo_calibration_check.main.display_images_side_by_side'):
            with patch('stereo_calibration_check.main.draw_epilines') as mock_epilines:
                with patch('stereo_calibration_check.main.cv2.imread') as mock_imread:
                    with patch('stereo_calibration_check.main.distort_image') as mock_distort:
                        dummy_image = np.zeros((100, 100), dtype=np.uint8)
                        mock_imread.return_value = dummy_image
                        mock_distort.return_value = dummy_image
                        mock_epilines.return_value = (dummy_image, dummy_image)

                        with patch('sys.argv', [
                            'stereo-calibration-check',
                            '--left_image', 'data/images/sync_left.png',
                            '--right_image', 'data/images/sync_right.png',
                            '--left_calib', 'data/calibration/thermal_left.yaml',
                            '--right_calib', 'data/calibration/thermal_right.yaml',
                            '--baseline', '0.15'
                        ]):
                            main()

    @pytest.mark.parametrize("baseline_value", ["0.05", "0.10", "0.15", "0.20", "0.30"])
    def test_cli_various_baseline_values(self, baseline_value):
        """Test CLI with various baseline values."""
        from stereo_calibration_check.main import main
        import numpy as np

        with patch('stereo_calibration_check.main.display_images_side_by_side'):
            with patch('stereo_calibration_check.main.draw_epilines') as mock_epilines:
                with patch('stereo_calibration_check.main.cv2.imread') as mock_imread:
                    with patch('stereo_calibration_check.main.distort_image') as mock_distort:
                        dummy_image = np.zeros((100, 100), dtype=np.uint8)
                        mock_imread.return_value = dummy_image
                        mock_distort.return_value = dummy_image
                        mock_epilines.return_value = (dummy_image, dummy_image)

                        with patch('sys.argv', [
                            'stereo-calibration-check',
                            '--left_image', 'data/images/sync_left.png',
                            '--right_image', 'data/images/sync_right.png',
                            '--left_calib', 'data/calibration/thermal_left.yaml',
                            '--right_calib', 'data/calibration/thermal_right.yaml',
                            '--baseline', baseline_value
                        ]):
                            main()

    def test_baseline_passed_to_get_projection_matrix(self):
        """Test that baseline argument is correctly passed to get_projection_matrix."""
        from stereo_calibration_check.main import main
        import numpy as np

        with patch('stereo_calibration_check.main.display_images_side_by_side'):
            with patch('stereo_calibration_check.main.cv2.imread') as mock_imread:
                with patch('stereo_calibration_check.main.get_projection_matrix') as mock_get_proj:
                    # Setup mocks
                    dummy_image = np.zeros((100, 100), dtype=np.uint8)
                    mock_imread.return_value = dummy_image
                    mock_get_proj.return_value = (
                        np.eye(3), np.eye(3),  # R1, R2
                        np.eye(3, 4), np.eye(3, 4)  # P1, P2
                    )

                    custom_baseline = 0.18
                    with patch('sys.argv', [
                        'stereo-calibration-check',
                        '--left_image', 'data/images/sync_left.png',
                        '--right_image', 'data/images/sync_right.png',
                        '--left_calib', 'data/calibration/thermal_left.yaml',
                        '--right_calib', 'data/calibration/thermal_right.yaml',
                        '--baseline', str(custom_baseline)
                    ]):
                        try:
                            main()
                        except:
                            pass

                    # Verify get_projection_matrix was called with correct baseline
                    assert mock_get_proj.called, "get_projection_matrix should be called"
                    call_args = mock_get_proj.call_args

                    # Check that baseline was passed (could be positional or keyword)
                    if call_args.kwargs:
                        assert 'baseline' in call_args.kwargs or len(call_args.args) >= 3
                        if 'baseline' in call_args.kwargs:
                            assert call_args.kwargs['baseline'] == custom_baseline
                        elif len(call_args.args) >= 3:
                            assert call_args.args[2] == custom_baseline
                    else:
                        # Positional argument
                        assert len(call_args.args) >= 3
                        assert call_args.args[2] == custom_baseline

    def test_cli_baseline_type_validation(self):
        """Test that invalid baseline values are rejected."""
        from stereo_calibration_check.main import main

        # Test with invalid (non-numeric) baseline
        with pytest.raises(SystemExit):
            with patch('sys.argv', [
                'stereo-calibration-check',
                '--baseline', 'invalid'
            ]):
                main()
