import cv2
import numpy as np
from ..utils.file_utils import load_yaml


def get_projection_matrix(calib_path1: str, calib_path2: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the projection matrices for stereo rectification from two calibration files.
    """
    intrinsics1 = load_yaml(calib_path1)
    intrinsics2 = load_yaml(calib_path2)
    K1 = np.array(intrinsics1['camera_matrix']['data']).reshape(3, 3)
    D1 = np.array(intrinsics1['distortion_coefficients']['data'])
    # R1 = np.array(intrinsics1['rectification_matrix']['data']).reshape(3, 3)
    P1 = np.array(intrinsics1['projection_matrix']['data']).reshape(3, 4)
    K2 = np.array(intrinsics2['camera_matrix']['data']).reshape(3, 3)
    D2 = np.array(intrinsics2['distortion_coefficients']['data'])
    # R2 = np.array(intrinsics2['rectification_matrix']['data']).reshape(3, 3)
    P2 = np.array(intrinsics2['projection_matrix']['data']).reshape(3, 4)
    R = np.eye(3)
    T = np.array([[0], [0.12], [0]])
    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        K1,
        D1,
        K2,
        D2, (640, 512),
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        balance=0.0)

    return P1, P2
