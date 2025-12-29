import cv2
import numpy as np


def get_projection_matrix(
    intrinsics1: dict, intrinsics2: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the projection matrices for stereo rectification from two calibration files.
    """
    K1 = intrinsics1['K']
    D1 = intrinsics1['D']
    K2 = intrinsics2['K']
    D2 = intrinsics2['D']
    w = intrinsics1['width']
    h = intrinsics1['height']

    R = np.eye(3)
    T = np.array([[0.12], [0], [0]])
    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        K1,
        D1,
        K2,
        D2, (w, h),
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        balance=0.0)
    return R1, R2, P1, P2
