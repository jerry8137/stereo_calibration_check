"""
Module to undistort and rectify fisheye images using camera calibration data.
"""
import cv2
import numpy as np


def rectify_image(image_path: str, output_path: [str | None], intrinsics: dict,
                  P_new: np.ndarray) -> np.ndarray:
    """
    Rectify a fisheye image using calibration data.
    """
    cv_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = cv_image.shape[:2]
    K = intrinsics['K']
    D = intrinsics['D']
    R = intrinsics['R']

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, R, P_new, (w, h),
                                                     cv2.CV_16SC2)
    rectified_image = cv2.remap(cv_image, map1, map2, cv2.INTER_LINEAR)

    if output_path is not None:
        cv2.imwrite(output_path, rectified_image)

    return rectified_image
