"""
Distort the rectified images back to their original distorted form.
"""
import cv2
import numpy as np
from ..utils.file_utils import load_yaml


def distort_image(rectified_image_path: str, calib_path: str,
                  output_path: [None | str], P_new: np.ndarray) -> np.ndarray:
    """
    Distort a rectified fisheye image back to its original distorted form using calibration data.
    rectified_image_path: Path to the rectified image.
    calib_path: Path to the camera calibration YAML file.
    output_path: Path to save the distorted image. If None, the image is not saved.
    P_new: New projection matrix used during rectification.
    Returns the distorted image as a numpy array.
    """
    cv_image = cv2.imread(rectified_image_path, cv2.IMREAD_GRAYSCALE)
    h, w = cv_image.shape[:2]
    intrinsics = load_yaml(calib_path)
    K = np.array(intrinsics['camera_matrix']['data']).reshape(3, 3)
    D = np.array(intrinsics['distortion_coefficients']['data'])
    R = np.array(intrinsics['rectification_matrix']['data']).reshape(3, 3)
    P = np.array(intrinsics['projection_matrix']['data']).reshape(3, 4)

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, R, P_new, (w, h),
                                                     cv2.CV_16SC2)
    distorted_image = cv2.remap(cv_image, map1, map2, cv2.INTER_LINEAR)

    if output_path is not None:
        cv2.imwrite(output_path, distorted_image)

    return distorted_image
