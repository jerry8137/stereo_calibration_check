from calibrate.stereo_calibrate import get_projection_matrix
from utils.file_utils import load_intrinsics
from calibrate.undistort import rectify_image
from utils.visualization import display_images_side_by_side

import cv2

if __name__ == "__main__":
    intrinsics1 = load_intrinsics('calibration/thermal_left.yaml')
    intrinsics2 = load_intrinsics('calibration/thermal_right.yaml')

    P1, P2 = get_projection_matrix(intrinsics1, intrinsics2)

    image1 = cv2.imread('left.png')
    image2 = cv2.imread('right.png')
    rect_image1 = rectify_image(image1, None, intrinsics1, P1)
    rect_image2 = rectify_image(image2, None, intrinsics2, P2)
    display_images_side_by_side(rect_image1, rect_image2, 'Rectified Images')
