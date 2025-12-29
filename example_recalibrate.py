from calibrate.stereo_calibrate import get_projection_matrix
from calibrate.undistort import rectify_image
from calibrate.distort import distort_image
from epipolar_calibration_check.epipolar_line import draw_epilines_sift as draw_epilines
from utils.file_utils import load_intrinsics
from utils.visualization import display_images_side_by_side

import cv2

if __name__ == "__main__":
    intrinsics1 = load_intrinsics('data/calibration/thermal_left.yaml')
    intrinsics2 = load_intrinsics('data/calibration/thermal_right.yaml')

    R1, R2, P1, P2 = get_projection_matrix(intrinsics1, intrinsics2)

    rect_image1 = cv2.imread('data/images/thermal_left_rectified.png',
                             cv2.IMREAD_GRAYSCALE)
    rect_image2 = cv2.imread('data/images/thermal_right_rectified.png',
                             cv2.IMREAD_GRAYSCALE)
    display_images_side_by_side(rect_image1, rect_image2, 'Rectified Images')

    epiline_image1, epiline_image2 = draw_epilines(rect_image1, rect_image2)
    display_images_side_by_side(epiline_image1, epiline_image2,
                                'Epilines on Rectified Images')

    distorted_image1 = distort_image(rect_image1, intrinsics1,
                                     intrinsics1['P'])
    distorted_image2 = distort_image(rect_image2, intrinsics2,
                                     intrinsics2['P'])
    display_images_side_by_side(distorted_image1, distorted_image2,
                                'Distorted Back Images')

    epiline_distorted_image1, epiline_distorted_image2 = draw_epilines(
        distorted_image1, distorted_image2)
    display_images_side_by_side(epiline_distorted_image1,
                                epiline_distorted_image2,
                                'Epilines on Distorted Back Images')

    rectified_back_image1 = rectify_image(distorted_image1, None, intrinsics1,
                                          R1, P1)
    rectified_back_image2 = rectify_image(distorted_image2, None, intrinsics2,
                                          R2, P2)
    epiline_rectified_back_image1, epiline_rectified_back_image2 = draw_epilines(
        rectified_back_image1, rectified_back_image2)
    display_images_side_by_side(epiline_rectified_back_image1,
                                epiline_rectified_back_image2,
                                'Epilines on Rectified Back Images')
