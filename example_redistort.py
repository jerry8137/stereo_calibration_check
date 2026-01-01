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

    image1 = cv2.imread('left2.png', cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread('right2.png', cv2.IMREAD_GRAYSCALE)
    rect_image1 = rectify_image(image1, None, intrinsics1, intrinsics1['R'],
                                intrinsics1['P'])
    rect_image2 = rectify_image(image2, None, intrinsics2, intrinsics2['R'],
                                intrinsics2['P'])
    display_images_side_by_side(rect_image1, rect_image2, 'Rectified Images')

    distorted_image1 = distort_image(rect_image1, intrinsics1, intrinsics1['P'])
    distorted_image2 = distort_image(rect_image2, intrinsics2, intrinsics2['P'])
    display_images_side_by_side(distorted_image1, distorted_image2,
                                'Distorted Back Images')
    cv2.imwrite('data/images/chess_left_distorted.png', distorted_image1)
    cv2.imwrite('data/images/chess_right_distorted.png', distorted_image2)
    image1 = cv2.imread('left2.png', cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread('right2.png', cv2.IMREAD_GRAYSCALE)
    overlay1 = cv2.addWeighted(distorted_image1, 0.5, image1, 0.5, 0)
    overlay2 = cv2.addWeighted(distorted_image2, 0.5, image2, 0.5, 0)
    display_images_side_by_side(overlay1, overlay2, 'Overlay Original and Distorted Images')

    image1_distorted = cv2.imread('data/images/chess_left_distorted.png', cv2.IMREAD_GRAYSCALE)
    image2_distorted = cv2.imread('data/images/chess_right_distorted.png', cv2.IMREAD_GRAYSCALE)
    overlay1 = cv2.addWeighted(image1_distorted, 0.5, image1, 0.5, 0)
    overlay2 = cv2.addWeighted(image2_distorted, 0.5, image2, 0.5, 0)
    display_images_side_by_side(overlay1, overlay2, 'Overlay Original and Distorted Images from File')
