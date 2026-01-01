from calibrate.stereo_calibrate import get_projection_matrix
from calibrate.undistort import rectify_image
from calibrate.distort import distort_image
from epipolar_calibration_check.epipolar_line import draw_epilines_sift as draw_epilines
from utils.file_utils import load_intrinsics
from utils.visualization import display_images_side_by_side

import cv2


def main():
    intrinsics1 = load_intrinsics('data/calibration/thermal_left.yaml')
    intrinsics2 = load_intrinsics('data/calibration/thermal_right.yaml')

    image1 = cv2.imread('data/images/chess_left_distorted.png',
                        cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread('data/images/chess_right_distorted.png',
                        cv2.IMREAD_GRAYSCALE)
    display_images_side_by_side(image1, image2, 'Original Distorted Images')

    image1_raw = cv2.imread('left2.png', cv2.IMREAD_GRAYSCALE)
    image2_raw = cv2.imread('right2.png', cv2.IMREAD_GRAYSCALE)
    overlay1 = cv2.addWeighted(image1_raw, 0.5, image1, 0.5, 0)
    overlay2 = cv2.addWeighted(image2_raw, 0.5, image2, 0.5, 0)
    display_images_side_by_side(overlay1, overlay2,
                                'Overlay Original and Distorted Images')

    R1, R2, P1, P2 = get_projection_matrix(intrinsics1, intrinsics2)
    rect_image1 = rectify_image(image1, None, intrinsics1, R1, P1)
    rect_image2 = rectify_image(image2, None, intrinsics2, R2, P2)
    display_images_side_by_side(rect_image1, rect_image2, 'Rectified Images')

    epiline_image1, epiline_image2 = draw_epilines(rect_image1, rect_image2)
    display_images_side_by_side(epiline_image1, epiline_image2,
                                'Epilines on Rectified Images')

    rect_image1_raw = rectify_image(image1_raw, None, intrinsics1, R1, P1)
    rect_image2_raw = rectify_image(image2_raw, None, intrinsics2, R2, P2)
    display_images_side_by_side(rect_image1_raw, rect_image2_raw, 'Rectified Images')

    epiline_image1, epiline_image2 = draw_epilines(rect_image1_raw, rect_image2_raw)
    display_images_side_by_side(epiline_image1, epiline_image2,
                                'Epilines on Rectified Images')

    overlay_rect1 = cv2.addWeighted(rect_image1_raw, 0.5, rect_image1, 0.5, 0)
    overlay_rect2 = cv2.addWeighted(rect_image2_raw, 0.5, rect_image2, 0.5, 0)
    display_images_side_by_side(overlay_rect1, overlay_rect2,
                                'Overlay Original and Rectified Images')
    diff1 = cv2.absdiff(rect_image1_raw, rect_image1)
    diff2 = cv2.absdiff(rect_image2_raw, rect_image2)
    display_images_side_by_side(diff1, diff2,
                                'Difference between Original and Rectified Images')


if __name__ == "__main__":
    main()
