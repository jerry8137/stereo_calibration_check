import cv2 as cv
import numpy as np


def corner_detection(img1: np.ndarray,
                     img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pattern_size = (8, 6)

    # 3. Find Chessboard Corners (Replaces SIFT)
    ret1, corners1 = cv.findChessboardCorners(img1, pattern_size, None)
    ret2, corners2 = cv.findChessboardCorners(img2, pattern_size, None)

    if ret1 and ret2:
        # 4. Refine corners to sub-pixel accuracy (Critical for calibration/epipolar geometry)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30,
                    0.001)
        corners1 = cv.cornerSubPix(img1, corners1, (11, 11), (-1, -1),
                                   criteria)
        corners2 = cv.cornerSubPix(img2, corners2, (11, 11), (-1, -1),
                                   criteria)

        # 5. Prepare points
        # Squeeze the shape from (N, 1, 2) to (N, 2) to match the original script's format
        pts1 = corners1.squeeze()
        pts2 = corners2.squeeze()

        # Convert to int32 as per the original script's requirement for drawing
        # (Though findFundamentalMat works better with floats, we stick to your original flow)
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)

    else:
        print(
            "Error: Chessboard corners not found in one or both images. Check pattern_size."
        )
        exit()

    return pts1, pts2


