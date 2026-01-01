import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 1. Load Images (Ensure these are your chessboard images)
img1 = cv.imread('images/rect_left.png',
                 cv.IMREAD_GRAYSCALE)  # queryimage # left image
img2 = cv.imread('images/rect_right.png',
                 cv.IMREAD_GRAYSCALE)  # trainimage # right image

# 2. Define Chessboard Parameters
# Change this to match the internal corners of your specific chessboard (e.g., 9x6, 7x7)
pattern_size = (8, 6)

# 3. Find Chessboard Corners (Replaces SIFT)
ret1, corners1 = cv.findChessboardCorners(img1, pattern_size, None)
ret2, corners2 = cv.findChessboardCorners(img2, pattern_size, None)

if ret1 and ret2:
    # 4. Refine corners to sub-pixel accuracy (Critical for calibration/epipolar geometry)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners1 = cv.cornerSubPix(img1, corners1, (11, 11), (-1, -1), criteria)
    corners2 = cv.cornerSubPix(img2, corners2, (11, 11), (-1, -1), criteria)

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

# --- The rest of the script remains largely the same ---

# Compute Fundamental Matrix
F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)

# We select only inlier points
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

plt.subplot(121), plt.imshow(img5)
plt.subplot(122), plt.imshow(img3)
plt.show()
