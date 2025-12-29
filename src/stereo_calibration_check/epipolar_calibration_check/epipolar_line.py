"""
Module to draw epipolar lines on images using corner detection.
"""
import cv2 as cv
import numpy as np
from feature_detection.corners import corner_detection
from feature_detection.sift import sift_feature_detection


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


def draw_epilines_corners(img1: np.ndarray,
                          img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts1, pts2 = corner_detection(img1, img2)
    # Compute Fundamental Matrix
    F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
    return img5, img3


def draw_epilines_sift(img1: np.ndarray,
                       img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts1, pts2 = sift_feature_detection(img1, img2)
    margin = 100
    width = img1.shape[1]
    height = img1.shape[0]

    mask_inliers = (pts1[:, 0] > margin) & (pts1[:, 0] < width - margin) & \
                   (pts1[:, 1] > margin) & (pts1[:, 1] < height - margin) & \
                   (pts2[:, 0] > margin) & (pts2[:, 0] < width - margin) & \
                   (pts2[:, 1] > margin) & (pts2[:, 1] < height - margin)
    pts1 = pts1[mask_inliers]
    pts2 = pts2[mask_inliers]
    # Compute Fundamental Matrix
    F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
    return img5, img3
