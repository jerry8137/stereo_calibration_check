import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np


def display_images_side_by_side(img1: np.ndarray,
                                img2: np.ndarray,
                                title1: str = "Image 1",
                                title2: str = "Image 2") -> None:
    """
    Display two images side by side for comparison.
    """
    plt.figure(figsize=(10, 5))

    stacked_images = np.hstack((img1, img2))
    plt.imshow(stacked_images, cmap='gray')
    plt.axis('off')
    plt.title(f"{title1} (Left) vs {title2} (Right)")
    plt.show()


def save_images_side_by_side(img1: np.ndarray, img2: np.ndarray,
                             output_path: str) -> None:
    """
    Save two images side by side to a single image file.
    """
    combined_image = np.hstack((img1, img2))
    cv.imwrite(output_path, combined_image)
