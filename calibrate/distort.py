import cv2
import numpy as np

def distort_image(image: np.ndarray, intrinsics: dict, P_new: np.ndarray) -> np.ndarray:
    """
    Distorts a rectified image back to the original fisheye lens geometry.
    """
    h, w = image.shape[:2]
    K = np.array(intrinsics['K'])
    D = np.array(intrinsics['D'])
    R = np.array(intrinsics['R'])

    # 1. Create a grid of coordinates for the DISTORTED image
    # These are the (u, v) pixels we want to fill in the final output
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    distorted_points = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 1, 2).astype(np.float32)

    # 2. Lift the distorted pixels to 3D camera coordinates (Normalized Image Plane)
    # This removes the effect of K and D
    undistorted_norm = cv2.fisheye.undistortPoints(distorted_points, K, D)

    # 3. Project these 3D points into the RECTIFIED image plane using P_new
    # Since undistortPoints gives us (x, y) on the z=1 plane, we add z=1 and rotate by R
    # However, it's easier to use the geometry: X_rect = R * X_distorted
    
    # Convert to homogeneous coordinates
    undistorted_norm_homo = np.insert(undistorted_norm, 2, 1.0, axis=2) # Shape (N, 1, 3)
    
    # Apply Rotation (Rectification rotation)
    # We use the inverse because we are going from Distorted Space -> Rectified Space
    rectified_3d = (R @ undistorted_norm_homo.reshape(-1, 3).T).T
    
    # Project onto the Rectified Camera Matrix (P_new)
    # P_new is typically [K_new | 0] or [K_new | translation]
    # We only need the 3x3 part for projection
    K_rect = P_new[:3, :3]
    
    # Project: x' = K_rect * (X / Z)
    rectified_pixels_homo = (K_rect @ rectified_3d.T).T
    map_x = rectified_pixels_homo[:, 0] / rectified_pixels_homo[:, 2]
    map_y = rectified_pixels_homo[:, 1] / rectified_pixels_homo[:, 2]

    # 4. Reshape maps and Remap
    map_x = map_x.reshape(h, w).astype(np.float32)
    map_y = map_y.reshape(h, w).astype(np.float32)

    redistorted_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

    return redistorted_image
