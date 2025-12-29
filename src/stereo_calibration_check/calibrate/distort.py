import cv2
import numpy as np

def distort_image(image: np.ndarray, intrinsics: dict, P_new: np.ndarray) -> np.ndarray:
    """
    Distorts a rectified image back to the original fisheye lens geometry.
    
    Args:
        image: The input RECTIFIED image.
        intrinsics: Dict containing 'K', 'D', 'R' of the ORIGINAL fisheye camera.
                    R is the rotation from Original -> Rectified Frame.
        P_new: The 3x4 projection matrix of the RECTIFIED camera.
    """
    h, w = image.shape[:2]
    K = np.array(intrinsics['K'])
    D = np.array(intrinsics['D'])
    R = np.array(intrinsics['R'])
    
    # 1. Create grid for the DESTINATION (Distorted) image
    # These are the pixels (u,v) in the final fisheye image
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    distorted_points = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 1, 2).astype(np.float32)

    # 2. Un-project distorted pixels to 3D rays in the ORIGINAL Camera Frame
    # undistortPoints with default R=Eye, P=Eye returns normalized coordinates (x, y)
    # on the plane z=1 in the original camera frame.
    undistorted_norm = cv2.fisheye.undistortPoints(distorted_points, K, D)
    
    # Convert to Homogeneous 3D vectors (x, y, 1)
    # Shape: (N, 3)
    vectors_orig = np.concatenate([undistorted_norm.reshape(-1, 2), np.ones((h*w, 1))], axis=1)

    # 3. Rotate rays from ORIGINAL Frame to RECTIFIED Frame
    # X_rect = R * X_orig
    # We transpose vectors_orig to (3, N) for matrix mult, then transpose back
    vectors_rect = (R @ vectors_orig.T).T

    # 4. Project these 3D rays onto the RECTIFIED Image Plane
    # P_new usually has the form [K_rect | Tx]. We only need K_rect for projection.
    K_rect = P_new[:3, :3]
    
    # Project: uv_homo = K_rect * vectors_rect
    rectified_pixels_homo = (K_rect @ vectors_rect.T).T
    
    # Normalize by Z to get pixel coordinates (u, v)
    map_x = rectified_pixels_homo[:, 0] / rectified_pixels_homo[:, 2]
    map_y = rectified_pixels_homo[:, 1] / rectified_pixels_homo[:, 2]

    # 5. Reshape and Remap
    map_x = map_x.reshape(h, w).astype(np.float32)
    map_y = map_y.reshape(h, w).astype(np.float32)

    # We use INTER_LINEAR or INTER_CUBIC. BORDER_CONSTANT is safe for out-of-bounds.
    redistorted_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    return redistorted_image
