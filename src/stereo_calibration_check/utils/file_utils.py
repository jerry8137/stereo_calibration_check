"""
Utility functions for file operations.
"""
import yaml
import numpy as np


def load_yaml(file_path: str) -> dict:
    """
    Load a YAML file and return its contents as a dictionary.
    """

    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def load_intrinsics(calib_path: str) -> dict:
    """
    Load camera intrinsics from a YAML file.
    """
    intrinsics = load_yaml(calib_path)
    data = dict()
    data['K'] = np.array(intrinsics['camera_matrix']['data']).reshape(3, 3)
    data['D'] = np.array(intrinsics['distortion_coefficients']['data'])
    data['R'] = np.array(intrinsics['rectification_matrix']['data']).reshape(
        3, 3)
    data['P'] = np.array(intrinsics['projection_matrix']['data']).reshape(3, 4)
    data['width'] = intrinsics['image_width']
    data['height'] = intrinsics['image_height']

    return data
