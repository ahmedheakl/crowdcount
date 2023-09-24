"""Utility functions for crowd counting"""
import os

import cv2
import numpy as np
from google_drive_downloader import GoogleDriveDownloader as gdd

from crowdcount import IMG_SIZE


def preprocess(img: np.ndarray) -> np.ndarray:
    """Preprocess image for crowd counting

    Args:
        img (np.array): Input unprocessed image

    Returns:
        np.array: Processed image
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32, copy=False)
    img = cv2.resize(img, IMG_SIZE)
    img = img.reshape((1, 1, img.shape[0], img.shape[1]))
    return img


def generate_density_map(density_map: np.ndarray) -> np.ndarray:
    """Generate density map from crowd count

    Args:
        density_map (np.ndarray): Input desnity map

    Returns:
        np.ndarray: processed density map
    """
    density_map = 255 * density_map / np.max(density_map)
    density_map = density_map[0][0]
    result_img = density_map.astype(np.uint8, copy=False)
    result_img = cv2.applyColorMap(result_img, cv2.COLORMAP_JET)
    return result_img


def download_file(file_url: str, file_name: str) -> str:
    """Download file from url

    Args:
        file_url (str): File URL
        file_name (str): Name of the file to be saved

    Raises:
        OSError: Check for the supported operating system

    Returns:
        str: Path to the downloaded file
    """

    # Define the path to the .cache directory for each operating system
    if os.name == "nt":  # Windows
        cache_dir = os.path.join(os.environ["LOCALAPPDATA"], ".cache/crowdcount")
    elif os.name == "posix":  # Linux or Mac
        cache_dir = os.path.join(os.environ["HOME"], ".cache/crowdcount")
    else:
        raise OSError("Unsupported operating system")

    os.makedirs(cache_dir, exist_ok=True)
    model_weights_path = os.path.join(cache_dir, file_name)

    if os.path.exists(model_weights_path):
        return model_weights_path

    url_id = file_url.split("/")[-2]

    gdd.download_file_from_google_drive(file_id=url_id, dest_path=model_weights_path)
    return model_weights_path
