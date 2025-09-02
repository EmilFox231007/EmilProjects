"""
Provides utility functions for post-processing binary masks.
Includes morphological operations such as closing to remove small holes and connect nearby regions.
"""

import logging

import cv2
import numpy as np

from axonrooter.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def preprocess_single_mask(mask, kernel_size=(32, 32)):
    """
    Apply morphological closing to a single binary mask to remove small holes
    and connect nearby components.

    Parameters:
        mask (np.ndarray): Input binary mask (0 and 1 or 0 and 255).
        kernel_size (tuple): Size of the elliptical structuring element.

    Returns:
        np.ndarray: Post-processed binary mask.
    """
    logger.info("Starting morphological preprocessing of mask")
    logger.debug(f"Kernel size: {kernel_size}")
    logger.debug(f"Input mask shape: {mask.shape}, dtype: {mask.dtype}")

    if mask.size == 0 or 0 in mask.shape:
        return mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    closed_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    logger.info("Morphological closing applied")
    logger.debug(f"Closed mask shape: {closed_mask.shape}, dtype: {closed_mask.dtype}")

    # Visualize
    logger.debug("Displaying closed mask")
    """plt.figure(figsize=(6, 6))
    plt.imshow(closed_mask, cmap="gray")
    plt.title("Closed Mask (Morphological)")
    plt.axis("off")
    plt.show()"""

    return closed_mask
