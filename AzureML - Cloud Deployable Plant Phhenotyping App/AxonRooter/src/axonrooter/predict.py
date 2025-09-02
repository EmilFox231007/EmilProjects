"""
Core inference pipeline for AxonRooter.

This module defines the `predict_root_tips` function, which runs the full
pipeline to detect root tip coordinates from either a real image or a
simulation plate. Steps include cropping, padding, model inference,
morphological cleaning, bounding box filtering, and tip extraction.
"""

import logging
import os

import cv2

from axonrooter.data.preprocessing import crop_initial, format_crop, padding_function
from axonrooter.logging_config import setup_logging
from axonrooter.models.inference import run_inference
from axonrooter.models.tips import extract_bottom_tips
from axonrooter.utils.bboxes import apply_strict_bboxes_with_top_cutoff_and_proximity
from axonrooter.utils.morphology import preprocess_single_mask

setup_logging()
logger = logging.getLogger(__name__)


def predict_root_tips(image_path: str):
    """
    Predicts root tip coordinates from an input image.

    This function processes the given image through a root detection pipeline,
    including cropping, formatting, padding, inference, and post-processing
    to extract bottom tip coordinates of detected root structures.

    Parameters
    ----------
    image_path : str
        Path to the input image file. The image should be in a format
        readable by OpenCV (e.g., .png, .jpg).

    Returns
    -------
    tips : list of tuple of int
        A list of (x, y) coordinate tuples representing the positions of
        detected root tips.

    Raises
    ------
    FileNotFoundError
        If the specified image file does not exist.
    ValueError
        If the image cannot be loaded (e.g., unsupported format or corrupted file).

    Notes
    -----
    The pipeline assumes the input image is in grayscale or convertible to grayscale.
    Internally, it applies a series of transformations including:
    - Cropping the root region
    - Formatting and padding the image
    - Running a prediction model to generate a mask
    - Post-processing the mask to extract root tips

    See Also
    --------
    crop_initial, format_crop, padding_function, run_inference,
    preprocess_single_mask, apply_strict_bboxes_with_top_cutoff_and_proximity,
    extract_bottom_tips
    """
    if not os.path.exists(image_path):
        logger.error(f"Image file does not exist: {image_path}")
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.error(f"Could not load image from: {image_path}")
        raise ValueError(f"Could not load image from: {image_path}")

    # Inference pipeline
    logger.info("Starting preprocessing pipeline...")

    # Step-by-step logging
    logger.info("Cropping root region...")
    cropped = crop_initial(image)

    logger.info("Formatting cropped region...")
    formatted = format_crop(cropped)

    logger.info("Padding formatted image...")
    padded = padding_function(formatted)

    logger.info("Running inference model...")
    pred_mask = run_inference(*padded)

    logger.info("Preprocessing predicted mask...")
    cleaned_mask = preprocess_single_mask(pred_mask)

    logger.info("Applying strict bounding box filter...")
    labeled_mask, final_boxes = apply_strict_bboxes_with_top_cutoff_and_proximity(
        cleaned_mask, padded[0]
    )

    logger.info("Extracting bottom tips from labeled mask...")
    tips = extract_bottom_tips(labeled_mask)

    logger.info(f"Prediction complete. Found {len(tips)} root tip(s).")
    return tips
