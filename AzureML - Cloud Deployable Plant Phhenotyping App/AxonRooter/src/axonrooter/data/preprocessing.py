"""
This module provides utility functions for loading, cropping, formatting, and padding
microscopy images in grayscale in preparation for further processing or inference.
The operations include border cropping, edge detection-based square cropping,
and padding to match a specific patch size for model input compatibility.
"""

import logging

# Import necessary libraries
import cv2

# -----------------------------
# Set up logging
# -----------------------------
from axonrooter.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def crop_initial(image, initial_crop=100):
    """
    Crop the borders of an image by removing a fixed number of pixels from each side.

    This function removes border artifacts and noise by cropping a fixed number of pixels
    from all sides of the input image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image array to be cropped.
    initial_crop : int, optional
        Number of pixels to crop from each side. Default is 100.

    Returns
    -------
    tuple
        cropped_image : numpy.ndarray
            Image with borders removed.
        crop_left : int
            Number of pixels cropped from the left (equals initial_crop).
        crop_top : int
            Number of pixels cropped from the top (equals initial_crop).
    """
    logger.debug(f"Cropping image by {initial_crop} pixels from each side.")
    cropped = image[initial_crop:-initial_crop, initial_crop:-initial_crop]
    return cropped, initial_crop, initial_crop


def find_edges(line, threshold=70):
    """
    Find the left and right edges in a 1D intensity line based on threshold crossing.
    This function scans a 1D intensity profile to detect edges where pixel values
    cross a specified threshold, indicating transitions between background and content.

    Parameters
    ----------
    line : numpy.ndarray
        1D array representing pixel intensity values along a line.
    threshold : int, optional
        Threshold value to detect edge transitions. Default is 70.

    Returns
    -------
    tuple of int or None
        A tuple containing:

        - **left_edge** (*int or None*): Index of the left edge where intensity crosses the threshold.
        - **right_edge** (*int or None*): Index of the right edge where intensity crosses the threshold.

        Returns (None, None) if no edges are found.
    """
    left_edge, right_edge = None, None
    n = len(line)

    # Find left edge where intensity crosses threshold going up or down
    for i in range(1, n):
        if line[i - 1] < threshold <= line[i] or line[i - 1] >= threshold > line[i]:
            left_edge = i
            break

    # Find right edge similarly but scanning from right to left
    for i in range(n - 1, 0, -1):
        if line[i] < threshold <= line[i - 1] or line[i] >= threshold > line[i - 1]:
            right_edge = i
            break

    if left_edge is None or right_edge is None:
        logger.warning(f"Could not find both edges. Left: {left_edge}, Right: {right_edge}")
        return None, None

    return left_edge, right_edge


def format_crop(image):
    """
    Crop the image to a square region based on edge detection in central profiles.

    This function analyzes the central horizontal and vertical intensity profiles
    to detect content boundaries, then creates a square crop that encompasses
    the detected content area.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image to be cropped.

    Returns
    -------
    tuple
        cropped_image : numpy.ndarray
            Square-cropped image based on detected edges.
        left_x : int
            X-coordinate offset of the crop from the original image.
        top_y : int
            Y-coordinate offset of the crop from the original image.
    """
    logger.info(f"Performing format crop on image of shape {image.shape}")

    # Take the center row (horizontal line) and center column (vertical line)
    center_row = image.shape[0] // 2
    center_col = image.shape[1] // 2

    horizontal_line = image[center_row, :]
    vertical_line = image[:, center_col]

    # Detect edges in the horizontal and vertical profiles
    left_x, right_x = find_edges(horizontal_line)
    top_y, bottom_y = find_edges(vertical_line)

    if None in (left_x, right_x, top_y, bottom_y):
        logger.warning("One or more edges were not found. Returning original image.")
        return image, 0, 0

    # Calculate dimensions
    detected_width = right_x - left_x
    detected_height = bottom_y - top_y

    logger.info(f"Detected content dimensions: {detected_width} x {detected_height}")

    # Calculate side length for a square crop
    side_length = max(detected_width, detected_height)

    # Define square crop coordinates
    right_x = left_x + side_length
    bottom_y = top_y + side_length

    # Check if the square crop would exceed image boundaries
    image_height, image_width = image.shape[:2]

    if right_x > image_width or bottom_y > image_height:
        logger.info("Square crop exceeds boundaries, adjusting position...")

        # Alternative approach: adjust the starting position to fit the square
        if right_x > image_width:
            # Shift left to fit within width
            shift_left = right_x - image_width
            left_x = max(0, left_x - shift_left)
            right_x = left_x + side_length

        if bottom_y > image_height:
            # Shift up to fit within height
            shift_up = bottom_y - image_height
            top_y = max(0, top_y - shift_up)
            bottom_y = top_y + side_length

        # If still exceeding boundaries, use the smaller dimension
        if right_x > image_width or bottom_y > image_height:
            logger.warning("Using smaller dimension to fit within image boundaries")
            available_width = image_width - left_x
            available_height = image_height - top_y
            side_length = min(available_width, available_height, detected_width, detected_height)
            right_x = left_x + side_length
            bottom_y = top_y + side_length

    # Final boundary check
    right_x = min(right_x, image_width)
    bottom_y = min(bottom_y, image_height)

    # Crop and return the region
    cropped_image = image[top_y:bottom_y, left_x:right_x]

    logger.info(f"Format crop completed. Final shape: {cropped_image.shape}")

    return cropped_image, left_x, top_y


def padding_function(formatted, PATCH_SIZE=256):
    """
    Pad the input image so its dimensions are multiples of PATCH_SIZE.

    This function ensures the image dimensions are compatible with patch-based
    processing by padding the image to the nearest multiple of PATCH_SIZE.

    Parameters
    ----------
    formatted : numpy.ndarray
        Input image to be padded.
    PATCH_SIZE : int, optional
        Target patch size multiple for padding. Default is 256.

    Returns
    -------
    tuple
        padded_image : numpy.ndarray
            Image padded to the nearest multiple of PATCH_SIZE.
        pad_top : int
            Number of pixels padded at the top.
        pad_bottom : int
            Number of pixels padded at the bottom.
        pad_left : int
            Number of pixels padded on the left.
        pad_right : int
            Number of pixels padded on the right.
    """
    h, w = formatted.shape
    pad_h = ((h // PATCH_SIZE) + 1) * PATCH_SIZE - h
    pad_w = ((w // PATCH_SIZE) + 1) * PATCH_SIZE - w

    # Distribute padding evenly on top/bottom and left/right
    pad_top, pad_bottom = pad_h // 2, pad_h - (pad_h // 2)
    pad_left, pad_right = pad_w // 2, pad_w - (pad_w // 2)

    logger.debug(
        f"Padding image: top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}"
    )

    # Apply constant padding with black (0) pixels
    padded_image = cv2.copyMakeBorder(
        formatted,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    return padded_image, pad_top, pad_bottom, pad_left, pad_right


def load_and_preprocess_image(image, PATCH_SIZE=256):
    """
    Complete preprocessing pipeline for microscopy images.

    This function applies the full preprocessing pipeline including initial border
    cropping, content-based format cropping, and padding for patch compatibility.

    Parameters
    ----------
    image : numpy.ndarray
        Input image (BGR or grayscale) to be preprocessed.
    PATCH_SIZE : int, optional
        Target patch size multiple for padding. Default is 256.

    Returns
    -------
    tuple
        padded_image : numpy.ndarray
            Fully preprocessed and padded grayscale image.
        pad_top : int
            Number of pixels padded at the top.
        pad_bottom : int
            Number of pixels padded at the bottom.
        pad_left : int
            Number of pixels padded on the left.
        pad_right : int
            Number of pixels padded on the right.
        total_crop_left : int
            Total horizontal offset from all cropping operations.
        total_crop_top : int
            Total vertical offset from all cropping operations.

    Notes
    -----
    The preprocessing pipeline consists of:
    1. Conversion to grayscale (if needed)
    2. Initial border cropping (removes 100px borders)
    3. Format cropping (content-based square crop)
    4. Padding (to nearest PATCH_SIZE multiple)
    """
    logger.info(f"Starting preprocessing pipeline for image of shape {image.shape}")

    # If image is not grayscale, convert to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logger.debug("Converted image to grayscale")

    # Step 1: Crop the initial borders
    cropped_image, crop_left, crop_top = crop_initial(image)
    logger.info(f"Initial crop completed. Shape: {cropped_image.shape}")

    # Step 2: Format crop based on content detection
    formatted_image, format_left, format_top = format_crop(cropped_image)
    logger.info(f"Format crop completed. Shape: {formatted_image.shape}")

    # Step 3: Pad the image for patch compatibility
    padded_image, pad_top, pad_bottom, pad_left, pad_right = padding_function(
        formatted_image, PATCH_SIZE
    )
    logger.info(f"Padding completed. Final shape: {padded_image.shape}")

    # Calculate total crop offsets for coordinate transformation
    total_crop_left = crop_left + format_left
    total_crop_top = crop_top + format_top

    logger.info("Image preprocessing pipeline completed successfully")

    return padded_image, pad_top, pad_bottom, pad_left, pad_right, total_crop_left, total_crop_top
