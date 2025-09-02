"""
This module provides utilities to:

1. Compute Intersection-over-Union (IoU) between bounding boxes.

2. Apply strict filtering to bounding boxes extracted from binary masks, including:

   - size filtering
   - position-based exclusion (e.g. top or bottom Y cutoffs)
   - IoU-based suppression (non-max suppression)
   - proximity filtering (remove horizontally close boxes)

3. Generate a labeled mask and visualize filtered boxes over the image.

Useful for post-processing binary segmentation masks in root or object detection pipelines.
"""
import logging

import matplotlib.pyplot as plt  # noqa: F401

# Import necessary libraries
import numpy as np
from scipy.ndimage import label

# --- Setup Logging ---
from axonrooter.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
        box1 (tuple): Bounding box 1 (x_min, y_min, x_max, y_max).
        box2 (tuple): Bounding box 2 (x_min, y_min, x_max, y_max).

    Returns:
        float: IoU value.
    """
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])

    # Calculate intersection area
    inter_width = max(0, x_max - x_min)
    inter_height = max(0, y_max - y_min)
    inter_area = inter_width * inter_height

    # Calculate area of both bounding boxes
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate union area
    union_area = area_box1 + area_box2 - inter_area

    iou = round(inter_area / union_area, 6) if union_area > 0 else 0
    logger.debug(f"Calculated IoU between {box1} and {box2}: {iou}")
    return iou


def apply_strict_bboxes_with_top_cutoff_and_proximity(
    preprocessed_mask,
    original_mask,
    iou_threshold=0.001,
    size_threshold=2000,
    bottom_y_cutoff_ratio=0.85,
    top_y_cutoff=100,
    min_x_distance=20,
):
    """
    Extract bounding boxes from a mask with multiple filtering steps including size threshold,
    bottom cutoff, IoU-based suppression, top cutoff, and proximity-based suppression on the X axis.

    Parameters
    ----------
    preprocessed_mask : np.ndarray
        Binary mask (2D array) after preprocessing, used for connected component detection.
    original_mask : np.ndarray
        Original binary mask (2D array) for labeling the final output.
    iou_threshold : float, optional
        Intersection-over-Union threshold for suppressing overlapping bounding boxes (default is 0.001).
    size_threshold : int, optional
        Minimum area (in pixels) of bounding boxes to keep (default is 2000).
    bottom_y_cutoff_ratio : float, optional
        Fraction of image height below which bounding boxes are considered (default is 0.85).
    top_y_cutoff : int, optional
        Minimum y-coordinate to exclude boxes close to the top of the image (default is 100).
    min_x_distance : int, optional
        Minimum allowed horizontal distance between bounding boxes; boxes closer than this will be suppressed (default is 20).

    Returns
    -------
    labeled_output : np.ndarray
        Labeled mask of the same shape as `original_mask`, with connected components numbered after filtering.
    final : list of tuples
        List of final bounding boxes as tuples:
        (x_min, y_min, x_max, y_max, component_id, area)
        representing coordinates and metadata of each bounding box kept after filtering.

    Notes
    -----
    The function follows these steps:
    1. Label connected components on the preprocessed mask.
    2. Filter boxes by size and bottom cutoff.
    3. Suppress overlapping boxes based on IoU threshold.
    4. Remove boxes intersecting with the top strip.
    5. Suppress boxes too close horizontally, keeping the larger one.
    6. Create a labeled output mask reflecting the final bounding boxes.

    """
    logger.info("Starting bounding box extraction with constraints")
    h, w = original_mask.shape
    bottom_cutoff = int(h * bottom_y_cutoff_ratio)
    logger.debug(f"Image height: {h}, bottom Y cutoff: {bottom_cutoff}")

    labeled_preprocessed, num_labels = label(preprocessed_mask > 0)
    logger.info(f"Found {num_labels} connected components")

    bboxes = []
    for comp_id in range(1, num_labels + 1):
        component = labeled_preprocessed == comp_id
        y_coords, x_coords = np.where(component)

        if len(x_coords) == 0 or len(y_coords) == 0:
            continue

        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        area = (x_max - x_min + 1) * (y_max - y_min + 1)

        if area >= size_threshold and y_min < bottom_cutoff:
            bboxes.append((x_min, y_min, x_max, y_max, comp_id, area))
            logger.debug(
                f"Accepted box #{comp_id} at {(x_min, y_min, x_max, y_max)} with area {area}"
            )

    logger.info(f"{len(bboxes)} bounding boxes passed size and bottom cutoff filtering")

    # --- IoU suppression ---
    bboxes = sorted(bboxes, key=lambda box: box[5], reverse=True)
    filtered = []
    for box in bboxes:
        keep = True
        for kept in filtered:
            if calculate_iou(box[:4], kept[:4]) > iou_threshold:
                if box[5] <= kept[5]:
                    logger.debug(f"Suppressed box {box[:4]} due to IoU with {kept[:4]}")
                    keep = False
                    break
        if keep:
            filtered.append(box)

    logger.info(f"{len(filtered)} boxes after IoU suppression")

    # --- Post-filter: remove boxes intersecting top cutoff (y < 100) ---
    filtered = [box for box in filtered if box[1] >= top_y_cutoff]
    logger.info(f"{len(filtered)} boxes after top strip exclusion (cutoff = {top_y_cutoff})")

    # --- Post-filter: remove boxes too close in X axis ---
    kept_indices = set(range(len(filtered)))
    for i in range(len(filtered)):
        if i not in kept_indices:
            continue
        x_min_i, _, x_max_i, _, _, area_i = filtered[i]
        for j in range(i + 1, len(filtered)):
            if j not in kept_indices:
                continue
            x_min_j, _, x_max_j, _, _, area_j = filtered[j]

            # Compute horizontal gap (can be negative if overlapping)
            x_gap = min(abs(x_min_i - x_max_j), abs(x_min_j - x_max_i))
            if x_gap < min_x_distance:
                if area_i >= area_j:
                    logger.debug(f"Removing box {j} due to proximity to box {i}")
                    kept_indices.discard(j)
                else:
                    logger.debug(f"Removing box {i} due to proximity to box {j}")
                    kept_indices.discard(i)
                    break  # i is removed, stop comparing it

    final = [filtered[k] for k in sorted(kept_indices)]
    logger.info(f"{len(final)} final boxes after X-axis proximity filtering")

    # --- Create labeled mask ---
    labeled_output = np.zeros_like(original_mask, dtype=np.uint16)
    for box_idx, (x_min, y_min, x_max, y_max, _, _) in enumerate(final, start=1):
        region = original_mask[y_min : y_max + 1, x_min : x_max + 1]
        labeled_region, _ = label(region > 0)
        for cid in range(1, labeled_region.max() + 1):
            comp = labeled_region == cid
            labeled_output[y_min : y_max + 1, x_min : x_max + 1][comp] = box_idx

    logger.info("Labeled output mask created")

    # --- Visualization ---
    logger.debug("Displaying output mask with bounding boxes")
    """plt.figure(figsize=(6, 6))
    plt.imshow(labeled_output, cmap="viridis")
    plt.title("STRICT Mask + Final Y Cutoff + X Proximity Filter")
    plt.axis("off")
    for box in final:
        x_min, y_min, x_max, y_max, *_ = box
        plt.gca().add_patch(
            plt.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                edgecolor="red",
                fill=False,
                linewidth=2,
            )
        )
    plt.show()"""

    return labeled_output, final
