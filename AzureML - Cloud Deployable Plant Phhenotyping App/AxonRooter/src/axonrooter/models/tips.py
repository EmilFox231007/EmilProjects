"""
This module extract the coordinates of bottom-most tips of segmented root structures
from a labeled mask. The approach uses skeletonization and graph analysis to identify the lowest
points in each root's path. The function visualizes and returns the tips as pixel coordinates.
"""

# Import necessary libraries
import logging

import matplotlib.pyplot as plt  # noqa: F401
import networkx as nx
import numpy as np
from skan.csr import skeleton_to_csgraph
from skimage.morphology import skeletonize

from axonrooter.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def extract_bottom_tips(labeled_mask, max_roots=5):
    """
    Extract bottom-most tip coordinates of roots from a labeled mask.

    This function:
    - Skeletonizes the binary mask of each labeled root.
    - Converts each skeleton into a graph and computes top and bottom nodes.
    - Selects only roots that start in the top half of the image.
    - Returns up to `max_roots` largest roots based on pixel area.
    - Plots the detected tips on the labeled mask.

    Parameters
    ----------
    labeled_mask : numpy.ndarray
        2D labeled array where each root has a unique integer label.
    max_roots : int, optional
        Maximum number of roots to process, by area. Default is 5.

    Returns
    -------
    tuple
        (tips, tips_image)
        tips: list of (y, x) pixel coordinates of the detected bottom tips.
        tips_image: np.ndarray, blank image with tips marked as dots (value 255).
    """
    tips = []
    logger.info("Starting extraction of bottom tips")

    skeleton = skeletonize(labeled_mask > 0)
    logger.info("Skeletonization completed")

    # Get unique root IDs, skipping background (0)
    root_ids = np.unique(labeled_mask)
    root_ids = root_ids[root_ids > 0]
    logger.info(f"Found {len(root_ids)} unique roots (excluding background)")

    h, w = labeled_mask.shape
    half_height = h // 2
    logger.info(f"Image height: {h}, half height cutoff: {half_height}")

    # Compute areas of roots
    root_areas = {root_id: np.sum(labeled_mask == root_id) for root_id in root_ids}
    logger.info(f"Computed areas for each root")

    # Sort root IDs by area descending and keep only top max_roots
    top_root_ids = sorted(root_areas, key=root_areas.get, reverse=True)[:max_roots]
    logger.info(f"Selected top {max_roots} roots by area")

    for root_id in top_root_ids:
        logger.info(f"Processing root ID {root_id} with area {root_areas[root_id]}")
        root_skel = (labeled_mask == root_id) & skeleton

        # Convert skeleton to graph
        graph, coords = skeleton_to_csgraph(root_skel)
        G = nx.Graph()
        for i, j in zip(*graph.nonzero()):
            G.add_edge(i, j, weight=graph[i, j])

        try:
            # Find top and bottom nodes in the graph
            top_node = min(G.nodes, key=lambda n: coords[0][n])  # min y (top)
            bottom_node = max(G.nodes, key=lambda n: coords[0][n])  # max y (bottom)
            top_y = coords[0][top_node]

            logger.info(f"Root {root_id}: top node y={top_y}")

            # Filter: Only accept roots starting above halfway point
            if top_y < half_height:
                tip_y = int(coords[0][bottom_node])
                tip_x = int(coords[1][bottom_node])
                tips.append((tip_y, tip_x))
                logger.info(f"Root {root_id}: bottom tip at (y={tip_y}, x={tip_x})")
            else:
                logger.info(f"Root {root_id} discarded: top node below half height")
        except (ValueError, IndexError, KeyError) as e:
            logger.warning(f"Failed to process root ID {root_id}: {e}")
            continue

    logger.info(f"Detected {len(tips)} bottom tips")

    return tips
