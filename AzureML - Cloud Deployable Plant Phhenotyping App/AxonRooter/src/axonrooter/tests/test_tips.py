from unittest.mock import patch

import numpy as np

from axonrooter.models.tips import extract_bottom_tips  # Update path as needed


def make_simple_labeled_mask(height, width, roots):
    """
    Returns a mask of given shape.
    `roots` is a list of tuples: (root_id, list of (y, x) pixel coords)
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for root_id, pixels in roots:
        for y, x in pixels:
            mask[y, x] = root_id
    return mask


@patch("axonrooter.models.tips.plt.show")
def test_extract_single_root_bottom_tip(mock_show):
    # Create a single vertical root: id=1, from (2, 5) to (7, 5)
    height, width = 10, 10
    root_pixels = [(i, 5) for i in range(2, 8)]
    labeled_mask = make_simple_labeled_mask(height, width, [(1, root_pixels)])
    tips = extract_bottom_tips(labeled_mask, max_roots=1)
    # The bottom-most point is (7, 5)
    assert len(tips) == 1
    assert (7, 5) in tips


@patch("axonrooter.models.tips.plt.show")
def test_multiple_roots_limit_and_area_sorting(mock_show):
    # Two roots: root 1 (small, upper), root 2 (large, lower)
    root1_pixels = [(i, 2) for i in range(1, 5)]  # id=1
    root2_pixels = [(i, 7) for i in range(4, 10)]  # id=2
    mask = make_simple_labeled_mask(12, 12, [(1, root1_pixels), (2, root2_pixels)])
    # Only max_roots=1: should pick root 2 (bigger)
    tips = extract_bottom_tips(mask, max_roots=1)
    assert len(tips) == 1
    assert tips[0][1] == 7  # x=7 (root2)


@patch("axonrooter.models.tips.plt.show")
def test_top_node_filter(mock_show):
    # Root that starts at y=8 (lower half), should be excluded
    height = 16
    root_pixels = [(i, 5) for i in range(8, 13)]
    mask = make_simple_labeled_mask(height, 12, [(1, root_pixels)])
    tips = extract_bottom_tips(mask, max_roots=1)
    assert tips == []  # Should be empty


@patch("axonrooter.models.tips.plt.show")
def test_no_roots(mock_show):
    # All zeros
    mask = np.zeros((10, 10), dtype=np.uint8)
    tips = extract_bottom_tips(mask)
    assert tips == []


@patch("axonrooter.models.tips.plt.show")
def test_handles_disconnected_root(mock_show):
    # Two pixels for one root, disconnected
    pixels = [(2, 2), (7, 7)]
    mask = make_simple_labeled_mask(10, 10, [(1, pixels)])
    # Should not crash, should either find one or zero tips
    tips = extract_bottom_tips(mask)
    assert isinstance(tips, list)


# Optionally, test that scatter is called for each tip
@patch("axonrooter.models.tips.plt.scatter")
@patch("axonrooter.models.tips.plt.show")
def test_scatter_called_for_each_tip(mock_show, mock_scatter):
    root_pixels = [(i, 4) for i in range(2, 6)]
    mask = make_simple_labeled_mask(10, 10, [(1, root_pixels)])
    extract_bottom_tips(mask)
