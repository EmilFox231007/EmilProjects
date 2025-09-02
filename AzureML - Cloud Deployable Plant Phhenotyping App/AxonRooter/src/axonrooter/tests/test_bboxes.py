from unittest.mock import patch

import numpy as np

from axonrooter.utils.bboxes import apply_strict_bboxes_with_top_cutoff_and_proximity


def make_box_mask(shape, *boxes):
    """
    Make a binary mask with one or more filled rectangles (x_min, y_min, x_max, y_max).
    """
    mask = np.zeros(shape, dtype=np.uint8)
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        mask[y_min : y_max + 1, x_min : x_max + 1] = 1
    return mask


@patch("axonrooter.utils.bboxes.plt.show")
def test_bbox_finds_single_box(mock_show):
    mask = make_box_mask((200, 200), (10, 20, 30, 60))
    labeled, bboxes = apply_strict_bboxes_with_top_cutoff_and_proximity(
        mask, mask, size_threshold=10, top_y_cutoff=0
    )
    assert len(bboxes) == 1
    x_min, y_min, x_max, y_max, *_ = bboxes[0]
    assert (x_min, y_min, x_max, y_max) == (10, 20, 30, 60)
    assert np.all(labeled[20:61, 10:31] == 1)
    assert np.sum(labeled == 1) > 0


@patch("axonrooter.utils.bboxes.plt.show")
def test_boxes_smaller_than_threshold_are_ignored(mock_show):
    mask = make_box_mask((100, 100), (10, 10, 12, 12))  # 2x2 = 4 area
    labeled, bboxes = apply_strict_bboxes_with_top_cutoff_and_proximity(
        mask, mask, size_threshold=10
    )
    assert len(bboxes) == 0
    assert np.all(labeled == 0)


@patch("axonrooter.utils.bboxes.plt.show")
def test_top_cutoff_removes_boxes_near_top(mock_show):
    mask = make_box_mask((200, 200), (5, 5, 30, 50))
    labeled, bboxes = apply_strict_bboxes_with_top_cutoff_and_proximity(
        mask, mask, top_y_cutoff=40, size_threshold=10
    )
    # Box y_min = 5 < 40, should be filtered
    assert len(bboxes) == 0


@patch("axonrooter.utils.bboxes.plt.show")
def test_iou_suppression_keeps_largest(mock_show):
    # Two overlapping boxes: large and small
    mask = make_box_mask((200, 200), (10, 10, 80, 80), (20, 20, 40, 40))
    labeled, bboxes = apply_strict_bboxes_with_top_cutoff_and_proximity(
        mask, mask, iou_threshold=0.01, size_threshold=10, top_y_cutoff=0
    )
    # Only one box (the largest) should survive IoU suppression
    assert len(bboxes) == 1
    x_min, y_min, x_max, y_max, *_ = bboxes[0]
    assert (x_min, y_min, x_max, y_max) == (10, 10, 80, 80)


@patch("axonrooter.utils.bboxes.plt.show")
def test_x_proximity_removes_smaller_box(mock_show):
    # Two boxes close in X, one much larger
    mask1 = make_box_mask((200, 200), (10, 60, 40, 100))
    mask2 = make_box_mask((200, 200), (45, 60, 60, 100))
    mask = mask1 + mask2
    labeled, bboxes = apply_strict_bboxes_with_top_cutoff_and_proximity(
        mask, mask, min_x_distance=10, size_threshold=10, top_y_cutoff=0
    )
    # Only the larger should be kept
    assert len(bboxes) == 1
    x_min, y_min, x_max, y_max, *_ = bboxes[0]
    assert (x_min, y_min, x_max, y_max) == (10, 60, 40, 100)


@patch("axonrooter.utils.bboxes.plt.show")
def test_empty_mask_returns_empty_labeled_and_no_bboxes(mock_show):
    mask = np.zeros((100, 100), dtype=np.uint8)
    labeled, bboxes = apply_strict_bboxes_with_top_cutoff_and_proximity(mask, mask)
    assert labeled.shape == mask.shape
    assert np.all(labeled == 0)
    assert bboxes == []
