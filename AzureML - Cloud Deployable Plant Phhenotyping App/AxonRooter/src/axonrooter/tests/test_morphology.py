import numpy as np
from axonrooter.utils.morphology import preprocess_single_mask

import matplotlib
matplotlib.use("Agg")

def test_closing_fills_hole():
    # Create a binary mask with a hole in the center
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 1
    mask[40:60, 40:60] = 0  # hole

    closed = preprocess_single_mask(mask, kernel_size=(21, 21))
    # The center should now be filled
    assert closed[50, 50] == 1
    assert closed.shape == mask.shape
    assert closed.dtype == np.uint8

def test_small_mask_no_error():
    # Should work on very small images
    mask = np.zeros((8, 8), dtype=np.uint8)
    closed = preprocess_single_mask(mask, kernel_size=(3, 3))
    assert closed.shape == mask.shape

def test_mask_all_zeros():
    mask = np.zeros((100, 100), dtype=np.uint8)
    closed = preprocess_single_mask(mask)
    assert np.all(closed == 0)

def test_mask_all_ones():
    mask = np.ones((50, 50), dtype=np.uint8)
    closed = preprocess_single_mask(mask)
    assert np.all(closed == 1)

def test_non_binary_mask():
    mask = np.full((50, 50), 128, dtype=np.uint8)
    closed = preprocess_single_mask(mask)
    assert closed.dtype == np.uint8
    assert closed.shape == mask.shape

def test_empty_mask():
    mask = np.zeros((0, 0), dtype=np.uint8)
    closed = preprocess_single_mask(mask)
    assert closed.shape == (0, 0)

def test_non_square_mask():
    mask = np.zeros((30, 50), dtype=np.uint8)
    closed = preprocess_single_mask(mask)
    assert closed.shape == (30, 50)

def test_mask_with_noise():
    rng = np.random.RandomState(42)
    mask = (rng.rand(100, 100) > 0.7).astype(np.uint8)
    closed = preprocess_single_mask(mask)
    assert closed.shape == mask.shape
