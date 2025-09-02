import numpy as np
import cv2
import pytest
from axonrooter.data.preprocessing import crop_initial, find_edges, format_crop, padding_function

import matplotlib
matplotlib.use("Agg")


#def test_crop_initial_basic():
 #   img = np.arange(200*200).reshape(200, 200).astype(np.uint8)
  #  result = crop_initial(img)
   # if isinstance(result, tuple):
   #     cropped = result[0]
   # else:
   #     cropped = result
   # print("DEBUG cropped.shape:", cropped.shape)
   # assert cropped.shape == (150, 100)


def test_crop_initial_too_large():
    img = np.arange(50*50).reshape(50, 50).astype(np.uint8)
    cropped = crop_initial(img, initial_crop=30)
    # Cropping removes too much; shape should have zero or negative dimension
    assert not isinstance(cropped, np.ndarray) or cropped.size == 0


def test_find_edges_simple_upward_transition():
    line = np.array([0, 0, 10, 80, 255])
    left, right = find_edges(line, threshold=70)
    assert left == 3
    assert right == 3  # Only one upward transition

def test_find_edges_no_transition():
    line = np.array([10, 20, 30, 40])
    left, right = find_edges(line, threshold=100)
    assert left is None
    assert right is None

def test_format_crop_square_crop():
    # Create a bright square in a dark image
    img = np.zeros((100, 100), dtype=np.uint8)
    img[30:70, 30:70] = 255
    cropped = format_crop(img)
    if isinstance(cropped, tuple):
        cropped = cropped[0]
    assert cropped.shape[0] == cropped.shape[1]
    assert np.all(cropped == 255)
    assert isinstance(cropped, np.ndarray)

'''def test_format_crop_non_square_content():
    img = np.zeros((120, 100), dtype=np.uint8)
    img[20:90, 40:90] = 200
    cropped = format_crop(img)
    assert cropped.shape[0] == cropped.shape[1]
    assert np.all(cropped == 200)'''

'''def test_format_crop_empty_image():
    img = np.zeros((100, 100), dtype=np.uint8)
    cropped = format_crop(img)
    assert cropped.shape[0] == cropped.shape[1]
    assert np.all(cropped == 0)'''

def test_padding_function_regular():
    img = np.ones((120, 120), dtype=np.uint8) * 127
    padded_image, pad_top, pad_bottom, pad_left, pad_right = padding_function(img, PATCH_SIZE=64)
    assert padded_image.shape[0] % 64 == 0
    assert padded_image.shape[1] % 64 == 0

def test_padding_function_small_image():
    img = np.zeros((30, 30), dtype=np.uint8)
    padded_image, pad_top, pad_bottom, pad_left, pad_right = padding_function(img, PATCH_SIZE=64)
    assert padded_image.shape == (64, 64)

def test_padding_function_zero_image():
    img = np.zeros((0, 0), dtype=np.uint8)
    padded_image, pad_top, pad_bottom, pad_left, pad_right = padding_function(img, PATCH_SIZE=64)
    # Should either remain empty or be properly padded
    assert padded_image.size == 0 or padded_image.shape == (64, 64)
